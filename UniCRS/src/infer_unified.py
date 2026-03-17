import argparse
import os
import sys
import json

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from loguru import logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from config import gpt2_special_tokens_dict
from dataset_dbpedia import DBpedia
from dataset_conv import CRSConvDataset, CRSConvDataCollator
from dataset_rec import CRSRecDataset, CRSRecDataCollator
from evaluate_conv import ConvEvaluator
from evaluate_rec import RecEvaluator
from model_gpt2 import PromptGPT2forCRS
from model_prompt import KGPrompt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible inference.")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
    
    # data
    parser.add_argument("--dataset", type=str, required=True, help="A file containing all data.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"], help="Dataset split to use.")
    parser.add_argument("--context_max_length", type=int, help="max input length in dataset.")
    parser.add_argument("--resp_max_length", type=int, help="max response length.")
    parser.add_argument("--prompt_max_length", type=int)
    parser.add_argument("--entity_max_length", type=int, help="max entity length in dataset.")
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--output_file", type=str, help="Where to store the inference results.")
    parser.add_argument("--ignore_pad_token_for_loss", action='store_true')
    
    # model
    parser.add_argument("--model", type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--num_bases", type=int, default=8, help="num_bases in RGCN.")
    parser.add_argument("--n_prefix_rec", type=int)
    parser.add_argument("--n_prefix_conv", type=int)
    parser.add_argument("--prompt_encoder", type=str)
    parser.add_argument("--max_gen_len", type=int, default=50)
    
    # batch size
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="Batch size (per device) for the evaluation dataloader.")
    
    # evaluation mode
    parser.add_argument("--eval_rec", action="store_true", help="Evaluate recommendation task.")
    parser.add_argument("--eval_conv", action="store_true", help="Evaluate conversation task.")
    parser.add_argument('--mixed_precision', type=str, default='no', choices=['no', 'fp16', 'bf16'])

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = vars(args)

    # Initialize the accelerator
    accelerator = Accelerator(device_placement=False, mixed_precision=args.mixed_precision)
    device = accelerator.device

    # Set up logging
    logger.remove()
    logger.add(sys.stderr, level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.info(config)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    # Set seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)

    # Create output directory if needed
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Get knowledge graph information
    kg = DBpedia(dataset=args.dataset, debug=args.debug).get_entity_kg_info()

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)
    model = PromptGPT2forCRS.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)

    # Initialize KGPrompt encoder
    prompt_encoder = KGPrompt(
        model.config.n_embd, model.config.n_embd, model.config.n_head, model.config.n_layer, 2,
        n_entity=kg['num_entities'], num_relations=kg['num_relations'], num_bases=args.num_bases,
        edge_index=kg['edge_index'], edge_type=kg['edge_type'],
        n_prefix_rec=args.n_prefix_rec, n_prefix_conv=args.n_prefix_conv
    )
    
    if args.prompt_encoder is not None:
        prompt_encoder.load(args.prompt_encoder)
    prompt_encoder = prompt_encoder.to(device)

    # Set model to evaluation mode
    model.eval()
    prompt_encoder.eval()

    results = {}
    
    # Evaluate recommendation task if requested
    if args.eval_rec:
        logger.info("Evaluating recommendation task...")
        
        # Initialize dataset and collator
        rec_dataset = CRSRecDataset(
            dataset=args.dataset, split=args.split, debug=args.debug,
            tokenizer=tokenizer, context_max_length=args.context_max_length, 
            entity_max_length=args.entity_max_length,
        )
        
        rec_collator = CRSRecDataCollator(
            tokenizer=tokenizer, device=device, debug=args.debug,
            context_max_length=args.context_max_length, 
            entity_max_length=args.entity_max_length,
            pad_entity_id=kg['pad_entity_id']
        )
        
        rec_dataloader = DataLoader(
            rec_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=rec_collator,
        )
        
        # Initialize evaluator
        rec_evaluator = RecEvaluator()
        
        # Prepare with accelerator
        rec_dataloader = accelerator.prepare(rec_dataloader)
        
        rec_predictions = []
        rec_references = []
        
        # Inference loop
        for batch in tqdm(rec_dataloader, disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                # Generate unified prompt embeddings
                unified_prompt_embeds = prompt_encoder(
                    entity_ids=batch['entity'],
                    output_entity=True,
                    use_rec_prefix=True,
                    use_conv_prefix=True
                )
                
                batch['context']['prompt_embeds'] = unified_prompt_embeds
                batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()
                
                outputs = model(**batch['context'], rec=True)
                
                # Get recommendations
                logits = outputs.rec_logits[:, kg['item_ids']]
                ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
                ranks = [[kg['item_ids'][rank] for rank in batch_rank] for batch_rank in ranks]
                labels = batch['context']['rec_labels']
                
                # Save predictions and references
                for pred, ref in zip(ranks, labels):
                    rec_predictions.append({'prediction': pred, 'top_1': pred[0] if pred else None})
                    rec_references.append({'reference': ref})
                
                # Evaluate
                rec_evaluator.evaluate(ranks, labels)
        
        # Calculate metrics
        rec_report = accelerator.gather(rec_evaluator.report())
        rec_metrics = {}
        for k, v in rec_report.items():
            if k != 'count':
                rec_metrics[f'rec_{k}'] = v.sum().item() / rec_report['count'].sum().item()
        
        logger.info(f"Recommendation metrics: {rec_metrics}")
        results['rec_metrics'] = rec_metrics
        results['rec_predictions'] = rec_predictions
        results['rec_references'] = rec_references
    
    # Evaluate conversation task if requested
    if args.eval_conv:
        logger.info("Evaluating conversation task...")
        
        # Initialize dataset and collator
        conv_dataset = CRSConvDataset(
            args.dataset, args.split, tokenizer, debug=args.debug,
            context_max_length=args.context_max_length, 
            resp_max_length=args.resp_max_length,
            entity_max_length=args.entity_max_length
        )
        
        conv_gen_collator = CRSConvDataCollator(
            tokenizer=tokenizer, 
            device=device, 
            pad_entity_id=kg['pad_entity_id'],
            ignore_pad_token_for_loss=args.ignore_pad_token_for_loss,
            context_max_length=args.context_max_length,
            resp_max_length=args.resp_max_length,
            entity_max_length=args.entity_max_length,
            use_amp=accelerator.use_fp16,
            debug=args.debug,
            gen=True
        )
        
        conv_dataloader = DataLoader(
            conv_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=conv_gen_collator,
        )
        
        # Initialize evaluator with output file
        gen_file_path = args.output_file if args.output_file else 'log/generation_output.jsonl'
        conv_evaluator = ConvEvaluator(tokenizer=tokenizer, log_file_path=gen_file_path)
        
        # Prepare with accelerator
        conv_dataloader = accelerator.prepare(conv_dataloader)
        
        conv_predictions = []
        conv_references = []
        
        # Inference loop
        for batch in tqdm(conv_dataloader, disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                # Generate unified prompt embeddings
                unified_prompt_embeds = prompt_encoder(
                    entity_ids=batch['entity'],
                    output_entity=False,
                    use_rec_prefix=True,
                    use_conv_prefix=True
                )
                
                batch['context']['prompt_embeds'] = unified_prompt_embeds
                
                # Generate responses
                gen_seqs = accelerator.unwrap_model(model).generate(
                    **batch['context'],
                    max_new_tokens=args.max_gen_len,
                    no_repeat_ngram_size=3,
                )
                
                gen_resp_ids = []
                for gen_seq, length in zip(gen_seqs, batch['context_len']):
                    gen_seq = [token_id for token_id in gen_seq if token_id != tokenizer.pad_token_id]
                    gen_resp_ids.append(gen_seq[length:])
                
                # Decode generated responses and references
                decoded_preds = tokenizer.batch_decode(gen_resp_ids, skip_special_tokens=False)
                decoded_preds = [decoded_pred.replace('<pad>', '').replace('<|endoftext|>', '') for decoded_pred in
                               decoded_preds]
                decoded_preds = [pred.strip() for pred in decoded_preds]
                
                decoded_labels = tokenizer.batch_decode(batch['resp'], skip_special_tokens=False)
                decoded_labels = [decoded_label.replace('<pad>', '').replace('<|endoftext|>', '') for decoded_label in
                                decoded_labels]
                decoded_labels = [label.strip() for label in decoded_labels]
                
                # Save predictions and references
                for pred, ref in zip(decoded_preds, decoded_labels):
                    conv_predictions.append({'prediction': pred})
                    conv_references.append({'reference': ref})
                
                # Evaluate
                conv_evaluator.evaluate(gen_resp_ids, batch['resp'], log=accelerator.is_local_main_process)
        
        # Calculate metrics
        conv_report = conv_evaluator.report()
        conv_metrics = {}
        for k, v in conv_report.items():
            conv_metrics[f'conv_{k}'] = v
        
        logger.info(f"Conversation metrics: {conv_metrics}")
        results['conv_metrics'] = conv_metrics
        results['conv_predictions'] = conv_predictions
        results['conv_references'] = conv_references
    
    # Save results to file if specified
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output_file}")