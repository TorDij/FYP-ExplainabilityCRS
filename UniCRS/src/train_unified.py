import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import transformers
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from loguru import logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer

from dataset_conv import CRSConvDataset, CRSConvDataCollator
from dataset_rec import CRSRecDataset, CRSRecDataCollator
from dataset_dbpedia import DBpedia
from evaluate_conv import ConvEvaluator
from evaluate_rec import RecEvaluator
from model_gpt2 import PromptGPT2forCRS
from model_prompt import KGPrompt
from community_prompt_enhancer import CommunityRecommenderPromptEnhancer

def parse_args():
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store the final model and logs.")
    parser.add_argument("--debug", action='store_true', help="Activate debug mode (use smaller dataset samples).")
    # Data
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name or path containing all data.")
    parser.add_argument("--context_max_length", type=int, help="Max length of dialogue context input.")
    parser.add_argument("--resp_max_length", type=int, help="Max length of response output for conversation task.")
    parser.add_argument("--entity_max_length", type=int, help="Max length of entity sequence.")
    parser.add_argument("--tokenizer", type=str, required=True, help="Tokenizer name or path (e.g., GPT-2 tokenizer).")
    # Model
    parser.add_argument("--model", type=str, required=True,
                        help="Path to pretrained PromptGPT2forCRS model or model identifier from huggingface.")
    parser.add_argument("--text_encoder", type=str, help="(Optional) Path to text encoder model (if using).")
    parser.add_argument("--prompt_encoder", type=str, help="(Optional) Path to a pretrained KGPrompt to load.")
    parser.add_argument("--n_prefix_conv", type=int, default=5, help="Number of prefix tokens for conversation prompt.")
    parser.add_argument("--n_prefix_rec", type=int, default=5, help="Number of prefix tokens for recommendation prompt.")
    parser.add_argument("--num_bases", type=int, default=8, help="Number of bases in RGCN for KG encoding.")
    # Optimization
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total training steps (overrides num_train_epochs if set).")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Train batch size per device.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Eval batch size per device.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Steps to accumulate gradients before updating.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer.")
    parser.add_argument("--num_warmup_steps", type=int, default=10000, help="Warmup steps for scheduler.")
    parser.add_argument("--max_grad_norm", type=float, default=None, help="Max gradient norm for clipping.")
    parser.add_argument("--mixed_precision", type=str, default='no', choices=['no', 'fp16', 'bf16'],
                        help="Mixed precision mode (fp16 or bf16) for acceleration.")
    parser.add_argument('--num_workers', type=int, default=0)
    # WandB logging
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for experiment tracking.")
    parser.add_argument("--entity", type=str, help="WandB entity/account name.")
    parser.add_argument("--project", type=str, help="WandB project name.")
    parser.add_argument("--name", type=str, help="WandB run name.")
    parser.add_argument("--log_all", action="store_true", help="Log from all processes (useful in multi-GPU setting).")
    # Enhancer usage
    parser.add_argument("--use_enhancer", action="store_true", help="Enable CommunityRecommenderPromptEnhancer for recommendations.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    config = vars(args)

    # Initialize accelerator
    accelerator = Accelerator(device_placement=False, mixed_precision=args.mixed_precision)
    device = accelerator.device

    # Setup logging
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logger.remove()
    # Only main process logs debug info; others log errors
    logger.add(sys.stderr, level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.add(f'log/{local_time}.log', level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.info(config)

    # Setup WandB run if requested&#8203;:contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}
    if args.use_wandb:
        if args.log_all:
            group = args.name if args.name else 'DDP_' + local_time
            run = wandb.init(entity=args.entity, project=args.project, group=group, config=config, name=args.name)
        else:
            if accelerator.is_local_main_process:
                run = wandb.init(entity=args.entity, project=args.project, config=config, name=args.name)
            else:
                run = None
    else:
        run = None

    # Set seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Create output directory if not exist
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load knowledge graph data&#8203;:contentReference[oaicite:2]{index=2}
    kg = DBpedia(dataset=args.dataset, debug=args.debug).get_entity_kg_info()

    # Initialize tokenizer and base model&#8203;:contentReference[oaicite:3]{index=3}
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    from config import gpt2_special_tokens_dict, prompt_special_tokens_dict
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)
    model = PromptGPT2forCRS.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)

    # Initialize KG prompt encoder (KGPrompt)&#8203;:contentReference[oaicite:4]{index=4}
    prompt_encoder = KGPrompt(
        hidden_size=model.config.n_embd,
        token_hidden_size=model.config.n_embd,
        n_head=model.config.n_head,
        n_layer=model.config.n_layer,
        n_block=2,
        n_entity=kg['num_entities'],
        num_relations=kg['num_relations'],
        num_bases=args.num_bases,
        edge_index=kg['edge_index'],
        edge_type=kg['edge_type'],
        n_prefix_rec=args.n_prefix_rec,
        n_prefix_conv=args.n_prefix_conv
    ).to(device)
    if args.prompt_encoder:
        prompt_encoder.load(args.prompt_encoder)
    # Freeze the base language model parameters (only train prompt_encoder)&#8203;:contentReference[oaicite:5]{index=5}
    model.requires_grad_(False)

    # Prepare optimizer for prompt_encoder parameters
    modules = [prompt_encoder]
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for module in modules for n, p in module.named_parameters()
                       if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for module in modules for n, p in module.named_parameters()
                       if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Load datasets for conversation and recommendation tasks
    train_conv_dataset = CRSConvDataset(
        dataset=args.dataset, split='train', tokenizer=tokenizer, debug=args.debug,
        context_max_length=args.context_max_length, resp_max_length=args.resp_max_length,
        entity_max_length=args.entity_max_length
    )
    valid_conv_dataset = CRSConvDataset(
        dataset=args.dataset, split='valid', tokenizer=tokenizer, debug=args.debug,
        context_max_length=args.context_max_length, resp_max_length=args.resp_max_length,
        entity_max_length=args.entity_max_length
    )
    test_conv_dataset = CRSConvDataset(
        dataset=args.dataset, split='test', tokenizer=tokenizer, debug=args.debug,
        context_max_length=args.context_max_length, resp_max_length=args.resp_max_length,
        entity_max_length=args.entity_max_length
    )
    train_rec_dataset = CRSRecDataset(
        dataset=args.dataset, split='train', tokenizer=tokenizer, debug=args.debug,
        context_max_length=args.context_max_length, entity_max_length=args.entity_max_length
    )
    valid_rec_dataset = CRSRecDataset(
        dataset=args.dataset, split='valid', tokenizer=tokenizer, debug=args.debug,
        context_max_length=args.context_max_length, entity_max_length=args.entity_max_length
    )
    test_rec_dataset = CRSRecDataset(
        dataset=args.dataset, split='test', tokenizer=tokenizer, debug=args.debug,
        context_max_length=args.context_max_length, entity_max_length=args.entity_max_length
    )

    # Initialize data collators
    conv_collator = CRSConvDataCollator(
        tokenizer=tokenizer, device=device, pad_entity_id=kg['pad_entity_id'],
        ignore_pad_token_for_loss=True,  # for conv, pad tokens should be ignored in loss
        context_max_length=(args.context_max_length or tokenizer.model_max_length) + (args.resp_max_length or 0),
        resp_max_length=args.resp_max_length, entity_max_length=args.entity_max_length,
        use_amp=(args.mixed_precision == 'fp16'), debug=args.debug, gen=False
    )
    rec_collator = CRSRecDataCollator(
        tokenizer=tokenizer, device=device, pad_entity_id=kg['pad_entity_id'],
        context_max_length=args.context_max_length, entity_max_length=args.entity_max_length,
        use_amp=(args.mixed_precision == 'fp16'), debug=args.debug
    )

    # Prepare data loaders
    train_conv_dataloader = DataLoader(
        train_conv_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=0 if args.debug else args.num_workers,
        collate_fn=conv_collator
    )
    train_rec_dataloader = DataLoader(
        train_rec_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=0 if args.debug else args.num_workers,
        collate_fn=rec_collator
    )
    valid_conv_dataloader = DataLoader(
        valid_conv_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=conv_collator
    )
    valid_rec_dataloader = DataLoader(
        valid_rec_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=rec_collator
    )
    test_conv_dataloader = DataLoader(
        test_conv_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=conv_collator
    )
    test_rec_dataloader = DataLoader(
        test_rec_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=rec_collator
    )

    # Prepare model, prompt_encoder, optimizer, and dataloaders with accelerator
    model, prompt_encoder, optimizer, train_conv_dataloader, train_rec_dataloader, valid_conv_dataloader, valid_rec_dataloader, test_conv_dataloader, test_rec_dataloader = accelerator.prepare(
        model, prompt_encoder, optimizer,
        train_conv_dataloader, train_rec_dataloader,
        valid_conv_dataloader, valid_rec_dataloader,
        test_conv_dataloader, test_rec_dataloader
    )
    device = accelerator.device  # Update device if needed (accelerator handles device placement)

    # Set up learning rate scheduler
    num_update_steps_per_epoch = math.ceil((len(train_conv_dataloader) + len(train_rec_dataloader)) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=args.max_train_steps
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)

    # Initialize evaluators for conversation and recommendation
    conv_evaluator = ConvEvaluator(tokenizer, log_file_path=f'log/{local_time}_conv_eval.log')
    rec_evaluator = RecEvaluator(device=device)

    # Initialize CommunityRecommenderPromptEnhancer if using enhanced prompts
    enhancer = CommunityRecommenderPromptEnhancer() if args.use_enhancer else None

    # Training loop
    logger.info("***** Running unified training *****")
    logger.info(f"  Num conversation examples = {len(train_conv_dataset)}")
    logger.info(f"  Num recommendation examples = {len(train_rec_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    best_metric = float('inf')  # Track best validation combined loss (lower is better)
    best_metric_dir = os.path.join(args.output_dir, 'best')
    os.makedirs(best_metric_dir, exist_ok=True)

    for epoch in range(args.num_train_epochs):
        prompt_encoder.train()
        # Initialize trackers for losses
        conv_train_losses = []
        rec_train_losses = []
        # Use iterators to interleave conv and rec training steps
        conv_iter = iter(train_conv_dataloader)
        rec_iter = iter(train_rec_dataloader)
        more_conv = more_rec = True
        # Gradient accumulation counter
        accum_step_count = 0

        while more_conv or more_rec:
            # 1. Conversation task training step (if available)
            if more_conv:
                try:
                    conv_batch = next(conv_iter)
                except StopIteration:
                    more_conv = False
                    conv_batch = None
                if conv_batch is not None:
                    # Generate conversation prompt embeddings
                    conv_prompt_embeds = prompt_encoder(
                        entity_ids=conv_batch['entity'],
                        output_entity=False,
                        use_conv_prefix=True
                    )
                    conv_batch['context']['prompt_embeds'] = conv_prompt_embeds
                    # Forward pass for conversation (with labels)
                    conv_loss = model(**conv_batch['context'], conv=True, conv_labels=conv_batch['resp']).conv_loss
                    conv_loss = conv_loss / args.gradient_accumulation_steps
                    accelerator.backward(conv_loss)
                    conv_train_losses.append(float(conv_loss))
                    accum_step_count += 1
            # 2. Recommendation task training step (if available)
            if more_rec:
                try:
                    rec_batch = next(rec_iter)
                except StopIteration:
                    more_rec = False
                    rec_batch = None
                if rec_batch is not None:
                    # Prepare dialogue text for enhancer
                    relationship_embeddings = None
                    if enhancer is not None:
                        # Use combined dialogue text (ground-truth conversation) for community search
                        dialogue_text = " ".join(rec_batch["dialogue"])
                        try:
                            relationship_embeddings = enhancer.get_enhanced_rec_prompt(completed_steps, dialogue_text)
                        except Exception as e:
                            logger.error(f"Enhancer error at step {completed_steps}: {e}")
                            relationship_embeddings = None
                        # Ensure tensor on correct device
                        if relationship_embeddings is not None:
                            relationship_embeddings = relationship_embeddings.to(device)
                    # Generate recommendation prompt embeddings with possible enhanced prompt
                    rec_prompt_embeds = prompt_encoder(
                        entity_ids=rec_batch['entity'],
                        output_entity=True,
                        use_rec_prefix=True,
                        relationship_embeddings=relationship_embeddings
                    )
                    rec_batch['context']['prompt_embeds'] = rec_prompt_embeds
                    rec_batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()
                    # Forward pass for recommendation (with labels)
                    rec_output = model(**rec_batch['context'], rec=True)
                    rec_loss = rec_output.rec_loss
                    rec_loss = rec_loss / args.gradient_accumulation_steps
                    accelerator.backward(rec_loss)
                    rec_train_losses.append(float(rec_loss))
                    accum_step_count += 1
            # Perform optimizer step if gradient accumulation is fulfilled or end of epoch
            if accum_step_count % args.gradient_accumulation_steps == 0 or (not more_conv and not more_rec):
                if args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(prompt_encoder.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                accum_step_count = 0  # reset counter after update

                progress_bar.update(1)
                completed_steps += 1
                # Log training loss to wandb (combined loss if both present)
                if run:
                    # Compute mean of accumulated conv and rec losses since last step
                    avg_conv_loss = np.mean(conv_train_losses) * args.gradient_accumulation_steps if conv_train_losses else 0.0
                    avg_rec_loss = np.mean(rec_train_losses) * args.gradient_accumulation_steps if rec_train_losses else 0.0
                    # Log the latest values (not resetting lists here to keep running mean)
                    metrics = {}
                    if conv_batch is not None:
                        metrics['train/conv_loss'] = float(avg_conv_loss)
                    if rec_batch is not None:
                        metrics['train/rec_loss'] = float(avg_rec_loss)
                    if metrics:
                        run.log(metrics)

                # Stop if we've reached the total training steps
                if completed_steps >= args.max_train_steps:
                    # Drain any remaining data without updating (to finish epoch loops)
                    more_conv = more_rec = False

        # End of epoch: evaluate on validation sets
        prompt_encoder.eval()
        # Conversation validation (loss and generation metrics)
        valid_conv_losses = []
        for batch in tqdm(valid_conv_dataloader, disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                prompt_embeds = prompt_encoder(
                    entity_ids=batch['entity'],
                    output_entity=False,
                    use_conv_prefix=True
                )
                batch['context']['prompt_embeds'] = prompt_embeds
                conv_loss = model(**batch['context'], conv=True, conv_labels=batch['resp']).conv_loss
                valid_conv_losses.append(float(conv_loss))
        # Conversation generation for evaluation metrics (e.g., BLEU, Distinct)
        conv_evaluator.log_file.write(f'\n\n*** valid-{conv_evaluator.log_cnt} ***\n\n')
        for batch in tqdm(valid_conv_dataloader, disable=not accelerator.is_local_main_process):
            # Note: we use a separate DataLoader for generation if needed (here reuse with caution)
            with torch.no_grad():
                prompt_embeds = prompt_encoder(
                    entity_ids=batch['entity'],
                    output_entity=False,
                    use_conv_prefix=True
                )
                batch['context']['prompt_embeds'] = prompt_embeds
                gen_seqs = accelerator.unwrap_model(model).generate(
                    **batch['context'],
                    max_new_tokens=args.resp_max_length or 50,  # generate up to resp_max_length tokens
                    no_repeat_ngram_size=3
                )
                # Post-process generated sequence to extract the response part
                gen_resp_ids = []
                for gen_seq, context_len in zip(gen_seqs, batch.get('context_len', [0]*len(gen_seqs))):
                    gen_seq = [token_id for token_id in gen_seq.tolist() if token_id != tokenizer.pad_token_id]
                    if 'context_len' in batch:
                        # If using generation mode collator, 'context_len' provided actual context length (excluding prompt)
                        resp_start = context_len
                    else:
                        # If no context length, fall back to finding the first 'System:' token of response
                        resp_start = 0
                        for idx in range(len(gen_seq)):
                            # Identify where the response likely starts (after the prompt "System:" token)
                            if tokenizer.decode([gen_seq[idx]]).strip() == 'System:':
                                resp_start = idx
                                break
                    gen_resp_ids.append(gen_seq[resp_start:])
                conv_evaluator.evaluate(gen_resp_ids, batch['resp'], log=accelerator.is_local_main_process)
        # Compile conversation metrics
        conv_metrics = conv_evaluator.report()
        conv_evaluator.reset_metric()
        conv_evaluator.log_cnt += 1

        # Recommendation validation (loss and ranking metrics)
        valid_rec_losses = []
        for batch in tqdm(valid_rec_dataloader, disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                relationship_embeddings = None
                if enhancer is not None:
                    # Use enhancer on validation dialogues as well (if used in training)
                    dialogue_text = " ".join(batch["dialogue"])
                    try:
                        relationship_embeddings = enhancer.get_enhanced_rec_prompt(-1, dialogue_text)
                    except Exception as e:
                        relationship_embeddings = None
                prompt_embeds = prompt_encoder(
                    entity_ids=batch['entity'],
                    output_entity=True,
                    use_rec_prefix=True,
                    relationship_embeddings=relationship_embeddings
                )
                batch['context']['prompt_embeds'] = prompt_embeds
                batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()
                outputs = model(**batch['context'], rec=True)
                valid_rec_losses.append(float(outputs.rec_loss))
                logits = outputs.rec_logits
                # Compute top-k indices and evaluate ranking metrics
                ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
                labels = batch['context']['rec_labels']
                rec_evaluator.evaluate(ranks, labels)
        rec_metrics = rec_evaluator.report()
        rec_evaluator.reset_metric()

        # Aggregate validation metrics for logging
        valid_loss_conv = np.mean(valid_conv_losses) if valid_conv_losses else 0.0
        valid_loss_rec = np.mean(valid_rec_losses) if valid_rec_losses else 0.0
        # Combine or average losses for selection metric
        combined_valid_loss = valid_loss_conv + valid_loss_rec
        valid_report = {}
        # Add conversation metrics to report
        for k, v in conv_metrics.items():
            # Skip 'sent_cnt' (count) or internal metrics not relevant
            if k == 'sent_cnt':
                continue
            valid_report[f'valid/{k}'] = v if isinstance(v, float) else v  # already normalized in ConvEvaluator.report()
        # Add recommendation metrics to report
        for k, v in rec_metrics.items():
            if k == 'count':
                continue
            # Average the metric over all samples
            if isinstance(v, torch.Tensor):
                v = v.sum().item() / rec_metrics['count'].sum().item()
            valid_report[f'valid/{k}'] = v
        # Add losses
        valid_report['valid/conv_loss'] = valid_loss_conv
        valid_report['valid/rec_loss'] = valid_loss_rec
        valid_report['epoch'] = epoch
        logger.info(f"Epoch {epoch} validation: {valid_report}")
        if run:
            run.log(valid_report)

        # Save best model if combined validation loss improved
        if combined_valid_loss < best_metric:
            best_metric = combined_valid_loss
            prompt_encoder.save(best_metric_dir)
            logger.info(f"New best model saved at epoch {epoch} with combined valid loss {best_metric:.4f}")

        # Test evaluation at end of each epoch (optional, can be done after training)
        prompt_encoder.eval()
        test_conv_losses = []
        for batch in tqdm(test_conv_dataloader, disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                prompt_embeds = prompt_encoder(
                    entity_ids=batch['entity'],
                    output_entity=False,
                    use_conv_prefix=True
                )
                batch['context']['prompt_embeds'] = prompt_embeds
                conv_loss = model(**batch['context'], conv=True, conv_labels=batch['resp']).conv_loss
                test_conv_losses.append(float(conv_loss))
        conv_evaluator.log_file.write(f'\n\n*** test-{conv_evaluator.log_cnt} ***\n\n')
        for batch in tqdm(test_conv_dataloader, disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                prompt_embeds = prompt_encoder(
                    entity_ids=batch['entity'],
                    output_entity=False,
                    use_conv_prefix=True
                )
                batch['context']['prompt_embeds'] = prompt_embeds
                gen_seqs = accelerator.unwrap_model(model).generate(
                    **batch['context'],
                    max_new_tokens=args.resp_max_length or 50,
                    no_repeat_ngram_size=3
                )
                gen_resp_ids = []
                for gen_seq, context_len in zip(gen_seqs, batch.get('context_len', [0]*len(gen_seqs))):
                    gen_seq = [token_id for token_id in gen_seq.tolist() if token_id != tokenizer.pad_token_id]
                    if 'context_len' in batch:
                        resp_start = context_len
                    else:
                        resp_start = 0
                        for idx in range(len(gen_seq)):
                            if tokenizer.decode([gen_seq[idx]]).strip() == 'System:':
                                resp_start = idx
                                break
                    gen_resp_ids.append(gen_seq[resp_start:])
                conv_evaluator.evaluate(gen_resp_ids, batch['resp'], log=accelerator.is_local_main_process)
        test_conv_metrics = conv_evaluator.report()
        conv_evaluator.reset_metric()
        conv_evaluator.log_cnt += 1

        test_rec_losses = []
        for batch in tqdm(test_rec_dataloader, disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                relationship_embeddings = None
                if enhancer is not None:
                    dialogue_text = " ".join(batch["dialogue"])
                    try:
                        relationship_embeddings = enhancer.get_enhanced_rec_prompt(-1, dialogue_text)
                    except Exception:
                        relationship_embeddings = None
                prompt_embeds = prompt_encoder(
                    entity_ids=batch['entity'],
                    output_entity=True,
                    use_rec_prefix=True,
                    relationship_embeddings=relationship_embeddings
                )
                batch['context']['prompt_embeds'] = prompt_embeds
                batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()
                outputs = model(**batch['context'], rec=True)
                test_rec_losses.append(float(outputs.rec_loss))
                logits = outputs.rec_logits
                ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
                labels = batch['context']['rec_labels']
                rec_evaluator.evaluate(ranks, labels)
        test_rec_metrics = rec_evaluator.report()
        rec_evaluator.reset_metric()

        test_loss_conv = np.mean(test_conv_losses) if test_conv_losses else 0.0
        test_loss_rec = np.mean(test_rec_losses) if test_rec_losses else 0.0
        test_report = {}
        for k, v in test_conv_metrics.items():
            if k == 'sent_cnt':
                continue
            test_report[f'test/{k}'] = v
        for k, v in test_rec_metrics.items():
            if k == 'count':
                continue
            if isinstance(v, torch.Tensor):
                v = v.sum().item() / test_rec_metrics['count'].sum().item()
            test_report[f'test/{k}'] = v
        test_report['test/conv_loss'] = test_loss_conv
        test_report['test/rec_loss'] = test_loss_rec
        test_report['epoch'] = epoch
        logger.info(f"Epoch {epoch} test: {test_report}")
        if run:
            run.log(test_report)

    # Save final model prompt encoder
    final_dir = os.path.join(args.output_dir, 'final')
    os.makedirs(final_dir, exist_ok=True)
    prompt_encoder.save(final_dir)
    logger.info("Saved final prompt encoder model.")
    if run:
        run.finish()
