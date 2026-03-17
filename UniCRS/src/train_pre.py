# train_pre.py
import argparse
import math
import os
import sys
import time
import json
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

from dataset_dbpedia import DBpedia
from dataset_pre import CRSDataset, CRSDataCollator
from evaluate_rec import RecEvaluator
from model_gpt2 import PromptGPT2forCRS
from config import gpt2_special_tokens_dict
from model_prompt import KGPrompt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, default='save', help="Where to store the final model.")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
    # data
    parser.add_argument("--dataset", type=str, required=True, help="A file containing all data.")
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument("--max_length", type=int, help="max input length in dataset.")
    parser.add_argument("--entity_max_length", type=int, help="max entity length in dataset.")
    parser.add_argument("--tokenizer", type=str)
    # model
    parser.add_argument("--model", type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--num_bases", type=int, default=8, help="num_bases in RGCN.")
    # optim
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--num_warmup_steps', type=int, default=10000)
    parser.add_argument('--mixed_precision', type=str, default='no', choices=['no', 'fp16', 'bf16'])
    # wandb
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--entity", type=str)
    parser.add_argument("--project", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--log_all", action="store_true")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    config = vars(args)

    # Initialize accelerator
    accelerator = Accelerator(device_placement=False, mixed_precision=args.mixed_precision)
    device = accelerator.device

    # Setup logging
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logger.remove()
    logger.add(sys.stderr, level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.add(f'log/{local_time}.log', level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.info(config)
    logger.info(accelerator.state)

    if args.use_wandb:
        name = args.name if args.name else local_time
        name += '_' + str(accelerator.process_index)

        if args.log_all:
            group = args.name if args.name else 'DDP_' + local_time
            run = wandb.init(entity=args.entity, project=args.project, group=group, config=config, name=name)
        else:
            if accelerator.is_local_main_process:
                run = wandb.init(entity=args.entity, project=args.project, config=config, name=name)
            else:
                run = None
    else:
        run = None

    # Set seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load knowledge graph
    kg = DBpedia(dataset=args.dataset, debug=args.debug).get_entity_kg_info()

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)
    
    model = PromptGPT2forCRS.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)

    # Initialize prompt encoder with RGCN
    prompt_encoder = KGPrompt(
        model.config.n_embd, model.config.n_embd, model.config.n_head, model.config.n_layer, 2,
        n_entity=kg['num_entities'], num_relations=kg['num_relations'], num_bases=args.num_bases,
        edge_index=kg['edge_index'], edge_type=kg['edge_type']
    ).to(device)

    # Freeze the base model
    model.requires_grad_(False)

    # Set up optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in prompt_encoder.named_parameters()
                      if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in prompt_encoder.named_parameters()
                      if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Initialize datasets
    train_dataset = CRSDataset(
        dataset=args.dataset, split='train', tokenizer=tokenizer, debug=args.debug,
        max_length=args.max_length, entity_max_length=args.entity_max_length
    )
    valid_dataset = CRSDataset(
        dataset=args.dataset, split='valid', tokenizer=tokenizer, debug=args.debug,
        max_length=args.max_length, entity_max_length=args.entity_max_length
    )
    test_dataset = CRSDataset(
        dataset=args.dataset, split='test', tokenizer=tokenizer, debug=args.debug,
        max_length=args.max_length, entity_max_length=args.entity_max_length
    )

    # Initialize data collator
    data_collator = CRSDataCollator(
        tokenizer=tokenizer, device=device, pad_entity_id=kg['pad_entity_id'],
        max_length=args.max_length, entity_max_length=args.entity_max_length,
        use_amp=args.mixed_precision == 'fp16', debug=args.debug
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator,
        shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
    )

    evaluator = RecEvaluator()

    # Prepare with accelerator
    prompt_encoder, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        prompt_encoder, optimizer, train_dataloader, valid_dataloader, test_dataloader
    )

    # Training setup
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, args.num_warmup_steps, args.max_train_steps
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)

    # Training loop
    completed_steps = 0
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    metric, mode = 'loss', -1
    best_metric = float('inf') if mode == -1 else 0
    best_metric_dir = os.path.join(args.output_dir, 'best')
    os.makedirs(best_metric_dir, exist_ok=True)

    for epoch in range(args.num_train_epochs):
        train_loss = []
        prompt_encoder.train()

        for step, batch in enumerate(train_dataloader):
            # Generate embeddings using only RGCN
            prompt_embeds = prompt_encoder(
                entity_ids=batch['entity'],
                output_entity=True
            )
            
            # Prepare batch
            batch['context']['prompt_embeds'] = prompt_embeds
            batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()

            # Forward pass
            loss = model(**batch['context'], rec=True).rec_loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            train_loss.append(float(loss))

            # Optimization step
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(prompt_encoder.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                completed_steps += 1
                if run:
                    run.log({'loss': np.mean(train_loss) * args.gradient_accumulation_steps})

            if completed_steps >= args.max_train_steps:
                break

        # Validation loop
        valid_loss = []
        prompt_encoder.eval()
        for batch in tqdm(valid_dataloader):
            with torch.no_grad():
                prompt_embeds = prompt_encoder(
                    entity_ids=batch['entity'],
                    output_entity=True
                )
                batch['context']['prompt_embeds'] = prompt_embeds
                batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()

                outputs = model(**batch['context'], rec=True)
                valid_loss.append(float(outputs.rec_loss))
                logits = outputs.rec_logits
                ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
                labels = batch['context']['rec_labels']
                evaluator.evaluate(ranks, labels)

        # Report metrics
        report = accelerator.gather(evaluator.report())
        valid_report = {}
        for k, v in report.items():
            if k != 'count':
                valid_report[f'valid/{k}'] = v.sum().item() / report['count'].sum().item()
        valid_report['valid/loss'] = np.mean(valid_loss)
        valid_report['epoch'] = epoch
        logger.info(f'{valid_report}')
        if run:
            run.log(valid_report)
        evaluator.reset_metric()

        # Save best model
        if valid_report[f'valid/{metric}'] * mode > best_metric * mode:
            prompt_encoder.save(best_metric_dir)
            best_metric = valid_report[f'valid/{metric}']
            logger.info(f'new best model with {metric}')

        # Test evaluation
        test_loss = []
        prompt_encoder.eval()
        for batch in tqdm(test_dataloader):
            with torch.no_grad():
                prompt_embeds = prompt_encoder(
                    entity_ids=batch['entity'],
                    output_entity=True
                )
                batch['context']['prompt_embeds'] = prompt_embeds
                batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()

                outputs = model(**batch['context'], rec=True)
                test_loss.append(float(outputs.rec_loss))
                logits = outputs.rec_logits
                ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
                labels = batch['context']['rec_labels']
                evaluator.evaluate(ranks, labels)

        # Report test metrics
        report = accelerator.gather(evaluator.report())
        test_report = {}
        for k, v in report.items():
            if k != 'count':
                test_report[f'test/{k}'] = v.sum().item() / report['count'].sum().item()
        test_report['test/loss'] = np.mean(test_loss)
        test_report['epoch'] = epoch
        logger.info(f'{test_report}')
        if run:
            run.log(test_report)
        evaluator.reset_metric()

    # Save final model
    final_dir = os.path.join(args.output_dir, 'final')
    os.makedirs(final_dir, exist_ok=True)
    prompt_encoder.save(final_dir)
    logger.info(f'save final model')