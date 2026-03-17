# train_rec.py
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
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel

from config import gpt2_special_tokens_dict, prompt_special_tokens_dict
from dataset_dbpedia import DBpedia
from dataset_rec import CRSRecDataset, CRSRecDataCollator
from evaluate_rec import RecEvaluator
from model_gpt2 import PromptGPT2forCRS
from model_prompt import KGPrompt

# sys.path.append('/home/Nema/UniCRS_GraphRAG') 
# sys.path.append('/home/Nema/UniCRS_GraphRAG/Recommendation_GraphRAG') 
# from UniCRS_GraphRAG.Recommendation_GraphRAG.pre_rec_util import RecommenderPromptEnhancer
sys.path.insert(0, '/home/Nema/UniCRS_GraphRAG/Recommendation_GraphRAG')
from pre_rec_util import RecommenderPromptEnhancer
from community_prompt_enhancer import CommunityRecommenderPromptEnhancer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, default='save', help="Where to store the final model.")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
    # data
    parser.add_argument("--dataset", type=str, required=True, help="A file containing all data.")
    parser.add_argument("--shot", type=float, default=1)
    parser.add_argument("--use_resp", action="store_true")
    parser.add_argument("--context_max_length", type=int, help="max input length in dataset.")
    parser.add_argument("--prompt_max_length", type=int)
    parser.add_argument("--entity_max_length", type=int, help="max entity length in dataset.")
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--text_tokenizer", type=str)
    # model
    parser.add_argument("--model", type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--text_encoder", type=str)
    parser.add_argument("--num_bases", type=int, default=8, help="num_bases in RGCN.")
    parser.add_argument("--n_prefix_rec", type=int)
    parser.add_argument("--prompt_encoder", type=str)
    # optim
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--num_warmup_steps', type=int)
    #parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--mixed_precision', type=str, default='no', choices=['no', 'fp16', 'bf16'])
    # wandb
    parser.add_argument("--use_wandb", action="store_true", help="whether to use wandb")
    parser.add_argument("--entity", type=str, help="wandb username")
    parser.add_argument("--project", type=str, help="wandb exp project")
    parser.add_argument("--name", type=str, help="wandb exp name")
    parser.add_argument("--log_all", action="store_true", help="log in all processes, otherwise only in rank0")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = vars(args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us.
    #accelerator = Accelerator(device_placement=False, fp16=args.fp16)
    accelerator = Accelerator(device_placement=False, mixed_precision=args.mixed_precision)
    device = accelerator.device

    # Make one log on every process with the configuration for debugging.
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logger.remove()
    logger.add(sys.stderr, level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.add(f'log/{local_time}.log', level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.info(accelerator.state)
    logger.info(config)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
    # wandb
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

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    kg = DBpedia(dataset=args.dataset, debug=args.debug).get_entity_kg_info()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)
    model = PromptGPT2forCRS.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)

    # text_tokenizer = AutoTokenizer.from_pretrained(args.text_tokenizer)
    # text_tokenizer.add_special_tokens(prompt_special_tokens_dict)
    # text_encoder = AutoModel.from_pretrained(args.text_encoder)
    # text_encoder.resize_token_embeddings(len(text_tokenizer))
    # text_encoder = text_encoder.to(device)

    prompt_encoder = KGPrompt(
    model.config.n_embd, model.config.n_embd, model.config.n_head, model.config.n_layer, 2,
    n_entity=kg['num_entities'], num_relations=kg['num_relations'], num_bases=args.num_bases,
    edge_index=kg['edge_index'], edge_type=kg['edge_type'],
    n_prefix_rec=args.n_prefix_rec
    )
    if args.prompt_encoder is not None:
        prompt_encoder.load(args.prompt_encoder)
    prompt_encoder = prompt_encoder.to(device)

    fix_modules = [model]
    for module in fix_modules:
        module.requires_grad_(False)

    # optim & amp
    modules = [prompt_encoder]
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for model in modules for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for model in modules for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    # data
    # CHANGE CRSDATASET TO ALSO TIE IN THE KG WITH THE OUTPUT
    train_dataset = CRSRecDataset(
    dataset=args.dataset, split='train', debug=args.debug,
    tokenizer=tokenizer, context_max_length=args.context_max_length, use_resp=args.use_resp,
    entity_max_length=args.entity_max_length,
    )
    shot_len = int(len(train_dataset) * args.shot)
    train_dataset = random_split(train_dataset, [shot_len, len(train_dataset) - shot_len])[0]
    assert len(train_dataset) == shot_len
    valid_dataset = CRSRecDataset(
    dataset=args.dataset, split='valid', debug=args.debug,
    tokenizer=tokenizer, context_max_length=args.context_max_length, use_resp=args.use_resp,
    entity_max_length=args.entity_max_length,
)
    test_dataset = CRSRecDataset(
    dataset=args.dataset, split='test', debug=args.debug,
    tokenizer=tokenizer, context_max_length=args.context_max_length, use_resp=args.use_resp,
    entity_max_length=args.entity_max_length,
)
    data_collator = CRSRecDataCollator(
    tokenizer=tokenizer, device=device, debug=args.debug,
    context_max_length=args.context_max_length, entity_max_length=args.entity_max_length,
    pad_entity_id=kg['pad_entity_id']
)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator,
        shuffle=False
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
    prompt_encoder, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        prompt_encoder, optimizer, train_dataloader, valid_dataloader, test_dataloader
    )
    # step, epoch, batch size
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    completed_steps = 0
    # lr_scheduler
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.num_warmup_steps, args.max_train_steps)
    lr_scheduler = accelerator.prepare(lr_scheduler)
    # training info
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    # save model with best metric
    metric, mode = 'loss', -1
    assert mode in (-1, 1)
    if mode == 1:
        best_metric = 0
    else:
        best_metric = float('inf')
    best_metric_dir = os.path.join(args.output_dir, 'best')
    os.makedirs(best_metric_dir, exist_ok=True)

    # recommenderPromptEnhancer = RecommenderPromptEnhancer()
    # enhancer = CommunityRecommenderPromptEnhancer()
    # # Create output directories
    # os.makedirs("/home/Nema/UniCRS_GraphRAG/Recommendation_GraphRAG/dialogue_outputs/train", exist_ok=True)
    # os.makedirs("/home/Nema/UniCRS_GraphRAG/Recommendation_GraphRAG/dialogue_outputs/valid", exist_ok=True)
    # os.makedirs("/home/Nema/UniCRS_GraphRAG/Recommendation_GraphRAG/dialogue_outputs/test", exist_ok=True)
    # # train loop
    # logger.info("Processing training set...")
    # for step, batch in enumerate(train_dataloader):
    #     # --------------- ADDED FOR ENHANCED REC PROMPT ----------------------
    #     dialogue_text = " ".join(batch["dialogue"])  # Handle list to string
    #     print(f"\n[DEBUG] Step {step}: Extracted Dialogue - {dialogue_text}\n")
    #     output_path = f"/home/Nema/UniCRS_GraphRAG/Recommendation_GraphRAG/dialogue_outputs/train/train_step{step}.txt"
    #     with open(output_path, 'w', encoding='utf-8') as f:
    #         f.write(dialogue_text)

    # # Process validation set
    # logger.info("Processing validation set...")
    # for step, batch in enumerate(tqdm(valid_dataloader)):
    #     dialogue_text = " ".join(batch["dialogue"])
    #     output_path = f"/home/Nema/UniCRS_GraphRAG/Recommendation_GraphRAG/dialogue_outputs/valid/valid_step{step}.txt"
    #     with open(output_path, 'w', encoding='utf-8') as f:
    #         f.write(dialogue_text)

    # # Process test set
    # logger.info("Processing test set...")
    # for step, batch in enumerate(tqdm(test_dataloader)):
    #     dialogue_text = " ".join(batch["dialogue"])
    #     output_path = f"/home/Nema/UniCRS_GraphRAG/Recommendation_GraphRAG/dialogue_outputs/test/test_step{step}.txt"
    #     with open(output_path, 'w', encoding='utf-8') as f:
    #         f.write(dialogue_text)

logger.info("Dialogue extraction complete!")
