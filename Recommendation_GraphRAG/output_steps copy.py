import os
import sys
import time
from loguru import logger
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from UniCRS.src.config import gpt2_special_tokens_dict
from UniCRS.src.dataset_dbpedia import DBpedia
from UniCRS.src.dataset_rec_copy import CRSRecDataset, CRSRecDataCollator
from UniCRS.src.model_gpt2 import PromptGPT2forCRS


# Setup basic logging
local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
logger.remove()
logger.add(sys.stderr, level='DEBUG')
logger.add(f'log/{local_time}.log', level='DEBUG')

# Create output directories
os.makedirs("dialogue_outputs/train", exist_ok=True)
os.makedirs("dialogue_outputs/valid", exist_ok=True)
os.makedirs("dialogue_outputs/test", exist_ok=True)

# Dataset settings
dataset_name = "redial_gen"
debug_mode = False
context_max_length = 200
entity_max_length = 32
use_resp = False

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
tokenizer.add_special_tokens(gpt2_special_tokens_dict)

# Initialize KG
kg = DBpedia(dataset=dataset_name, debug=debug_mode).get_entity_kg_info()

# Create datasets
train_dataset = CRSRecDataset(
    dataset=dataset_name, split='train', debug=debug_mode,
    tokenizer=tokenizer, context_max_length=context_max_length, use_resp=use_resp,
    entity_max_length=entity_max_length,
)
valid_dataset = CRSRecDataset(
    dataset=dataset_name, split='valid', debug=debug_mode,
    tokenizer=tokenizer, context_max_length=context_max_length, use_resp=use_resp,
    entity_max_length=entity_max_length,
)
test_dataset = CRSRecDataset(
    dataset=dataset_name, split='test', debug=debug_mode,
    tokenizer=tokenizer, context_max_length=context_max_length, use_resp=use_resp,
    entity_max_length=entity_max_length,
)

# Data collator
data_collator = CRSRecDataCollator(
    tokenizer=tokenizer, device="cpu", debug=debug_mode,
    context_max_length=context_max_length, entity_max_length=entity_max_length,
    pad_entity_id=kg['pad_entity_id']
)

# Create dataloaders
train_dataloader = DataLoader(
    train_dataset,
    batch_size=64,  # Set to 1 to process dialogues one at a time
    collate_fn=data_collator,
    shuffle=False  # No need to shuffle for dialogue extraction
)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=64,
    collate_fn=data_collator,
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=64,
    collate_fn=data_collator,
)

logger.info("***** Extracting dialogues *****")
logger.info(f"Train examples: {len(train_dataset)}")
logger.info(f"Valid examples: {len(valid_dataset)}")
logger.info(f"Test examples: {len(test_dataset)}")

# Process train set
logger.info("Processing training set...")
for step, batch in enumerate(tqdm(train_dataloader)):
    # dialogue_text = " ".join(batch["dialogue"])
    print(step, batch)
    if step > 2:
        break

#     output_path = f"dialogue_outputs/train/train_step{step}.txt"
#     with open(output_path, 'w', encoding='utf-8') as f:
#         f.write(dialogue_text)

# # Process validation set
# logger.info("Processing validation set...")
# for step, batch in enumerate(tqdm(valid_dataloader)):
#     dialogue_text = " ".join(batch["dialogue"])
#     output_path = f"dialogue_outputs/valid/valid_step{step}.txt"
#     with open(output_path, 'w', encoding='utf-8') as f:
#         f.write(dialogue_text)

# # Process test set
# logger.info("Processing test set...")
# for step, batch in enumerate(tqdm(test_dataloader)):
#     dialogue_text = " ".join(batch["dialogue"])
#     output_path = f"dialogue_outputs/test/test_step{step}.txt"
#     with open(output_path, 'w', encoding='utf-8') as f:
#         f.write(dialogue_text)

# logger.info("Dialogue extraction complete!")