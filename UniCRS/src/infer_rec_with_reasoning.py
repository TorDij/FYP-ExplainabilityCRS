"""
Inference script for recommendation model with KECR-style anchored KG path explanation.

Key change from previous version:
Instead of finding KG paths independently and reranking, we now:
1. Get top-5 predictions from the recommendation module
2. Convert predicted IDs to movie names
3. Find the best KG path FROM any mentioned entity TO any predicted movie
4. This gives a faithful explanation — the path is anchored to what was actually recommended
"""
import argparse
import json
import os
import re
import sys
import torch
import numpy as np
from tqdm import tqdm
from loguru import logger

sys.path.insert(0, '../../Recommendation_GraphRAG')

from transformers import AutoTokenizer
from config import gpt2_special_tokens_dict
from dataset_dbpedia import DBpedia
from dataset_rec import CRSRecDataset, CRSRecDataCollator
from model_gpt2 import PromptGPT2forCRS
from model_prompt import KGPrompt
from community_prompt_enhancer import CommunityRecommenderPromptEnhancer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="redial_gen")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--tokenizer", type=str, default="microsoft/DialoGPT-small")
    parser.add_argument("--model", type=str, default="microsoft/DialoGPT-small")
    parser.add_argument("--prompt_encoder", type=str, required=True)
    parser.add_argument("--num_bases", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--output_file", type=str, default="./results/reasoning_analysis.json")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--embedding_weight", type=float, default=0.7,
                        help="Weight for RGCN embedding similarity vs edge weight in path scoring")
    parser.add_argument("--top_k_explain", type=int, default=5,
                        help="Number of top predictions to attempt path explanation for")
    return parser.parse_args()


def extract_user_context_embed(prompt_encoder, batch_entity_ids, device):
    """
    Extract a user-level context embedding from the RGCN prompt encoder.
    Takes the mean of RGCN embeddings for entities mentioned in this dialogue turn.
    """
    with torch.no_grad():
        all_entity_embeds = prompt_encoder.get_entity_embeds()
        batch_size = batch_entity_ids.shape[0]
        pad_id = all_entity_embeds.shape[0] - 1

        user_embeds = []
        for i in range(batch_size):
            entity_ids = batch_entity_ids[i]
            valid_mask = entity_ids != pad_id
            valid_ids = entity_ids[valid_mask]

            if len(valid_ids) > 0:
                embeds = all_entity_embeds[valid_ids]
                user_embed = embeds.mean(dim=0)
            else:
                user_embed = torch.zeros(all_entity_embeds.shape[1], device=device)

            user_embeds.append(user_embed)

        return torch.stack(user_embeds, dim=0)


def extract_mentioned_movies(dialogue_text: str, entity_name_to_id: dict) -> list:
    """
    Extract movie names mentioned in the dialogue by matching against
    known entity names from entity2id.json.

    Returns a list of lowercase movie name strings found in the dialogue.
    """
    if not dialogue_text:
        return []

    dialogue_lower = dialogue_text.lower()
    mentioned = []

    for name in entity_name_to_id:
        if re.search(r'\(\d{4}\)', name) and name in dialogue_lower:
            mentioned.append(name)

    # Deduplicate preserving order
    seen = set()
    unique = []
    for m in mentioned:
        if m not in seen:
            seen.add(m)
            unique.append(m)

    return unique

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading models and data...")

    # Load KG
    kg = DBpedia(dataset=args.dataset, debug=False).get_entity_kg_info()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)
    model = PromptGPT2forCRS.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)
    model.eval()

    # Load prompt encoder
    prompt_encoder = KGPrompt(
        model.config.n_embd, model.config.n_embd, model.config.n_head, model.config.n_layer, 2,
        n_entity=kg['num_entities'], num_relations=kg['num_relations'], num_bases=args.num_bases,
        edge_index=kg['edge_index'], edge_type=kg['edge_type'],
        n_prefix_rec=10
    )
    prompt_encoder.load(args.prompt_encoder)
    prompt_encoder = prompt_encoder.to(device)
    prompt_encoder.eval()

    # Load enhancer (which contains KGPathReasoner internally)
    enhancer = CommunityRecommenderPromptEnhancer(use_path_reasoning=True)

    # Set RGCN entity embeddings on the path reasoner
    if enhancer.use_path_reasoning and enhancer.path_reasoner is not None:
        logger.info("Setting RGCN entity embeddings on KGPathReasoner...")
        with torch.no_grad():
            rgcn_embeds = prompt_encoder.get_entity_embeds()
        enhancer.path_reasoner.set_entity_embeddings(rgcn_embeds)
        logger.info(f"RGCN embeddings set: {rgcn_embeds.shape}")
    else:
        logger.warning("Path reasoning not available")

    # Load complete entity2id mapping covering both GraphRAG and DBpedia entities
    entity2id_path = '/home/torrey/Project-UniCRSxGraphRAG/UniCRS/src/data/redial/entity2id.json'
    with open(entity2id_path) as f:
        raw_entity2id = json.load(f)
    # Normalise keys to lowercase for matching
    entity_name_to_id = {k.lower(): v for k, v in raw_entity2id.items()}
    # Reverse mapping: id -> lowercase name
    id_to_entity_name = {v: k.lower() for k, v in raw_entity2id.items()}

    logger.info(f"Loaded entity2id: {len(entity_name_to_id)} entities")

    # Load dataset
    test_dataset = CRSRecDataset(
        dataset=args.dataset, split=args.split, debug=False,
        tokenizer=tokenizer, context_max_length=200, use_resp=False,
        entity_max_length=32,
    )

    if args.max_samples and args.max_samples < len(test_dataset):
        logger.info(f"Limiting to first {args.max_samples} samples")
        test_dataset = torch.utils.data.Subset(test_dataset, range(args.max_samples))

    data_collator = CRSRecDataCollator(
        tokenizer=tokenizer, device=device, debug=False,
        context_max_length=200, entity_max_length=32,
        pad_entity_id=kg['pad_entity_id'],
        use_amp=False,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
    )

    logger.info(f"Analyzing {len(test_dataset)} samples...")
    logger.info(f"Embedding weight: {args.embedding_weight}")
    logger.info(f"Explaining top-{args.top_k_explain} predictions")

    all_results = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch_size = batch['entity'].shape[0]
            dialogues = batch.get("dialogue", [""] * batch_size)

            # Extract user context embeddings from RGCN
            user_context_embeds = extract_user_context_embed(
                prompt_encoder, batch['entity'], device
            )

            # Get community-enhanced relationship embeddings
            relationship_embeddings_list = []
            for idx, dialogue_text in enumerate(dialogues):
                relationship_emb = enhancer.get_enhanced_rec_prompt(
                    step=len(all_results) + idx,
                    dialogue_text=dialogue_text
                )
                relationship_embeddings_list.append(relationship_emb)

            if all(emb is not None for emb in relationship_embeddings_list):
                relationship_embeddings_list = [
                    emb.to(device) for emb in relationship_embeddings_list
                ]
                relationship_embeddings = torch.stack(relationship_embeddings_list)
            else:
                relationship_embeddings = None

            # Get prompt embeddings
            prompt_embeds = prompt_encoder(
                entity_ids=batch['entity'],
                output_entity=True,
                use_rec_prefix=True,
                relationship_embeddings=relationship_embeddings
            )

            batch['context']['prompt_embeds'] = prompt_embeds
            batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()

            # Get model predictions
            outputs = model(**batch['context'], rec=True)
            logits = outputs.rec_logits[:, kg['item_ids']]
            top_k = torch.topk(logits, k=50, dim=-1)

            for i in range(batch_size):
                top_indices = top_k.indices[i].cpu().tolist()
                top_items = [kg['item_ids'][idx] for idx in top_indices]

                ground_truth_idx = batch['context']['rec_labels'][i].item()
                if ground_truth_idx < 0 or ground_truth_idx >= len(kg['item_ids']):
                    continue
                ground_truth = kg['item_ids'][ground_truth_idx]

                dialogue_text = dialogues[i] if i < len(dialogues) else ""

                # ---------------------------------------------------------------
                # KECR-style anchored explanation
                # Step 1: Get top-5 predicted movie names from recommendation module
                # ---------------------------------------------------------------
                top_5_ids = top_items[:args.top_k_explain]
                top_5_names = [
                    id_to_entity_name[item_id]
                    for item_id in top_5_ids
                    if item_id in id_to_entity_name
                ]

                # Step 2: Extract movies mentioned in dialogue
                mentioned_movies = extract_mentioned_movies(dialogue_text, entity_name_to_id)

                # Step 3: Find best KG path FROM any mentioned movie TO any predicted movie
                explanation_path = None
                if (mentioned_movies and top_5_names and
                        enhancer.path_reasoner is not None):
                    user_embed = user_context_embeds[i]
                    explanation_path = enhancer.path_reasoner.find_best_explanation_path(
                        mentioned_entities=mentioned_movies,
                        top_predicted_names=top_5_names,
                        max_hops=2,
                        user_context_embed=user_embed,
                        embedding_weight=args.embedding_weight
                    )

                # Step 4: Build result
                path_found = explanation_path is not None

                # Strip bulky GraphRAG fields before saving
                if path_found:
                    explanation_path = {
                        k: v for k, v in explanation_path.items()
                        if k not in ('target_entity', 'path_details')
                    }

                result = {
                    'sample_idx': len(all_results),
                    'dialogue': dialogue_text[:500],
                    'ground_truth_id': ground_truth,
                    'ground_truth_name': id_to_entity_name.get(ground_truth, ''),
                    # Recommendation module outputs (unchanged)
                    'top_1_prediction': top_items[0],
                    'top_1_prediction_name': id_to_entity_name.get(top_items[0], ''),
                    'top_10_predictions': top_items[:10],
                    'top_50_predictions': top_items[:50],
                    # Correctness
                    'is_correct_top_1': top_items[0] == ground_truth,
                    'is_correct_top_10': ground_truth in top_items[:10],
                    'is_correct_top_50': ground_truth in top_items[:50],
                    # Explanation path
                    'path_found': path_found,
                    'path_note': '' if path_found else 'No KG path found between any mentioned entity and top-5 predictions',
                    'mentioned_movies': mentioned_movies,
                    'top_5_predicted_names': top_5_names,
                    'explanation_path': explanation_path if path_found else [],
                    'scoring_method': (
                        explanation_path.get('scoring_method', 'weight')
                        if path_found else 'none'
                    ),
                }

                all_results.append(result)

    # Statistics
    total = len(all_results)
    paths_found = sum(1 for r in all_results if r['path_found'])
    stats = {
        'total_samples': total,
        'embedding_weight': args.embedding_weight,
        'top_k_explain': args.top_k_explain,
        # Recommendation performance (unchanged from base model)
        'recall@1': sum(1 for r in all_results if r['is_correct_top_1']) / total * 100,
        'recall@10': sum(1 for r in all_results if r['is_correct_top_10']) / total * 100,
        'recall@50': sum(1 for r in all_results if r['is_correct_top_50']) / total * 100,
        # Explanation coverage
        'paths_found': paths_found,
        'path_coverage': paths_found / total * 100,
        'avg_mentioned_movies': sum(len(r['mentioned_movies']) for r in all_results) / total,
        # Path coverage on correct predictions specifically
        'paths_found_on_correct_top10': sum(
            1 for r in all_results if r['path_found'] and r['is_correct_top_10']
        ),
        'path_coverage_on_correct_top10': sum(
            1 for r in all_results if r['path_found'] and r['is_correct_top_10']
        ) / max(sum(1 for r in all_results if r['is_correct_top_10']), 1) * 100,
    }

    # Sample results: correct top-10 predictions with explanation paths
    samples_with_explanation = [
        r for r in all_results
        if r['path_found'] and r['is_correct_top_10']
    ][:10]

    output = {
        'statistics': stats,
        'sample_explanations': samples_with_explanation,
        'all_results': all_results
    }

    os.makedirs(
        os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else '.',
        exist_ok=True
    )
    with open(args.output_file, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    logger.info(f"\n{'='*80}")
    logger.info("KECR-STYLE ANCHORED EXPLANATION RESULTS")
    logger.info(f"{'='*80}")
    logger.info(f"Total samples: {stats['total_samples']}")
    logger.info(f"\n--- Recommendation Performance ---")
    logger.info(f"Recall@1:   {stats['recall@1']:.2f}%")
    logger.info(f"Recall@10:  {stats['recall@10']:.2f}%")
    logger.info(f"Recall@50:  {stats['recall@50']:.2f}%")
    logger.info(f"\n--- Explanation Coverage ---")
    logger.info(f"Paths found: {stats['paths_found']} / {total} ({stats['path_coverage']:.2f}%)")
    logger.info(f"Avg mentioned movies per sample: {stats['avg_mentioned_movies']:.2f}")
    logger.info(f"Path coverage on correct@10 predictions: {stats['path_coverage_on_correct_top10']:.2f}%")
    logger.info(f"\nResults saved to: {args.output_file}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()