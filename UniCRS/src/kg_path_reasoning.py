import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import os
import torch
import torch.nn.functional as F


# Entity types that receive an informativeness bonus during path scoring.
# Actor and director connections produce more specific, informative explanations
# than genre connections (Zhang & Chen, 2020), so paths through these nodes
# are preferred when their score is otherwise comparable.
INFORMATIVE_ENTITY_TYPES = {'ACTOR', 'DIRECTOR'}
DEFAULT_ENTITY_TYPE_BONUS = 0.15  # additive bonus applied to hop_score


class KGPathReasoner:
    def __init__(self, entities_path: str, relationships_path: str):
        print("Loading KG data...")
        self.entities = pd.read_parquet(entities_path)
        self.relationships = pd.read_parquet(relationships_path)

        self.entity_dict = {
            row['title'].lower(): row.to_dict()
            for _, row in self.entities.iterrows()
        }
        self.entity_name_to_id = {
            row['title'].lower(): int(row['human_readable_id'])
            for _, row in self.entities.iterrows()
        }
        self.entity_id_to_name = {
            int(row['human_readable_id']): row['title'].lower()
            for _, row in self.entities.iterrows()
        }
        # Build entity type lookup for bonus application
        self.entity_type = {
            row['title'].lower(): str(row.get('type', '')).upper()
            for _, row in self.entities.iterrows()
        }
        self._build_relationship_index()
        self.rgcn_entity_embeds = None
        print(f"Loaded {len(self.entities)} entities and {len(self.relationships)} relationships")

    def _build_relationship_index(self):
        self.neighbors = {}
        for _, row in self.relationships.iterrows():
            source = row['source'].lower()
            target = row['target'].lower()
            if source not in self.neighbors:
                self.neighbors[source] = []
            self.neighbors[source].append({
                'target': target,
                'description': row.get('description', ''),
                'weight': float(row.get('weight', 1.0)),
                'id': row.get('id', '')
            })

    def set_entity_embeddings(self, entity_embeds: torch.Tensor):
        self.rgcn_entity_embeds = entity_embeds.detach().cpu()
        print(f"Set RGCN entity embeddings: {entity_embeds.shape}")

    def _get_embedding_score(self, entity_name: str, user_context_embed: torch.Tensor) -> float:
        if self.rgcn_entity_embeds is None:
            return 1.0
        entity_id = self.entity_name_to_id.get(entity_name)
        if entity_id is None or entity_id >= self.rgcn_entity_embeds.shape[0]:
            return 0.0
        entity_embed = self.rgcn_entity_embeds[entity_id]
        sim = F.cosine_similarity(
            entity_embed.unsqueeze(0),
            user_context_embed.unsqueeze(0)
        ).item()
        return (sim + 1.0) / 2.0

    def _get_entity_type_bonus(self, entity_name: str,
                                bonus: float = DEFAULT_ENTITY_TYPE_BONUS) -> float:
        """
        Return additive bonus if entity is ACTOR or DIRECTOR.
        These intermediate node types produce more informative explanations
        than genre nodes (Zhang & Chen, 2020).
        """
        entity_type = self.entity_type.get(entity_name, '')
        return bonus if entity_type in INFORMATIVE_ENTITY_TYPES else 0.0

    def find_entity(self, entity_name: str) -> Optional[Dict]:
        entity_name_lower = entity_name.lower().strip()
        if entity_name_lower in self.entity_dict:
            return self.entity_dict[entity_name_lower]
        for key, entity in self.entity_dict.items():
            if entity_name_lower in key or key in entity_name_lower:
                return entity
        return None

    def get_neighbors(self, entity_name: str) -> List[Dict]:
        return self.neighbors.get(entity_name.lower(), [])

    def find_path(self, start_entity, target_type="MOVIE", max_hops=2,
                  top_k=5, length_penalty=0.85, min_weight=1.0):
        start_entity_obj = self.find_entity(start_entity)
        if not start_entity_obj:
            return []
        start_entity_lower = start_entity_obj['title'].lower()
        paths = []
        visited = set()
        queue = [([start_entity_lower], [], 1.0)]
        for hop in range(max_hops):
            new_queue = []
            for path, path_details, raw_score in queue:
                current = path[-1]
                for neighbor in self.get_neighbors(current):
                    target = neighbor['target']
                    weight = neighbor['weight']
                    if weight < min_weight or target in path:
                        continue
                    sig = tuple(path + [target])
                    if sig in visited:
                        continue
                    visited.add(sig)
                    new_path = path + [target]
                    new_details = path_details + [(current, neighbor['description'], weight)]
                    new_raw = raw_score * weight
                    new_score = new_raw * (length_penalty ** len(new_path))
                    te = self.entity_dict.get(target)
                    if te and te.get('type') == target_type:
                        paths.append({'path': new_path, 'path_details': new_details,
                                      'target': target, 'score': new_score,
                                      'raw_score': new_raw, 'length': len(new_path),
                                      'target_entity': te, 'scoring_method': 'weight'})
                    elif hop < max_hops - 1:
                        new_queue.append((new_path, new_details, new_raw))
            queue = new_queue
        paths.sort(key=lambda x: x['score'], reverse=True)
        seen = set()
        return [p for p in paths if p['target'] not in seen and not seen.add(p['target'])][:top_k]

    def find_path_with_embeddings(self, start_entity, user_context_embed,
                                   target_type="MOVIE", max_hops=2, top_k=5,
                                   length_penalty=0.85, embedding_weight=0.7,
                                   entity_type_bonus=DEFAULT_ENTITY_TYPE_BONUS):
        if self.rgcn_entity_embeds is None:
            return self.find_path(start_entity, target_type, max_hops, top_k, length_penalty)
        start_obj = self.find_entity(start_entity)
        if not start_obj:
            return []
        start_lower = start_obj['title'].lower()
        user_context_embed = user_context_embed.detach().cpu().float()
        paths = []
        visited = set()
        queue = [([start_lower], [], 1.0)]
        for hop in range(max_hops):
            new_queue = []
            for path, path_details, raw_score in queue:
                current = path[-1]
                neighbors = self.get_neighbors(current)
                if not neighbors:
                    continue
                max_weight = max(n['weight'] for n in neighbors) or 1.0
                for neighbor in neighbors:
                    target = neighbor['target']
                    edge_weight = neighbor['weight']
                    if edge_weight < 1.0 or target in path:
                        continue
                    sig = tuple(path + [target])
                    if sig in visited:
                        continue
                    visited.add(sig)
                    embed_sim = self._get_embedding_score(target, user_context_embed)
                    norm_w = edge_weight / max_weight
                    type_bonus = self._get_entity_type_bonus(target, entity_type_bonus)
                    hop_score = (embedding_weight * embed_sim +
                                 (1 - embedding_weight) * norm_w +
                                 type_bonus)
                    new_path = path + [target]
                    new_details = path_details + [(current, neighbor['description'], edge_weight, embed_sim)]
                    new_raw = raw_score * hop_score
                    new_score = new_raw * (length_penalty ** len(new_path))
                    te = self.entity_dict.get(target)
                    if te and te.get('type') == target_type:
                        paths.append({'path': new_path, 'path_details': new_details,
                                      'target': target, 'score': new_score,
                                      'raw_score': new_raw, 'length': len(new_path),
                                      'target_entity': te, 'scoring_method': 'embedding'})
                    elif hop < max_hops - 1:
                        new_queue.append((new_path, new_details, new_raw))
            queue = new_queue
        paths.sort(key=lambda x: x['score'], reverse=True)
        seen = set()
        return [p for p in paths if p['target'] not in seen and not seen.add(p['target'])][:top_k]

    def find_path_to_target(self, start_entity, target_entity, max_hops=2,
                             length_penalty=0.85, user_context_embed=None,
                             embedding_weight=0.7,
                             entity_type_bonus=DEFAULT_ENTITY_TYPE_BONUS):
        """
        KECR-style anchored path finding. Only returns a path if it reaches
        the specific target_entity.

        Actor/director intermediate nodes receive an additive bonus
        (entity_type_bonus) to favour more informative explanation paths
        over generic genre connections (Zhang & Chen, 2020).
        The bonus is only applied to intermediate nodes, never to the
        final target which must always be a MOVIE.
        """
        start_obj = self.find_entity(start_entity)
        if not start_obj:
            return None
        start_lower = start_obj['title'].lower()

        target_obj = self.find_entity(target_entity)
        if not target_obj:
            return None
        target_lower = target_obj['title'].lower()

        if start_lower == target_lower:
            return None

        use_embeddings = (user_context_embed is not None and
                          self.rgcn_entity_embeds is not None)
        if use_embeddings:
            user_context_embed = user_context_embed.detach().cpu().float()

        matching_paths = []
        visited = set()
        queue = [([start_lower], [], 1.0)]

        for hop in range(max_hops):
            new_queue = []
            for path, path_details, raw_score in queue:
                current = path[-1]
                neighbors = self.get_neighbors(current)
                if not neighbors:
                    continue
                max_weight = max(n['weight'] for n in neighbors) or 1.0

                for neighbor in neighbors:
                    nb_target = neighbor['target']
                    edge_weight = neighbor['weight']

                    if edge_weight < 1.0 or nb_target in path:
                        continue
                    sig = tuple(path + [nb_target])
                    if sig in visited:
                        continue
                    visited.add(sig)

                    # Bonus applies to intermediate nodes only, not the final target
                    is_intermediate = (nb_target != target_lower)
                    type_bonus = (self._get_entity_type_bonus(nb_target, entity_type_bonus)
                                  if is_intermediate else 0.0)

                    if use_embeddings:
                        embed_sim = self._get_embedding_score(nb_target, user_context_embed)
                        norm_w = edge_weight / max_weight
                        hop_score = (embedding_weight * embed_sim +
                                     (1 - embedding_weight) * norm_w +
                                     type_bonus)
                        detail_entry = (current, neighbor['description'], edge_weight, embed_sim)
                        scoring_method = 'embedding'
                    else:
                        hop_score = edge_weight + type_bonus
                        detail_entry = (current, neighbor['description'], edge_weight)
                        scoring_method = 'weight'

                    new_path = path + [nb_target]
                    new_details = path_details + [detail_entry]
                    new_raw = raw_score * hop_score
                    new_score = new_raw * (length_penalty ** len(new_path))

                    if nb_target == target_lower:
                        matching_paths.append({
                            'path': new_path,
                            'path_details': new_details,
                            'source': start_lower,
                            'target': target_lower,
                            'score': new_score,
                            'raw_score': new_raw,
                            'length': len(new_path),
                            'target_entity': target_obj,
                            'scoring_method': scoring_method
                        })
                    elif hop < max_hops - 1:
                        new_queue.append((new_path, new_details, new_raw))

            queue = new_queue

        if not matching_paths:
            return None
        matching_paths.sort(key=lambda x: x['score'], reverse=True)
        return matching_paths[0]

    def find_best_explanation_path(self, mentioned_entities, top_predicted_names,
                                    max_hops=2, user_context_embed=None,
                                    embedding_weight=0.7,
                                    entity_type_bonus=DEFAULT_ENTITY_TYPE_BONUS):
        """
        Try all combinations of mentioned_entity x predicted_movie.
        Returns highest scoring path found. Actor/director intermediates
        receive bonus via entity_type_bonus.
        """
        best_path = None
        for predicted_name in top_predicted_names:
            for source_name in mentioned_entities:
                path = self.find_path_to_target(
                    start_entity=source_name,
                    target_entity=predicted_name,
                    max_hops=max_hops,
                    user_context_embed=user_context_embed,
                    embedding_weight=embedding_weight,
                    entity_type_bonus=entity_type_bonus
                )
                if path is not None:
                    if best_path is None or path['score'] > best_path['score']:
                        best_path = path
        return best_path

    def rerank_predictions(self, top_item_ids, reasoning_paths,
                           entity_name_to_id, path_boost=1.5):
        path_target_ids = set()
        for path in reasoning_paths:
            target = path.get('target', '')
            if target:
                tid = (entity_name_to_id.get(target.lower()) or
                       entity_name_to_id.get(target.upper()))
                if tid is not None:
                    path_target_ids.add(tid)
        if not path_target_ids:
            return top_item_ids
        scored = [(iid, (1.0/(r+1)) * (path_boost if iid in path_target_ids else 1.0),
                   iid in path_target_ids)
                  for r, iid in enumerate(top_item_ids)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in scored]

    def explain_path(self, path: Dict, include_weights: bool = True) -> str:
        if not path or not path.get('path'):
            return "No reasoning path found"
        entities = path['path']
        path_details = path.get('path_details', [])
        target = path['target']
        scoring_method = path.get('scoring_method', 'weight')
        parts = []
        for i, detail in enumerate(path_details):
            source, rel_desc, weight = detail[0], detail[1], detail[2]
            embed_sim = detail[3] if len(detail) > 3 else None
            target_node = entities[i + 1]
            se = self.entity_dict.get(source, {})
            te = self.entity_dict.get(target_node, {})
            src_name = se.get('title', source).title()
            tgt_name = te.get('title', target_node).title()
            tgt_type = te.get('type', 'entity')
            type_bonus = self._get_entity_type_bonus(target_node)
            bonus_str = f", type_bonus={type_bonus:.2f}" if type_bonus > 0 else ""
            if include_weights:
                if embed_sim is not None:
                    step = f"{src_name} → {tgt_name} ({tgt_type}, edge={weight:.0f}, embed_sim={embed_sim:.3f}{bonus_str})"
                else:
                    step = f"{src_name} → {tgt_name} ({tgt_type}, weight={weight:.0f}{bonus_str})"
            else:
                step = f"{src_name} → {tgt_name} ({tgt_type})"
            parts.append(step)
        te = path.get('target_entity', {})
        tgt_title = te.get('title', target).title()
        tgt_desc = te.get('description', '')
        out = f"Reasoning Path [{scoring_method}]:\n  " + "\n  ".join(parts) + "\n\n"
        out += f"Recommendation: {tgt_title}\n"
        out += f"Score: {path['score']:.4f} (raw: {path['raw_score']:.4f}, length: {path['length']})\n"
        if tgt_desc:
            out += f"Description: {tgt_desc[:300]}..."
        return out

    def get_recommendation_with_reasoning(self, mentioned_entity, max_hops=2,
                                          top_k=5, user_context_embed=None):
        if user_context_embed is not None and self.rgcn_entity_embeds is not None:
            paths = self.find_path_with_embeddings(
                mentioned_entity, user_context_embed=user_context_embed,
                target_type="MOVIE", max_hops=max_hops, top_k=top_k)
        else:
            paths = self.find_path(mentioned_entity, target_type="MOVIE",
                                   max_hops=max_hops, top_k=top_k)
        if not paths:
            return {'recommendation': None, 'reasoning_path': [], 'path_details': [],
                    'score': 0.0, 'explanation': f"No recommendations for {mentioned_entity}",
                    'all_recommendations': [], 'scoring_method': 'none'}
        best = paths[0]
        return {
            'recommendation': best['target'],
            'reasoning_path': best['path'],
            'path_details': best['path_details'],
            'score': best['score'],
            'raw_score': best['raw_score'],
            'explanation': self.explain_path(best, include_weights=True),
            'all_recommendations': [{'rank': i+1, 'movie': p['target'], 'score': p['score'],
                                      'path_length': p['length'], 'path': p['path'],
                                      'scoring_method': p.get('scoring_method', 'weight')}
                                     for i, p in enumerate(paths)],
            'scoring_method': best.get('scoring_method', 'weight')
        }


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    entities_path = os.path.join(base_dir, "../../GraphRAG/output/successful_20250129-110435/artifacts/create_final_entities.parquet")
    relationships_path = os.path.join(base_dir, "../../GraphRAG/output/successful_20250129-110435/artifacts/create_final_relationships.parquet")
    reasoner = KGPathReasoner(entities_path=entities_path, relationships_path=relationships_path)
    print("\n" + "="*80)
    print("TESTING: find_path_to_target")
    print("="*80)
    path = reasoner.find_path_to_target("moana (2016)", "coco (2017)", max_hops=2)
    print(reasoner.explain_path(path) if path else "No path found")
    print("\n" + "="*80)
    print("TESTING: find_best_explanation_path")
    print("="*80)
    result = reasoner.find_best_explanation_path(
        mentioned_entities=["moana (2016)", "frozen (2013)"],
        top_predicted_names=["coco (2017)", "finding nemo (2003)", "the little mermaid (1989)"],
        max_hops=2)
    print(reasoner.explain_path(result) if result else "No path found")