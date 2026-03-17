import pandas as pd
import numpy as np
from collections import defaultdict
import json
import re
import html
from tqdm.auto import tqdm

# --------- Create file item_ids.json --------------
# Regular expression to match movie IDs in text
movie_pattern = re.compile(r'@\d+')

def simplify_and_normalize(name):
    """Simplifies and normalizes entity or movie names."""
    # Replace underscores with spaces, convert to lowercase, and remove parenthetical text
    name = name.replace("_", " ").lower()
    name = re.sub(r"\s*\(.*?\)", "", name).strip()
    return name

def match_entity_by_name(name, entity_name_map):
    """
    Find a matching entity in entity_name_map using substring matching.
    Returns the entity ID instead of the name.
    """
    simplified_name = simplify_and_normalize(name)
    for entity, entity_id in entity_name_map.items():
        normalized_entity = simplify_and_normalize(entity)
        if simplified_name in normalized_entity:  # Substring matching
            return entity_id
    return None

def process_utt(utt, movieid2name, replace_movieId):
    """
    Replaces movie IDs in text with their corresponding names, normalizing the text.
    """
    def convert(match):
        movieid = match.group(0)[1:]
        if movieid in movieid2name:
            movie_name = movieid2name[movieid]
            movie_name = ' '.join(movie_name.split())
            return movie_name
        else:
            return match.group(0)

    if replace_movieId:
        utt = re.sub(movie_pattern, convert, utt)
    utt = ' '.join(utt.split())
    utt = html.unescape(utt)

    return utt

def process(data_file, out_file, movie_set, entity2id):
    """
    Processes a data file, replaces movie IDs with names in text but stores entity IDs in lists.
    """
    with open(data_file, 'r', encoding='utf-8') as fin, open(out_file, 'w', encoding='utf-8') as fout:
        for line in tqdm(fin):
            dialog = json.loads(line)
            if len(dialog['messages']) == 0:
                continue

            movieid2name = dialog['movieMentions']
            user_id, resp_id = dialog['initiatorWorkerId'], dialog['respondentWorkerId']
            context, resp = [], ''
            entity_list = []  # Will store IDs instead of names
            messages = dialog['messages']
            turn_i = 0
            
            while turn_i < len(messages):
                worker_id = messages[turn_i]['senderWorkerId']
                utt_turn = []
                entity_turn = []
                movie_turn = []

                turn_j = turn_i
                while turn_j < len(messages) and messages[turn_j]['senderWorkerId'] == worker_id:
                    utt = process_utt(messages[turn_j]['text'], movieid2name, replace_movieId=True)
                    utt_turn.append(utt)

                    # Match entities and store their IDs
                    entity_ids = [
                        match_entity_by_name(entity, entity2id)
                        for entity in messages[turn_j]['entity']
                    ]
                    entity_turn.extend([eid for eid in entity_ids if eid is not None])

                    # Match movies and store their IDs
                    movie_ids = [
                        match_entity_by_name(movie, entity2id)
                        for movie in messages[turn_j]['movie']
                    ]
                    movie_turn.extend([mid for mid in movie_ids if mid is not None])

                    turn_j += 1

                utt = ' '.join(utt_turn)
                resp = utt

                # Flatten and deduplicate entity IDs from context
                context_entity_list = [entity for entity_l in entity_list for entity in entity_l]
                context_entity_list = list(set(context_entity_list))

                if len(context) == 0:
                    context.append('')

                turn = {
                    'context': context,
                    'resp': resp,
                    'rec': list(set(movie_turn + entity_turn)),  # Store unique IDs
                    'entity': context_entity_list,  # Store IDs from context
                }
                fout.write(json.dumps(turn, ensure_ascii=False) + '\n')

                context.append(resp)
                entity_list.append(movie_turn + entity_turn)
                movie_set |= set(movie_turn)

                turn_i = turn_j


with open('/home/Nema/UniCRS_GraphRAG/UniCRS/data/redial/entity2id.json', 'r', encoding='utf-8') as f:
    entity2id = json.load(f)

item_set = set()

# Process files
process('/home/Nema/UniCRS_GraphRAG/UniCRS/data/redial/valid_data_dbpedia.jsonl', 
        '/home/Nema/UniCRS_GraphRAG/UniCRS/data/redial/valid_data_processed.jsonl', 
        item_set,
        entity2id)

process('/home/Nema/UniCRS_GraphRAG/UniCRS/data/redial/test_data_dbpedia.jsonl', 
        '/home/Nema/UniCRS_GraphRAG/UniCRS/data/redial/test_data_processed.jsonl', 
        item_set,
        entity2id)

process('/home/Nema/UniCRS_GraphRAG/UniCRS/data/redial/train_data_dbpedia.jsonl', 
        '/home/Nema/UniCRS_GraphRAG/UniCRS/data/redial/train_data_processed.jsonl', 
        item_set,
        entity2id)

# Save item IDs
with open('/home/Nema/UniCRS_GraphRAG/UniCRS/data/redial/item_ids.json', 'w', encoding='utf-8') as f:
    json.dump(list(item_set), f, ensure_ascii=False)

print(f'#item: {len(item_set)}')
print("Completed Creating file item_ids.json!")