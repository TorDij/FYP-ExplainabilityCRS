import pandas as pd
import numpy as np
from collections import defaultdict
import json
import re
import html
from tqdm.auto import tqdm

entities_file_path = "/home/Nema/UniCRS_GraphRAG/GraphRAG/output/successful_20250129-110435/artifacts/create_final_entities.parquet"
relationship_file_path = "/home/Nema/UniCRS_GraphRAG/GraphRAG/output/successful_20250129-110435/artifacts/create_final_relationships.parquet"
entities_df = pd.read_parquet(entities_file_path)
relationship_df = pd.read_parquet(relationship_file_path)

# --------- Assign Relationship Label --------------

# Prepare mappings for entities
entity_id_map = dict(zip(entities_df['title'], entities_df['human_readable_id']))
entity_type_map = dict(zip(entities_df['title'], entities_df['type']))

# Create Relationships DataFrame with initial mappings
relationships_clean_df = relationship_df.assign(
    entity_id_source=relationship_df['source'].map(entity_id_map),
    entity_type_source=relationship_df['source'].map(entity_type_map),
    entity_id_target=relationship_df['target'].map(entity_id_map),
    entity_type_target=relationship_df['target'].map(entity_type_map)
)

# Define relationship type mapping with IDs
relationship_type_mapping = {
    ('MOVIE', 'ACTOR'): ('features', 0),
    ('ACTOR', 'MOVIE'): ('acted_in', 1),
    ('MOVIE', 'GENRE'): ('belongs_to_genre', 2),
    ('GENRE', 'MOVIE'): ('categorizes', 3),
    ('MOVIE', 'DIRECTOR'): ('directed_by', 4),
    ('DIRECTOR', 'MOVIE'): ('directed', 5),
    ('MOVIE', 'CHARACTER'): ('has_character', 6),
    ('CHARACTER', 'MOVIE'): ('featured_in', 7),
    ('ACTOR', 'CHARACTER'): ('portrays', 8),
    ('CHARACTER', 'ACTOR'): ('portrayed_by', 9),
    ('GENRE', 'DIRECTOR'): ('prefers_to_direct', 10),
    ('DIRECTOR', 'GENRE'): ('has_preference_for', 11),
    ('CHARACTER', 'DIRECTOR'): ('created_by', 12),
    ('DIRECTOR', 'CHARACTER'): ('conceived', 13),
    ('MOVIE', 'MOVIE'): ('similar', 14)
}

# Reverse the relationships and add them to the original DataFrame
relationships_reversed_df = relationships_clean_df.rename(columns={
    'entity_id_source': 'entity_id_target',
    'entity_name_source': 'entity_name_target',
    'entity_type_source': 'entity_type_target',
    'entity_id_target': 'entity_id_source',
    'entity_name_target': 'entity_name_source',
    'entity_type_target': 'entity_type_source'
}).copy()

# Combine the original and reversed relationships
relationships_combined_df = pd.concat([relationships_clean_df, relationships_reversed_df], ignore_index=True)

# Assign relationship types and IDs based on source and target entity types
relationships_combined_df[['relationship_type', 'relationship_type_id']] = relationships_combined_df.apply(
    lambda row: pd.Series(relationship_type_mapping.get(
        (row['entity_type_source'], row['entity_type_target']), ('unknown', -1)
    )),
    axis=1
)

kg_df = relationships_combined_df[['entity_id_source', 'source', 'entity_type_source',  'entity_id_target', 'target', 'entity_type_target', 'relationship_type', 'relationship_type_id']]

print("Completed Assigning Relationship Label!")

# --------- Create file dbpedia_subkg.json --------------

# Group data by source entity
json_structure = defaultdict(list)

# Replace 'kg_df' with the DataFrame containing the KG relationships
for _, row in kg_df.iterrows():
    source_id = row['entity_id_source']
    target_id = row['entity_id_target']
    relationship_id = row['relationship_type_id']
    if relationship_id != -1:  # Only include valid relationships
        json_structure[source_id].append([relationship_id, target_id])

# Convert the defaultdict to a regular dictionary
json_output = {str(source): [[rel_id, target] for rel_id, target in targets]
               for source, targets in json_structure.items()}

# Save the compact JSON structure to a file
compact_json_file_path = "/home/Nema/UniCRS_GraphRAG/UniCRS/data/redial/dbpedia_subkg.json"
with open(compact_json_file_path, "w") as json_file:
    json.dump(json_output, json_file, separators=(',', ':'))

print("Created file dbpedia_subkg.json!")


# --------- Create file entity2id.json --------------

# Create a mapping of entity: id in the required format
entity_id_mapping = {
    f"{row['title']}" : row['human_readable_id']
    for _, row in entities_df.iterrows()
}

# Save the mapping as a JSON file
entity_json_file_path = "/home/Nema/UniCRS_GraphRAG/UniCRS/data/redial/entity2id.json"
with open(entity_json_file_path, "w") as json_file:
    json.dump(entity_id_mapping, json_file, separators=(',', ':'))

print("Created file entity2id.json!")

# --------- Create file relation2id.json --------------

# Create a mapping of relationship_type: id
relationship_type_id_mapping = {relation: rel_id for _, (relation, rel_id) in relationship_type_mapping.items()}

# Save the mapping as a JSON file
relationship_json_file_path = "/home/Nema/UniCRS_GraphRAG/UniCRS/data/redial/relation2id.json"
with open(relationship_json_file_path, "w") as json_file:
    json.dump(relationship_type_id_mapping, json_file, separators=(',', ':'))

print("Created file relation2id.json!")

# --------- Format the train, test, valid data files ---------

# Load entity2id mapping and normalize keys
with open('/home/Nema/UniCRS_GraphRAG/UniCRS/data/redial/entity2id.json', encoding='utf-8') as f:
    entity2id = {key.lower(): value for key, value in json.load(f).items()}  # Convert to lowercase

def simplify_uri(uri):
    """Extracts the suffix from a URI and normalizes it."""
    if uri.startswith("<http://dbpedia.org/resource/"):
        # Remove URI prefix and convert underscores to spaces
        simplified = uri.split("/")[-1].strip(">").replace("_", " ").lower()
        # Remove parenthetical annotations like "(film)"
        return re.sub(r"\s*\(.*?\)", "", simplified).strip()
    return uri.lower()

def find_matching_entity(simplified_entity, entity2id):
    """Find a matching entity in entity2id using substring matching."""
    for key in entity2id:
        # Normalize key (entity2id keys are already lowercase)
        normalized_key = re.sub(r"\s*\(.*?\)", "", key).strip()
        if simplified_entity in normalized_key:  # Substring matching
            return key
    return None

def process_file(src_file, tgt_file, entity2id):
    """
    Processes the input JSONL file to:
    1. Simplify all URIs in entities and movies.
    2. Normalize and filter entities and movies based on entity2id using substring matching.
    3. Save the processed data into a new JSONL file.

    Args:
        src_file (str): Path to the input JSONL file.
        tgt_file (str): Path to the output JSONL file.
        entity2id (dict): Mapping of entities to IDs.
    """
    with open(src_file, encoding='utf-8') as f, open(tgt_file, 'w', encoding='utf-8') as tgt:
        for line in tqdm(f, desc=f"Processing {src_file}"):
            record = json.loads(line)
            for message in record['messages']:
                # Simplify and filter entities
                new_entity, new_entity_name = [], []
                for j, entity in enumerate(message['entity']):
                    simplified_entity = simplify_uri(entity)
                    matching_entity = find_matching_entity(simplified_entity, entity2id)
                    if matching_entity:
                        new_entity.append(matching_entity)
                        new_entity_name.append(message['entity_name'][j])
                message['entity'] = new_entity
                message['entity_name'] = new_entity_name

                # Simplify and filter movies
                new_movie, new_movie_name = [], []
                for j, movie in enumerate(message['movie']):
                    simplified_movie = simplify_uri(movie)
                    matching_movie = find_matching_entity(simplified_movie, entity2id)
                    if matching_movie:
                        new_movie.append(matching_movie)
                        new_movie_name.append(message['movie_name'][j])
                message['movie'] = new_movie
                message['movie_name'] = new_movie_name

            # Write the processed record to the target file
            tgt.write(json.dumps(record, ensure_ascii=False) + '\n')

# Define input and output file paths
src_files = [
    '/home/Nema/UniCRS_GraphRAG/UniCRS/data/redial/test_data_dbpedia_raw.jsonl',
    '/home/Nema/UniCRS_GraphRAG/UniCRS/data/redial/valid_data_dbpedia_raw.jsonl',
    '/home/Nema/UniCRS_GraphRAG/UniCRS/data/redial/train_data_dbpedia_raw.jsonl'
]

tgt_files = [
    '/home/Nema/UniCRS_GraphRAG/UniCRS/data/redial/test_data_dbpedia.jsonl',
    '/home/Nema/UniCRS_GraphRAG/UniCRS/data/redial/valid_data_dbpedia.jsonl',
    '/home/Nema/UniCRS_GraphRAG/UniCRS/data/redial/train_data_dbpedia.jsonl'
]

# Process each file
for src_file, tgt_file in zip(src_files, tgt_files):
    process_file(src_file, tgt_file, entity2id)

print("Completed Formating the train, test, valid data files!")

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