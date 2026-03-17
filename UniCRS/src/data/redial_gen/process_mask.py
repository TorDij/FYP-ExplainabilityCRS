
import json
import re
import html
from tqdm.auto import tqdm

movie_pattern = re.compile(r'@\d+')

def process_utt(utt, movieid2name, replace_movieId, remove_movie=False):
    def convert(match):
        movieid = match.group(0)[1:]
        if movieid in movieid2name:
            if remove_movie:
                return '<movie>'
            movie_name = movieid2name[movieid]
            return movie_name
        else:
            return match.group(0)

    if replace_movieId:
        utt = re.sub(movie_pattern, convert, utt)
    utt = ' '.join(utt.split())
    utt = html.unescape(utt)

    return utt

def process(data_file, out_file, movie_set, entity2id):
    with open(data_file, 'r', encoding='utf-8') as fin, open(out_file, 'w', encoding='utf-8') as fout:
        for line in tqdm(fin):
            dialog = json.loads(line)
            if len(dialog['messages']) == 0:
                continue

            movieid2name = dialog['movieMentions']
            user_id, resp_id = dialog['initiatorWorkerId'], dialog['respondentWorkerId']
            context, resp = [], ''
            entity_list = []
            messages = dialog['messages']
            turn_i = 0

            while turn_i < len(messages):
                worker_id = messages[turn_i]['senderWorkerId']
                utt_turn = []
                entity_turn = []
                movie_turn = []
                mask_utt_turn = []

                turn_j = turn_i
                while turn_j < len(messages) and messages[turn_j]['senderWorkerId'] == worker_id:
                    utt = process_utt(messages[turn_j]['text'], movieid2name, replace_movieId=True, remove_movie=False)
                    utt_turn.append(utt)

                    mask_utt = process_utt(messages[turn_j]['text'], movieid2name, replace_movieId=True, remove_movie=True)
                    mask_utt_turn.append(mask_utt)

                    # Convert entity/movie names to uppercase for lookup
                    entity_ids = [entity2id.get(entity.upper(), None) for entity in messages[turn_j]['entity']]
                    movie_ids = [entity2id.get(movie.upper(), None) for movie in messages[turn_j]['movie']]

                    entity_ids = [e for e in entity_ids if e is not None]  # Remove None values
                    movie_ids = [m for m in movie_ids if m is not None]

                    entity_turn.extend(entity_ids)
                    movie_turn.extend(movie_ids)

                    turn_j += 1

                utt = ' '.join(utt_turn)
                mask_utt = ' '.join(mask_utt_turn)

                if worker_id == user_id:
                    context.append(utt)
                    entity_list.append(entity_turn + movie_turn)
                else:
                    resp = utt

                    context_entity_list = [entity for entity_l in entity_list for entity in entity_l]
                    context_entity_list_extend = list(set(context_entity_list))  # Ensure uniqueness

                    if len(context) == 0:
                        context.append('')
                    turn = {
                        'context': context,
                        'resp': mask_utt,
                        'rec': movie_turn,
                        'entity': context_entity_list_extend,
                    }
                    fout.write(json.dumps(turn, ensure_ascii=False) + '\n')

                    context.append(resp)
                    entity_list.append(movie_turn + entity_turn)
                    movie_set |= set(movie_turn)

                turn_i = turn_j

if __name__ == '__main__':
    with open('/home/Nema/UniCRS_GraphRAG/UniCRS/src/data/redial/entity2id.json', 'r', encoding='utf-8') as f:
        entity2id = json.load(f)
    movie_set = set()

    process('/home/Nema/UniCRS_GraphRAG/UniCRS/src/data/redial/test_data_dbpedia.jsonl', '/home/Nema/UniCRS_GraphRAG/UniCRS/src/data/redial/test_data_processed.jsonl', movie_set, entity2id)
    process('/home/Nema/UniCRS_GraphRAG/UniCRS/src/data/redial/valid_data_dbpedia.jsonl', '/home/Nema/UniCRS_GraphRAG/UniCRS/src/data/redial/valid_data_processed.jsonl', movie_set, entity2id)
    process('/home/Nema/UniCRS_GraphRAG/UniCRS/src/data/redial/train_data_dbpedia.jsonl', '/home/Nema/UniCRS_GraphRAG/UniCRS/src/data/redial/train_data_processed.jsonl', movie_set, entity2id)

    with open('/home/Nema/UniCRS_GraphRAG/UniCRS/src/data/redial/movie_ids.json', 'w', encoding='utf-8') as f:
        json.dump(list(movie_set), f, ensure_ascii=False)

    print(f'#movies: {len(movie_set)}')
