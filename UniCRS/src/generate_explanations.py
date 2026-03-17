"""
generate_explanations.py - Conversational recommendation response generation using Mistral-7B-Instruct

Generates a unified response conditioned on:
  1. Full dialogue history
  2. The recommended movie (top-1 prediction from recommendation module)
  3. The KG reasoning path explaining the recommendation

Output is a natural, conversational reply that both makes the recommendation
and explains the KG-grounded reason behind it.
"""
import argparse, json, os, re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reasoning',  default='./results/reasoning_analysis_kecr.json')
    parser.add_argument('--output',     default='./results/explanations_mistral.jsonl')
    parser.add_argument('--model',      default='mistralai/Mistral-7B-Instruct-v0.2')
    parser.add_argument('--hf_token',   default=None, help='HuggingFace token for gated model access')
    parser.add_argument('--quantize',   default='4bit', choices=['none', '4bit', '8bit'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_input',  type=int, default=768)
    parser.add_argument('--max_output', type=int, default=120)
    parser.add_argument('--only_with_path', action='store_true',
                        help='Only generate responses for samples where a KG path was found')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit number of samples to process (for testing)')
    return parser.parse_args()


# ── Utilities ─────────────────────────────────────────────────────────────────

def title_case(text):
    """Best-effort title casing for entity names stored in lowercase in the KG."""
    LOWER_WORDS = {'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor',
                   'on', 'at', 'to', 'by', 'in', 'of', 'up', 'as', 'is'}
    tokens = text.split()
    if not tokens:
        return text
    result = []
    for idx, tok in enumerate(tokens):
        if tok.startswith('(') and tok.endswith(')'):
            result.append(tok)
        elif idx == 0:
            result.append(tok.capitalize())
        else:
            result.append(tok if tok in LOWER_WORDS else tok.capitalize())
    return ' '.join(result)


def format_dialogue(dialogue: str) -> str:
    """
    Reformat raw dialogue string into clean Recommender / User turns.
    Raw format: 'System: ...User: ...System: ...'
    """
    if not dialogue:
        return "(no dialogue context)"
    parts = re.split(r'(System:|User:)', dialogue)
    turns = []
    i = 1
    while i < len(parts) - 1:
        speaker = parts[i].strip().rstrip(':')
        content = parts[i + 1].strip()
        if content:
            label = "Recommender" if speaker == "System" else "User"
            turns.append(f"  {label}: {content}")
        i += 2
    return '\n'.join(turns) if turns else "(no dialogue context)"


def format_path(path: list) -> str:
    """Format KG path list as arrow-connected string with title casing."""
    return ' -> '.join(title_case(p) for p in path)


# ── Prompt construction ───────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a conversational movie recommender assistant. "
    "You will be given a dialogue history between a user and a recommender, "
    "a recommended movie, and a knowledge graph reasoning path of the form "
    "Source -> Intermediate -> Target. "
    "Generate a single natural, friendly conversational response that recommends "
    "the target movie and explicitly references the intermediate node "
    "(e.g. the shared genre, actor, or director) as the reason for the recommendation. "
    "Be concise and conversational. Do not repeat the path verbatim. "
    "Output one or two sentences only."
)

FEW_SHOT_EXAMPLES = """Here are examples of good responses:

Example 1:
Dialogue:
  User: I really enjoyed The Terminator, any suggestions?
  Recommender: Great choice! Do you like sci-fi in general?
Recommended movie: The Matrix (1999)
Reasoning path: The Terminator (1984) -> Sci-Fi -> The Matrix (1999)
Response: Based on your love of The Terminator, I think you'd really enjoy The Matrix (1999) as both are iconic sci-fi films that blend action with thought-provoking themes!

Example 2:
Dialogue:
  User: I just watched Superbad and loved it.
  Recommender: Glad you liked it! Are you into comedies?
Recommended movie: Pineapple Express (2008)
Reasoning path: Superbad (2007) -> Comedy -> Pineapple Express (2008)
Response: Since you enjoyed Superbad, you might love Pineapple Express (2008) as it's another hilarious comedy with a similar irreverent humour!

Example 3:
Dialogue:
  User: Have you seen anything with Christopher Nolan?
  Recommender: I love his work! Inception is a masterpiece.
Recommended movie: Dunkirk (2017)
Reasoning path: Inception (2010) -> Christopher Nolan -> Dunkirk (2017)
Response: If you're a fan of Inception, I'd recommend Dunkirk (2017) as Christopher Nolan directed both, and Dunkirk delivers the same intense, masterful filmmaking.

Now generate a response for:
"""


def build_messages(dialogue: str, recommended_movie: str, path: list) -> list:
    """
    Build Mistral chat message list conditioned on dialogue history,
    recommended movie, and KG reasoning path.
    """
    formatted_dialogue = format_dialogue(dialogue)
    formatted_path = format_path(path)
    recommended_tc = title_case(recommended_movie)

    user_content = (
        FEW_SHOT_EXAMPLES
        + f"Dialogue:\n{formatted_dialogue}\n"
        + f"Recommended movie: {recommended_tc}\n"
        + f"Reasoning path: {formatted_path}\n"
        + "Response:"
    )

    return [
        {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + user_content},
    ]


# ── Post-process ──────────────────────────────────────────────────────────────

def clean_response(text: str) -> str:
    """Strip artefacts and truncate cleanly at sentence boundary."""
    text = re.sub(r'^(Response:|Recommendation:|Assistant:)\s*', '', text,
                  flags=re.IGNORECASE).strip()
    text = re.sub(r'\s*->\s*', ' -> ', text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return ' '.join(sentences[:2]).strip()


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_name, quantize, hf_token):
    bnb_config = None
    if quantize == '4bit':
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )
    elif quantize == '8bit':
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        padding_side='left',
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        quantization_config=bnb_config,
        device_map='auto',
        torch_dtype=torch.float16 if quantize == 'none' else None,
    )
    model.eval()
    return tokenizer, model


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print(f"Loading reasoning results from {args.reasoning}...")
    with open(args.reasoning) as f:
        data = json.load(f)
    all_results = data.get('all_results', [])
    print(f"Loaded {len(all_results)} samples.")
    if args.max_samples:
        all_results = all_results[:args.max_samples]
        print(f'Limited to {len(all_results)} samples.')

    records = []
    skipped_no_path = 0
    skipped_no_mention = 0

    for sample in all_results:
        path_found = sample.get('path_found', False)

        if args.only_with_path and not path_found:
            skipped_no_path += 1
            continue

        mentioned = sample.get('mentioned_movies', [])
        if not mentioned:
            skipped_no_mention += 1
            continue

        dialogue = sample.get('dialogue', '')
        path     = sample.get('explanation_path', {}).get('path', []) if path_found else []

        # Use path target as recommended movie for faithfulness —
        # ensures the response recommends exactly what the KG path explains
        if path:
            recommended_movie = path[-1]
        else:
            recommended_movie = sample.get('top_1_prediction_name', '')
            path = [mentioned[0], '?', recommended_movie]

        messages = build_messages(dialogue, recommended_movie, path)

        records.append({
            'sample_idx':        sample.get('sample_idx'),
            'dialogue':          dialogue,
            'ground_truth_id':   sample.get('ground_truth_id'),
            'ground_truth_name': sample.get('ground_truth_name', ''),
            'recommended_movie': recommended_movie,
            'is_correct_top_1':  sample.get('is_correct_top_1', False),
            'is_correct_top_10': sample.get('is_correct_top_10', False),
            'path_found':        path_found,
            'path':              path,
            'messages':          messages,
        })

    print(f"Records to process:    {len(records)}")
    print(f"Skipped (no path):     {skipped_no_path}")
    print(f"Skipped (no mentions): {skipped_no_mention}")

    print(f"\nLoading {args.model} (quantize={args.quantize})...")
    tokenizer, model = load_model(args.model, args.quantize, args.hf_token)
    print("Model loaded.\n")

    os.makedirs(
        os.path.dirname(args.output) if os.path.dirname(args.output) else '.',
        exist_ok=True
    )

    out_f = open(args.output, 'w')
    total = len(records)

    for batch_start in range(0, total, args.batch_size):
        batch = records[batch_start:batch_start + args.batch_size]

        prompts = [
            tokenizer.apply_chat_template(
                r['messages'],
                tokenize=False,
                add_generation_prompt=True,
            )
            for r in batch
        ]

        encodings = tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=args.max_input,
        ).to(model.device)

        input_len = encodings['input_ids'].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **encodings,
                max_new_tokens=args.max_output,
                do_sample=False,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = output_ids[:, input_len:]

        for i, rec in enumerate(batch):
            raw = tokenizer.decode(generated[i], skip_special_tokens=True).strip()
            response = clean_response(raw)
            result = {
                'sample_idx':        rec['sample_idx'],
                'dialogue':          rec['dialogue'][:400],
                'ground_truth_name': rec['ground_truth_name'],
                'recommended_movie': rec['recommended_movie'],
                'is_correct_top_1':  rec['is_correct_top_1'],
                'is_correct_top_10': rec['is_correct_top_10'],
                'path_found':        rec['path_found'],
                'path':              rec['path'],
                'response':          response,
            }
            out_f.write(json.dumps(result) + '\n')

        if (batch_start // args.batch_size) % 10 == 0:
            print(f"  {batch_start + len(batch)}/{total} processed...")

    out_f.close()
    print(f"\nDone. Saved {total} responses to {args.output}")


if __name__ == '__main__':
    main()