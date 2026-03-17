import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def per_sentence_distinct_n(response, n):
    tokens = response.lower().split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)

def compute_corpus_distinct_n(responses, n):
    total_ngrams = []
    for resp in responses:
        tokens = resp.lower().split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        total_ngrams.extend(ngrams)
    if not total_ngrams:
        return 0.0
    return len(set(total_ngrams)) / len(total_ngrams)

# Load generated responses
responses = []
with open('explanations_mistral.jsonl') as f:
    for line in f:
        r = json.loads(line)
        responses.append(r)

generated = [r['response'] for r in responses]
references = [r['dialogue'] for r in responses]

# BLEU
smoother = SmoothingFunction().method1
bleu1_scores = []
bleu2_scores = []

for gen, ref in zip(generated, references):
    ref_tokens = [ref.lower().split()]
    gen_tokens = gen.lower().split()
    if gen_tokens:
        bleu1_scores.append(sentence_bleu(ref_tokens, gen_tokens,
                            weights=(1,0,0,0), smoothing_function=smoother))
        bleu2_scores.append(sentence_bleu(ref_tokens, gen_tokens,
                            weights=(0.5,0.5,0,0), smoothing_function=smoother))

# Per-sentence Distinct-N (1 through 4)
ps_d1, ps_d2, ps_d3, ps_d4 = [], [], [], []
for r in generated:
    ps_d1.append(per_sentence_distinct_n(r, 1))
    ps_d2.append(per_sentence_distinct_n(r, 2))
    ps_d3.append(per_sentence_distinct_n(r, 3))
    ps_d4.append(per_sentence_distinct_n(r, 4))

avg_len = sum(len(r.split()) for r in generated) / len(generated)

print(f"Total samples:              {len(generated)}")
print(f"Avg response length:        {avg_len:.1f} tokens")
print()
print(f"BLEU-1:                     {sum(bleu1_scores)/len(bleu1_scores):.4f}")
print(f"BLEU-2:                     {sum(bleu2_scores)/len(bleu2_scores):.4f}")
print()
print("--- Corpus-level Distinct-N ---")
print(f"Corpus Distinct-1:          {compute_corpus_distinct_n(generated, 1):.4f}")
print(f"Corpus Distinct-2:          {compute_corpus_distinct_n(generated, 2):.4f}")
print(f"Corpus Distinct-3:          {compute_corpus_distinct_n(generated, 3):.4f}")
print(f"Corpus Distinct-4:          {compute_corpus_distinct_n(generated, 4):.4f}")
print()
print("--- Per-sentence Distinct-N ---")
print(f"Per-sentence Distinct-1:    {sum(ps_d1)/len(ps_d1):.4f}")
print(f"Per-sentence Distinct-2:    {sum(ps_d2)/len(ps_d2):.4f}")
print(f"Per-sentence Distinct-3:    {sum(ps_d3)/len(ps_d3):.4f}")
print(f"Per-sentence Distinct-4:    {sum(ps_d4)/len(ps_d4):.4f}")