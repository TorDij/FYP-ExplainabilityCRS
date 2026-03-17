import json

with open('reasoning_analysis_kecr.json') as f:
    data = json.load(f)

results = data['all_results']
total = len(results)

correct_top5 = [
    r for r in results
    if r['ground_truth_id'] in r['top_10_predictions'][:5]
]

paths_found_on_correct_top5 = [
    r for r in correct_top5
    if r['path_found']
]

pc5 = len(paths_found_on_correct_top5) / max(len(correct_top5), 1) * 100
recall5 = len(correct_top5) / total * 100

print(f"Total samples:              {total}")
print(f"Correct@5:                  {len(correct_top5)} ({recall5:.2f}%)")
print(f"Paths found on correct@5:   {len(paths_found_on_correct_top5)}")
print(f"PC@5:                       {pc5:.2f}%")