"""Evaluate agent.py on HotPotQA using F1 score."""

import json
import re
import subprocess
import sys
from collections import Counter


def normalize(s: str) -> str:
    """Lowercase, remove punctuation/articles, strip whitespace."""
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s]', '', s)
    return ' '.join(s.split())


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize(prediction).split()
    gt_tokens = normalize(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def main():
    with open(sys.argv[1]) as f:
        problems = [json.loads(line) for line in f]

    total = len(problems)
    total_f1 = 0.0
    exact = 0

    print(f"Evaluating {total} problems...", file=sys.stderr)

    for item in problems:
        try:
            result = subprocess.run(
                ["python3", "agent.py"],
                input=json.dumps(item), capture_output=True, text=True, timeout=60,
            )
            got = result.stdout.strip()
        except (subprocess.TimeoutExpired, Exception):
            got = ""

        f1 = f1_score(got, item["answer"])
        total_f1 += f1
        if normalize(got) == normalize(item["answer"]):
            exact += 1

    avg_f1 = total_f1 / total
    em = exact / total
    print("---")
    print(f"f1:               {avg_f1:.6f}")
    print(f"exact_match:      {em:.6f}")
    print(f"exact_correct:    {exact}")
    print(f"total:            {total}")


if __name__ == "__main__":
    main()
