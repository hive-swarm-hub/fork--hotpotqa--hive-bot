#!/usr/bin/env bash
set -euo pipefail
mkdir -p data

echo "Downloading HotPotQA..."
python3 -c "
from datasets import load_dataset
import json, pathlib, random

random.seed(42)
ds = load_dataset('hotpotqa/hotpot_qa', 'distractor', split='validation')
samples = list(ds)
random.shuffle(samples)
samples = samples[:50]

out = pathlib.Path('data/test.jsonl')
with out.open('w') as f:
    for row in samples:
        # include supporting context
        context_texts = []
        for title, sents in zip(row['context']['title'], row['context']['sentences']):
            context_texts.append(f'{title}: {\" \".join(sents)}')
        f.write(json.dumps({
            'question': row['question'],
            'answer': row['answer'],
            'context': context_texts,
            'type': row.get('type', ''),
        }) + '\n')

print(f'Wrote {len(samples)} problems to {out}')
"

echo "Done. $(wc -l < data/test.jsonl) problems in data/test.jsonl"
