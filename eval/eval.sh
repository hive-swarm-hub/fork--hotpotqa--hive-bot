#!/usr/bin/env bash
set -euo pipefail
DATA="data/test.jsonl"
[ ! -f "$DATA" ] && echo "ERROR: $DATA not found. Run: bash prepare.sh" >&2 && exit 1
python3 eval/judge.py "$DATA"
