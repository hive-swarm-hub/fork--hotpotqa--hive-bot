# HotPotQA

Improve a multi-hop question answering solver to maximize F1 score on HotPotQA.

## Setup

1. Read these files for full context:
   - `prepare.sh` — downloads HotPotQA dataset. Do not modify.
   - `eval/eval.sh` — runs evaluation. Do not modify.
   - `agent.py` — the file you modify. The solver.
2. Verify data exists: check that `data/` contains `test.jsonl`. If not, run `bash prepare.sh`.
3. Create `results.tsv` with just the header row.

## Experimentation

Each experiment runs on the test set (50 problems). You launch it as: `bash eval/eval.sh`.

**What you CAN do:**
- Modify `agent.py` — everything is fair game: prompting strategy, chain-of-thought, context selection, step-by-step reasoning, answer extraction.

**What you CANNOT do:**
- Modify `prepare.sh` or `eval/eval.sh`. They are read-only.
- Modify the test data.
- Change the model (set via `SOLVER_MODEL` env var).
- Install new packages beyond what's in `requirements.txt`.

**The goal: get the highest F1 score on HotPotQA.** Questions require reasoning across multiple context paragraphs. Answers are usually short (a few words).

**The first run**: establish the baseline by running eval as-is.

## Output format

```
---
f1:               0.6500
exact_match:      0.4800
exact_correct:    24
total:            50
```

## Logging results

Log to `results.tsv` (tab-separated, do not commit):

```
commit	f1	exact_match	status	description
a1b2c3d	0.650000	0.480000	keep	baseline
```

## The experiment loop

LOOP FOREVER:

1. **THINK** — decide what to try next.
2. Modify `agent.py`.
3. git commit
4. Run: `bash eval/eval.sh > run.log 2>&1`
5. Check: `grep "^f1:" run.log`
6. If crashed, check `tail -n 50 run.log`.
7. Log to results.tsv.
8. If improved, keep. If not, `git reset --hard HEAD~1`.

**NEVER STOP.** The loop runs until interrupted.
