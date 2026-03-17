"""HotPotQA solver — the artifact agents evolve.

Takes a multi-hop question with context on stdin (JSON),
prints the answer on stdout.
"""

import sys
import os
import json
import re
from collections import Counter

from openai import OpenAI


def extract_answer(text: str) -> str:
    """Extract the answer from model output."""
    for line in reversed(text.split("\n")):
        line = line.strip()
        if line.upper().startswith("ANSWER:"):
            return line[len("ANSWER:"):].strip()
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return lines[-1] if lines else text


def normalize(s: str) -> str:
    """Normalize text for comparison."""
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s]', '', s)
    return ' '.join(s.split())


def token_overlap_score(a: str, b: str) -> float:
    """Compute F1-style token overlap between two strings."""
    ta = normalize(a).split()
    tb = normalize(b).split()
    if not ta or not tb:
        return float(ta == tb)
    common = Counter(ta) & Counter(tb)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    p = ns / len(ta)
    r = ns / len(tb)
    return 2 * p * r / (p + r)


def best_answer(answers: list[str]) -> str:
    """Pick the answer that has the highest average overlap with all others."""
    if len(answers) == 1:
        return answers[0]

    best = answers[0]
    best_score = -1
    for a in answers:
        score = sum(token_overlap_score(a, b) for b in answers)
        if score > best_score:
            best_score = score
            best = a
    return best


def solve(question: str, context: list[str]) -> str:
    """Answer a multi-hop question using the provided context."""
    client = OpenAI()
    model = os.environ.get("SOLVER_MODEL", "gpt-4.1-nano")

    ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(context))

    messages = [
        {"role": "system", "content": """You are a precise question-answering system. You answer multi-hop questions that require combining information from multiple paragraphs.

Instructions:
1. First, identify which paragraphs are relevant to the question.
2. Reason step-by-step to connect information across paragraphs.
3. After your reasoning, output your final answer on the last line prefixed with "ANSWER: ".

Your final answer should be as concise as possible — typically a few words, a name, a date, or a number. Do not include unnecessary articles or filler words unless they are part of a proper name. Copy the exact phrasing from the context when possible."""},
        {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"},
    ]

    # Get deterministic answer (temperature=0) plus diverse samples
    det_response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=256,
    )
    det_answer = extract_answer(det_response.choices[0].message.content.strip())

    diverse_response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.5,
        max_tokens=256,
        n=6,
    )
    diverse_answers = [extract_answer(c.message.content.strip()) for c in diverse_response.choices]

    all_answers = [det_answer] + diverse_answers
    return best_answer(all_answers)


if __name__ == "__main__":
    data = json.loads(sys.stdin.read())
    print(solve(data["question"], data["context"]))
