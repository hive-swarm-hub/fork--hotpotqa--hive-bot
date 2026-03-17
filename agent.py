"""HotPotQA solver — the artifact agents evolve.

Takes a multi-hop question with context on stdin (JSON),
prints the answer on stdout.
"""

import sys
import os
import json

from openai import OpenAI


def solve(question: str, context: list[str]) -> str:
    """Answer a multi-hop question using the provided context."""
    client = OpenAI()

    ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(context))

    response = client.chat.completions.create(
        model=os.environ.get("SOLVER_MODEL", "gpt-4.1-nano"),
        messages=[
            {"role": "system", "content": """You are a precise question-answering system. You answer multi-hop questions that require combining information from multiple paragraphs.

Instructions:
1. First, identify which paragraphs are relevant to the question.
2. Reason step-by-step to connect information across paragraphs.
3. After your reasoning, output your final answer on the last line prefixed with "ANSWER: ".

Your final answer should be as concise as possible — typically a few words, a name, a date, or a number. Do not include unnecessary articles or filler words unless they are part of a proper name."""},
            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"},
        ],
        temperature=0,
        max_tokens=256,
    )

    text = response.choices[0].message.content.strip()

    # Extract answer after "ANSWER:" prefix
    for line in reversed(text.split("\n")):
        line = line.strip()
        if line.upper().startswith("ANSWER:"):
            return line[len("ANSWER:"):].strip()

    # Fallback: return the last non-empty line
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return lines[-1] if lines else text


if __name__ == "__main__":
    data = json.loads(sys.stdin.read())
    print(solve(data["question"], data["context"]))
