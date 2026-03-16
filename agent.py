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

    ctx = "\n\n".join(context)

    response = client.chat.completions.create(
        model=os.environ.get("SOLVER_MODEL", "gpt-4.1-nano"),
        messages=[
            {"role": "system", "content": "Answer the question using the provided context. Give ONLY the answer, nothing else. Be concise — usually a few words."},
            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"},
        ],
        temperature=0,
        max_tokens=64,
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    data = json.loads(sys.stdin.read())
    print(solve(data["question"], data["context"]))
