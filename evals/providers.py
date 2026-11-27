"""Deterministic extractive provider and prompt rendering for the smoke suite.

The provider sees only the rendered prompt (question + context text). Golden expectations
(must_contain, gold chunk ids) never enter a prompt, so isolation holds by construction.
"""

import re

from rag_platform.domain.ports import Completion

SERVED_MODEL = "fake-extractive"
REFUSAL = "I don't know."
_STOP = {"what", "which", "when", "where", "does", "that", "this", "with", "from", "have", "who"}


def render_prompt(question: str, context: list[str]) -> str:
    body = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(context))
    return f"QUESTION: {question}\nCONTEXT:\n{body}\n"


def keywords(text: str) -> set[str]:
    words = (w.rstrip("s") for w in re.findall(r"[a-z0-9]+", text.lower()))
    return {w for w in words if len(w) > 2 and w not in _STOP}


def overlap(question: str, text: str) -> int:
    return len(keywords(question) & keywords(text))


class ExtractiveFakeProvider:
    """Returns the context sentence that best overlaps the question, else a refusal."""

    async def complete(self, prompt: str, *, max_tokens: int) -> Completion:
        head, _, ctx = prompt.partition("\nCONTEXT:\n")
        question = head.removeprefix("QUESTION: ")
        best, best_score = REFUSAL, 0
        for line in ctx.splitlines():
            line = re.sub(r"^\[\d+\]\s*", "", line)
            for sent in re.split(r"(?<=[.!?])\s+", line):
                score = overlap(question, sent)
                if score > best_score:
                    best, best_score = sent.strip(), score
        words = best.split()[:max_tokens]
        return Completion(
            text=" ".join(words),
            tokens_in=len(prompt.split()),
            tokens_out=len(words),
            usd=0.0,
            model=SERVED_MODEL,
        )
