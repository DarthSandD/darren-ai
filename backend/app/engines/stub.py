"""
Stub detector — NO GPU, NO model download.

Purpose: let the whole stack (API, web, desktop, android, CI, tests) run and be
developed without a GPU. It is a *heuristic*, not a real detector: it looks at
surface AI-tells (templated transitions, uniform sentence length, low lexical
variety). Do NOT ship this as the product detector — it exists so the plumbing
works everywhere and so tests are deterministic.

The real engine (LAPD) is a drop-in replacement behind the same interface.
"""
from __future__ import annotations

import re
import statistics

from .base import DetectResult, Span

AI_TELLS = [
    "furthermore", "moreover", "in conclusion", "it is important to note",
    "delve", "tapestry", "underscore", "pivotal", "crucial role",
    "in today's world", "plays a vital role", "navigating the",
    "it is worth noting", "a testament to", "in the realm of",
]

_SENT_RE = re.compile(r"[^.!?]+[.!?]?")


def _sentences(text: str) -> list[tuple[str, int, int]]:
    out = []
    for m in _SENT_RE.finditer(text):
        s = m.group().strip()
        if len(s) < 8:
            continue
        out.append((s, m.start(), m.end()))
    return out


def _sentence_score(sent: str) -> float:
    low = sent.lower()
    tells = sum(1 for t in AI_TELLS if t in low)
    words = low.split()
    # lexical diversity of the sentence (low = more repetitive/AI-ish)
    uniq = len(set(words)) / max(len(words), 1)
    score = 0.0
    score += min(tells * 0.25, 0.6)
    score += max(0.0, (0.75 - uniq)) * 1.2      # low diversity -> higher
    if len(words) > 30:
        score += 0.1                            # rambling long sentence
    return max(0.0, min(1.0, score))


class StubDetector:
    name = "stub"

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def score(self, text: str) -> DetectResult:
        sents = _sentences(text)
        if len(text.split()) < 25 or len(sents) < 2:
            return DetectResult(
                score=0.0, verdict="insufficient_text", confidence="low",
                engine=self.name,
                caveats=["Text is too short for a reliable estimate (need ~25+ words)."],
            )

        spans: list[Span] = []
        scores: list[float] = []
        for s, a, b in sents:
            sc = _sentence_score(s)
            scores.append(sc)
            spans.append(Span(text=s, start=a, end=b, score=sc, flagged=sc >= self.threshold))

        overall = sum(scores) / len(scores)
        # burstiness: uniform sentence lengths read as AI
        lengths = [len(s.split()) for s, _, _ in sents]
        if len(lengths) > 2:
            cv = statistics.pstdev(lengths) / max(statistics.mean(lengths), 1e-6)
            if cv < 0.35:
                overall = min(1.0, overall + 0.12)

        if overall < 0.35:
            verdict, conf = "human", "medium"
        elif overall < 0.6:
            verdict, conf = "mixed", "low"
        else:
            verdict, conf = "ai", "medium"

        return DetectResult(
            score=round(overall, 4), verdict=verdict, confidence=conf,
            engine=self.name, spans=spans,
            caveats=["Heuristic stub engine — install LAPD for real accuracy."],
        )
