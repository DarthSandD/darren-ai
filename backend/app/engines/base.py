"""Engine interfaces: a detector and a rewriter are swappable backends."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Span:
    """A sentence-level score used for the UI heatmap."""
    text: str
    start: int
    end: int
    score: float          # 0..1, higher = more AI-like
    flagged: bool


@dataclass
class DetectResult:
    score: float                          # 0..1 overall AI-likelihood
    verdict: str                          # "human" | "mixed" | "ai" | "insufficient_text"
    confidence: str                       # "low" | "medium" | "high"
    engine: str
    spans: list[Span] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)


@dataclass
class RewriteResult:
    text: str
    similarity: float                     # meaning preservation 0..1


class Detector(Protocol):
    name: str
    def score(self, text: str) -> DetectResult: ...


class Rewriter(Protocol):
    name: str
    def rewrite(self, text: str, *, target_spans: list[str] | None = None,
                strength: int = 60) -> RewriteResult: ...
