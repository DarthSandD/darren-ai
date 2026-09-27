"""Pydantic request/response schemas for the API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class DetectRequest(BaseModel):
    text: str = Field(..., min_length=1)
    threshold: float | None = None


class SpanOut(BaseModel):
    text: str
    start: int
    end: int
    score: float
    flagged: bool


class DetectResponse(BaseModel):
    score: float
    verdict: str
    confidence: str
    engine: str
    spans: list[SpanOut] = []
    caveats: list[str] = []


class HumanizeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    strength: int = Field(60, ge=0, le=100)
    max_rounds: int | None = Field(None, ge=1, le=10)


class HealthResponse(BaseModel):
    status: str
    engine: str
    rewriter: str
    version: str
