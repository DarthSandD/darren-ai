"""
Pangram detector — hosted, commercial-grade AI-text detection via API.

Why this engine exists
----------------------
LAPD needs a GPU. A small classifier needs a CPU. This engine needs NEITHER —
it proxies to Pangram's servers, so the heavy ML runs on their infrastructure.
That makes the backend a thin, stateless proxy that can run on free edge hosts
(Cloudflare Workers, Vercel, Render free tier).

Accuracy context (independent benchmarks, 2026):
  * University of Chicago Booth (2025): Pangram was the ONLY detector to hold
    false positives at/below 0.5% while still detecting AI reliably.
  * Detects humanized / paraphrased text, which defeats most competitors.
  * ~99.98% claimed accuracy, ~0.01% (1-in-10,000) false-positive rate.

Pricing: ~$0.05 per 1,000 words. There is no free tier at this accuracy level.

API shape (docs.pangram.com/quickstart-rest)
--------------------------------------------
  POST https://text.external-api.pangram.com/task      {"text": ..., "model": "default"}
       -> {"task_id": "..."}
  GET  https://text.external-api.pangram.com/task/{id}  (poll)
       -> {"stage": "STAGE_SUCCESS", "fraction_ai": 0.7, "headline": "AI Detected",
           "windows": [{"text": ..., "label": ..., "ai_assistance_score": ...,
                        "start_index": ..., "end_index": ...}]}

Errors: 401 bad key, 402 insufficient credits, 429 rate limited.
"""
from __future__ import annotations

import time
from typing import Optional

from .base import DetectResult, Span

BASE_URL = "https://text.external-api.pangram.com"
TASK_URL = f"{BASE_URL}/task"
POLL_INTERVAL = 1.0
POLL_TIMEOUT = 90.0

# Pangram window labels -> our 0..1 score mapping
_LABEL_SCORE = {
    "AI-Generated": 0.95,
    "AI-Assisted": 0.65,
    "Human-Written": 0.10,
    "Human": 0.10,
}


class PangramDetector:
    name = "pangram"

    def __init__(self, api_key: str, *, model: str = "default",
                 threshold: float = 0.5, timeout: float = POLL_TIMEOUT) -> None:
        self.api_key = api_key
        self.model = model
        self.threshold = threshold
        self.timeout = timeout

    # -- HTTP helpers -----------------------------------------------------
    def _headers(self) -> dict:
        return {"Content-Type": "application/json", "x-api-key": self.api_key}

    def _submit(self, text: str) -> str:
        import httpx

        r = httpx.post(
            TASK_URL,
            headers=self._headers(),
            json={"text": text, "model": self.model, "public_dashboard_link": False},
            timeout=30,
        )
        if r.status_code == 401:
            raise RuntimeError("Pangram: invalid API key (401).")
        if r.status_code == 402:
            raise RuntimeError("Pangram: insufficient credits (402) — top up your account.")
        if r.status_code == 429:
            raise RuntimeError("Pangram: rate limited (429) — slow down.")
        r.raise_for_status()
        return r.json()["task_id"]

    def _poll(self, task_id: str) -> dict:
        import httpx

        deadline = time.time() + self.timeout
        while time.time() < deadline:
            r = httpx.get(f"{TASK_URL}/{task_id}", headers=self._headers(), timeout=30)
            r.raise_for_status()
            data = r.json()
            stage = data.get("stage", "")
            if stage == "STAGE_SUCCESS":
                return data
            if stage == "STAGE_FAILED":
                raise RuntimeError("Pangram: task failed.")
            time.sleep(POLL_INTERVAL)
        raise TimeoutError("Pangram: timed out waiting for result.")

    # -- mapping ----------------------------------------------------------
    def _to_result(self, data: dict, text: str) -> DetectResult:
        fraction_ai = float(data.get("fraction_ai", 0.0) or 0.0)
        fraction_assisted = float(data.get("fraction_ai_assisted", 0.0) or 0.0)
        # AI-assisted counts partially toward the score (it IS machine involvement)
        score = min(1.0, fraction_ai + 0.6 * fraction_assisted)

        spans: list[Span] = []
        for w in data.get("windows", []) or []:
            label = w.get("label", "")
            sc = _LABEL_SCORE.get(label)
            if sc is None:
                sc = float(w.get("ai_assistance_score", 0.0) or 0.0)
            spans.append(Span(
                text=w.get("text", ""),
                start=int(w.get("start_index", 0) or 0),
                end=int(w.get("end_index", 0) or 0),
                score=round(sc, 4),
                flagged=sc >= self.threshold,
            ))

        headline = (data.get("headline") or "").lower()
        if "human" in headline and "ai" not in headline:
            verdict = "human"
        elif "mixed" in headline or "assist" in headline:
            verdict = "mixed"
        else:
            verdict = "ai"
        if score < 0.35:
            verdict = "human"

        conf = "high"  # Pangram reports segment confidence; high overall by design

        return DetectResult(
            score=round(score, 4),
            verdict=verdict,
            confidence=conf,
            engine=self.name,
            spans=spans,
            caveats=[
                "Pangram (hosted API) — independent benchmarks rank it top-tier "
                "for low false positives.",
                "No detector proves authorship; results are probabilistic.",
                f"Pangram verdict: {data.get('prediction_short') or data.get('headline') or 'n/a'}",
            ],
        )

    # -- public -----------------------------------------------------------
    def score(self, text: str) -> DetectResult:
        if not self.api_key:
            raise RuntimeError(
                "Pangram engine selected but DARREN_PANGRAM_API_KEY is unset."
            )
        if len(text.split()) < 25:
            return DetectResult(
                score=0.0, verdict="insufficient_text", confidence="low",
                engine=self.name,
                caveats=["Need ~25+ words for a reliable estimate."],
            )
        task_id = self._submit(text)
        data = self._poll(task_id)
        return self._to_result(data, text)
