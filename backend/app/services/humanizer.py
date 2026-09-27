"""
Humanizer service — the closed detector-feedback loop.

This is the "untell-style" approach: score -> rewrite only the flagged spans ->
re-score -> repeat until the detector stops flagging OR we run out of rounds.
Every candidate rewrite is gated on a meaning-similarity check so we never
silently mangle the text.

Honest note baked into the response: the local detector is a *proxy*. Clearing
it does not guarantee clearing Turnitin / GPTZero / Originality.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Callable

from ..config import get_settings
from ..engines.factory import get_detector, get_rewriter


def humanize(
    text: str,
    *,
    strength: int = 60,
    max_rounds: int | None = None,
    on_step: Callable[[dict], None] | None = None,
) -> dict:
    s = get_settings()
    detector = get_detector()
    rewriter = get_rewriter()

    rounds = max_rounds or s.humanizer_max_rounds
    target = s.humanizer_target
    best_of = s.humanizer_best_of
    min_sim = s.humanizer_min_similarity

    baseline = detector.score(text)
    current_text = text
    current = baseline
    history = [{
        "round": 0,
        "score": baseline.score,
        "verdict": baseline.verdict,
        "similarity": 1.0,
    }]
    if on_step:
        on_step(history[-1])

    accepted = None
    for rnd in range(1, rounds + 1):
        if current.score <= target:
            break

        flagged = [sp.text for sp in current.spans if sp.flagged][:12]

        # best-of-N: generate several candidates, keep the lowest scorer that
        # still preserves meaning.
        candidates = []
        for _ in range(max(1, best_of)):
            try:
                cand = rewriter.rewrite(current_text, target_spans=flagged, strength=strength)
            except Exception as exc:  # keep the loop alive
                if on_step:
                    on_step({"round": rnd, "error": str(exc)})
                break
            if cand.similarity >= min_sim:
                candidates.append(cand)

        if not candidates:
            break

        scored = []
        for cand in candidates:
            res = detector.score(cand.text)
            scored.append((res.score, cand, res))
        scored.sort(key=lambda x: x[0])
        best_score, best_cand, best_res = scored[0]

        if best_score < current.score:
            current_text = best_cand.text
            current = best_res
            accepted = best_cand

        history.append({
            "round": rnd,
            "score": round(best_score, 4),
            "verdict": best_res.verdict,
            "similarity": best_cand.similarity,
        })
        if on_step:
            on_step(history[-1])

    improved = round(baseline.score - current.score, 4)
    return {
        "input": text,
        "output": current_text,
        "before": {
            "score": baseline.score, "verdict": baseline.verdict,
            "spans": [asdict(sp) for sp in baseline.spans],
        },
        "after": {
            "score": current.score, "verdict": current.verdict,
            "spans": [asdict(sp) for sp in current.spans],
        },
        "rounds": history,
        "improved": improved,
        "meaning_similarity": accepted.similarity if accepted else 1.0,
        "engine": detector.name,
        "rewriter": rewriter.name,
        "caveats": [
            "Local detector is a PROXY — passing it does not guarantee passing "
            "Turnitin, GPTZero or Originality.ai.",
            "Always review the output: verify facts, numbers and citations survived.",
            "Best-of-N is the biggest lever — rerun for a different sample if the score is high.",
        ],
    }
