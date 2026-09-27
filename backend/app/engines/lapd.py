"""
LAPD detector — "Alignment Imprint" zero-shot AI-text detection.

Implements the mechanism from:
  "Alignment Imprint: Zero-Shot AI-Generated Text Detection via Provable
   Preference Discrepancy" (creator-xi/LAPD, arXiv:2604.16923)

Core idea
---------
Modern user-facing LLMs are alignment-tuned (SFT + preference tuning). That
leaves a measurable *distributional imprint*: an aligned model assigns extra
probability to text it (or a sibling aligned model) produced, beyond what the
general language modelling of its own base model would predict.

For each token we compute the per-token log-prob under:
    - the BASE model      (logp_base)
    - the ALIGNED model   (logp_aligned)
and form the "Alignment Imprint" discrepancy. LAPD standardizes it with an
information weight so high-entropy regions don't dominate, then averages.

    imprint_t   = logp_aligned_t - logp_base_t
    weight_t    = p_base_t  (information weight; low-entropy tokens count more)
    LAPD        = sum_t weight_t * imprint_t / sum_t weight_t

AI text tends to score HIGHER than human text. We map the raw statistic to a
0..1 score with a logistic centred on `threshold`.

This file is model-agnostic: pass any base/aligned pair (e.g. Qwen2.5-1.5B /
Qwen2.5-1.5B-Instruct). Heavy imports (torch/transformers) are lazy so the API
boots on a CPU-only box that never calls the real engine.
"""
from __future__ import annotations

import math
import re
from typing import Optional

from .base import DetectResult, Span

_SENT_RE = re.compile(r"[^.!?]+[.!?]?")


def _split_sentences(text: str) -> list[tuple[str, int, int]]:
    out = []
    for m in _SENT_RE.finditer(text):
        s = m.group().strip()
        if len(s) >= 8:
            out.append((s, m.start(), m.end()))
    return out


class LAPDDetector:
    name = "lapd"

    def __init__(
        self,
        base_model: str,
        aligned_model: str,
        *,
        device: str = "auto",
        max_tokens: int = 512,
        threshold: float = 0.5,
        torch_dtype: str = "float16",
    ) -> None:
        self.base_model_name = base_model
        self.aligned_model_name = aligned_model
        self.max_tokens = max_tokens
        self.threshold = threshold
        self._device_pref = device
        self._dtype_pref = torch_dtype
        self._base = None
        self._aligned = None
        self._tok = None
        self._device = None

    # -- lazy load --------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._base is not None:
            return
        import torch  # noqa: WPS433 (lazy)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self._device_pref == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self._device_pref

        dtype = getattr(torch, self._dtype_pref, torch.float32)
        if self._device == "cpu":
            dtype = torch.float32

        self._tok = AutoTokenizer.from_pretrained(self.base_model_name)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token

        self._base = (
            AutoModelForCausalLM.from_pretrained(self.base_model_name, torch_dtype=dtype)
            .to(self._device).eval()
        )
        self._aligned = (
            AutoModelForCausalLM.from_pretrained(self.aligned_model_name, torch_dtype=dtype)
            .to(self._device).eval()
        )

    # -- core math --------------------------------------------------------
    def _token_logprobs(self, text: str) -> tuple[list[float], list[float]]:
        """Return per-token (logp_base, logp_aligned) for text."""
        import torch

        enc = self._tok(text, return_tensors="pt", truncation=True,
                        max_length=self.max_tokens)
        ids = enc["input_ids"].to(self._device)
        if ids.shape[1] < 3:
            return [], []

        with torch.inference_mode():
            out_b = self._base(ids).logits
            out_a = self._aligned(ids).logits

        # shift: predict token t from tokens < t
        logp_b = torch.log_softmax(out_b[:, :-1, :].float(), dim=-1)
        logp_a = torch.log_softmax(out_a[:, :-1, :].float(), dim=-1)
        tgt = ids[:, 1:]

        lp_b = logp_b.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]
        lp_a = logp_a.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]
        return lp_b.tolist(), lp_a.tolist()

    def _raw_lapd(self, text: str) -> Optional[float]:
        lp_b, lp_a = self._token_logprobs(text)
        if not lp_b:
            return None
        num = 0.0
        den = 0.0
        for b, a in zip(lp_b, lp_a):
            imprint = a - b                 # alignment imprint for this token
            weight = math.exp(b)            # p_base: information weight
            num += weight * imprint
            den += weight
        if den == 0:
            return None
        return num / den

    def _to_score(self, raw: float) -> float:
        """Logistic map of the raw statistic to 0..1 (threshold -> 0.5)."""
        # scale so a 1.0-nat shift spans most of the range; tune with data
        k = 4.0
        return 1.0 / (1.0 + math.exp(-k * (raw - self._calibration_shift())))

    def _calibration_shift(self) -> float:
        # raw imprint for AI text is typically small positive; centre the
        # logistic near 0 by default. Replace with a measured value.
        return 0.0

    # -- public -----------------------------------------------------------
    def score(self, text: str) -> DetectResult:
        words = text.split()
        if len(words) < 25:
            return DetectResult(
                score=0.0, verdict="insufficient_text", confidence="low",
                engine=self.name,
                caveats=["LAPD needs ~25+ words; short text is unreliable."],
            )
        self._ensure_loaded()

        raw = self._raw_lapd(text)
        if raw is None:
            return DetectResult(0.0, "insufficient_text", "low", self.name,
                                caveats=["Could not tokenize text."])

        overall = self._to_score(raw)

        # sentence-level heatmap (recompute per sentence — costlier but gives
        # the UI its per-span view; disabled for very long inputs)
        spans: list[Span] = []
        if len(words) <= 900:
            for s, a, b in _split_sentences(text):
                r = self._raw_lapd(s)
                sc = self._to_score(r) if r is not None else overall
                spans.append(Span(s, a, b, round(sc, 4), sc >= self.threshold))

        if overall < 0.35:
            verdict, conf = "human", "medium"
        elif overall < 0.6:
            verdict, conf = "mixed", "low"
        else:
            verdict, conf = "ai", "high"

        return DetectResult(
            score=round(overall, 4), verdict=verdict, confidence=conf,
            engine=self.name, spans=spans,
            caveats=[
                "Detectors are probabilistic; not proof of authorship.",
                "Accuracy drops on paraphrased/humanized or heavily edited text.",
            ],
        )
