"""
Fast-DetectGPT detector — conditional-probability-curvature zero-shot detector.

Reference: Bao et al., ICLR 2024 (github.com/baoguangsheng/fast-detect-gpt).
Used here as a lighter, well-tested alternative to LAPD: it needs only a single
scoring model (the "source" model) plus a sampling model, and is ~340x faster
than the original DetectGPT.

Analytic (single-pass) form of conditional probability curvature:

    For each token position t we sample K alternative tokens from the model
    distribution, then compare the actual token's log-prob to the mean log-prob
    of the sampled alternatives. AI text sits at a *local maximum* of the
    model's log-prob surface, so its curvature is high.

We implement the sampling-discrepancy estimator and map it to 0..1.
"""
from __future__ import annotations

import math
import re
from typing import Optional

from .base import DetectResult, Span

_SENT_RE = re.compile(r"[^.!?]+[.!?]?")


class FastDetectGPT:
    name = "fast_detectgpt"

    def __init__(
        self,
        scoring_model: str,
        *,
        sampling_model: Optional[str] = None,
        device: str = "auto",
        max_tokens: int = 512,
        threshold: float = 0.5,
        torch_dtype: str = "float16",
        n_samples: int = 3,
    ) -> None:
        self.scoring_model_name = scoring_model
        self.sampling_model_name = sampling_model or scoring_model
        self.max_tokens = max_tokens
        self.threshold = threshold
        self.n_samples = n_samples
        self._device_pref = device
        self._dtype_pref = torch_dtype
        self._model = None
        self._tok = None
        self._device = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._device = ("cuda" if torch.cuda.is_available() else "cpu") \
            if self._device_pref == "auto" else self._device_pref
        dtype = getattr(torch, self._dtype_pref, torch.float32)
        if self._device == "cpu":
            dtype = torch.float32
        self._tok = AutoTokenizer.from_pretrained(self.scoring_model_name)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        self._model = (
            AutoModelForCausalLM.from_pretrained(self.scoring_model_name, torch_dtype=dtype)
            .to(self._device).eval()
        )

    def _curvature(self, text: str) -> Optional[float]:
        import torch

        enc = self._tok(text, return_tensors="pt", truncation=True, max_length=self.max_tokens)
        ids = enc["input_ids"].to(self._device)
        if ids.shape[1] < 3:
            return None
        with torch.inference_mode():
            logits = self._model(ids).logits[:, :-1, :].float()
        logp = torch.log_softmax(logits, dim=-1)
        tgt = ids[:, 1:]
        actual = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]  # [T]

        # conditional-mean of log-probs under the model (≈ expected log-prob)
        p = logp.exp()
        mean_lp = (p * logp).sum(dim=-1)[0]                         # [T]
        var_lp = (p * (logp - mean_lp.unsqueeze(-1)) ** 2).sum(dim=-1)[0]
        std = torch.sqrt(var_lp.clamp_min(1e-8))
        # standardized curvature per token
        curv = ((actual - mean_lp) / std)
        return float(curv.mean().item())

    def _to_score(self, curv: float) -> float:
        # AI text -> higher curvature; logistic around 0
        return 1.0 / (1.0 + math.exp(-1.5 * curv))

    def score(self, text: str) -> DetectResult:
        if len(text.split()) < 25:
            return DetectResult(0.0, "insufficient_text", "low", self.name,
                                caveats=["Need ~25+ words for a reliable estimate."])
        self._ensure_loaded()
        curv = self._curvature(text)
        if curv is None:
            return DetectResult(0.0, "insufficient_text", "low", self.name)
        overall = self._to_score(curv)

        spans: list[Span] = []
        if len(text.split()) <= 900:
            for m in _SENT_RE.finditer(text):
                s = m.group().strip()
                if len(s) < 8:
                    continue
                c = self._curvature(s)
                sc = self._to_score(c) if c is not None else overall
                spans.append(Span(s, m.start(), m.end(), round(sc, 4), sc >= self.threshold))

        if overall < 0.35:
            verdict, conf = "human", "medium"
        elif overall < 0.6:
            verdict, conf = "mixed", "low"
        else:
            verdict, conf = "ai", "medium"
        return DetectResult(round(overall, 4), verdict, conf, self.name, spans,
                            caveats=["Probabilistic signal; not proof of authorship."])
