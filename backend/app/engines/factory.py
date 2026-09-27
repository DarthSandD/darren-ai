"""Factory: build the configured detector / rewriter from settings."""
from __future__ import annotations

from functools import lru_cache

from ..config import get_settings
from .base import Detector, Rewriter
from .stub import StubDetector


@lru_cache
def get_detector() -> Detector:
    s = get_settings()
    engine = s.detector_engine.lower()
    if engine == "lapd":
        from .lapd import LAPDDetector
        return LAPDDetector(
            s.base_model, s.aligned_model,
            device=s.device, max_tokens=s.max_tokens,
            threshold=s.detect_threshold, torch_dtype=s.torch_dtype,
        )
    if engine in ("fast_detectgpt", "fast-detectgpt"):
        from .fast_detectgpt import FastDetectGPT
        return FastDetectGPT(
            s.base_model, sampling_model=s.aligned_model,
            device=s.device, max_tokens=s.max_tokens,
            threshold=s.detect_threshold, torch_dtype=s.torch_dtype,
        )
    return StubDetector(threshold=s.detect_threshold)


def get_rewriter() -> Rewriter:
    s = get_settings()
    backend = s.humanizer_rewriter.lower()
    if backend == "dipper":
        from .rewriters import DipperRewriter
        return DipperRewriter(device=s.device)
    if backend == "llm":
        from .rewriters import LLMRewriter
        if not s.llm_base_url or not s.llm_api_key:
            raise RuntimeError("LLM rewriter selected but DARREN_LLM_API_KEY is unset.")
        return LLMRewriter(s.llm_base_url, s.llm_api_key, s.llm_model)
    from .rewriters import TemplateRewriter
    return TemplateRewriter()
