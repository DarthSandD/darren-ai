"""
Darren Ai backend configuration.

Everything is env-driven so the same image runs locally, on Colab/Kaggle (dev),
on Hugging Face ZeroGPU (free demo), or on RunPod (paid production).
"""
from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DARREN_", env_file=".env", extra="ignore")

    app_name: str = "Darren Ai"
    version: str = "0.1.0"

    # --- Detector engine -------------------------------------------------
    # Which detector backend to use.
    #   "pangram"       -> hosted commercial API (BEST accuracy, no GPU/CPU
    #                      needed locally — proxies to Pangram's servers)
    #   "lapd"          -> LAPD (alignment-imprint; needs base+aligned pair + GPU)
    #   "fast_detectgpt"-> Fast-DetectGPT fallback (single model pair + GPU)
    #   "stub"          -> deterministic heuristic, NO GPU (CI / dev only)
    detector_engine: str = "stub"

    # Pangram hosted API (detector_engine="pangram"). Key comes from the
    # environment — never hard-code it. Get one at pangram.com.
    pangram_api_key: str = ""
    pangram_model: str = "default"

    # Base + aligned model pair for LAPD / Fast-DetectGPT.
    # Small pair chosen so it fits a free 16GB T4.
    base_model: str = "Qwen/Qwen2.5-1.5B"
    aligned_model: str = "Qwen/Qwen2.5-1.5B-Instruct"

    device: str = "auto"          # auto | cuda | cpu
    max_tokens: int = 512         # truncate long documents
    torch_dtype: str = "float16"  # float16 on GPU, float32 on CPU

    # Detection decision threshold on the normalized LAPD statistic.
    # Higher score => more AI-like. Calibrate on your own data.
    detect_threshold: float = 0.5

    # --- Humanizer (detector-feedback loop) ------------------------------
    # "rewriter" backend: "template" (no deps), "dipper" (GPU), "llm" (API key)
    humanizer_rewriter: str = "template"
    humanizer_max_rounds: int = 5
    humanizer_target: float = 0.35        # stop when score drops below this
    humanizer_best_of: int = 3            # independent rewrites per round
    humanizer_min_similarity: float = 0.6   # reject meaning-changing rewrites

    # Optional LLM rewriter (OpenAI-compatible). Leave blank to disable.
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"

    # --- Limits ----------------------------------------------------------
    max_input_chars: int = 60_000
    cors_origins: str = "*"   # comma-separated list in prod


@lru_cache
def get_settings() -> Settings:
    return Settings()
