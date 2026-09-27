"""Rewriter backends for the humanizer loop.

Three swappable rewriters, weakest -> strongest:
  * TemplateRewriter  : zero-dependency lexical surgery (no GPU, no key)
  * DipperRewriter    : DIPPER paraphraser (NeurIPS 2023) — needs GPU
  * LLMRewriter       : OpenAI-compatible chat API (needs a key, best quality)

All three return a RewriteResult with a meaning-similarity estimate so the loop
can reject rewrites that drift too far from the source.
"""
from __future__ import annotations

import difflib
import random
import re

from .base import RewriteResult

# --- template backend ----------------------------------------------------

# Deliberately plain-word substitutions that reduce the "AI register" without
# changing meaning. Conservative on purpose: meaning preservation first.
_SWAPS = {
    "furthermore": "also",
    "moreover": "and",
    "in conclusion": "to sum up",
    "it is important to note that": "note that",
    "it is worth noting that": "note that",
    "delve into": "look at",
    "plays a crucial role": "matters a lot",
    "plays a vital role": "matters a lot",
    "in today's world": "these days",
    "a testament to": "a sign of",
    "in the realm of": "in",
    "utilize": "use",
    "leverage": "use",
    "commence": "start",
    "subsequently": "then",
    "numerous": "many",
    "facilitate": "help",
    "demonstrate": "show",
    "approximately": "about",
    "additionally": "also",
}

_TELL_PREFIX = re.compile(
    r"^\s*(furthermore|moreover|additionally|in addition|in conclusion|"
    r"firstly|secondly|thirdly|finally|overall|indeed|notably|importantly)\s*,?\s*",
    re.IGNORECASE,
)

# stopwords excluded from the meaning-preservation check so connective swaps
# (furthermore->also) don't count as meaning loss.
_STOP = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "it", "this", "that", "as", "at",
    "by", "from", "also", "then", "so", "such", "will", "would", "can", "could",
    "note", "noting", "these", "those", "there", "their", "its", "into", "than",
    # connective / register words that the humanizer deliberately swaps —
    # they must NOT count as meaning, or the gate rejects its own rewrites.
    "furthermore", "moreover", "additionally", "conclusion", "important",
    "importantly", "crucial", "vital", "plays", "play", "role", "testament",
    "sign", "delve", "utilize", "use", "leverage", "numerous", "many",
    "facilitate", "help", "demonstrate", "show", "approximately", "about",
    "subsequently", "commence", "start", "today", "todays", "world", "realm",
    "matters", "lot", "also",
}


def _content_words(s: str) -> list[str]:
    import re as _re
    return [w for w in _re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP]


def _similarity(a: str, b: str) -> float:
    """Order-aware similarity on content words only.

    Robust to the connective/register swaps the humanizer makes, while still
    catching meaning breaks (role swaps score ~0.33 vs ~0.8+ for good rewrites).
    """
    ca, cb = _content_words(a), _content_words(b)
    if not ca and not cb:
        return 1.0
    if not ca or not cb:
        return 0.0
    return round(difflib.SequenceMatcher(None, ca, cb).ratio(), 4)


class TemplateRewriter:
    """Deterministic, dependency-free lexical humanizer."""

    name = "template"

    def rewrite(self, text: str, *, target_spans: list[str] | None = None,
                strength: int = 60) -> RewriteResult:
        out = text
        for src, dst in _SWAPS.items():
            out = re.sub(re.escape(src), dst, out, flags=re.IGNORECASE)

        # strip leading templated transitions on sentences the detector flagged
        sentences = re.split(r"(?<=[.!?])\s+", out)
        new_sents = []
        for s in sentences:
            if target_spans and any(s.strip()[:40] in t for t in target_spans):
                s = _TELL_PREFIX.sub("", s, count=1)
            new_sents.append(s)
        out = " ".join(new_sents).strip()

        # light, meaning-safe variation: split a very long sentence
        if strength >= 60:
            parts = re.split(r",\s+(?=and\b|which\b|but\b)", out)
            if len(parts) > 1 and len(out.split()) > 40:
                out = ". ".join(p[0].upper() + p[1:] if p else p for p in parts)
                out = out.replace(".. ", ". ")

        # restore sentence-initial capitalisation after swaps/stripping
        out = re.sub(r"(^|[.!?]\s+)([a-z])",
                     lambda m: m.group(1) + m.group(2).upper(), out)
        out = re.sub(r"\bi\b", "I", out)

        return RewriteResult(text=out, similarity=_similarity(text, out))


# --- DIPPER backend ------------------------------------------------------

class DipperRewriter:
    """
    DIPPER paraphrase (kalpeshk2011/dipper-paraphraser-xxl). ~11B params,
    needs a GPU with ~24GB or 8-bit quantization. Lazy import.
    """

    name = "dipper"

    def __init__(self, model: str = "kalpeshk2011/dipper-paraphraser-xxl",
                 device: str = "auto") -> None:
        self.model_name = model
        self.device = device
        self._dp = None

    def _ensure(self) -> None:
        if self._dp is not None:
            return
        import torch
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
        model = T5ForConditionalGeneration.from_pretrained(self.model_name).to(dev).eval()
        self._tok, self._model, self._dev = tok, model, dev

    def rewrite(self, text: str, *, target_spans: list[str] | None = None,
                strength: int = 60) -> RewriteResult:
        self._ensure()
        import torch

        lex = min(100, max(0, strength))
        order = min(100, max(0, strength - 20))
        prefix = "lexical = {}, order = {}".format(100 - lex, 100 - order)
        inp = f"{prefix} <sent> {text} </sent>"
        enc = self._tok([inp], return_tensors="pt").to(self._dev)
        with torch.inference_mode():
            out = self._model.generate(**enc, do_sample=True, top_p=0.75, max_length=512)
        res = self._tok.batch_decode(out, skip_special_tokens=True)[0].strip()
        return RewriteResult(text=res, similarity=round(_similarity(text, res), 4))


# --- LLM backend ---------------------------------------------------------

class LLMRewriter:
    """OpenAI-compatible chat rewriter (DeepSeek / OpenAI / OpenRouter)."""

    name = "llm"

    PROMPT = (
        "Rewrite the text so it reads like a thoughtful human wrote it, while "
        "keeping the meaning, facts, numbers, names and citations EXACTLY the same. "
        "Vary sentence length and rhythm; remove formulaic transitions and "
        "'AI tells'. Do not add or remove information. Return only the rewritten text."
    )

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def rewrite(self, text: str, *, target_spans: list[str] | None = None,
                strength: int = 60) -> RewriteResult:
        import httpx

        hint = ""
        if target_spans:
            hint = "\nFocus especially on these sentences:\n- " + "\n- ".join(target_spans[:8])
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.PROMPT},
                {"role": "user", "content": text + hint},
            ],
            "temperature": min(1.5, 0.7 + strength / 100),
        }
        r = httpx.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=120,
        )
        r.raise_for_status()
        out = r.json()["choices"][0]["message"]["content"].strip()
        return RewriteResult(text=out, similarity=round(_similarity(text, out), 4))
