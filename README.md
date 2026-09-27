# Darren Ai — AI Text Detector & Humanizer

Detect AI-generated text with the **LAPD** zero-shot detector, then rewrite
flagged sentences until they read human — with a detector-in-the-loop and a
meaning-preservation gate. One codebase ships as a **website**, a **desktop app**
and an **Android app**.

> ⚠️ **Honest scope.** AI detectors are probabilistic and **not proof of
> authorship**. Accuracy drops sharply on paraphrased, humanized, translated or
> heavily edited text — for every detector, including Turnitin and ZeroGPT.
> Review all output. Do not use this to misrepresent authorship or evade
> institutional policies.

---

## Architecture

```
  web (Next-less static UI) ─┐
  Tauri desktop             ─┼─▶  FastAPI backend  ─▶ LAPD detector (GPU)
  Capacitor Android         ─┘                     ─▶ Humanizer loop
```

- **Detector** (`backend/app/engines/lapd.py`): LAPD "Alignment Imprint" —
  compares per-token log-probs under a base model and its aligned twin.
- **Humanizer** (`backend/app/services/humanizer.py`): score → rewrite flagged
  spans → re-score → repeat, keeping the lowest scorer that preserves meaning.
- **Rewriters**: `template` (no deps), `dipper` (GPU), `llm` (API key).
- **Stub engine**: heuristic, no GPU — lets the whole stack run in dev/CI.

## Quick start (no GPU needed)

```bash
cd backend
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install fastapi "uvicorn[standard]" pydantic pydantic-settings httpx pytest
DARREN_DETECTOR_ENGINE=stub uvicorn app.main:app --reload --port 8000
```

Open <http://localhost:8000> — the UI is served from the same origin.
Run tests: `pytest -q`.

## Running the real LAPD engine (GPU)

```bash
pip install torch transformers accelerate sentencepiece
export DARREN_DETECTOR_ENGINE=lapd
export DARREN_BASE_MODEL=Qwen/Qwen2.5-1.5B
export DARREN_ALIGNED_MODEL=Qwen/Qwen2.5-1.5B-Instruct
uvicorn app.main:app --port 8000
```

Pick a base/aligned pair from the same family. Larger pairs = better accuracy,
more VRAM. The 1.5B pair fits a 16 GB T4.

## Desktop (Tauri)

```bash
npm install
npm run desktop:dev      # live dev window
npm run desktop:build    # .exe / .dmg / .AppImage
```

## Android (Capacitor)

```bash
npm install
npm run android:add
npm run android:sync
npm run android:open     # builds in Android Studio
```

Set the API host in `capacitor.config.ts` (or `window.DARREN_API`) to your
deployed backend.

## Deploy

| Target | Where | Notes |
|---|---|---|
| Free demo | `deploy/hf-space/` | HF ZeroGPU, 3.5 GPU-min/day |
| Production | `deploy/runpod/Dockerfile` | Rented GPU, real traffic |
| Web + API | `deploy/Dockerfile.backend` | CPU image (stub or small models) |

## Configuration

All settings are env vars prefixed `DARREN_` (see `backend/app/config.py`).

## License & ethics

Provided for research and personal use. You are responsible for how you use it.
