"""FastAPI application: detection + humanizer endpoints."""
from __future__ import annotations

import json
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .config import get_settings
from .engines.factory import get_detector, get_rewriter
from .schemas import (DetectRequest, DetectResponse, HealthResponse,
                      HumanizeRequest)
from .services.humanizer import humanize

settings = get_settings()
app = FastAPI(title=settings.app_name, version=settings.version)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    det = get_detector()
    try:
        rw = get_rewriter().name
    except Exception:
        rw = "unavailable"
    return HealthResponse(status="ok", engine=det.name, rewriter=rw,
                          version=settings.version)


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest) -> DetectResponse:
    if len(req.text) > settings.max_input_chars:
        raise HTTPException(413, f"Input exceeds {settings.max_input_chars} chars.")
    detector = get_detector()
    if req.threshold is not None:
        detector.threshold = req.threshold  # type: ignore[attr-defined]
    res = detector.score(req.text)
    payload = asdict(res)
    return DetectResponse(**payload)


@app.post("/humanize")
def humanize_endpoint(req: HumanizeRequest) -> StreamingResponse:
    """
    Server-Sent-Events stream so the UI can animate each loop round live.
    """
    if len(req.text) > settings.max_input_chars:
        raise HTTPException(413, f"Input exceeds {settings.max_input_chars} chars.")

    def event_stream():
        queue: list[dict] = []

        def on_step(step: dict) -> None:
            queue.append(step)

        # run synchronously but stream intermediate steps as they land
        result = humanize(req.text, strength=req.strength,
                          max_rounds=req.max_rounds, on_step=on_step)
        for step in queue:
            yield f"event: step\ndata: {json.dumps(step)}\n\n"
        yield f"event: result\ndata: {json.dumps(result)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api")
def root() -> dict:
    return {"app": settings.app_name, "docs": "/docs", "health": "/health"}


# --- serve the web UI from the same origin (production) -----------------
# Mounted last so it never shadows the API routes above.
_web_dir = Path(__file__).resolve().parents[2] / "web"
if _web_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(_web_dir), html=True), name="web")
