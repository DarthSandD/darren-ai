"""Tests for the Darren Ai backend (stub engine, no GPU)."""
import json
import os

os.environ["DARREN_DETECTOR_ENGINE"] = "stub"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)

AI_TEXT = (
    "Furthermore, it is important to note that artificial intelligence plays a "
    "crucial role in today's world. Moreover, it is a testament to human "
    "ingenuity. In conclusion, AI will delve into every realm of society and "
    "facilitate numerous improvements for humanity going forward."
)

HUMAN_TEXT = (
    "I burnt the toast again. My kid laughed, I laughed, and we ate cereal "
    "instead. Some mornings just go like that, you know? No big deal."
)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["engine"] == "stub"


def test_detect_ai_scores_higher_than_human():
    ai = client.post("/detect", json={"text": AI_TEXT}).json()
    hu = client.post("/detect", json={"text": HUMAN_TEXT}).json()
    assert ai["score"] > hu["score"]


def test_detect_short_text_guard():
    r = client.post("/detect", json={"text": "Too short."}).json()
    assert r["verdict"] == "insufficient_text"


def test_humanize_reduces_score():
    r = client.post("/humanize", json={"text": AI_TEXT, "strength": 80})
    assert r.status_code == 200
    # SSE stream: find the final result event
    result = None
    for line in r.text.splitlines():
        if line.startswith("data: ") and '"before"' in line:
            result = json.loads(line[6:])
    assert result is not None
    assert result["after"]["score"] <= result["before"]["score"]
    assert result["output"]


def test_input_limit():
    big = "x " * 40000
    r = client.post("/detect", json={"text": big})
    assert r.status_code == 413
