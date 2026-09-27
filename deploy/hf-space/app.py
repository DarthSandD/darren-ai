"""
Hugging Face ZeroGPU Space for Darren Ai.

Free personal accounts can host up to 2 ZeroGPU Spaces (3.5 GPU-min/day).
This exposes the LAPD detector as a small Gradio demo. For a *product* with
real traffic, deploy the FastAPI backend on a paid GPU (see deploy/runpod).

The @spaces.GPU decorator requests the shared GPU only while scoring runs.
"""
import gradio as gr
import spaces  # provided by HF Spaces

from app.config import get_settings
from app.engines.factory import get_detector

settings = get_settings()


@spaces.GPU(duration=30)
def detect(text: str):
    det = get_detector()
    res = det.score(text)
    label = {"human": "🟢 Likely human",
             "mixed": "🟡 Mixed / uncertain",
             "ai": "🔴 Likely AI",
             "insufficient_text": "⚪ Too short"}[res.verdict]
    return f"{label} — {round(res.score*100)}% AI-like (engine: {res.engine})"


demo = gr.Interface(
    fn=detect,
    inputs=gr.Textbox(lines=10, label="Text to analyse"),
    outputs=gr.Textbox(label="Verdict"),
    title="Darren Ai — AI Text Detector (LAPD)",
    description="Zero-shot AI-text detection. Probabilistic — not proof of authorship.",
)

if __name__ == "__main__":
    demo.launch()
