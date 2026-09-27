# Darren Ai backend for GPU hosts (RunPod / Vast / Lambda).
# CUDA 12.1 + torch cu121, then installs app deps.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip git curl && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cu121 \
    && pip install -r requirements.txt

COPY backend/app ./app
EXPOSE 8000

# Pre-download the LAPD model pair at build time so cold start is fast.
ARG BASE_MODEL=Qwen/Qwen2.5-1.5B
ARG ALIGNED_MODEL=Qwen/Qwen2.5-1.5B-Instruct
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
AutoTokenizer.from_pretrained('${BASE_MODEL}'); \
AutoModelForCausalLM.from_pretrained('${BASE_MODEL}'); \
AutoTokenizer.from_pretrained('${ALIGNED_MODEL}'); \
AutoModelForCausalLM.from_pretrained('${ALIGNED_MODEL}')" || echo "prefetch skipped"

ENV DARREN_DETECTOR_ENGINE=lapd
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
