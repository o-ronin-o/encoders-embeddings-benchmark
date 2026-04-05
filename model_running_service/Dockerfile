FROM python:3.11-slim

WORKDIR /app

# ── System deps (libsndfile for librosa/soundfile, ffmpeg as audio backend) ──
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps — split into two layers so heavy torch layer is cached ────────
#    Layer 1: torch CPU wheel (largest, changes rarely)
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

#    Layer 2: everything else (changes more often)
RUN pip install --no-cache-dir \
        transformers>=4.36 \
        huggingface_hub>=0.20 \
        numpy>=1.24 \
        librosa>=0.10 \
        soundfile \
        mlflow>=2.10

# ── Copy encoder scripts last (changes most often → cheapest cache miss) ──────
COPY *.py .

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

# Default: overridden by docker-compose command:
CMD ["python", "run_wav2vec.py"]
