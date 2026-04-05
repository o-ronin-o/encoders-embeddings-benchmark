#!/usr/bin/env python3
"""
run_wav2vec.py — Wav2Vec 2.0 Encoder Runner
=============================================
Model  : facebook/wav2vec2-base-960h  (HuggingFace)
Input  : (batch, sequence_length) float32 waveform @ 16 kHz
Output : embeddings.npy  shape (N, 768)  — mean-pooled last hidden state

Environment variables
---------------------
SAMPLING_RATE           int   source sampling rate of the processed windows
MLFLOW_TRACKING_URI     str   MLflow server URL
DATA_PATH               str   root data directory  [default: /app/data]
MLFLOW_PARENT_RUN_ID    str   set by orchestrator for nested run  [optional]
BATCH_SIZE              int   windows per forward pass  [default: 16]
"""

import time
import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model

import encoder_utils as U

ENCODER_NAME = "wav2vec"
MODEL_ID     = "facebook/wav2vec2-base-960h"
TARGET_SR    = 8_000


def main() -> None:
    cfg = U.load_env_config(ENCODER_NAME)

    print(f"[{ENCODER_NAME}] ── Starting encoder ──────────────────────────", flush=True)
    print(f"[{ENCODER_NAME}] sampling_rate={cfg['sampling_rate']}  batch_size={cfg['batch_size']}", flush=True)

    if U.embeddings_exist(cfg):
        return

    # ── Load data ─────────────────────────────────────────────────────────────
    windows = U.load_windows(cfg)          # (N, window_size)
    N, window_size = windows.shape

    # ── Resample to 16 kHz if needed ─────────────────────────────────────────
    windows = U.resample_batch(windows, cfg["sampling_rate"], TARGET_SR)

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{ENCODER_NAME}] device={device}", flush=True)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[{ENCODER_NAME}] Loading {MODEL_ID} ...", flush=True)
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model     = Wav2Vec2Model.from_pretrained(MODEL_ID)
    model.eval().to(device)
    print(f"[{ENCODER_NAME}] Model ready.", flush=True)

    # ── Encode ────────────────────────────────────────────────────────────────
    all_embeddings = []
    t0 = time.perf_counter()

    with torch.no_grad():
        for start, batch in U.batch_iter(windows, cfg["batch_size"]):
            end = min(start + cfg["batch_size"], N)
            print(f"[{ENCODER_NAME}] Encoding [{start:>6} – {end:>6}] / {N}", flush=True)

            # processor normalises to zero-mean unit-variance per sample
            inputs = processor(
                list(batch.astype(np.float32)),
                sampling_rate=TARGET_SR,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs.input_values.to(device)   # (B, T)

            outputs      = model(input_values)
            hidden       = outputs.last_hidden_state          # (B, T', H)
            pooled       = hidden.mean(dim=1)                 # (B, H)

            print(f"  batch hidden shape={tuple(hidden.shape)}  pooled={tuple(pooled.shape)}", flush=True)
            all_embeddings.append(pooled.cpu().numpy())

    runtime    = time.perf_counter() - t0
    embeddings = np.concatenate(all_embeddings, axis=0)     # (N, H)
    print(f"[{ENCODER_NAME}] Final embeddings shape: {embeddings.shape}  runtime={runtime:.1f}s", flush=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path  = U.save_embeddings(embeddings, cfg)
    meta_path = U.write_metadata(
        cfg,
        model_id      = MODEL_ID,
        windows_shape = (N, window_size),
        embedding_dim = embeddings.shape[1],
        runtime_sec   = runtime,
        out_path      = out_path,
    )

    # ── MLflow ────────────────────────────────────────────────────────────────
    with U.mlflow_start_run(cfg):
        U.mlflow_log_all(
            cfg,
            model_id      = MODEL_ID,
            windows_shape = (N, window_size),
            embedding_dim = embeddings.shape[1],
            runtime_sec   = runtime,
            out_path      = out_path,
            meta_path     = meta_path,
            device        = str(device),
        )

    print(f"[{ENCODER_NAME}] ── Done ──────────────────────────────────────", flush=True)


if __name__ == "__main__":
    main()
