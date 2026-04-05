#!/usr/bin/env python3
"""
run_ts2vec.py — TS2Vec Time-Series Encoder Runner
==================================================
Architecture : Lightweight dilated causal-TCN (Yue et al., 2022).
               Self-supervised; trained briefly on the target data
               (no labels) then used for inference.

SR behaviour (key difference from HuggingFace encoders)
--------------------------------------------------------
TS2Vec has NO hardcoded sampling rate — it treats audio as a generic
time series.  This runner therefore:

  1. Tries to find processed data for the requested SAMPLING_RATE.
  2. If found  → runs on that SR, logs one MLflow run.
  3. If NOT found:
       - Logs a 'data_not_found' MLflow run for the requested SR.
       - Scans data/processed/ for ALL available SRs.
       - Runs on each available SR independently.
       - Logs a separate MLflow run per SR (each nested under the
         same parent) so every SR is visible in comparisons.

This means even a stale SAMPLING_RATE env var will not silently
produce no output — the encoder always does something useful.

Checkpoint strategy
-------------------
Each (SR, hidden_dim, epochs) combination gets its own checkpoint file
inside data/embeddings/ts2vec/<SR>/.  Subsequent runs skip training.

Environment variables
---------------------
SAMPLING_RATE           int   preferred SR (fallback logic applies if missing)
MLFLOW_TRACKING_URI     str   MLflow server URL
DATA_PATH               str   root data directory  [default: /app/data]
MLFLOW_PARENT_RUN_ID    str   set by orchestrator for nested run  [optional]
BATCH_SIZE              int   inference batch size  [default: 32]
TS2VEC_EPOCHS           int   training epochs if no checkpoint  [default: 10]
TS2VEC_TRAIN_BATCH      int   training batch size  [default: 8]
TS2VEC_HIDDEN           int   encoder hidden dim  [default: 320]
"""

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import encoder_utils as U

ENCODER_NAME = "ts2vec"
MODEL_ID     = "ts2vec-dilated-tcn-v1"


# ══════════════════════════════════════════════════════════════════════════════
# Architecture
# ══════════════════════════════════════════════════════════════════════════════

class _DilatedResBlock(nn.Module):
    """Dilated causal conv block with instance norm + residual connection."""

    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation          # causal left-only padding
        self.conv         = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad)
        self.norm         = nn.InstanceNorm1d(channels, affine=True)
        self.act          = nn.GELU()
        self.drop         = nn.Dropout(p=0.1)
        self._causal_trim = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv(x)
        if self._causal_trim:
            out = out[:, :, : -self._causal_trim]
        return self.act(self.drop(self.norm(out))) + residual


class TS2VecNet(nn.Module):
    """Stacked dilated causal TCN — SR-agnostic time-series encoder."""

    def __init__(self, input_dim: int = 1, hidden_dim: int = 320, n_layers: int = 10, kernel_size: int = 3) -> None:
        super().__init__()
        self.input_proj  = nn.Conv1d(input_dim, hidden_dim, 1)
        self.blocks      = nn.ModuleList([
            _DilatedResBlock(hidden_dim, kernel_size, dilation=2 ** i)
            for i in range(n_layers)
        ])
        self.output_proj = nn.Conv1d(hidden_dim, hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1) → permute → (B, C, T)
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_proj(x)
        return x.mean(dim=-1)     # (B, C)


# ══════════════════════════════════════════════════════════════════════════════
# Contrastive loss
# ══════════════════════════════════════════════════════════════════════════════

def _contrastive_loss(model: TS2VecNet, x: torch.Tensor, device: torch.device, temp: float = 0.07) -> torch.Tensor:
    B, T, _ = x.shape
    half    = T // 2
    t1      = torch.randint(0, T - half, (1,)).item()
    t2      = torch.randint(0, T - half, (1,)).item()

    z1 = F.normalize(model(x[:, t1: t1 + half, :]), dim=-1)
    z2 = F.normalize(model(x[:, t2: t2 + half, :]), dim=-1)

    logits = torch.mm(z1, z2.T) / temp
    labels = torch.arange(B, device=device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


# ══════════════════════════════════════════════════════════════════════════════
# Train / encode helpers
# ══════════════════════════════════════════════════════════════════════════════

def _train(model: TS2VecNet, windows: np.ndarray, device: torch.device, epochs: int, train_batch: int) -> None:
    x      = torch.tensor(windows[:, :, None].astype(np.float32))
    loader = DataLoader(TensorDataset(x), batch_size=train_batch, shuffle=True, drop_last=True)
    opt    = torch.optim.AdamW(model.parameters(), lr=3e-4)

    model.train()
    for epoch in range(1, epochs + 1):
        total = sum(
            _contrastive_loss(model, b[0].to(device), device).backward() or
            (torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0), opt.step(), opt.zero_grad(), b[0].shape[0])[3]
            for b in loader
        )
        # simpler loop for clarity:
        model.train()
        epoch_loss = 0.0
        for (batch,) in loader:
            loss = _contrastive_loss(model, batch.to(device), device)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
        print(f"  [ts2vec train] epoch {epoch}/{epochs}  loss={epoch_loss / len(loader):.4f}", flush=True)
    model.eval()


@torch.no_grad()
def _encode(model: TS2VecNet, windows: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    all_embs = []
    N = len(windows)
    for start, batch_np in U.batch_iter(windows, batch_size):
        end = min(start + batch_size, N)
        print(f"  [ts2vec infer] [{start:>6} – {end:>6}] / {N}", flush=True)
        t   = torch.tensor(batch_np[:, :, None].astype(np.float32)).to(device)
        all_embs.append(model(t).cpu().numpy())
    return np.concatenate(all_embs, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# Per-SR run logic
# ══════════════════════════════════════════════════════════════════════════════

def _run_for_sr(
    cfg: dict,
    sr: int,
    device: torch.device,
    hidden_dim: int,
    epochs: int,
    train_batch: int,
    is_fallback: bool = False,
) -> None:
    """
    Run encoding for a single SR.
    Logs one MLflow run (nested under parent if parent_run_id is set).
    """
    label = f"{'fallback ' if is_fallback else ''}SR={sr} Hz"
    print(f"\n[{ENCODER_NAME}] Processing {label} ...", flush=True)

    if U.embeddings_exist(cfg, sr=sr):
        return

    # Load
    windows = U.load_windows(cfg, sr=sr)
    N, window_size = windows.shape
    print(f"[{ENCODER_NAME}] windows shape={windows.shape}  SR={sr} Hz", flush=True)

    # Checkpoint
    ckpt_dir  = U.get_output_dir(cfg, sr=sr)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"ts2vec_h{hidden_dim}_e{epochs}.pt"

    # Build model
    model = TS2VecNet(input_dim=1, hidden_dim=hidden_dim).to(device)

    t0 = time.perf_counter()

    if ckpt_path.exists():
        print(f"[{ENCODER_NAME}] Loading checkpoint: {ckpt_path}", flush=True)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
    else:
        print(f"[{ENCODER_NAME}] Training {epochs} epochs on SR={sr} Hz data ...", flush=True)
        _train(model, windows, device, epochs=epochs, train_batch=train_batch)
        torch.save(model.state_dict(), ckpt_path)
        print(f"[{ENCODER_NAME}] Checkpoint saved → {ckpt_path}", flush=True)

    # Encode
    embeddings = _encode(model, windows, device, batch_size=cfg["batch_size"])
    runtime    = time.perf_counter() - t0

    print(f"[{ENCODER_NAME}] Embeddings shape={embeddings.shape}  runtime={runtime:.1f}s", flush=True)

    # Save
    out_path  = U.save_embeddings(embeddings, cfg, sr=sr)
    meta_path = U.write_metadata(
        cfg,
        model_id      = MODEL_ID,
        windows_shape = (N, window_size),
        embedding_dim = embeddings.shape[1],
        runtime_sec   = runtime,
        out_path      = out_path,
        sr            = sr,
        extra={
            "hidden_dim":   hidden_dim,
            "train_epochs": epochs,
            "ckpt_path":    str(ckpt_path),
            "is_fallback":  is_fallback,
        },
    )

    # MLflow — one run per SR, tagged so they're easy to filter
    with U.mlflow_start_run(cfg, sr=sr):
        import mlflow
        if is_fallback:
            mlflow.set_tag("fallback", "true")
            mlflow.set_tag("requested_sr", str(cfg["sampling_rate"]))
        U.mlflow_log_all(
            cfg,
            model_id      = MODEL_ID,
            windows_shape = (N, window_size),
            embedding_dim = embeddings.shape[1],
            runtime_sec   = runtime,
            out_path      = out_path,
            meta_path     = meta_path,
            device        = str(device),
            sr            = sr,
            extra_params={
                "ts2vec_hidden_dim":   hidden_dim,
                "ts2vec_train_epochs": epochs,
                "ckpt_path":           str(ckpt_path),
                "is_fallback":         str(is_fallback),
            },
        )


def _log_data_not_found(cfg: dict, sr: int, available_srs: list[int]) -> None:
    """Log a 'data_not_found' MLflow run for the requested-but-missing SR."""
    import mlflow

    mlflow.set_tracking_uri(cfg["mlflow_uri"])
    run_name = f"{cfg['encoder_name']}_{sr}hz"
    kwargs: dict = {"run_name": run_name}
    if cfg["parent_run_id"]:
        kwargs["nested"] = True

    reason = (
        f"processed_{sr}hz/ not found. "
        f"TS2Vec will run on all available SRs instead: {available_srs}."
    )
    print(f"[{ENCODER_NAME}] DATA NOT FOUND for SR={sr} Hz — {reason}", flush=True)

    with mlflow.start_run(**kwargs):
        mlflow.set_tag("status", "data_not_found")
        mlflow.set_tag("fallback_srs", str(available_srs))
        mlflow.log_params({
            "encoder_name":   cfg["encoder_name"],
            "requested_sr":   sr,
            "available_srs":  str(available_srs),
            "status":         "data_not_found",
        })
        mlflow.log_text(reason, "data_not_found_reason.txt")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    cfg = U.load_env_config(ENCODER_NAME)

    epochs      = int(os.environ.get("TS2VEC_EPOCHS",      "10"))
    train_batch = int(os.environ.get("TS2VEC_TRAIN_BATCH", "8"))
    hidden_dim  = int(os.environ.get("TS2VEC_HIDDEN",      "320"))

    print(f"[{ENCODER_NAME}] ── Starting encoder ──────────────────────────", flush=True)
    print(f"[{ENCODER_NAME}] Requested SR={cfg['sampling_rate']} Hz  hidden_dim={hidden_dim}  epochs={epochs}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{ENCODER_NAME}] device={device}", flush=True)

    requested_sr  = cfg["sampling_rate"]
    processed_dir = U.get_processed_dir(cfg, sr=requested_sr)

    run_kwargs = dict(
        cfg         = cfg,
        device      = device,
        hidden_dim  = hidden_dim,
        epochs      = epochs,
        train_batch = train_batch,
    )

    if processed_dir.exists() and (processed_dir / "combined_windows.npy").exists():
        # ── Happy path: requested SR data exists ──────────────────────────────
        _run_for_sr(**run_kwargs, sr=requested_sr, is_fallback=False)

    else:
        # ── Fallback: scan for all available SRs ─────────────────────────────
        available_srs = U.find_available_srs(cfg)

        if not available_srs:
            print(
                f"[{ENCODER_NAME}] No processed data found anywhere under "
                f"{cfg['data_path'] / 'processed'}. "
                "Run the preprocessing service first.",
                flush=True,
            )
            # Still log so the missing run is visible in MLflow
            _log_data_not_found(cfg, sr=requested_sr, available_srs=[])
            return

        print(
            f"[{ENCODER_NAME}] SR={requested_sr} Hz not found. "
            f"Available SRs: {available_srs}. Running on all of them.",
            flush=True,
        )

        # Log the 'data_not_found' entry for the requested SR
        _log_data_not_found(cfg, sr=requested_sr, available_srs=available_srs)

        # Run on each available SR
        for sr in available_srs:
            _run_for_sr(**run_kwargs, sr=sr, is_fallback=True)

    print(f"[{ENCODER_NAME}] ── Done ──────────────────────────────────────", flush=True)


if __name__ == "__main__":
    main()
