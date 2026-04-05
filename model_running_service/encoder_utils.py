"""
encoder_utils.py — Shared helpers for all encoder runner scripts.
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Generator, Optional

import numpy as np


# ── Environment ───────────────────────────────────────────────────────────────

def load_env_config(encoder_name: str) -> dict:
    def _require(key: str) -> str:
        val = os.environ.get(key)
        if val is None:
            print(f"[{encoder_name}][ERROR] Required env var '{key}' is not set.", flush=True)
            sys.exit(1)
        return val

    return {
        "encoder_name":  encoder_name,
        "data_path":     Path(os.environ.get("DATA_PATH", "/app/data")),
        "sampling_rate": int(_require("SAMPLING_RATE")),
        "mlflow_uri":    os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001"),
        "parent_run_id": os.environ.get("MLFLOW_PARENT_RUN_ID"),
        "batch_size":    int(os.environ.get("BATCH_SIZE", "16")),
    }


# ── Path helpers ──────────────────────────────────────────────────────────────

def get_processed_dir(cfg: dict, sr: Optional[int] = None) -> Path:
    sr = sr if sr is not None else cfg["sampling_rate"]
    return cfg["data_path"] / "processed" / f"processed_{sr}hz"


def get_output_dir(cfg: dict, sr: Optional[int] = None) -> Path:
    sr = sr if sr is not None else cfg["sampling_rate"]
    return cfg["data_path"] / "embeddings" / cfg["encoder_name"] / str(sr)


def embeddings_exist(cfg: dict, sr: Optional[int] = None) -> bool:
    out = get_output_dir(cfg, sr) / "embeddings.npy"
    if out.exists():
        print(
            f"[{cfg['encoder_name']}] Embeddings already exist at {out}. Skipping.",
            flush=True,
        )
        return True
    return False


# ── SR discovery ──────────────────────────────────────────────────────────────

def find_available_srs(cfg: dict) -> list[int]:
    """
    Scan data/processed/ and return sorted list of SRs that have
    a processed_<SR>hz/ directory containing combined_windows.npy.
    """
    processed_root = cfg["data_path"] / "processed"
    if not processed_root.exists():
        return []

    pattern = re.compile(r"^processed_(\d+)hz$")
    available = []
    for d in sorted(processed_root.iterdir()):
        m = pattern.match(d.name)
        if m and (d / "combined_windows.npy").exists():
            available.append(int(m.group(1)))

    return sorted(available)


# ── SR guard for HuggingFace speech models ────────────────────────────────────

def check_sr_or_skip(cfg: dict, required_sr: int = 16_000) -> bool:
    """
    Returns True  → caller should exit (SR mismatch, skip logged to MLflow).
    Returns False → SR is correct, proceed.

    HuggingFace speech models (wav2vec2, HuBERT, WavLM) are all pretrained
    on 16 kHz audio. Their convolutional feature extractor is hardcoded to
    that temporal resolution.

    Upsampling a low-SR signal to 16 kHz does NOT recover information above
    the original Nyquist limit (SR/2). It produces interpolated zeros in the
    missing frequency bands, giving the model a silently corrupted input that
    would produce meaningless embeddings.  Downsampling is fine — information
    is lost but what remains is accurate.
    """
    if cfg["sampling_rate"] == required_sr:
        return False

    sr = cfg["sampling_rate"]
    if sr < required_sr and sr * 2 < required_sr:
        reason = (
            f"UPSAMPLING REJECTED: {cfg['encoder_name']} requires {required_sr} Hz. "
            f"Source SR={sr} Hz → Nyquist limit={sr // 2} Hz. "
            f"Upsampling to {required_sr} Hz cannot recover frequencies above {sr // 2} Hz. "
            f"Run the preprocessing service with TARGET_SR=16000 to get valid input."
        )
    else:
        # sr > 16000 — downsampling is safe but unusual
        reason = (
            f"SR MISMATCH: {cfg['encoder_name']} requires {required_sr} Hz. "
            f"Source SR={sr} Hz. Downsampling would be safe but the pipeline is "
            f"designed to use processed_16000hz/ directly. Skipping."
        )

    print(f"\n[{cfg['encoder_name']}] ⚠ SKIP — {reason}\n", flush=True)
    _log_skip_run(cfg, reason=reason, required_sr=required_sr)
    return True


def _log_skip_run(cfg: dict, *, reason: str, required_sr: int) -> None:
    """Log a fully-tagged 'skipped' MLflow run for visibility in the comparison UI."""
    import mlflow

    mlflow.set_tracking_uri(cfg["mlflow_uri"])
    run_name = f"{cfg['encoder_name']}_{cfg['sampling_rate']}hz"
    kwargs: dict = {"run_name": run_name}
    if cfg["parent_run_id"]:
        kwargs["nested"] = True

    with mlflow.start_run(**kwargs):
        mlflow.set_tag("status", "skipped")
        mlflow.set_tag("skip_reason", "sr_mismatch")
        mlflow.log_params({
            "encoder_name":  cfg["encoder_name"],
            "sampling_rate": cfg["sampling_rate"],
            "required_sr":   required_sr,
            "status":        "skipped",
        })
        # Log full reason as a text artifact so it shows in the artifact browser
        mlflow.log_text(reason, "skip_reason.txt")
        print(f"[{cfg['encoder_name']}] Skip logged to MLflow run: {run_name}", flush=True)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_windows(cfg: dict, sr: Optional[int] = None) -> np.ndarray:
    processed_dir = get_processed_dir(cfg, sr)
    path = processed_dir / "combined_windows.npy"

    if not path.exists():
        print(f"[{cfg['encoder_name']}][ERROR] Not found: {path}", flush=True)
        sys.exit(1)

    windows = np.load(path)
    assert windows.ndim == 2, f"Expected 2-D array, got shape {windows.shape}"
    actual_sr = sr if sr is not None else cfg["sampling_rate"]
    print(
        f"[{cfg['encoder_name']}] Loaded windows → shape={windows.shape}  SR={actual_sr} Hz",
        flush=True,
    )
    return windows


# ── Resampling (downsample only) ──────────────────────────────────────────────

def resample_batch(windows: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Downsample only.  Raises if upsampling is attempted.

    Upsampling cannot recover frequencies above the Nyquist limit of the
    source signal and would corrupt the encoder input.
    """
    if orig_sr == target_sr:
        return windows

    if orig_sr < target_sr:
        raise ValueError(
            f"Upsampling from {orig_sr} Hz to {target_sr} Hz is not allowed. "
            f"Source signal has no information above {orig_sr // 2} Hz. "
            "Use processed_16000hz/ data directly for HuggingFace encoders."
        )

    import librosa

    print(
        f"  Downsampling {orig_sr} Hz → {target_sr} Hz  ({windows.shape[0]} windows) ...",
        flush=True,
    )
    resampled = np.stack([
        librosa.resample(w.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)
        for w in windows
    ])
    print(f"  Downsampled shape: {resampled.shape}", flush=True)
    return resampled


# ── Batch iteration ───────────────────────────────────────────────────────────

def batch_iter(
    data: np.ndarray,
    batch_size: int,
) -> Generator[tuple[int, np.ndarray], None, None]:
    n = len(data)
    for start in range(0, n, batch_size):
        yield start, data[start : start + batch_size]


# ── Save / Metadata ───────────────────────────────────────────────────────────

def save_embeddings(embeddings: np.ndarray, cfg: dict, sr: Optional[int] = None) -> Path:
    out_dir = get_output_dir(cfg, sr)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "embeddings.npy"
    np.save(out_path, embeddings)
    print(
        f"[{cfg['encoder_name']}] Saved embeddings → {out_path}  shape={embeddings.shape}",
        flush=True,
    )
    return out_path


def write_metadata(
    cfg: dict,
    *,
    model_id: str,
    windows_shape: tuple,
    embedding_dim: int,
    runtime_sec: float,
    out_path: Path,
    sr: Optional[int] = None,
    extra: Optional[dict] = None,
) -> Path:
    actual_sr = sr if sr is not None else cfg["sampling_rate"]
    meta = {
        "encoder":         cfg["encoder_name"],
        "model_id":        model_id,
        "sampling_rate":   actual_sr,
        "num_windows":     int(windows_shape[0]),
        "window_size":     int(windows_shape[1]),
        "embedding_dim":   int(embedding_dim),
        "runtime_sec":     round(runtime_sec, 3),
        "embeddings_path": str(out_path),
        "status":          "completed",
    }
    if extra:
        meta.update(extra)

    meta_path = get_output_dir(cfg, sr) / "metadata.json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[{cfg['encoder_name']}] Metadata → {meta_path}", flush=True)
    return meta_path


# ── MLflow helpers ────────────────────────────────────────────────────────────

def mlflow_start_run(cfg: dict, sr: Optional[int] = None):
    import mlflow

    mlflow.set_tracking_uri(cfg["mlflow_uri"])
    actual_sr = sr if sr is not None else cfg["sampling_rate"]
    run_name = f"{cfg['encoder_name']}_{actual_sr}hz"
    kwargs: dict = {"run_name": run_name}
    if cfg["parent_run_id"]:
        kwargs["nested"] = True

    return mlflow.start_run(**kwargs)


def mlflow_log_all(
    cfg: dict,
    *,
    model_id: str,
    windows_shape: tuple,
    embedding_dim: int,
    runtime_sec: float,
    out_path: Path,
    meta_path: Path,
    device: str,
    sr: Optional[int] = None,
    extra_params: Optional[dict] = None,
    extra_metrics: Optional[dict] = None,
) -> None:
    import mlflow

    actual_sr = sr if sr is not None else cfg["sampling_rate"]

    mlflow.set_tag("status", "completed")
    mlflow.set_tag("encoder", cfg["encoder_name"])
    mlflow.set_tag("sampling_rate_hz", str(actual_sr))

    mlflow.log_params({
        "encoder_name":  cfg["encoder_name"],
        "model_id":      model_id,
        "sampling_rate": actual_sr,
        "num_windows":   int(windows_shape[0]),
        "window_size":   int(windows_shape[1]),
        "device":        device,
        "status":        "completed",
        **(extra_params or {}),
    })
    mlflow.log_metrics({
        "embedding_dim": float(embedding_dim),
        "runtime_sec":   round(runtime_sec, 3),
        **(extra_metrics or {}),
    })
    mlflow.log_param("embeddings_path", str(out_path))
    mlflow.log_artifact(str(meta_path))
