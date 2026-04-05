#!/usr/bin/env python3
"""
run_pipeline.py — Audio Embeddings Benchmark Orchestrator
==========================================================
Runs encoder services as Docker containers via docker-compose and tracks
everything under a single parent MLflow run.

SR behaviour per encoder type
------------------------------
  wav2vec / hubert / wavlm  → require 16000 Hz; any other SR is logged as
                               'skipped' with a Nyquist explanation and the
                               container exits cleanly (exit code 0).
  ts2vec                    → SR-agnostic; if the requested SR is not found
                               it falls back to ALL available SRs automatically,
                               logging each one separately in MLflow.

The orchestrator itself does NOT enforce these rules — the encoder scripts do.
The orchestrator simply collects exit codes and logs the overall summary.

Usage
-----
  # All encoders, 16 kHz (full run — all encoders active)
  python run_pipeline.py --sampling_rate 16000 --encoders wav2vec hubert wavlm ts2vec

  # Low-SR run: HuggingFace encoders will be skipped, ts2vec will run
  python run_pipeline.py --sampling_rate 1000 --encoders wav2vec hubert wavlm ts2vec

  # Parallel execution
  python run_pipeline.py --sampling_rate 16000 --encoders wav2vec hubert --parallel

  # Dry run (print commands, don't execute)
  python run_pipeline.py --sampling_rate 8000 --encoders ts2vec --dry_run
"""

import argparse
import os
import sys
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import mlflow


VALID_ENCODERS = ["wav2vec", "hubert", "wavlm", "ts2vec"]
# Encoders that hard-require 16 kHz (for the orchestrator summary log only)
HF_ENCODERS    = {"wav2vec", "hubert", "wavlm"}
REQUIRED_SR    = 16_000


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audio Embeddings Benchmark Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sampling_rate", "-sr", type=int, required=True,
                   choices=[1000, 2000, 4000, 8000,14000, 16000])
    p.add_argument("--encoders", "-e", nargs="+", choices=VALID_ENCODERS,
                   required=True, metavar="ENCODER")
    p.add_argument("--parallel", action="store_true",
                   help="Run encoders in parallel (default: sequential).")
    p.add_argument("--mlflow_uri",
                   default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    p.add_argument("--data_path",
                   default=os.environ.get("DATA_PATH", str(Path(__file__).parent / "preprocessing_service/data")))
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--compose_file",
                   default=str(Path(__file__).parent / "docker-compose.yml"))
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


# ── Pre-flight summary ────────────────────────────────────────────────────────

def print_sr_plan(encoders: list[str], sampling_rate: int) -> dict[str, str]:
    """
    Print a human-readable table of what will happen for each encoder
    at the given SR.  Returns a dict of encoder → expected_status.
    """
    expected: dict[str, str] = {}
    print(f"\n{'─'*60}")
    print(f"  SR Plan for {sampling_rate} Hz")
    print(f"{'─'*60}")
    for enc in encoders:
        if enc in HF_ENCODERS and sampling_rate != REQUIRED_SR and sampling_rate * 2 < REQUIRED_SR:
            status = f"WILL SKIP (requires {REQUIRED_SR} Hz, upsampling forbidden)"
        elif enc == "ts2vec":
            status = "WILL RUN (SR-agnostic; fallback if data missing)"
        else:
            status = "WILL RUN"
        expected[enc] = status
        print(f"  {enc:<10}  {status}")
    print(f"{'─'*60}\n")
    return expected


# ── Encoder runner ────────────────────────────────────────────────────────────

def run_encoder(
    encoder: str,
    *,
    sampling_rate: int,
    mlflow_uri: str,
    parent_run_id: str,
    data_path: str,
    batch_size: int,
    compose_file: str,
    dry_run: bool = False,
) -> tuple[str, int]:
    env = {
        **os.environ,
        "SAMPLING_RATE":        str(sampling_rate),
        "MLFLOW_TRACKING_URI":  mlflow_uri,
        "MLFLOW_PARENT_RUN_ID": parent_run_id,
        "DATA_PATH":            "/app/data",
        "BATCH_SIZE":           str(batch_size),
        "HOST_DATA_PATH":       str(data_path),
    }

    cmd = [
        "docker-compose", "-f", compose_file,
        "run", "--rm",
        "-e", f"SAMPLING_RATE={sampling_rate}",
        "-e", f"MLFLOW_TRACKING_URI={mlflow_uri}",
        "-e", f"MLFLOW_PARENT_RUN_ID={parent_run_id}",
        "-e", f"BATCH_SIZE={batch_size}",
        encoder,
    ]

    print(f"\n[pipeline] ▶ Starting {encoder}  SR={sampling_rate} Hz", flush=True)
    if dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}", flush=True)
        return encoder, 0

    t0     = time.perf_counter()
    result = subprocess.run(cmd, env=env)
    elapsed = time.perf_counter() - t0

    icon   = "✓" if result.returncode == 0 else "✗"
    print(f"[pipeline] {icon} {encoder} finished  exit={result.returncode}  wall={elapsed:.1f}s", flush=True)
    return encoder, result.returncode


# ── Orchestrator ──────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Pre-flight checks ────────────────────────────────────────────────────
    processed_root = Path(args.data_path) / "processed"
    if not processed_root.exists():
        print(
            f"[pipeline][ERROR] No processed data found at {processed_root}\n"
            f"  Run preprocessing first:\n"
            f"  TARGET_SR={args.sampling_rate} docker-compose run --rm preprocess",
            file=sys.stderr,
        )
        sys.exit(1)

    expected_statuses = print_sr_plan(args.encoders, args.sampling_rate)

    # ── MLflow parent run ────────────────────────────────────────────────────
    mlflow.set_tracking_uri(args.mlflow_uri)
    run_name = f"benchmark_{args.sampling_rate}hz"

    print(f"[pipeline] Starting parent MLflow run: '{run_name}'", flush=True)

    t_total = time.perf_counter()

    with mlflow.start_run(run_name=run_name) as parent_run:
        parent_run_id = parent_run.info.run_id

        # Encoders expected to be skipped (for logging clarity)
        will_skip = [e for e, s in expected_statuses.items() if "SKIP" in s]
        will_run  = [e for e, s in expected_statuses.items() if "SKIP" not in s]

        mlflow.log_params({
            "sampling_rate":        args.sampling_rate,
            "encoders_requested":   ",".join(args.encoders),
            "encoders_expected_run": ",".join(will_run),
            "encoders_expected_skip": ",".join(will_skip),
            "parallel":             args.parallel,
            "batch_size":           args.batch_size,
            "data_path":            args.data_path,
        })

        # ── Run containers ───────────────────────────────────────────────────
        runner_kwargs = dict(
            sampling_rate = args.sampling_rate,
            mlflow_uri    = args.mlflow_uri,
            parent_run_id = parent_run_id,
            data_path     = args.data_path,
            batch_size    = args.batch_size,
            compose_file  = args.compose_file,
            dry_run       = args.dry_run,
        )
        results: dict[str, int] = {}

        if args.parallel:
            print(f"[pipeline] Running {len(args.encoders)} encoders in PARALLEL ...", flush=True)
            with ThreadPoolExecutor(max_workers=len(args.encoders)) as pool:
                futures = {pool.submit(run_encoder, enc, **runner_kwargs): enc for enc in args.encoders}
                for f in as_completed(futures):
                    enc, code = f.result()
                    results[enc] = code
        else:
            print(f"[pipeline] Running {len(args.encoders)} encoders SEQUENTIALLY ...", flush=True)
            for enc in args.encoders:
                _, code = run_encoder(enc, **runner_kwargs)
                results[enc] = code

        # ── Final summary ────────────────────────────────────────────────────
        total_time = time.perf_counter() - t_total
        n_ok     = sum(1 for c in results.values() if c == 0)
        n_failed = len(results) - n_ok

        mlflow.log_metrics({
            "n_encoders_total":  float(len(results)),
            "n_encoders_ok":     float(n_ok),
            "n_encoders_failed": float(n_failed),
            "total_wall_sec":    round(total_time, 2),
        })

        print(f"\n{'═'*60}", flush=True)
        print(f"  Pipeline summary  —  SR={args.sampling_rate} Hz  ({total_time:.1f}s)", flush=True)
        print(f"{'─'*60}", flush=True)
        for enc in args.encoders:
            code   = results[enc]
            expect = expected_statuses[enc]
            icon   = "✓" if code == 0 else "✗"
            print(f"  {icon}  {enc:<10}  exit={code}   [{expect}]", flush=True)
        print(f"{'─'*60}", flush=True)
        print(f"  {n_ok}/{len(results)} containers exited OK", flush=True)
        if will_skip:
            print(f"  Note: {will_skip} logged 'skipped' runs in MLflow (SR mismatch)", flush=True)
        print(f"  MLflow run ID : {parent_run_id}", flush=True)
        print(f"  MLflow UI     : {args.mlflow_uri}", flush=True)
        print(f"{'═'*60}\n", flush=True)

        if n_failed:
            sys.exit(1)


if __name__ == "__main__":
    main()
