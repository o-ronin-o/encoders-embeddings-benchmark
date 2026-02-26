#!/usr/bin/env python3
"""
Heart Sound Embedding Benchmark — main entry point.

Compares multiple audio embedding models on heart sound classification
using the MSEB framework for standardised embedding extraction.

Architecture:
    MSEB DirectRunner  →  embedding extraction (standardised across models)
    Custom Evaluator   →  linear probe / nearest centroid classification

Usage examples:
    # Compare all models with a linear probe (needs train + test split):
    python run_benchmark.py \\
        --audio_dir      heart_sounds/test/audio \\
        --labels_csv     heart_sounds/test/labels.csv \\
        --train_audio_dir  heart_sounds/train/audio \\
        --train_labels_csv heart_sounds/train/labels.csv \\
        --model all \\
        --strategy linear_probe \\
        --output_json results.json

    # Quick sanity check with nearest centroid (no train split needed):
    python run_benchmark.py \\
        --audio_dir  heart_sounds/audio \\
        --labels_csv heart_sounds/labels.csv \\
        --model whisper_base \\
        --strategy nearest_centroid

    # TS2Vec only (you must provide a checkpoint):
    python run_benchmark.py \\
        --audio_dir          heart_sounds/test/audio \\
        --labels_csv         heart_sounds/test/labels.csv \\
        --train_audio_dir    heart_sounds/train/audio \\
        --train_labels_csv   heart_sounds/train/labels.csv \\
        --model ts2vec \\
        --ts2vec_checkpoint  /path/to/model.pkl \\
        --strategy linear_probe
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from typing import Any, Callable, Optional

# Configure logging before any other imports so that all loggers
# (including those inside MSEB) format with our style.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Heart sound audio embedding benchmark (MSEB framework).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data paths ─────────────────────────────────────────────────────────────
    data = p.add_argument_group("Data")
    data.add_argument(
        "--audio_dir", required=True, metavar="PATH",
        help="Directory containing TEST WAV files.",
    )
    data.add_argument(
        "--labels_csv", required=True, metavar="PATH",
        help="CSV with columns [audio_id, label] for the TEST set.",
    )
    data.add_argument(
        "--train_audio_dir", default=None, metavar="PATH",
        help="Directory containing TRAINING WAV files. "
             "Required when --strategy=linear_probe.",
    )
    data.add_argument(
        "--train_labels_csv", default=None, metavar="PATH",
        help="CSV with columns [audio_id, label] for the TRAINING set. "
             "Required when --strategy=linear_probe.",
    )

    # ── Output ─────────────────────────────────────────────────────────────────
    out = p.add_argument_group("Output")
    out.add_argument(
        "--output_json", default="benchmark_results.json", metavar="PATH",
        help="JSON file to save all results.",
    )

    # ── Model selection ────────────────────────────────────────────────────────
    mdl = p.add_argument_group("Model selection")
    mdl.add_argument(
        "--model", default="all",
        choices=[
            "all",
            "ts2vec",
            "clap",
            "whisper_tiny",
            "whisper_base",
            "whisper_medium",
        ],
        help=(
            "Which encoder(s) to benchmark.  'all' runs every available model. "
            "ts2vec requires --ts2vec_checkpoint."
        ),
    )

    # ── TS2Vec options ─────────────────────────────────────────────────────────
    ts = p.add_argument_group("TS2Vec options")
    ts.add_argument(
        "--ts2vec_checkpoint", default=None, metavar="PATH",
        help="Path to a pretrained TS2Vec checkpoint (.pkl). "
             "Required when --model=ts2vec or --model=all.",
    )
    ts.add_argument(
        "--ts2vec_output_dim", type=int, default=320,
        help="Embedding dimension of the TS2Vec model.",
    )
    ts.add_argument(
        "--ts2vec_encoding_window",
        default="full_series",
        choices=["full_series", "multiscale"],
        help=(
            "'full_series': global embedding (faster, good for classification). "
            "'multiscale': multi-resolution then mean-pool."
        ),
    )

    # ── CLAP options ───────────────────────────────────────────────────────────
    clap = p.add_argument_group("CLAP options")
    clap.add_argument(
        "--clap_model",
        default="laion/clap-htsat-unfused",
        help="HuggingFace model identifier for CLAP.",
    )

    # ── Evaluation options ─────────────────────────────────────────────────────
    ev = p.add_argument_group("Evaluation")
    ev.add_argument(
        "--strategy",
        default="linear_probe",
        choices=["linear_probe", "nearest_centroid"],
        help=(
            "'linear_probe': logistic regression on frozen embeddings. "
            "Most informative comparison — requires --train_audio_dir. "
            "'nearest_centroid': cosine distance to class centroids. "
            "Works without training data (but less discriminative)."
        ),
    )
    ev.add_argument(
        "--batch_size", type=int, default=8,
        help="Forward-pass batch size for all encoders.",
    )

    return p


# ── Encoder factory ───────────────────────────────────────────────────────────

EncoderFactory = Callable[[], Any]   # () → MultiModalEncoder


def _build_encoder_configs(args: argparse.Namespace) -> list[dict]:
    """
    Return a list of {'name': str, 'factory': Callable} dicts
    for each encoder that should run given the CLI flags.
    """
    configs: list[dict] = []

    # ── TS2Vec ─────────────────────────────────────────────────────────────────
    if args.model in ("all", "ts2vec"):
        if args.ts2vec_checkpoint is None:
            logger.warning(
                "Skipping TS2Vec: --ts2vec_checkpoint is not set.  "
                "Provide a path to a pretrained TS2Vec .pkl file."
            )
        else:
            def _make_ts2vec(a=args) -> Any:  # closure captures args
                from encoders.ts2vec_encoder import TS2VecHeartSoundEncoder
                return TS2VecHeartSoundEncoder(
                    model_path=a.ts2vec_checkpoint,
                    output_dim=a.ts2vec_output_dim,
                    encoding_window=a.ts2vec_encoding_window,
                    batch_size=a.batch_size,
                )
            configs.append({
                "name":    f"ts2vec_{args.ts2vec_encoding_window}",
                "factory": _make_ts2vec,
            })

    # ── CLAP ───────────────────────────────────────────────────────────────────
    if args.model in ("all", "clap"):
        def _make_clap(a=args) -> Any:
            from encoders.clap_encoder import CLAPHeartSoundEncoder
            return CLAPHeartSoundEncoder(
                model_name=a.clap_model,
                batch_size=a.batch_size,
            )
        configs.append({
            "name":    f"clap_{args.clap_model.split('/')[-1]}",
            "factory": _make_clap,
        })

    # ── Whisper variants ───────────────────────────────────────────────────────
    whisper_sizes = {
        "whisper_tiny":   "tiny",
        "whisper_base":   "base",
        "whisper_medium": "medium",
    }
    for key, size in whisper_sizes.items():
        if args.model in ("all", key):
            def _make_whisper(sz=size, a=args) -> Any:
                from encoders.whisper_pooled_encoder import WhisperPooledHeartSoundEncoder
                return WhisperPooledHeartSoundEncoder(
                    model_size=sz,
                    batch_size=a.batch_size,
                )
            configs.append({
                "name":    f"whisper_{size}_pooled",
                "factory": _make_whisper,
            })

    return configs


# ── Embedding extraction ──────────────────────────────────────────────────────

def _run_encoder(encoder: Any, task: Any) -> dict:
    """
    Use MSEB's DirectRunner to extract embeddings.

    From the MSEB notebook (confirmed):
        runner = DirectRunner(encoder=encoder)
        embeddings = runner.run(task.sounds())   # → {audio_id: SoundEmbedding}

    The runner calls the encoder in batches internally.

    Returns:
        {audio_id: SoundEmbedding} dict — MSEB's "Embedding Cache".
    """
    from mseb.runner import DirectRunner  # noqa: PLC0415
    runner = DirectRunner(encoder=encoder)
    return runner.run(task.sounds())


# ── Results display ───────────────────────────────────────────────────────────

def _print_table(all_results: list[dict]) -> None:
    """Print a formatted comparison table, sorted by macro_f1 descending."""
    # Sort by macro_f1, treat missing as 0
    ranked = sorted(
        all_results,
        key=lambda r: r.get("metrics", {}).get("macro_f1", 0.0),
        reverse=True,
    )

    try:
        _print_rich_table(ranked)
    except ImportError:
        _print_plain_table(ranked)


def _print_rich_table(ranked: list[dict]) -> None:
    """Rich-formatted table (coloured, aligned)."""
    from rich.console import Console       # noqa: PLC0415
    from rich.table import Table           # noqa: PLC0415
    from rich import box                   # noqa: PLC0415

    console = Console()

    # ── Summary table ──────────────────────────────────────────────────────────
    t = Table(
        title="Heart Sound Embedding Benchmark",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold cyan",
    )
    t.add_column("Rank", justify="center", style="dim")
    t.add_column("Model",                  style="bold", min_width=32)
    t.add_column("Accuracy",   justify="right")
    t.add_column("Macro F1",   justify="right", style="green")
    t.add_column("AUROC",      justify="right")
    t.add_column("Time (s)",   justify="right", style="dim")
    t.add_column("Strategy",   justify="center", style="dim")

    for rank, r in enumerate(ranked, start=1):
        m = r.get("metrics", {})
        if "error" in r:
            t.add_row(
                str(rank), r["model_name"],
                "[red]FAILED[/red]", "-", "-", "-", "-",
            )
        else:
            auroc = m.get("auroc_binary")
            t.add_row(
                str(rank),
                r["model_name"],
                f"{m.get('accuracy', 0):.3f}",
                f"{m.get('macro_f1', 0):.3f}",
                f"{auroc:.3f}" if auroc is not None else "N/A",
                f"{r.get('encoding_time_s', 0):.1f}",
                m.get("strategy", "-"),
            )
    console.print(t)

    # ── Per-class F1 breakdown ─────────────────────────────────────────────────
    has_per_class = any(
        "per_class_f1" in r.get("metrics", {}) for r in ranked
    )
    if has_per_class:
        all_classes = sorted({
            cls
            for r in ranked
            for cls in r.get("metrics", {}).get("per_class_f1", {})
        })
        b = Table(
            title="Per-Class F1 Breakdown",
            box=box.SIMPLE,
            header_style="bold",
        )
        b.add_column("Model", style="bold", min_width=32)
        for cls in all_classes:
            b.add_column(f"F1({cls})", justify="right")

        for r in ranked:
            m = r.get("metrics", {})
            pc = m.get("per_class_f1", {})
            if "error" not in r:
                b.add_row(
                    r["model_name"],
                    *[f"{pc.get(cls, 0.0):.3f}" for cls in all_classes],
                )
        console.print(b)


def _print_plain_table(ranked: list[dict]) -> None:
    """Plain-text fallback table (no rich dependency)."""
    sep = "=" * 90
    print(f"\n{sep}")
    print("HEART SOUND EMBEDDING BENCHMARK RESULTS")
    print(sep)
    h = f"{'Rank':<5} {'Model':<33} {'Accuracy':>10} {'Macro F1':>10} {'AUROC':>8} {'Time(s)':>9}"
    print(h)
    print("-" * 90)
    for rank, r in enumerate(ranked, start=1):
        m = r.get("metrics", {})
        if "error" in r:
            print(f"{rank:<5} {r['model_name']:<33} {'FAILED':>10}")
            continue
        auroc = m.get("auroc_binary")
        auroc_s = f"{auroc:.3f}" if auroc is not None else "N/A"
        print(
            f"{rank:<5} {r['model_name']:<33} "
            f"{m.get('accuracy', 0):>10.3f} "
            f"{m.get('macro_f1', 0):>10.3f} "
            f"{auroc_s:>8} "
            f"{r.get('encoding_time_s', 0):>9.1f}"
        )
    print(sep)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = _build_parser().parse_args()

    # ── Validate linear probe requirement ─────────────────────────────────────
    if args.strategy == "linear_probe":
        if args.train_audio_dir is None or args.train_labels_csv is None:
            logger.error(
                "linear_probe requires --train_audio_dir and --train_labels_csv. "
                "Either provide training data or use --strategy nearest_centroid."
            )
            sys.exit(1)

    # ── MSEB availability check ────────────────────────────────────────────────
    try:
        from mseb.runner import DirectRunner  # noqa: F401, PLC0415
    except ImportError:
        logger.error(
            "MSEB is not installed. "
            "Run: pip install git+https://github.com/google-research/mseb"
        )
        sys.exit(1)

    # ── Build tasks ───────────────────────────────────────────────────────────
    from heart_sound_task import HeartSoundClassificationTask           # noqa: PLC0415
    from evaluators.classification_evaluator import HeartSoundClassificationEvaluator  # noqa: PLC0415

    logger.info("Loading TEST task  →  %s", args.audio_dir)
    test_task       = HeartSoundClassificationTask(args.audio_dir, args.labels_csv)
    test_label_dict = test_task.label_dict()

    train_task       = None
    train_label_dict = None
    if args.train_audio_dir and args.train_labels_csv:
        logger.info("Loading TRAIN task  →  %s", args.train_audio_dir)
        train_task       = HeartSoundClassificationTask(args.train_audio_dir, args.train_labels_csv)
        train_label_dict = train_task.label_dict()

    evaluator = HeartSoundClassificationEvaluator(strategy=args.strategy)
    configs   = _build_encoder_configs(args)

    if not configs:
        logger.error(
            "No encoders configured.  "
            "Check your --model flag and --ts2vec_checkpoint if using ts2vec."
        )
        sys.exit(1)

    logger.info(
        "Starting benchmark: %d encoder(s) — %s",
        len(configs), [c["name"] for c in configs],
    )

    all_results: list[dict] = []

    # ── Per-encoder loop ──────────────────────────────────────────────────────
    for config in configs:
        name = config["name"]
        logger.info("\n%s\nEncoder: %s\n%s", "=" * 60, name, "=" * 60)

        result_entry: dict = {"model_name": name}

        try:
            # 1. Instantiate and set up the encoder
            encoder = config["factory"]()
            encoder.setup()   # loads model weights

            # 2. Encode the TEST set via MSEB DirectRunner
            logger.info("Encoding TEST set …")
            t0 = time.perf_counter()
            test_embeddings = _run_encoder(encoder, test_task)
            enc_time = time.perf_counter() - t0

            n_test = len(test_embeddings)
            logger.info(
                "TEST encoding done: %d samples in %.1f s (%.3f s/sample)",
                n_test, enc_time, enc_time / max(n_test, 1),
            )

            # 3. Encode the TRAIN set (if available)
            train_embeddings = None
            if train_task is not None:
                logger.info("Encoding TRAIN set …")
                train_embeddings = _run_encoder(encoder, train_task)
                logger.info("TRAIN encoding done: %d samples.", len(train_embeddings))

            # 4. Evaluate
            logger.info("Evaluating with strategy: %s", args.strategy)
            metrics = evaluator.evaluate(
                embeddings_dict      = test_embeddings,
                ground_truth_dict    = test_label_dict,
                train_embeddings_dict= train_embeddings,
                train_labels_dict    = train_label_dict,
            )

            result_entry.update({
                "encoding_time_s": round(enc_time, 2),
                "n_test_samples":  n_test,
                "n_train_samples": len(train_embeddings) if train_embeddings else None,
                "metrics":         metrics,
            })

        except Exception as exc:
            logger.error("Encoder %s FAILED: %s", name, exc, exc_info=True)
            result_entry["error"] = str(exc)

        finally:
            # Free GPU memory before loading the next model
            try:
                del encoder  # type: ignore[possibly-undefined]
            except NameError:
                pass
            try:
                import torch  # noqa: PLC0415
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()

        all_results.append(result_entry)

    # ── Display results ───────────────────────────────────────────────────────
    _print_table(all_results)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Results saved → %s", args.output_json)


if __name__ == "__main__":
    main()
