"""
Heart sound classification task, compatible with the MSEB framework.

What we know for certain from the MSEB notebook (all 827 lines):
  - mseb.types.Sound(waveform=np.ndarray, context=SoundContextParams(id=..., sample_rate=...))
  - mseb.types.TextPrediction(prediction=str, context=PredictionContextParams(id=..., debug_text=...))
  - mseb.types.SoundContextParams(id=str, sample_rate=int)
  - mseb.types.PredictionContextParams(id=str, debug_text=str)
  - Tasks expose: sounds(), examples(sub_task), setup(runner=None), metadata
  - DirectRunner calls task.sounds() directly — it is a generator

The MSEB runner does NOT call any base class hooks — it purely duck-types the task.
We therefore implement the interface without requiring BaseTask inheritance,
with an optional import so the class degrades gracefully if MSEB is not installed.
"""

from __future__ import annotations

import logging
import os
from typing import Iterator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Import MSEB types ────────────────────────────────────────────────────────
try:
    from mseb import types as mseb_types
    _MSEB_AVAILABLE = True
except ImportError:
    _MSEB_AVAILABLE = False
    logger.warning(
        "MSEB not installed. Install via: "
        "pip install git+https://github.com/google-research/mseb"
    )

# ── Constants ────────────────────────────────────────────────────────────────
TARGET_SAMPLE_RATE: int = 16_000          # Hz — standard for speech/audio models
VALID_LABELS: tuple[str, ...] = ("normal", "murmur", "other")


class HeartSoundClassificationTask:
    """
    MSEB-compatible classification task for heart sound recordings.

    Classifies each recording into one of:
        normal  — healthy cardiac sound
        murmur  — systolic or diastolic murmur
        other   — other pathology (click, rub, gallop, etc.)

    Dataset layout expected on disk:
        audio_dir/
            rec_001.wav     (mono WAV, any sample rate, any duration)
            rec_002.wav
            ...
        labels_csv          (CSV with columns: audio_id, label)
                            audio_id must match filename stem (no extension)

    MSEB Interface implemented:
        sounds()            → generator of mseb.types.Sound
        examples(sub_task)  → generator of mseb.types.TextPrediction (ground truth)
        setup(runner)       → no-op for classification
        metadata            → class attribute dict
    """

    metadata: dict = {
        "name": "HeartSoundClassification",
        "description": (
            "Classify heart sound audio recordings into normal, murmur, or other "
            "pathology categories. Designed to compare audio embedding models "
            "for cardiac audio analysis without task-specific fine-tuning."
        ),
        "labels": list(VALID_LABELS),
        "metrics": ["accuracy", "macro_f1", "auroc_binary"],
        "primary_metric": "macro_f1",
        "domain": "bioacoustics / cardiac",
        "input_modality": "audio",
        "output_modality": "text_label",
    }

    def __init__(self, audio_dir: str, labels_csv: str) -> None:
        """
        Args:
            audio_dir:  Path to directory containing WAV files.
            labels_csv: Path to CSV file with columns [audio_id, label].
                        audio_id must match the WAV filename without extension.

        Raises:
            FileNotFoundError: If audio_dir or labels_csv does not exist.
            ValueError:        If labels_csv is missing required columns.
        """
        if not _MSEB_AVAILABLE:
            raise RuntimeError(
                "MSEB must be installed to use HeartSoundClassificationTask. "
                "Run: pip install git+https://github.com/google-research/mseb"
            )

        if not os.path.isdir(audio_dir):
            raise FileNotFoundError(f"Audio directory not found: {audio_dir!r}")
        if not os.path.isfile(labels_csv):
            raise FileNotFoundError(f"Labels CSV not found: {labels_csv!r}")

        self._audio_dir = audio_dir

        self._labels_df = pd.read_csv(labels_csv, dtype=str).fillna("")
        required_cols = {"audio_id", "label"}
        missing = required_cols - set(self._labels_df.columns)
        if missing:
            raise ValueError(
                f"labels_csv is missing required columns: {missing}. "
                f"Found columns: {list(self._labels_df.columns)}"
            )

        # Normalise whitespace
        self._labels_df["audio_id"] = self._labels_df["audio_id"].str.strip()
        self._labels_df["label"]    = self._labels_df["label"].str.strip().str.lower()

        # Warn about unexpected labels but do not crash — the dataset may have
        # more granular labels that the user wants to keep.
        unknown = set(self._labels_df["label"].unique()) - set(VALID_LABELS)
        if unknown:
            logger.warning(
                "Labels CSV contains label values not in %s: %s  "
                "These will be passed through as-is.",
                VALID_LABELS, unknown,
            )

        label_counts = self._labels_df["label"].value_counts().to_dict()
        logger.info(
            "HeartSoundClassificationTask initialised — %d samples | "
            "label distribution: %s",
            len(self._labels_df), label_counts,
        )

    # ── Public helpers ────────────────────────────────────────────────────────

    def label_dict(self) -> dict[str, str]:
        """Return {audio_id: label} mapping for every row in labels_csv."""
        return dict(
            zip(
                self._labels_df["audio_id"],
                self._labels_df["label"],
            )
        )

    def audio_ids(self) -> list[str]:
        """Return list of all audio IDs in insertion order."""
        return self._labels_df["audio_id"].tolist()

    # ── MSEB Task Interface ───────────────────────────────────────────────────

    def sounds(self) -> Iterator[mseb_types.Sound]:
        """
        Generator that yields one MSEB Sound per audio file.

        Sound fields populated:
            waveform   — float32 numpy array, shape (T,), values in [-1, 1]
            context.id          — audio_id string (used as unique key by runner)
            context.sample_rate — TARGET_SAMPLE_RATE (16000 Hz)

        Audio files that cannot be loaded are skipped with a warning.
        """
        for _, row in self._labels_df.iterrows():
            audio_id: str = row["audio_id"]
            audio_path = os.path.join(self._audio_dir, f"{audio_id}.wav")

            if not os.path.isfile(audio_path):
                logger.warning("WAV file not found — skipping: %s", audio_path)
                continue

            try:
                waveform = _load_and_preprocess_audio(audio_path, TARGET_SAMPLE_RATE)
            except Exception as exc:
                logger.warning(
                    "Failed to load audio %s — skipping. Error: %s",
                    audio_path, exc,
                )
                continue

            yield mseb_types.Sound(
                waveform=waveform,
                context=mseb_types.SoundContextParams(
                    id=audio_id,
                    sample_rate=TARGET_SAMPLE_RATE,
                ),
            )

    def examples(self, sub_task: str = "classification") -> Iterator[mseb_types.TextPrediction]:
        """
        Generator that yields ground-truth labels as TextPrediction objects.

        The MSEB runner matches these to encoder outputs via context.id.
        Our custom evaluator uses the returned dict directly, but we follow
        the MSEB interface so this task can also be used with built-in evaluators.

        Args:
            sub_task: Ignored — kept for MSEB interface compatibility.

        Yields:
            mseb.types.TextPrediction
                .prediction  — ground-truth label string ("normal", "murmur", "other")
                .context.id  — audio_id (must match Sound.context.id)
        """
        for _, row in self._labels_df.iterrows():
            audio_id: str = row["audio_id"]
            label:    str = row["label"]

            yield mseb_types.TextPrediction(
                prediction=label,
                context=mseb_types.PredictionContextParams(
                    id=audio_id,
                    debug_text=f"ground_truth_label={label}",
                ),
            )

    def setup(self, runner=None) -> None:  # noqa: ANN001
        """
        Task pre-processing hook required by the MSEB interface.

        Classification tasks have no corpus to index or pre-process,
        so this is intentionally a no-op.

        Args:
            runner: MSEB runner instance (unused for classification).
        """
        pass  # no-op


# ── Audio loading utility ─────────────────────────────────────────────────────

def _load_and_preprocess_audio(path: str, target_sr: int) -> np.ndarray:
    """
    Load a WAV file, convert to mono float32, resample, and normalise to [-1, 1].

    We prefer soundfile (fast, no hidden dependencies) and fall back to
    librosa if soundfile cannot decode the file.

    Args:
        path:      Absolute path to a WAV (or any audio) file.
        target_sr: Target sample rate in Hz.

    Returns:
        float32 numpy array of shape (T,), values in [-1, 1].

    Raises:
        RuntimeError: If neither soundfile nor librosa can decode the file.
    """
    waveform: np.ndarray | None = None
    sr: int | None = None

    # Attempt 1 — soundfile (fast, pure C extension)
    try:
        import soundfile as sf
        waveform, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        pass

    # Attempt 2 — librosa (handles more formats including MP3)
    if waveform is None:
        try:
            import librosa
            waveform, sr = librosa.load(path, sr=None, mono=False, dtype=np.float32)
        except Exception as exc:
            raise RuntimeError(
                f"Could not decode audio file {path!r}. "
                "Install soundfile and/or librosa."
            ) from exc

    # --- Mono conversion ---
    if waveform.ndim == 2:
        # soundfile returns (T, C); librosa returns (C, T)
        # Detect shape by checking which dim is larger
        if waveform.shape[0] > waveform.shape[1]:
            # soundfile layout: (T, C) → mean over channels
            waveform = waveform.mean(axis=1)
        else:
            # librosa layout: (C, T) → mean over channels
            waveform = waveform.mean(axis=0)

    # --- Resample ---
    if sr != target_sr:
        try:
            import librosa
            waveform = librosa.resample(
                waveform.astype(np.float32),
                orig_sr=sr,
                target_sr=target_sr,
            )
        except ImportError:
            raise RuntimeError(
                f"librosa is required for resampling from {sr} Hz to {target_sr} Hz. "
                "Run: pip install librosa"
            )

    # --- Peak normalise to [-1, 1] ---
    peak = float(np.abs(waveform).max())
    if peak > 0.0:
        waveform = waveform / peak

    return waveform.astype(np.float32)
