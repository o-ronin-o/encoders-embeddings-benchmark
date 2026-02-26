"""
CLAP (Contrastive Language-Audio Pretraining) encoder — MSEB MultiModalEncoder interface.

CLAP architecture:
  - Dual encoder: audio branch (HTSAT-tiny transformer) + text branch (BERT/RoBERTa)
  - We use ONLY the audio branch: model.get_audio_features(**inputs) → (B, 512)
  - Input requirement: 48 000 Hz.  Our heart sounds are 16 000 Hz → resample inside.
  - Outputs are L2-normalised so cosine similarity = dot product.

HuggingFace models available:
  'laion/clap-htsat-unfused'              — general audio + music + speech (512-dim)
  'laion/larger_clap_music_and_speech'    — larger model, same output dim

Verified API (transformers >= 4.36):
  ClapProcessor.from_pretrained(model_name)
  ClapModel.from_pretrained(model_name)
  processor(audios=[np.array,...], sampling_rate=48000, return_tensors='pt', padding=True)
  model.get_audio_features(**inputs)   → Tensor (B, 512)
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch

from mseb import encoder as encoder_lib
from mseb import types as mseb_types

logger = logging.getLogger(__name__)

# CLAP requires audio at exactly 48 kHz.
_CLAP_SR: int = 48_000
# Our heart sounds (after HeartSoundClassificationTask.sounds()) are at 16 kHz.
_SOURCE_SR: int = 16_000
# Output embedding dimension for all CLAP HTSAT models.
_CLAP_DIM: int = 512


class CLAPHeartSoundEncoder(encoder_lib.MultiModalEncoder):
    """
    Wraps a LAION-CLAP model to produce fixed-size L2-normalised audio embeddings.

    The audio-encoder branch of CLAP maps variable-length waveforms to a single
    512-dimensional vector.  This is directly comparable to other fixed-size
    embedding approaches (TS2Vec, Whisper-pooled) for downstream classification.

    Note on cardiac audio suitability:
        CLAP was pretrained on FreeSound + music data.  It has never seen clinical
        heart sounds, but its acoustic representations generalise surprisingly well
        to other bioacoustic domains (cf. the MSEB paper's FSD50K results, 0.43 mAP).
    """

    def __init__(
        self,
        model_name: str = "laion/clap-htsat-unfused",
        batch_size: int = 8,
    ) -> None:
        """
        Args:
            model_name:  HuggingFace model identifier. Options:
                           'laion/clap-htsat-unfused'            (recommended)
                           'laion/larger_clap_music_and_speech'
            batch_size:  Sounds encoded per GPU forward pass. Reduce if OOM.
        """
        super().__init__()
        self._model_name = model_name
        self._batch_size = batch_size
        self._model      = None
        self._processor  = None
        self._device     = None

    # ── MSEB required overrides ───────────────────────────────────────────────

    def _setup(self) -> None:
        """Load ClapModel and ClapProcessor from HuggingFace Hub."""
        try:
            from transformers import ClapModel, ClapProcessor  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "transformers is not installed. Run: pip install transformers>=4.36"
            ) from exc

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("CLAP: loading %s | device=%s", self._model_name, self._device)

        self._processor = ClapProcessor.from_pretrained(self._model_name)
        self._model     = ClapModel.from_pretrained(self._model_name)
        self._model.eval()
        self._model.to(self._device)

        logger.info("CLAP model loaded successfully. Output dim: %d", _CLAP_DIM)

    def _check_input_types(
        self, batch: Sequence[mseb_types.MultiModalObject]
    ) -> None:
        """Assert every item in the batch is a Sound object."""
        bad = [type(x).__name__ for x in batch if not isinstance(x, mseb_types.Sound)]
        if bad:
            raise ValueError(
                f"CLAPHeartSoundEncoder requires all inputs to be "
                f"mseb.types.Sound, but received: {set(bad)}"
            )

    def _encode(
        self,
        batch: Sequence[mseb_types.Sound],
    ) -> list[mseb_types.SoundEmbedding]:
        """
        Encode a batch of Sound objects using CLAP's audio encoder.

        Per-sample pipeline:
          1. Resample waveform from 16 kHz → 48 kHz (CLAP requirement)
          2. Collate into a batch via ClapProcessor (padding, mel-spec)
          3. Forward through model.get_audio_features() → (B, 512) tensor
          4. L2-normalise each row so cosine sim = dot product

        Args:
            batch: Sequence of mseb.types.Sound, waveforms at 16 kHz float32.

        Returns:
            List of mseb.types.SoundEmbedding with L2-normalised 512-dim vectors.
        """
        results: list[mseb_types.SoundEmbedding] = []

        for i in range(0, len(batch), self._batch_size):
            sub_batch = batch[i : i + self._batch_size]
            results.extend(self._encode_sub_batch(sub_batch))

        return results

    # ── Private helpers ───────────────────────────────────────────────────────

    def _resample_to_clap_sr(self, waveform: np.ndarray) -> np.ndarray:
        """
        Resample a 16 kHz waveform to 48 kHz for CLAP.

        We use librosa.resample because it handles edge cases (very short
        clips, non-power-of-two ratios) robustly.
        """
        try:
            import librosa  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "librosa is required for CLAP resampling. "
                "Run: pip install librosa"
            ) from exc

        return librosa.resample(
            waveform.astype(np.float32),
            orig_sr=_SOURCE_SR,
            target_sr=_CLAP_SR,
        )

    def _encode_sub_batch(
        self, sub_batch: Sequence[mseb_types.Sound]
    ) -> list[mseb_types.SoundEmbedding]:
        """
        Encode one sub-batch of sounds.

        Sounds that fail resampling are replaced with silence so the rest of
        the batch still processes. The whole-batch forward pass is protected
        by a try/except that falls back to zero vectors if PyTorch raises.
        """
        # ── Step 1: resample each waveform ────────────────────────────────────
        waveforms_48k: list[np.ndarray] = []
        sounds_in_batch: list[mseb_types.Sound] = []

        for sound in sub_batch:
            try:
                wav_48k = self._resample_to_clap_sr(sound.waveform)
            except Exception as exc:
                logger.warning(
                    "CLAP resampling failed for sample %s: %s  →  using silence.",
                    sound.context.id, exc,
                )
                # 1-second silence at 48 kHz
                wav_48k = np.zeros(_CLAP_SR, dtype=np.float32)

            waveforms_48k.append(wav_48k)
            sounds_in_batch.append(sound)

        # ── Step 2–4: processor → model → L2-normalise ───────────────────────
        try:
            inputs = self._processor(
                audios=waveforms_48k,
                sampling_rate=_CLAP_SR,
                return_tensors="pt",
                padding=True,
            )
            # Move all processor outputs to the model's device
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                # audio_features: Tensor of shape (B, 512)
                audio_features = self._model.get_audio_features(**inputs)

            embeddings = audio_features.cpu().numpy().astype(np.float32)  # (B, 512)

            # L2-normalise row-wise; guard against zero-norm vectors
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms < 1e-8, 1.0, norms)
            embeddings = embeddings / norms

            return [
                mseb_types.SoundEmbedding(
                    embedding=embeddings[j],
                    context=sound.context,
                )
                for j, sound in enumerate(sounds_in_batch)
            ]

        except Exception as exc:
            logger.warning(
                "CLAP forward pass failed for sub-batch of %d samples: %s  "
                "→  falling back to zero vectors.",
                len(sounds_in_batch), exc,
            )
            return [
                mseb_types.SoundEmbedding(
                    embedding=np.zeros(_CLAP_DIM, dtype=np.float32),
                    context=sound.context,
                )
                for sound in sounds_in_batch
            ]
