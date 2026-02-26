"""
Whisper audio encoder with mean-pooling — MSEB MultiModalEncoder interface.

This encoder uses ONLY Whisper's audio encoder (the transformer that converts
mel-spectrograms to a sequence of hidden states).  The decoder is never loaded
or run, so there is no transcription step.

The output token sequence (shape: B × T' × D) is mean-pooled over T' to
produce a single fixed-size vector of shape (D,) per audio clip.

Whisper-specific details (verified from whisper source and openai/whisper README):
  - All models operate on 16 kHz audio — no resampling needed.
  - Audio is padded / trimmed to exactly 30 seconds = 480 000 samples.
    whisper.pad_or_trim(tensor, length=480_000) handles this.
  - Log-mel spectrogram: whisper.log_mel_spectrogram(audio_tensor) → (80, 3000)
    The 80-channel, 3000-frame format is fixed across all Whisper variants.
  - Encoder output shapes by model size:
      tiny    → D=384,  sequence length ≈ 1500
      base    → D=512
      small   → D=768
      medium  → D=1024
      large-* → D=1280

Note on heart sound suitability:
    Whisper was pretrained on 680 000 hours of multilingual speech.
    Its mel-spectrogram frontend + transformer layers capture rich acoustic
    features that transfer to non-speech domains.  Pooled Whisper embeddings
    have been used effectively for speaker verification and environmental audio.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch

from mseb import encoder as encoder_lib
from mseb import types as mseb_types

logger = logging.getLogger(__name__)

# These are the exact embedding dimensions reported in the Whisper paper / code.
_WHISPER_EMBEDDING_DIM: dict[str, int] = {
    "tiny":     384,
    "tiny.en":  384,
    "base":     512,
    "base.en":  512,
    "small":    768,
    "small.en": 768,
    "medium":   1024,
    "medium.en":1024,
    "large":    1280,
    "large-v1": 1280,
    "large-v2": 1280,
    "large-v3": 1280,
}

_WHISPER_SR:           int = 16_000     # Hz — Whisper's fixed input sample rate
_WHISPER_MAX_SAMPLES:  int = 480_000    # 30 s × 16 000 Hz


class WhisperPooledHeartSoundEncoder(encoder_lib.MultiModalEncoder):
    """
    Uses Whisper's audio encoder (no decoder) with mean-pooling to embed audio.

    Since our HeartSoundClassificationTask already resamples all audio to 16 kHz,
    no resampling is performed here.  The encoder simply:
        1. Pads / trims each waveform to exactly 30 s.
        2. Converts to an 80-channel log-mel spectrogram.
        3. Runs the Whisper encoder transformer.
        4. Mean-pools the output token sequence → single vector.

    This gives a representation that captures the acoustic texture of the
    entire clip, which is exactly what we need for heart sound classification.
    """

    def __init__(
        self,
        model_size: str = "base",
        batch_size: int = 8,
    ) -> None:
        """
        Args:
            model_size: One of tiny, base, small, medium, large, large-v2, large-v3.
                        Larger models have richer embeddings but are slower.
                        'base' is a good starting point (512-dim, fast).
            batch_size: Number of mel spectrograms in one GPU forward pass.
                        All 30-second mels have the same shape (80, 3000),
                        so true batching is possible here (unlike TS2Vec).
        """
        super().__init__()
        if model_size not in _WHISPER_EMBEDDING_DIM:
            raise ValueError(
                f"model_size must be one of {sorted(_WHISPER_EMBEDDING_DIM)}, "
                f"got: {model_size!r}"
            )
        self._model_size   = model_size
        self._batch_size   = batch_size
        self._embedding_dim = _WHISPER_EMBEDDING_DIM[model_size]

        # Set in _setup
        self._audio_encoder      = None
        self._log_mel_spectrogram = None   # whisper function, stored for convenience
        self._pad_or_trim         = None   # whisper function
        self._device              = None

    # ── MSEB required overrides ───────────────────────────────────────────────

    def _setup(self) -> None:
        """
        Load Whisper and extract only the audio encoder component.

        We do NOT load the decoder — this saves memory and is faster.
        The decoder is reconstructed by whisper.load_model internally;
        we simply detach the encoder and discard the rest.
        """
        try:
            import whisper  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "openai-whisper is not installed. "
                "Run: pip install openai-whisper"
            ) from exc

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(
            "WhisperPooled: loading model=%s | device=%s | embed_dim=%d",
            self._model_size, self._device, self._embedding_dim,
        )

        # Load the full model — whisper.load_model always loads both encoder
        # and decoder, but we only keep the encoder to save GPU memory.
        full_model = whisper.load_model(self._model_size, device=self._device)
        self._audio_encoder = full_model.encoder
        self._audio_encoder.eval()
        # Explicitly delete the rest to free memory
        del full_model

        # Store whisper utility functions
        self._log_mel_spectrogram = whisper.log_mel_spectrogram
        self._pad_or_trim         = whisper.pad_or_trim

        logger.info("WhisperPooled encoder ready.")

    def _check_input_types(
        self, batch: Sequence[mseb_types.MultiModalObject]
    ) -> None:
        """Assert every item in the batch is a Sound object."""
        bad = [type(x).__name__ for x in batch if not isinstance(x, mseb_types.Sound)]
        if bad:
            raise ValueError(
                f"WhisperPooledHeartSoundEncoder requires all inputs to be "
                f"mseb.types.Sound, but received: {set(bad)}"
            )

    def _encode(
        self,
        batch: Sequence[mseb_types.Sound],
    ) -> list[mseb_types.SoundEmbedding]:
        """
        Encode a batch of Sound objects into mean-pooled Whisper embeddings.

        Unlike TS2Vec (variable-length) and CLAP (handled by processor padding),
        Whisper mel-spectrograms always have shape (80, 3000) after pad_or_trim,
        so we can process true mini-batches on the GPU.

        Args:
            batch: Sequence of mseb.types.Sound objects at 16 kHz.

        Returns:
            List of mseb.types.SoundEmbedding, one per input sound.
        """
        results: list[mseb_types.SoundEmbedding] = []

        for i in range(0, len(batch), self._batch_size):
            sub_batch = batch[i : i + self._batch_size]
            results.extend(self._encode_sub_batch(sub_batch))

        return results

    # ── Private helpers ───────────────────────────────────────────────────────

    def _waveform_to_mel(self, waveform: np.ndarray) -> torch.Tensor | None:
        """
        Convert a 16 kHz waveform to an 80-channel log-mel spectrogram.

        Args:
            waveform: float32 array of shape (T,).

        Returns:
            torch.Tensor of shape (80, 3000), or None on failure.
        """
        try:
            # pad_or_trim requires a torch.Tensor
            audio = torch.from_numpy(waveform.astype(np.float32))
            # Clip to / pad up to exactly 30 seconds
            audio = self._pad_or_trim(audio, length=_WHISPER_MAX_SAMPLES)
            # log_mel_spectrogram returns a CPU Tensor of shape (80, 3000)
            mel   = self._log_mel_spectrogram(audio)
            return mel
        except Exception as exc:
            logger.debug("mel spectrogram computation failed: %s", exc)
            return None

    def _encode_sub_batch(
        self, sub_batch: Sequence[mseb_types.Sound]
    ) -> list[mseb_types.SoundEmbedding]:
        """
        Encode one sub-batch of sounds via a single GPU forward pass.

        Sounds that fail mel computation are assigned zero vectors so the
        rest of the batch still processes.
        """
        # ── Build mel batch ───────────────────────────────────────────────────
        mels:   list[torch.Tensor | None] = []
        sounds: list[mseb_types.Sound]    = []

        for sound in sub_batch:
            mel = self._waveform_to_mel(sound.waveform)
            if mel is None:
                logger.warning(
                    "WhisperPooled: mel failed for %s  →  zero vector.",
                    sound.context.id,
                )
            mels.append(mel)
            sounds.append(sound)

        # ── Separate valid and failed samples ─────────────────────────────────
        valid_mels:   list[torch.Tensor] = []
        valid_indices: list[int]         = []
        failed_indices: list[int]        = []

        for idx, mel in enumerate(mels):
            if mel is not None:
                valid_mels.append(mel)
                valid_indices.append(idx)
            else:
                failed_indices.append(idx)

        # ── GPU forward pass for valid mels ───────────────────────────────────
        embeddings_map: dict[int, np.ndarray] = {}

        if valid_mels:
            try:
                # Stack into (B, 80, 3000) and move to GPU
                mel_batch = torch.stack(valid_mels, dim=0).to(self._device)

                with torch.no_grad():
                    # Whisper encoder output: (B, T', embedding_dim)
                    # T' ≈ 1500 for all model sizes
                    encoder_out = self._audio_encoder(mel_batch)

                # Mean-pool over the time dimension: (B, T', D) → (B, D)
                pooled = encoder_out.mean(dim=1)            # (B, embedding_dim)
                pooled_np = pooled.cpu().numpy().astype(np.float32)

                for local_i, global_idx in enumerate(valid_indices):
                    embeddings_map[global_idx] = pooled_np[local_i]

            except Exception as exc:
                logger.warning(
                    "WhisperPooled forward pass failed for sub-batch: %s  "
                    "→  using zero vectors for all %d samples.",
                    exc, len(valid_mels),
                )
                for global_idx in valid_indices:
                    embeddings_map[global_idx] = np.zeros(
                        self._embedding_dim, dtype=np.float32
                    )

        # Fill failed samples with zero vectors
        for global_idx in failed_indices:
            embeddings_map[global_idx] = np.zeros(
                self._embedding_dim, dtype=np.float32
            )

        # ── Assemble results in original order ────────────────────────────────
        return [
            mseb_types.SoundEmbedding(
                embedding=embeddings_map[idx],
                context=sound.context,
            )
            for idx, sound in enumerate(sounds)
        ]
