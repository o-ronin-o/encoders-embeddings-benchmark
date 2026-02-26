"""
TS2Vec encoder wrapper — MSEB MultiModalEncoder interface.

What we know from the MSEB notebook:
  - Encoders must subclass mseb.encoder.MultiModalEncoder
  - Must implement: _setup(), _check_input_types(batch), _encode(batch)
  - _encode() receives a Sequence[MultiModalObject] and returns a Sequence[MultiModalObject]
  - The public encode(batch) method calls _check_input_types then _encode
  - setup() is the public method that calls _setup(); call it once before encoding

TS2Vec API (from https://github.com/zhihanyue/ts2vec):
  - TS2Vec(input_dims=1, device=0, output_dims=320, ...)
  - model.load(path)     — loads checkpoint
  - model.encode(data, encoding_window)
      data: np.ndarray of shape (n_samples, n_timesteps, n_features)
      encoding_window='full_series' → returns (n_samples, output_dims)
      encoding_window=None          → returns (n_samples, n_timesteps, output_dims)
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch

from mseb import encoder as encoder_lib
from mseb import types as mseb_types

logger = logging.getLogger(__name__)


class TS2VecHeartSoundEncoder(encoder_lib.MultiModalEncoder):
    """
    Wraps a pretrained TS2Vec model to produce fixed-size audio embeddings.

    TS2Vec is a universal time-series self-supervised learning framework.
    We treat each raw waveform as a univariate time series:
        waveform shape (T,)  →  TS2Vec input (1, T, 1)

    Two encoding modes:
        'full_series'  — TS2Vec hierarchically pools across all time steps
                         Returns (1, output_dim). One global vector per clip.
                         Best for global classification (normal / murmur / other).

        'multiscale'   — TS2Vec returns token representations at every time step
                         shape (1, T, output_dim). We mean-pool over T to get a
                         fixed-size vector. Captures finer temporal detail.

    Reference: https://github.com/zhihanyue/ts2vec
    """

    def __init__(
        self,
        model_path: str,
        output_dim: int = 320,
        encoding_window: str = "full_series",
        batch_size: int = 8,
    ) -> None:
        """
        Args:
            model_path:       Path to pretrained TS2Vec checkpoint (.pkl file).
                              Produced by ts2vec.TS2Vec.save(path).
            output_dim:       Embedding dimension — must match the trained model.
                              Default 320 follows the TS2Vec paper.
            encoding_window:  'full_series' (recommended) or 'multiscale'.
                              'full_series' is faster and works well for global
                              cardiac sound classification.
            batch_size:       Sounds processed per forward pass. Reduce if OOM.
        """
        super().__init__()
        if encoding_window not in ("full_series", "multiscale"):
            raise ValueError(
                f"encoding_window must be 'full_series' or 'multiscale', "
                f"got: {encoding_window!r}"
            )
        self._model_path    = model_path
        self._output_dim    = output_dim
        self._encoding_window = encoding_window
        self._batch_size    = batch_size
        self._model         = None
        self._device        = None

    # ── MSEB required overrides ───────────────────────────────────────────────

    def _setup(self) -> None:
        """
        Load TS2Vec model from checkpoint.

        Called once by the public setup() method before any encoding.
        """
        try:
            from ts2vec import TS2Vec  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "ts2vec is not installed. Install via:\n"
                "  pip install ts2vec\n"
                "or for the original repo:\n"
                "  pip install git+https://github.com/zhihanyue/ts2vec"
            ) from exc

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_id    = 0 if torch.cuda.is_available() else -1

        logger.info(
            "TS2Vec: loading checkpoint from %s | device=%s | "
            "output_dim=%d | encoding_window=%s",
            self._model_path, self._device, self._output_dim, self._encoding_window,
        )

        # TS2Vec uses device_id=-1 for CPU, 0..n for GPU
        self._model = TS2Vec(
            input_dims=1,
            device=device_id,
            output_dims=self._output_dim,
        )
        self._model.load(self._model_path)
        logger.info("TS2Vec model loaded successfully.")

    def _check_input_types(
        self, batch: Sequence[mseb_types.MultiModalObject]
    ) -> None:
        """Assert every item in the batch is a Sound object."""
        bad = [type(x).__name__ for x in batch if not isinstance(x, mseb_types.Sound)]
        if bad:
            raise ValueError(
                f"TS2VecHeartSoundEncoder requires all inputs to be "
                f"mseb.types.Sound, but received: {set(bad)}"
            )

    def _encode(
        self,
        batch: Sequence[mseb_types.Sound],
    ) -> list[mseb_types.SoundEmbedding]:
        """
        Encode a batch of Sound objects into SoundEmbedding vectors.

        Sounds are processed individually (variable-length waveforms cannot be
        naively stacked into a batch tensor), then returned as a list.

        Args:
            batch: List of mseb.types.Sound objects. Waveforms may have
                   different lengths — handled correctly.

        Returns:
            List of mseb.types.SoundEmbedding, one per input sound.
            Failed samples get a zero vector so the pipeline does not crash.
        """
        results: list[mseb_types.SoundEmbedding] = []

        for i in range(0, len(batch), self._batch_size):
            sub_batch = batch[i : i + self._batch_size]
            for sound in sub_batch:
                results.append(self._encode_one(sound))

        return results

    # ── Private helpers ───────────────────────────────────────────────────────

    def _encode_one(self, sound: mseb_types.Sound) -> mseb_types.SoundEmbedding:
        """Encode a single Sound, returning a SoundEmbedding."""
        try:
            embedding = self._forward(sound.waveform)
        except Exception as exc:
            logger.warning(
                "TS2Vec failed on sample %s: %s  →  using zero vector.",
                sound.context.id, exc,
            )
            embedding = np.zeros(self._output_dim, dtype=np.float32)

        return mseb_types.SoundEmbedding(
            embedding=embedding,
            context=sound.context,
        )

    def _forward(self, waveform: np.ndarray) -> np.ndarray:
        """
        Run the TS2Vec forward pass.

        Args:
            waveform: float32 array of shape (T,).

        Returns:
            float32 array of shape (output_dim,).
        """
        # TS2Vec expects (n_samples, n_timesteps, n_features)
        x = waveform.astype(np.float32).reshape(1, -1, 1)  # (1, T, 1)

        if self._encoding_window == "full_series":
            # Returns (1, output_dim)
            out = self._model.encode(x, encoding_window="full_series")
            return out.squeeze(0).astype(np.float32)          # (output_dim,)

        else:  # "multiscale"
            # encoding_window=None returns per-timestep representations
            # Returns (1, T, output_dim) — we mean-pool over T
            out = self._model.encode(x, encoding_window=None)
            return out.squeeze(0).mean(axis=0).astype(np.float32)  # (output_dim,)
