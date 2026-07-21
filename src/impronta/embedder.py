"""Speaker-embedding backends.

The :class:`Embedder` protocol is the seam that keeps torch out of the
default test tier: the real :class:`EcapaEmbedder` imports speechbrain/torch
lazily inside ``_get_model`` and is process-cached, so a warm Cloud Run
instance loads the model exactly once.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .config import ImprontaConfig


def l2_normalize(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (rows / norms).astype(np.float32)


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns audio clips into L2-normalized embedding rows."""

    dim: int

    def embed_batch(self, clips: list[np.ndarray], sample_rate: int) -> np.ndarray:
        """(n clips) -> float32 array of shape (n, dim) with unit rows."""
        ...


_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}
_MODEL_LOCK = threading.Lock()


class EcapaEmbedder:
    """ECAPA-TDNN speaker embeddings via speechbrain (192-dim).

    The model is downloaded to ``config.model_cache_dir`` on first use and
    cached per (source, device, cache_dir) for the process lifetime.
    """

    dim = 192

    def __init__(self, config: ImprontaConfig | None = None):
        self._config = config or ImprontaConfig()

    def _get_model(self) -> Any:  # pragma: no cover - exercised in integration tier
        cfg = self._config
        key = (cfg.model_source, cfg.device, cfg.model_cache_dir)
        model = _MODEL_CACHE.get(key)
        if model is not None:
            return model
        with _MODEL_LOCK:
            model = _MODEL_CACHE.get(key)
            if model is not None:
                return model
            from speechbrain.inference.speaker import EncoderClassifier

            savedir = Path(cfg.model_cache_dir) / "ecapa"
            model = EncoderClassifier.from_hparams(
                source=cfg.model_source,
                savedir=str(savedir),
                run_opts={"device": cfg.device},
            )
            _MODEL_CACHE[key] = model
            return model

    def embed_batch(
        self, clips: list[np.ndarray], sample_rate: int
    ) -> np.ndarray:  # pragma: no cover - exercised in integration tier
        import torch

        if not clips:
            return np.zeros((0, self.dim), dtype=np.float32)
        model = self._get_model()
        max_len = max(c.shape[0] for c in clips)
        batch = torch.zeros(len(clips), max_len)
        lens = torch.zeros(len(clips))
        for i, clip in enumerate(clips):
            t = torch.from_numpy(np.ascontiguousarray(clip, dtype=np.float32))
            batch[i, : t.shape[0]] = t
            lens[i] = t.shape[0] / max_len
        with torch.inference_mode():
            emb = model.encode_batch(batch, lens)  # (n, 1, 192)
        rows = emb.squeeze(1).cpu().numpy().astype(np.float32)
        return l2_normalize(rows)
