"""Shared fixtures: scribe-response builder, synthetic voices, fake embedder.

The default tier must stay hermetic: no network (pytest-socket enforces
this) and no torch import (checked in ``pytest_sessionfinish`` below).

Synthetic voices
----------------
``make_voice_audio`` produces speech-like audio: a sine carrier at a
voice-specific frequency, amplitude-modulated by Gamma(0.4)-distributed
noise — the amplitude distribution WADA-SNR models speech with. Clean clips
read as very high SNR; mixing in white noise at a known ratio drags the
estimate down predictably.

``FakeEmbedder`` recovers the carrier frequency via FFT and maps it to a
configured unit vector, so tests control similarity outcomes exactly.
"""

from __future__ import annotations

import os
import sys
from typing import Any

# macOS only: the pip wheels of faiss-cpu and torch each bundle their own
# libomp, which crashes the process the moment both are active (integration
# tier). Single-threading OpenMP and allowing the duplicate runtime is the
# standard workaround; production Linux wheels share one runtime and are
# unaffected. Must be set before torch is imported.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest

from impronta.embedder import l2_normalize

SR = 16_000
DIM = 8


# ---------------------------------------------------------------------------
# scribe response builder
# ---------------------------------------------------------------------------


def make_words(
    spec: list[tuple[str, float | None, float | None, str | None]],
    logprob: float | None = -0.05,
) -> list[dict[str, Any]]:
    """[(text, start, end, speaker_id), ...] -> schema-exact word dicts."""
    return [
        {
            "text": text,
            "start": start,
            "end": end,
            "type": "word",
            "speaker_id": sid,
            "logprob": logprob,
            "channel_index": None,
        }
        for text, start, end, sid in spec
    ]


def speech_words(
    speaker_id: str,
    start: float,
    end: float,
    word_sec: float = 0.28,
    logprob: float | None = -0.05,
) -> list[dict[str, Any]]:
    """A continuous run of words for one speaker covering [start, end)."""
    words = []
    t = start
    i = 0
    while t < end - 1e-9:
        w_end = min(t + word_sec, end)
        words.append(
            {
                "text": f"w{i}",
                "start": round(t, 3),
                "end": round(w_end, 3),
                "type": "word",
                "speaker_id": speaker_id,
                "logprob": logprob,
                "channel_index": None,
            }
        )
        t = w_end
        i += 1
    return words


def make_scribe_response(
    words: list[dict[str, Any]],
    language: str = "en",
    transcription_id: str | None = "tx-1",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resp: dict[str, Any] = {
        "language_code": language,
        "language_probability": 0.98,
        "text": " ".join(w["text"] for w in words if w["type"] == "word"),
        "words": words,
        "channel_index": None,
        "audio_duration_secs": max((w["end"] or 0) for w in words) if words else 0.0,
    }
    if transcription_id is not None:
        resp["transcription_id"] = transcription_id
    if extra:
        resp.update(extra)
    return resp


def make_multichannel_response(
    channel_words: list[list[dict[str, Any]]],
    language: str = "en",
    transcription_id: str = "tx-multi",
) -> dict[str, Any]:
    return {
        "transcription_id": transcription_id,
        "transcripts": [
            {
                "language_code": language,
                "words": words,
                "channel_index": i,
                "audio_duration_secs": max((w["end"] or 0) for w in words) if words else 0.0,
            }
            for i, words in enumerate(channel_words)
        ],
    }


# ---------------------------------------------------------------------------
# synthetic audio
# ---------------------------------------------------------------------------


def make_voice_audio(
    duration: float,
    voice_freq: float,
    *,
    sr: int = SR,
    noise_db: float | None = None,
    seed: int = 7,
) -> np.ndarray:
    """Speech-like audio for one synthetic voice; optionally add noise.

    ``noise_db`` is the approximate target SNR in dB (None = clean).
    """
    rng = np.random.default_rng(seed + int(voice_freq))
    n = int(duration * sr)
    t = np.arange(n) / sr
    envelope = rng.gamma(shape=0.4, scale=1.0, size=n)
    signal = envelope * np.sin(2 * np.pi * voice_freq * t)
    signal = signal / (np.abs(signal).max() + 1e-9)
    if noise_db is not None:
        sig_power = float((signal**2).mean())
        noise_power = sig_power / (10 ** (noise_db / 10))
        noise = rng.normal(0.0, np.sqrt(noise_power), size=n)
        signal = signal + noise
        signal = signal / (np.abs(signal).max() + 1e-9)
    return signal.astype(np.float32)


def compose_timeline(
    duration: float,
    parts: list[tuple[float, float, float]],
    *,
    sr: int = SR,
    noise_db: float | None = None,
) -> np.ndarray:
    """Build a recording: [(start, end, voice_freq), ...] on a silent bed."""
    out = np.zeros(int(duration * sr), dtype=np.float32)
    for start, end, freq in parts:
        clip = make_voice_audio(end - start, freq, sr=sr, noise_db=noise_db)
        i = int(start * sr)
        out[i : i + clip.shape[0]] = clip
    return out


# ---------------------------------------------------------------------------
# fake embedder
# ---------------------------------------------------------------------------


def basis(axis: int, dim: int = DIM) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[axis] = 1.0
    return v


def blend(a: np.ndarray, b: np.ndarray, sim: float) -> np.ndarray:
    """A unit vector with cosine similarity ``sim`` to unit vector ``a``."""
    return (sim * a + np.sqrt(max(0.0, 1 - sim**2)) * b).astype(np.float32)


class FakeEmbedder:
    """Deterministic embedder: FFT peak frequency -> configured unit vector."""

    def __init__(self, voices: dict[float, np.ndarray], dim: int = DIM):
        self.voices = {float(f): l2_normalize(v[np.newaxis, :])[0] for f, v in voices.items()}
        self.dim = dim

    def _embed_one(self, clip: np.ndarray, sample_rate: int) -> np.ndarray:
        spectrum = np.abs(np.fft.rfft(clip))
        peak_freq = float(np.argmax(spectrum) * sample_rate / clip.shape[0])
        nearest = min(self.voices, key=lambda f: abs(f - peak_freq))
        return self.voices[nearest]

    def embed_batch(self, clips: list[np.ndarray], sample_rate: int) -> np.ndarray:
        if not clips:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._embed_one(c, sample_rate) for c in clips])


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    """Voices: 440 Hz -> e0 ('alice'), 880 -> e1 ('bob'), 1320 -> e2, 1760 -> e3."""
    return FakeEmbedder(
        {440.0: basis(0), 880.0: basis(1), 1320.0: basis(2), 1760.0: basis(3)}
    )


# ---------------------------------------------------------------------------
# hermeticity: the default tier must never import torch
# ---------------------------------------------------------------------------


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    markexpr = session.config.getoption("markexpr", default="") or ""
    if "not integration" in markexpr and "torch" in sys.modules:
        raise pytest.UsageError(
            "torch was imported during the default (hermetic) test tier — "
            "a lazy-import seam has leaked"
        )
