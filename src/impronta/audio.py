"""Audio decoding, resampling, channel handling, and slicing.

Decoding uses ``soundfile`` (bundled libsndfile: wav/flac/ogg and, since
libsndfile 1.1, mp3) with an optional PyAV fallback for m4a/aac via the
``impronta[m4a]`` extra. Resampling uses ``soxr`` so the default code path
never imports torch.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import BinaryIO, Union

import numpy as np
import soundfile as sf

from .exceptions import AudioDecodeError

AudioInput = Union[str, os.PathLike, bytes, bytearray, BinaryIO]  # noqa: UP007 - runtime alias


@dataclass(slots=True)
class DecodedAudio:
    """Decoded audio, resampled to the target rate.

    ``samples`` is float32 with shape (channels, n); mono audio keeps an
    explicit channel axis of size 1.
    """

    samples: np.ndarray
    sample_rate: int

    @property
    def num_channels(self) -> int:
        return self.samples.shape[0]

    @property
    def duration(self) -> float:
        return self.samples.shape[1] / self.sample_rate


def _as_readable(source: AudioInput) -> tuple[object, str]:
    if isinstance(source, (bytes, bytearray)):
        return io.BytesIO(bytes(source)), "<bytes>"
    if isinstance(source, (str, os.PathLike)):
        return os.fspath(source), os.fspath(source)
    if hasattr(source, "read"):
        return source, getattr(source, "name", "<stream>")
    raise AudioDecodeError(f"unsupported audio input type: {type(source).__name__}")


def _decode_with_pyav(readable: object, label: str) -> tuple[np.ndarray, int]:  # pragma: no cover
    """Fallback for formats libsndfile can't read (m4a/aac). Needs impronta[m4a]."""
    try:
        import av
    except ImportError as exc:
        raise AudioDecodeError(
            f"could not decode {label!r} with libsndfile, and PyAV is not installed; "
            "install the m4a extra (pip install 'impronta[m4a]') for m4a/aac support"
        ) from exc
    if isinstance(readable, io.BytesIO):
        readable.seek(0)
    try:
        with av.open(readable) as container:
            stream = next(s for s in container.streams if s.type == "audio")
            chunks = []
            rate = None
            for frame in container.decode(stream):
                arr = frame.to_ndarray()  # (channels, n) or (n,) depending on layout
                if arr.ndim == 1:
                    arr = arr[np.newaxis, :]
                if np.issubdtype(arr.dtype, np.integer):
                    arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max
                chunks.append(arr.astype(np.float32))
                rate = frame.sample_rate
            if not chunks or rate is None:
                raise AudioDecodeError(f"no audio frames decoded from {label!r}")
            return np.concatenate(chunks, axis=1), int(rate)
    except AudioDecodeError:
        raise
    except Exception as exc:
        raise AudioDecodeError(f"failed to decode {label!r}: {exc}") from exc


def load_audio(source: AudioInput, target_sr: int = 16_000) -> DecodedAudio:
    """Decode any supported input to float32 (channels, n) at ``target_sr``."""
    readable, label = _as_readable(source)
    try:
        data, sr = sf.read(readable, dtype="float32", always_2d=True)
        samples = np.ascontiguousarray(data.T)  # (channels, n)
    except (sf.LibsndfileError, RuntimeError, OSError):
        samples, sr = _decode_with_pyav(readable, label)
    if samples.shape[1] == 0:
        raise AudioDecodeError(f"audio input {label!r} contains no samples")
    if sr != target_sr:
        import soxr

        resampled = [
            soxr.resample(np.ascontiguousarray(ch), sr, target_sr) for ch in samples
        ]
        samples = np.stack([r.astype(np.float32) for r in resampled])
    return DecodedAudio(samples=samples.astype(np.float32), sample_rate=target_sr)


def select_channel(audio: DecodedAudio, channel: int | None) -> np.ndarray:
    """Pick one channel as a 1-D array; ``None`` (or out-of-range) downmixes.

    A multichannel Scribe response paired with mono audio is a real-world
    case (the caller re-exported the file); falling back to the downmix keeps
    that working instead of crashing.
    """
    if channel is not None and 0 <= channel < audio.num_channels:
        return audio.samples[channel]
    if audio.num_channels == 1:
        return audio.samples[0]
    return audio.samples.mean(axis=0).astype(np.float32)


def slice_seconds(x: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    """Slice [start, end) seconds, clamped to the signal bounds (may be empty)."""
    i = max(0, int(round(start * sr)))
    j = min(x.shape[0], int(round(end * sr)))
    if j <= i:
        return x[:0]
    return x[i:j]
