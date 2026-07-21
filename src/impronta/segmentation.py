"""Merging word timelines into embeddable utterance segments.

Individual Scribe words are typically 100–400 ms — far too short for a
reliable ECAPA-TDNN embedding. We merge contiguous same-speaker words into
utterance segments bounded by ``min_segment_sec``/``max_segment_sec``, with a
gap tolerance absorbing short pauses (and the holes left by words that carry
null timestamps).
"""

from __future__ import annotations

import math

from .config import ImprontaConfig
from .models import Segment, Word


def _mean_logprob_confidence(logprobs: list[float]) -> float:
    """Geometric-mean word probability in [0, 1]; 1.0 when no data."""
    if not logprobs:
        return 1.0
    return math.exp(sum(logprobs) / len(logprobs))


def segment_words(words: tuple[Word, ...], cfg: ImprontaConfig) -> dict[str, list[Segment]]:
    """Merge each speaker's words into time-ordered segments.

    Only ``type == 'word'`` entries with non-null start/end/speaker_id
    participate; ``spacing`` and ``audio_event`` entries and null-timed words
    are skipped (the gap tolerance absorbs the resulting holes). Segments
    shorter than ``min_segment_sec`` are dropped; a segment is closed at a
    word boundary once extending it would exceed ``max_segment_sec``.
    """
    by_speaker: dict[str, list[Word]] = {}
    for w in words:
        if w.type != "word" or w.speaker_id is None or w.start is None or w.end is None:
            continue
        if w.end < w.start:
            continue
        if (w.end - w.start) > cfg.max_segment_sec:
            continue  # a single "word" longer than a max segment is garbage timing data
        by_speaker.setdefault(w.speaker_id, []).append(w)

    out: dict[str, list[Segment]] = {}
    for speaker_id, speaker_words in by_speaker.items():
        speaker_words.sort(key=lambda w: (w.start, w.end))
        segments: list[Segment] = []
        seg_start: float | None = None
        seg_end = 0.0
        logprobs: list[float] = []
        for w in speaker_words:
            assert w.start is not None and w.end is not None
            if seg_start is None:
                seg_start, seg_end = w.start, w.end
                logprobs = [w.logprob] if w.logprob is not None else []
                continue
            gap_ok = (w.start - seg_end) <= cfg.gap_tolerance_sec
            length_ok = (w.end - seg_start) <= cfg.max_segment_sec
            if gap_ok and length_ok:
                seg_end = max(seg_end, w.end)
                if w.logprob is not None:
                    logprobs.append(w.logprob)
            else:
                _close_segment(segments, speaker_id, seg_start, seg_end, logprobs, cfg)
                seg_start, seg_end = w.start, w.end
                logprobs = [w.logprob] if w.logprob is not None else []
        _close_segment(segments, speaker_id, seg_start, seg_end, logprobs, cfg)
        if segments:
            out[speaker_id] = segments
    return out


def _close_segment(
    segments: list[Segment],
    speaker_id: str,
    seg_start: float | None,
    seg_end: float,
    logprobs: list[float],
    cfg: ImprontaConfig,
) -> None:
    if seg_start is not None and (seg_end - seg_start) >= cfg.min_segment_sec:
        segments.append(
            Segment(
                speaker_id=speaker_id,
                start=seg_start,
                end=seg_end,
                confidence=_mean_logprob_confidence(logprobs),
            )
        )


def segment_windows(duration: float, cfg: ImprontaConfig) -> list[Segment]:
    """Non-overlapping fixed windows over a raw clip (direct enrollment).

    The final partial window is kept only if it reaches ``min_segment_sec``.
    """
    windows: list[Segment] = []
    t = 0.0
    while t < duration:
        end = min(t + cfg.direct_window_sec, duration)
        if (end - t) >= cfg.min_segment_sec:
            windows.append(Segment(speaker_id=None, start=t, end=end, confidence=1.0))
        t = end
    return windows
