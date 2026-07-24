"""The shared preparation pipeline: segments -> filters -> embeddings.

Both enrollment and identification run the same steps per speaker:
audio slicing + blind SNR measurement -> confidence gate -> SNR filter
(progressive relaxation + rescue) -> quality-ranked cap -> ECAPA embedding.
Keeping one code path guarantees enrolled and queried embeddings are
produced identically.

The pipeline is split into three stages so the threshold-tuning harness can
replay everything downstream of embedding from a precomputed cache:

- :func:`measure_segments` — needs audio; applies only the STRUCTURAL gate
  (clip length >= min_segment_sec) and computes WADA SNR per clip.
- :func:`select_segments` — pure metadata (no audio, no torch): confidence
  gate, SNR tiers, rescue, quality cap.
- :func:`prepare_segments` — measure -> select -> embed (production path).
- :func:`assemble_prepared` — select -> row-index precomputed embeddings
  (tuning path).

CACHE-IDENTITY KNOBS: ``min_segment_sec``, ``max_segment_sec`` and
``gap_tolerance_sec`` participate in segmentation/measurement and therefore
define which segments exist at all. A precomputed embedding cache is only
valid for the exact values it was built with — these knobs are excluded
from threshold tuning.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from .audio import slice_seconds
from .config import ImprontaConfig
from .embedder import Embedder, l2_normalize
from .models import Segment, composite_quality
from .snr import filter_by_snr, wada_snr


@dataclass(slots=True)
class PreparedSpeaker:
    """One speaker's usable segments and their embeddings.

    ``embeddings is None`` means the speaker is unusable (nothing survived
    the filters); ``best_snr_db`` is kept for diagnostics in that case.
    """

    segments_total: int
    segments: list[Segment]  # surviving segments, snr_db filled in
    embeddings: np.ndarray | None  # (n, dim) L2-normalized, aligned with segments
    quality_tier: str | None
    best_snr_db: float | None


@dataclass(slots=True)
class MeasuredSegments:
    """Segments that survived the structural gate, with SNR measured.

    No confidence or SNR *filtering* has been applied — every segment here
    is a candidate for :func:`select_segments` under any config that shares
    the cache-identity knobs.
    """

    segments: list[Segment]  # snr_db filled in
    clips: list[np.ndarray]  # aligned audio slices
    segments_total: int  # count before slicing (diagnostics parity)


@dataclass(slots=True)
class SelectionResult:
    """Outcome of the metadata-only selection stage."""

    kept_indices: list[int]  # into MeasuredSegments.segments, final (time) order
    quality_tier: str | None
    best_snr_db: float | None  # max SNR among confidence-passing segments


def measure_segments(
    segments: list[Segment],
    audio_mono: np.ndarray,
    cfg: ImprontaConfig,
) -> MeasuredSegments:
    """Slice all segments and measure blind SNR; structural gate only."""
    out_segments: list[Segment] = []
    clips: list[np.ndarray] = []
    min_samples = int(cfg.min_segment_sec * cfg.sample_rate)
    for seg in segments:
        clip = slice_seconds(audio_mono, cfg.sample_rate, seg.start, seg.end)
        if clip.shape[0] >= min_samples:
            out_segments.append(replace(seg, snr_db=wada_snr(clip)))
            clips.append(clip)
    return MeasuredSegments(
        segments=out_segments, clips=clips, segments_total=len(segments)
    )


def select_segments(
    segments: Sequence[Segment], cfg: ImprontaConfig
) -> SelectionResult:
    """Pick the segments to embed. Pure metadata — no audio, no torch.

    Steps: ElevenLabs-confidence gate (hard, never relaxed) -> SNR tiers
    with progressive relaxation -> rescue supplement up to
    ``min_kept_speech_sec`` above ``snr_floor_db`` (demotes tier to "low")
    -> quality-ranked cap at ``max_embeddings_per_enroll`` -> time order.
    """
    confident = [
        i for i, s in enumerate(segments) if s.confidence >= cfg.min_segment_confidence
    ]
    if not confident:
        return SelectionResult([], None, None)

    snrs = [segments[i].snr_db for i in confident]
    best_snr = max(snrs)  # type: ignore[type-var]
    kept_conf_pos, tier = filter_by_snr(snrs, cfg)  # type: ignore[arg-type]
    kept = [confident[p] for p in kept_conf_pos]

    # rescue supplement: real telephony audio reads 5-15 dB on WADA, so the
    # tiers alone can starve a 20-minute call down to seconds of evidence
    # and the resulting profile is a channel fingerprint, not a voiceprint.
    kept_speech = sum(segments[i].duration for i in kept)
    if kept_speech < cfg.min_kept_speech_sec:
        kept_set = set(kept)
        rescuable = [
            i
            for i in confident
            if i not in kept_set
            and segments[i].snr_db is not None
            and segments[i].snr_db >= cfg.snr_floor_db  # type: ignore[operator]
        ]
        rescuable.sort(
            key=lambda i: composite_quality(
                segments[i].snr_db, segments[i].confidence, segments[i].duration
            ),
            reverse=True,
        )
        for i in rescuable:
            if kept_speech >= cfg.min_kept_speech_sec:
                break
            kept.append(i)
            kept_speech += segments[i].duration
            tier = "low"

    if not kept:
        return SelectionResult([], None, best_snr)

    # quality-ranked cap: keep the best segments, not the first ones
    kept.sort(
        key=lambda i: composite_quality(
            segments[i].snr_db, segments[i].confidence, segments[i].duration
        ),
        reverse=True,
    )
    kept = kept[: cfg.max_embeddings_per_enroll]
    kept.sort(key=lambda i: segments[i].start)  # restore time order
    return SelectionResult(kept, tier, best_snr)


def prepare_segments(
    segments: list[Segment],
    audio_mono: np.ndarray,
    embedder: Embedder,
    cfg: ImprontaConfig,
) -> PreparedSpeaker:
    """Run one speaker's segments through the full filter + embed pipeline."""
    measured = measure_segments(segments, audio_mono, cfg)
    selection = select_segments(measured.segments, cfg)
    if not selection.kept_indices:
        return PreparedSpeaker(
            measured.segments_total, [], None, None, selection.best_snr_db
        )
    clips = [measured.clips[i] for i in selection.kept_indices]
    embeddings = l2_normalize(embedder.embed_batch(clips, cfg.sample_rate))
    return PreparedSpeaker(
        segments_total=measured.segments_total,
        segments=[measured.segments[i] for i in selection.kept_indices],
        embeddings=embeddings,
        quality_tier=selection.quality_tier,
        best_snr_db=selection.best_snr_db,
    )


def assemble_prepared(
    segments: Sequence[Segment],
    embeddings_all: np.ndarray,
    segments_total: int,
    cfg: ImprontaConfig,
) -> PreparedSpeaker:
    """Tuning entry point: select over cached measurements, no embedding.

    ``segments`` must be measured segments (snr_db and confidence filled)
    row-aligned with ``embeddings_all`` (L2-normalized, as produced by the
    production embed step). Produces exactly what :func:`prepare_segments`
    would for the same inputs.
    """
    selection = select_segments(segments, cfg)
    if not selection.kept_indices:
        return PreparedSpeaker(segments_total, [], None, None, selection.best_snr_db)
    rows = np.ascontiguousarray(
        embeddings_all[np.asarray(selection.kept_indices)], dtype=np.float32
    )
    return PreparedSpeaker(
        segments_total=segments_total,
        segments=[segments[i] for i in selection.kept_indices],
        embeddings=rows,
        quality_tier=selection.quality_tier,
        best_snr_db=selection.best_snr_db,
    )


def cohesion(embeddings: np.ndarray) -> float:
    """Mean pairwise cosine similarity among L2-normalized rows.

    Low cohesion for a single diarized speaker means the segments likely
    contain more than one voice (diarization error). Returns 1.0 for fewer
    than two rows.
    """
    n = embeddings.shape[0]
    if n < 2:
        return 1.0
    total = embeddings.sum(axis=0)
    # sum over all ordered pairs of dot products = ||sum||^2 - n (unit rows)
    return float((np.dot(total, total) - n) / (n * (n - 1)))
