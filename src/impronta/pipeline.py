"""The shared preparation pipeline: segments -> filters -> embeddings.

Both enrollment and identification run the same steps per speaker:
confidence gate -> audio slicing -> blind SNR filter (progressive
relaxation) -> quality-ranked cap -> ECAPA embedding. Keeping one code path
guarantees enrolled and queried embeddings are produced identically.
"""

from __future__ import annotations

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


def prepare_segments(
    segments: list[Segment],
    audio_mono: np.ndarray,
    embedder: Embedder,
    cfg: ImprontaConfig,
) -> PreparedSpeaker:
    """Run one speaker's segments through the full filter + embed pipeline."""
    total = len(segments)

    # 1. ElevenLabs-confidence gate (hard, never relaxed)
    confident = [s for s in segments if s.confidence >= cfg.min_segment_confidence]

    # 2. slice audio; drop segments that fall outside the actual signal
    sliced: list[tuple[Segment, np.ndarray]] = []
    min_samples = int(cfg.min_segment_sec * cfg.sample_rate)
    for seg in confident:
        clip = slice_seconds(audio_mono, cfg.sample_rate, seg.start, seg.end)
        if clip.shape[0] >= min_samples:
            sliced.append((seg, clip))

    if not sliced:
        return PreparedSpeaker(total, [], None, None, None)

    # 3. blind SNR + progressive relaxation
    with_snr = [
        (replace(seg, snr_db=wada_snr(clip)), clip) for seg, clip in sliced
    ]
    snrs = [seg.snr_db for seg, _ in with_snr]
    kept_idx, tier = filter_by_snr(snrs, cfg)  # type: ignore[arg-type]
    best_snr = max(snrs) if snrs else None  # type: ignore[type-var]
    kept = [with_snr[i] for i in kept_idx]

    # 3b. rescue supplement: real telephony audio reads 5-15 dB on WADA, so
    # the tiers alone can starve a 20-minute call down to seconds of
    # evidence and the resulting profile is a channel fingerprint, not a
    # voiceprint. Top up with the best remaining segments above the
    # absolute floor until kept speech reaches min_kept_speech_sec; any
    # rescued audio demotes the tier to "low".
    kept_speech = sum(seg.duration for seg, _ in kept)
    if kept_speech < cfg.min_kept_speech_sec:
        kept_ids = set(kept_idx)
        rescuable = [
            pair
            for i, pair in enumerate(with_snr)
            if i not in kept_ids
            and pair[0].snr_db is not None
            and pair[0].snr_db >= cfg.snr_floor_db
        ]
        rescuable.sort(
            key=lambda pair: composite_quality(
                pair[0].snr_db, pair[0].confidence, pair[0].duration
            ),
            reverse=True,
        )
        for pair in rescuable:
            if kept_speech >= cfg.min_kept_speech_sec:
                break
            kept.append(pair)
            kept_speech += pair[0].duration
            tier = "low"

    if not kept:
        return PreparedSpeaker(total, [], None, None, best_snr)

    # 4. quality-ranked cap: keep the best segments, not the first ones
    kept.sort(
        key=lambda pair: composite_quality(
            pair[0].snr_db, pair[0].confidence, pair[0].duration
        ),
        reverse=True,
    )
    kept = kept[: cfg.max_embeddings_per_enroll]
    kept.sort(key=lambda pair: pair[0].start)  # restore time order

    # 5. embed
    clips = [clip for _, clip in kept]
    embeddings = l2_normalize(embedder.embed_batch(clips, cfg.sample_rate))
    return PreparedSpeaker(
        segments_total=total,
        segments=[seg for seg, _ in kept],
        embeddings=embeddings,
        quality_tier=tier,
        best_snr_db=best_snr,
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
