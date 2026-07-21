"""Tunable configuration for the impronta pipeline.

:class:`ImprontaConfig` is the single tuning surface of the package. The
docstrings on each field explain what the knob does and when to change it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

#: Ordering used when comparing quality tiers (higher = better).
TIER_ORDER = {"high": 3, "medium": 2, "low": 1}

#: Language codes that are ONE language for voice-matching purposes.
#: The first entry of each group is the canonical code entries are stored
#: under. Default: BCMS (Bosnian/Croatian/Serbian/Montenegrin) — phonetically
#: a single language, but STT language detection flips between the codes from
#: one recording to the next; without grouping, the hard language filter
#: would silently split one speaker db into three. "hbs" is the ISO 639-3
#: macrolanguage code for Serbo-Croatian.
DEFAULT_LANGUAGE_GROUPS: tuple[tuple[str, ...], ...] = (
    ("hbs", "bos", "hrv", "srp", "cnr", "sh", "bs", "hr", "sr"),
)


def default_cache_dir() -> str:
    """Model cache directory: $IMPRONTA_CACHE_DIR, else ~/.cache/impronta."""
    env = os.environ.get("IMPRONTA_CACHE_DIR")
    if env:
        return env
    return str(Path.home() / ".cache" / "impronta")


@dataclass
class ImprontaConfig:
    """All pipeline tunables with production-tested defaults.

    Segmentation
    ------------
    min_segment_sec: segments shorter than this are dropped — ECAPA-TDNN
        embeddings on sub-second audio are unreliable.
    max_segment_sec: an open segment is closed once extending it would exceed
        this duration (split happens at a word boundary).
    gap_tolerance_sec: consecutive same-speaker words closer than this are
        merged into one segment; larger gaps start a new segment.
    direct_window_sec: window size used by ``add_speaker_from_audio`` when
        slicing a raw clip (no transcript) into embedding windows.

    Filtering
    ---------
    snr_tiers: progressive relaxation ladder for the blind (WADA) SNR filter.
        The first threshold with at least one passing segment wins and its
        tier label is reported on results.
    min_kept_speech_sec: if the tier pass keeps less than this much speech,
        the pipeline supplements with the best remaining segments (composite
        quality order, above ``snr_floor_db``) and reports tier "low".
        Real telephony audio reads 5-15 dB on WADA — without this rescue a
        20-minute call can reduce to seconds of evidence, and the resulting
        profiles are channel fingerprints instead of voiceprints (measured
        on production calls). Set to 0 to disable the rescue.
    snr_floor_db: absolute floor — segments below it are never used, even
        by the rescue (this is what keeps pure noise unusable).
    min_segment_confidence: segments whose ElevenLabs transcription
        confidence (``exp(mean logprob)``) is below this are dropped before
        SNR filtering. Hard gate — never relaxed. Segments with no logprob
        data pass (confidence defaults to 1.0).

    Matching
    --------
    similarity_threshold: minimum cosine similarity for a stored speaker to
        receive credit from a segment. Below it the segment votes "unknown".
        Calibrated on production call recordings: genuine same-speaker
        matches score 0.5-0.8 (with the pipeline's speech-duration rescue in
        place), while wrong-person scores cluster at 0.3-0.45 — 0.5 rejects
        those. Lower it (0.30-0.35) only if you observe true matches being
        reported unknown AND can tolerate more false names.
    search_k: neighbours fetched per segment during the vote. Each distinct
        speaker among them is credited with their best score, so a 0.37 vs
        0.34 near-tie between two stored speakers surfaces in `candidates`
        instead of being winner-take-all.
    enroll_outlier_margin: enrollment segments whose centroid-similarity
        falls this far below the batch MEDIAN are dropped — diarization
        sometimes attributes another person's words to the labeled speaker,
        and one wrong segment poisons the profile. Relative to the median so
        it adapts to audio quality (clean batches agree at ~0.9, telephony
        batches at ~0.4-0.6). Requires >= 4 segments to activate.
    merge_threshold: stricter bar used for DESTRUCTIVE decisions — merging an
        unknown into a named speaker at enroll time, and deduplicating
        proposals against existing unknowns at commit time. Keep this well
        above ``similarity_threshold``: borderline scores must never corrupt
        a profile.
    gray_zone_margin: if the best named match scores within this margin
        BELOW ``similarity_threshold``, the speaker is reported unknown but
        NOT proposed for auto-enrollment (prevents one person fragmenting
        into a named speaker plus a near-duplicate unknown). Near-miss
        candidates are reported so the app can ask the user.

    Unknown proposal gating
    -----------------------
    min_proposal_segments: strangers with fewer usable segments than this are
        never proposed (one-off voices: waiters, TVs).
    min_proposal_tier: minimum SNR quality tier ("high"/"medium"/"low")
        required to propose a stranger.
    cohesion_threshold: minimum mean pairwise cosine among a stranger's own
        segment embeddings. Low cohesion means the diarizer likely merged two
        people into one speaker_id — proposing would create a poisoned
        identity. Calibrated on real phone recordings: single-speaker
        segments score ~0.2-0.55, so the default only rejects egregious
        mixes; don't raise it above ~0.3 without measuring on your audio.

    Growth control
    --------------
    max_embeddings_per_enroll: at most this many segments are embedded and
        stored per enroll/identify call (the highest-quality ones are kept).
    max_embeddings_per_speaker: hard cap per speaker in the store; exceeding
        entries are evicted lowest-composite-quality-first.
    merge_unknowns_on_enroll: when enrolling a NAMED speaker, absorb existing
        unknowns that match above ``merge_threshold``.

    Runtime
    -------
    device: torch device for the embedder ("cpu", "cuda", "mps").
    model_source: HuggingFace/speechbrain source for the ECAPA model.
    model_cache_dir: where model weights are cached. Defaults to
        $IMPRONTA_CACHE_DIR or ~/.cache/impronta.
    sample_rate: internal processing sample rate (ECAPA expects 16 kHz).
    """

    # segmentation
    min_segment_sec: float = 1.0
    max_segment_sec: float = 10.0
    gap_tolerance_sec: float = 0.5
    direct_window_sec: float = 5.0
    # filtering
    snr_tiers: tuple[tuple[float, str], ...] = ((20.0, "high"), (15.0, "medium"), (10.0, "low"))
    min_kept_speech_sec: float = 15.0
    snr_floor_db: float = 3.0
    min_segment_confidence: float = 0.5
    # matching
    similarity_threshold: float = 0.5
    merge_threshold: float = 0.6
    gray_zone_margin: float = 0.1
    search_k: int = 8
    enroll_outlier_margin: float = 0.3
    # unknown proposal gating
    min_proposal_segments: int = 2
    min_proposal_tier: str = "medium"
    cohesion_threshold: float = 0.2
    # growth control
    max_embeddings_per_enroll: int = 20
    max_embeddings_per_speaker: int = 100
    merge_unknowns_on_enroll: bool = True
    # language equivalence (see DEFAULT_LANGUAGE_GROUPS)
    language_groups: tuple[tuple[str, ...], ...] = DEFAULT_LANGUAGE_GROUPS
    # runtime
    device: str = "cpu"
    model_source: str = "speechbrain/spkrec-ecapa-voxceleb"
    model_cache_dir: str = field(default_factory=default_cache_dir)
    sample_rate: int = 16_000

    def __post_init__(self) -> None:
        if self.min_proposal_tier not in TIER_ORDER:
            raise ValueError(
                f"min_proposal_tier must be one of {sorted(TIER_ORDER)}, "
                f"got {self.min_proposal_tier!r}"
            )
        if self.merge_threshold < self.similarity_threshold:
            raise ValueError(
                "merge_threshold must be >= similarity_threshold "
                "(merging is destructive and needs a stricter bar)"
            )
        for _threshold, tier in self.snr_tiers:
            if tier not in TIER_ORDER:
                raise ValueError(f"unknown SNR tier label {tier!r}")

    def tier_at_least(self, tier: str, minimum: str) -> bool:
        """True if ``tier`` is at least as good as ``minimum``."""
        return TIER_ORDER[tier] >= TIER_ORDER[minimum]

    def canonical_language(self, code: str | None) -> str:
        """Normalize a language code through ``language_groups``.

        Applied at both enroll and query time, so any member of a group
        matches any other (e.g. an enrollment tagged ``hrv`` matches a
        query tagged ``srp`` — both store and search as ``hbs``).
        """
        c = (code or "und").lower()
        for group in self.language_groups:
            if c in group:
                return group[0]
        return c
