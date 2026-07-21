"""Duration x similarity weighted voting over a speaker's segments.

Each segment fetches its ``search_k`` nearest neighbours; every DISTINCT
speaker among them scoring at or above the similarity threshold is credited
``duration x best_score`` (a segment can credit several close speakers — a
0.37 vs 0.34 near-tie must surface as a near-tie, not a winner-take-all
flip). A segment where no speaker clears the threshold contributes
``duration`` to the unknown bucket. The highest total wins; one long strong
match outweighs several short weak ones.

The normalized totals are exposed as ranked :class:`Candidate` shares so the
application can late-fuse voice evidence with other signals.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .config import ImprontaConfig
from .models import UNKNOWN_BUCKET, Candidate, SearchFilter, Segment
from .store.base import VectorStore


@dataclass(frozen=True, slots=True)
class VoteOutcome:
    winner_key: str  # UNKNOWN_BUCKET when no known speaker wins
    winner_namespace: str | None
    winner_display_name: str | None
    candidates: tuple[Candidate, ...]
    mean_similarity: float | None  # over the winner's contributing segments
    best_named_score: float | None  # best score seen for ANY stored speaker
    near_misses: tuple[tuple[str, float], ...]


def run_vote(
    embeddings: np.ndarray,
    segments: Sequence[Segment],
    store: VectorStore,
    namespaces: Sequence[str],
    language: str | None,
    cfg: ImprontaConfig,
) -> VoteOutcome:
    """Vote one speaker's segment embeddings against the store.

    ``language=None`` disables the language hard filter.
    """
    totals: dict[str, float] = {UNKNOWN_BUCKET: 0.0}
    sims: dict[str, list[float]] = {}
    meta: dict[str, tuple[str | None, str | None]] = {UNKNOWN_BUCKET: (None, None)}
    best_scores: dict[str, float] = {}

    search_filter = SearchFilter(language=language) if language is not None else None
    for row, seg in zip(embeddings, segments, strict=True):
        hits = store.search(row, namespaces, k=cfg.search_k, filter=search_filter)
        # best score per distinct speaker among this segment's neighbours
        per_speaker: dict[str, float] = {}
        for hit in hits:
            key = hit.entry.speaker_key
            if key not in per_speaker or hit.score > per_speaker[key]:
                per_speaker[key] = hit.score
                if hit.score >= cfg.similarity_threshold:
                    meta[key] = (hit.entry.namespace, hit.entry.display_name)
        matched = False
        for key, score in per_speaker.items():
            best_scores[key] = max(best_scores.get(key, -1.0), score)
            if score >= cfg.similarity_threshold:
                totals[key] = totals.get(key, 0.0) + seg.duration * score
                sims.setdefault(key, []).append(score)
                matched = True
        if not matched:
            totals[UNKNOWN_BUCKET] += seg.duration

    def mean_sim(key: str) -> float | None:
        values = sims.get(key)
        return float(np.mean(values)) if values else None

    # deterministic winner: total, then mean similarity (unknown sorts last), then key
    def sort_key(key: str) -> tuple[float, float, str]:
        ms = mean_sim(key)
        return (totals[key], ms if ms is not None else -1.0, key)

    winner = max(totals, key=sort_key)
    if totals[winner] == 0.0:
        winner = UNKNOWN_BUCKET

    grand_total = sum(totals.values())
    candidates = tuple(
        sorted(
            (
                Candidate(
                    speaker_key=key,
                    display_name=meta.get(key, (None, None))[1],
                    score_share=(totals[key] / grand_total) if grand_total > 0 else 0.0,
                    mean_similarity=mean_sim(key),
                )
                for key in totals
                if totals[key] > 0 or key == winner
            ),
            key=lambda c: (-c.score_share, c.speaker_key),
        )
    )

    low = cfg.similarity_threshold - cfg.gray_zone_margin
    near_misses = tuple(
        sorted(
            (
                (key, score)
                for key, score in best_scores.items()
                if low <= score < cfg.similarity_threshold
            ),
            key=lambda pair: -pair[1],
        )
    )

    namespace, display_name = meta.get(winner, (None, None))
    return VoteOutcome(
        winner_key=winner,
        winner_namespace=namespace,
        winner_display_name=display_name,
        candidates=candidates,
        mean_similarity=mean_sim(winner),
        best_named_score=max(best_scores.values()) if best_scores else None,
        near_misses=near_misses,
    )
