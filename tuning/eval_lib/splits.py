"""Tune/test splitting and the cluster generalization holdout."""

from __future__ import annotations

import math

from .dataset import Person, RecordingRow, account_clusters


def order_recordings(rows: list[RecordingRow]) -> list[RecordingRow]:
    """Chronological order within an account (timestamp, then rid)."""
    return sorted(rows, key=lambda r: (r.timestamp or "9999", r.rid))


def chrono_split(rows: list[RecordingRow], tune_frac: float = 0.7) -> dict[str, str]:
    """(uid, rid)-keyed bucket assignment: earliest ~70% tune, rest test.

    Returns {rid: "tune"|"test"} for one account's usable recordings.
    """
    ordered = order_recordings(rows)
    n_tune = math.ceil(len(ordered) * tune_frac)
    return {
        r.rid: ("tune" if i < n_tune else "test") for i, r in enumerate(ordered)
    }


def holdout_uids(
    persons: dict[str, Person], holdout_cluster: str | None
) -> set[str]:
    """Accounts excluded from optimization when a cluster is held out."""
    if not holdout_cluster:
        return set()
    clusters = account_clusters(persons)
    return {uid for uid, c in clusters.items() if c == holdout_cluster}
