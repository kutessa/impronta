"""Scoring identification outcomes against ground-truth annotations.

Outcome categories per scored (recording, qid) row — ``p`` is the annotated
person, ``enrolled(p)`` means p was enrolled strictly before this recording
(and not from it):

- ``correct_name``          match names p
- ``wrong_name``            match names someone else while p is enrolled
- ``false_accept_stranger`` match names anyone while p is NOT enrolled
- ``missed_known``          no name while p is enrolled
- ``correct_unknown``       no name while p is NOT enrolled
- ``no_usable``             speaker unidentifiable under this config

The tuning objective: **wrong-name rate <= 2%** (numerator = wrong_name +
false_accept_stranger; denominator excludes no_usable — the system emitted
nothing there), maximize known recall subject to it. Recall counts
no_usable-when-enrolled rows as misses so a config cannot win by filtering
everything away.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from impronta.models import SpeakerMatch

CATEGORIES = (
    "correct_name",
    "wrong_name",
    "false_accept_stranger",
    "missed_known",
    "correct_unknown",
    "no_usable",
)


@dataclass(frozen=True, slots=True)
class ScoredRow:
    uid: str
    rid: str
    qid: str
    person: str
    outcome: str
    enrolled: bool  # was the annotated person enrolled before this recording
    matched_key: str | None
    mean_similarity: float | None
    bucket: str  # "tune" | "test" | "all"
    multi_speaker: bool


def categorize(match: SpeakerMatch, person: str, enrolled: bool) -> str:
    """Map one SpeakerMatch + ground truth to an outcome category."""
    if not match.identifiable:
        return "no_usable"
    named = match.display_name is not None
    if named:
        if match.speaker_key == person:
            return "correct_name"
        return "wrong_name" if enrolled else "false_accept_stranger"
    # unknown outcome (incl. recognized-but-unlabeled unknown keys)
    return "missed_known" if enrolled else "correct_unknown"


@dataclass(slots=True)
class Metrics:
    counts: Counter = field(default_factory=Counter)
    n: int = 0

    def add(self, outcome: str, *, enrolled: bool) -> None:
        self.counts[outcome] += 1
        if outcome == "no_usable" and enrolled:
            self.counts["no_usable_enrolled"] += 1
        self.n += 1

    # -- headline numbers ---------------------------------------------------

    @property
    def wrong_names(self) -> int:
        return self.counts["wrong_name"] + self.counts["false_accept_stranger"]

    @property
    def emitted(self) -> int:
        """Rows where the system produced any judgment (excludes no_usable)."""
        return self.n - self.counts["no_usable"]

    @property
    def wrong_rate(self) -> float:
        return self.wrong_names / self.emitted if self.emitted else 0.0

    @property
    def known_total(self) -> int:
        return (
            self.counts["correct_name"]
            + self.counts["wrong_name"]
            + self.counts["missed_known"]
            + self.counts["no_usable_enrolled"]
        )

    @property
    def recall_known(self) -> float:
        return (
            self.counts["correct_name"] / self.known_total if self.known_total else 0.0
        )

    @property
    def unknown_precision(self) -> float:
        """Of true strangers, how many were correctly left unknown."""
        strangers = (
            self.counts["correct_unknown"] + self.counts["false_accept_stranger"]
        )
        return self.counts["correct_unknown"] / strangers if strangers else 1.0

    def wrong_rate_interval(self, z: float = 1.96) -> tuple[float, float]:
        """Wilson score interval for the wrong-name rate."""
        n = self.emitted
        if n == 0:
            return (0.0, 1.0)
        p = self.wrong_rate
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        half = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
        return (max(0.0, center - half), min(1.0, center + half))

    def to_dict(self) -> dict:
        lo, hi = self.wrong_rate_interval()
        return {
            "n": self.n,
            "emitted": self.emitted,
            "wrong_names": self.wrong_names,
            "wrong_rate": round(self.wrong_rate, 5),
            "wrong_rate_ci95": [round(lo, 5), round(hi, 5)],
            "recall_known": round(self.recall_known, 5),
            "known_total": self.known_total,
            "unknown_precision": round(self.unknown_precision, 5),
            **{c: self.counts[c] for c in CATEGORIES},
        }


def aggregate(rows: list[ScoredRow]) -> Metrics:
    m = Metrics()
    for row in rows:
        m.add(row.outcome, enrolled=row.enrolled)
    return m
