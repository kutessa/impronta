"""Search-space definition with invariant-safe reparametrization.

``ImprontaConfig.__post_init__`` enforces ``merge_threshold >=
similarity_threshold`` and ordered/valid SNR tiers; sampling raw values
would reject points. We sample deltas instead, so every point in the box is
a valid config:

- ``merge_threshold  = similarity_threshold + merge_delta``
- SNR tiers built from ``t_low`` and positive deltas: low = t_low,
  medium = t_low + d_mid, high = t_low + d_mid + d_high
- ``snr_floor_db     = snr_floor_frac * t_low``  (floor stays below tiers)

Segmentation knobs (min/max_segment_sec, gap_tolerance_sec) are FROZEN —
they define embedding-cache identity (see impronta.pipeline docstring).
"""

from __future__ import annotations

from dataclasses import dataclass

from impronta import ImprontaConfig


@dataclass(frozen=True, slots=True)
class TrialParams:
    """Search parametrization. Defaults MUST mirror ImprontaConfig defaults
    (guarded by a test) so `evaluate.py` with no overrides measures the
    shipped configuration."""

    similarity_threshold: float = 0.4
    merge_delta: float = 0.2  # merge_threshold 0.6
    gray_zone_margin: float = 0.08
    reinforce_margin: float = 0.12
    search_k: int = 10
    t_low: float = 12.0
    d_mid: float = 3.0
    d_high: float = 5.0  # tiers 20/15/12
    snr_floor_frac: float = -2.0 / 12.0  # snr_floor_db -2
    min_kept_speech_sec: float = 7.0
    min_segment_confidence: float = 0.66
    cohesion_threshold: float = 0.3
    min_proposal_segments: int = 3
    min_proposal_tier: str = "medium"
    max_embeddings_per_enroll: int = 30
    max_embeddings_per_speaker: int = 75

    def to_config(self) -> ImprontaConfig:
        return ImprontaConfig(
            similarity_threshold=self.similarity_threshold,
            merge_threshold=self.similarity_threshold + self.merge_delta,
            gray_zone_margin=self.gray_zone_margin,
            reinforce_margin=self.reinforce_margin,
            search_k=self.search_k,
            snr_tiers=(
                (self.t_low + self.d_mid + self.d_high, "high"),
                (self.t_low + self.d_mid, "medium"),
                (self.t_low, "low"),
            ),
            snr_floor_db=self.snr_floor_frac * self.t_low,
            min_kept_speech_sec=self.min_kept_speech_sec,
            min_segment_confidence=self.min_segment_confidence,
            cohesion_threshold=self.cohesion_threshold,
            min_proposal_segments=self.min_proposal_segments,
            min_proposal_tier=self.min_proposal_tier,
            max_embeddings_per_enroll=self.max_embeddings_per_enroll,
            max_embeddings_per_speaker=self.max_embeddings_per_speaker,
        )

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


#: (low, high) bounds / choices for Optuna. Kept in one place so search and
#: report agree on the space.
FLOAT_BOUNDS: dict[str, tuple[float, float]] = {
    "similarity_threshold": (0.30, 0.65),
    "merge_delta": (0.0, 0.25),
    "gray_zone_margin": (0.0, 0.20),
    "reinforce_margin": (0.0, 0.30),
    "t_low": (5.0, 15.0),
    "d_mid": (2.0, 8.0),
    "d_high": (2.0, 8.0),
    "snr_floor_frac": (0.0, 0.8),
    "min_kept_speech_sec": (0.0, 30.0),
    "min_segment_confidence": (0.0, 0.8),
    "cohesion_threshold": (0.0, 0.4),
}
INT_BOUNDS: dict[str, tuple[int, int]] = {
    "search_k": (3, 16),
    "min_proposal_segments": (1, 5),
}
CAT_CHOICES: dict[str, list] = {
    "min_proposal_tier": ["high", "medium", "low"],
    "max_embeddings_per_enroll": [5, 10, 20, 40],
    "max_embeddings_per_speaker": [50, 100, 200],
}


def sample_params(trial) -> TrialParams:  # optuna.Trial
    kwargs: dict = {}
    for name, (lo, hi) in FLOAT_BOUNDS.items():
        kwargs[name] = trial.suggest_float(name, lo, hi)
    for name, (lo, hi) in INT_BOUNDS.items():
        kwargs[name] = trial.suggest_int(name, lo, hi)
    for name, choices in CAT_CHOICES.items():
        kwargs[name] = trial.suggest_categorical(name, choices)
    return TrialParams(**kwargs)


def grid_stage_a(
    similarity_step: float = 0.025, merge_step: float = 0.05
) -> list[TrialParams]:
    """The coarse similarity x merge_delta landscape; all else default."""
    out = []
    sim_lo, sim_hi = FLOAT_BOUNDS["similarity_threshold"]
    md_lo, md_hi = 0.0, 0.20
    sim = sim_lo
    while sim <= sim_hi + 1e-9:
        md = md_lo
        while md <= md_hi + 1e-9:
            out.append(
                TrialParams(similarity_threshold=round(sim, 3), merge_delta=round(md, 3))
            )
            md += merge_step
        sim += similarity_step
    return out
