"""Parity: the tuning replay path must equal the production pipeline.

The tuning harness precomputes embeddings for ALL structural segments and
re-runs selection via ``assemble_prepared``. If that ever diverges from
``prepare_segments``, every tuning result is a lie — this test is the lock.
"""

import numpy as np
import pytest
from conftest import FakeEmbedder, basis, compose_timeline

from impronta import ImprontaConfig
from impronta.embedder import l2_normalize
from impronta.models import Segment
from impronta.pipeline import (
    assemble_prepared,
    measure_segments,
    prepare_segments,
    select_segments,
)

EMB = FakeEmbedder({440.0: basis(0), 880.0: basis(1)})


def seg(start, end, confidence=0.9):
    return Segment(speaker_id="s0", start=start, end=end, confidence=confidence)


def cache_like(segments, audio, cfg):
    """What precompute_cache.py stores: measured segments + ALL embeddings."""
    measured = measure_segments(segments, audio, cfg)
    if measured.clips:
        embeddings_all = l2_normalize(EMB.embed_batch(measured.clips, cfg.sample_rate))
    else:
        embeddings_all = np.zeros((0, EMB.dim), dtype=np.float32)
    return measured.segments, embeddings_all, measured.segments_total


SCENARIOS = {
    "clean": dict(
        parts=[(0.0, 6.0, 440.0), (7.0, 13.0, 440.0), (14.0, 20.0, 440.0)],
        segments=[seg(0, 6), seg(7, 13), seg(14, 20)],
        cfg=ImprontaConfig(),
    ),
    "noisy_rescue": dict(
        parts=[(0.0, 20.0, 440.0)],
        noise_db=6.0,
        segments=[seg(0, 6), seg(7, 13), seg(14, 20)],
        cfg=ImprontaConfig(),
    ),
    "low_confidence_mix": dict(
        parts=[(0.0, 20.0, 440.0)],
        segments=[seg(0, 6, confidence=0.2), seg(7, 13), seg(14, 20, confidence=0.4)],
        cfg=ImprontaConfig(),
    ),
    "all_gated": dict(
        parts=[(0.0, 13.5, 440.0)],
        segments=[seg(0, 6, confidence=0.1), seg(7, 13, confidence=0.2)],
        cfg=ImprontaConfig(),
    ),
    "cap_applied": dict(
        parts=[(0.0, 30.0, 440.0)],
        segments=[seg(i * 6, i * 6 + 5) for i in range(5)],
        cfg=ImprontaConfig(max_embeddings_per_enroll=2),
    ),
    "outside_audio": dict(
        parts=[(0.0, 6.5, 440.0)],
        segments=[seg(0, 6), seg(50, 60)],
        cfg=ImprontaConfig(),
    ),
}


@pytest.mark.parametrize("name", SCENARIOS)
def test_prepared_parity_prod_vs_cache(name):
    sc = SCENARIOS[name]
    duration = max(e for s, e, *_ in [(s.start, s.end) for s in sc["segments"]]) + 0.5
    audio = compose_timeline(
        max(duration, max(e for _, e, _ in sc["parts"]) + 0.5),
        sc["parts"],
        noise_db=sc.get("noise_db"),
    )
    cfg = sc["cfg"]

    prod = prepare_segments(sc["segments"], audio, EMB, cfg)
    cached_segments, embeddings_all, total = cache_like(sc["segments"], audio, cfg)
    replayed = assemble_prepared(cached_segments, embeddings_all, total, cfg)

    assert replayed.segments_total == prod.segments_total
    assert replayed.quality_tier == prod.quality_tier
    assert replayed.best_snr_db == prod.best_snr_db
    assert replayed.segments == prod.segments
    if prod.embeddings is None:
        assert replayed.embeddings is None
    else:
        np.testing.assert_array_equal(replayed.embeddings, prod.embeddings)


def test_selection_is_pure_metadata():
    """select_segments must work on segments with snr_db already set —
    exactly what the cache provides — without audio."""
    segments = [
        Segment(speaker_id="s0", start=0, end=6, confidence=0.9, snr_db=25.0),
        Segment(speaker_id="s0", start=7, end=13, confidence=0.9, snr_db=8.0),
        Segment(speaker_id="s0", start=14, end=20, confidence=0.2, snr_db=30.0),
    ]
    sel = select_segments(segments, ImprontaConfig())
    # 25dB passes "high"; 8dB rescued (tier -> low); 0.2-confidence gated out
    assert sel.kept_indices == [0, 1]
    assert sel.quality_tier == "low"
    assert sel.best_snr_db == 25.0


def test_selection_config_sensitivity():
    """The same cached measurements produce different selections per config —
    the property the whole tuning harness rests on."""
    segments = [
        Segment(speaker_id="s0", start=0, end=6, confidence=0.6, snr_db=18.0),
        Segment(speaker_id="s0", start=7, end=13, confidence=0.9, snr_db=12.0),
    ]
    strict = select_segments(
        segments, ImprontaConfig(min_segment_confidence=0.7, min_kept_speech_sec=0.0)
    )
    lax = select_segments(
        segments, ImprontaConfig(min_segment_confidence=0.5, min_kept_speech_sec=0.0)
    )
    rescued = select_segments(
        segments, ImprontaConfig(min_segment_confidence=0.5, min_kept_speech_sec=15.0)
    )
    assert strict.kept_indices == [1]  # 0.6-confidence segment gated out
    assert lax.kept_indices == [0]  # medium tier (18 dB) wins, no rescue
    assert rescued.kept_indices == [0, 1]  # rescue tops up with the 12 dB one
    assert rescued.quality_tier == "low"
