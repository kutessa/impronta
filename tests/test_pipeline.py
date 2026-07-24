"""prepare_segments filter ordering and the cohesion heuristic."""

import numpy as np
import pytest
from conftest import FakeEmbedder, basis, blend, compose_timeline

from impronta import ImprontaConfig
from impronta.models import Segment
from impronta.pipeline import cohesion, prepare_segments

CFG = ImprontaConfig()
EMB = FakeEmbedder({440.0: basis(0), 880.0: basis(1)})


def seg(start, end, confidence=0.9):
    return Segment(speaker_id="s0", start=start, end=end, confidence=confidence)


def test_confidence_gate_drops_low_and_keeps_null_default():
    audio = compose_timeline(12.2, [(0.0, 12.0, 440.0)])
    segments = [seg(0, 4, confidence=0.2), seg(5, 9, confidence=0.9), seg(10, 12, 1.0)]
    prepared = prepare_segments(segments, audio, EMB, CFG)
    assert prepared.segments_total == 3
    assert [s.start for s in prepared.segments] == [5, 10]


def test_all_low_confidence_is_unusable():
    audio = compose_timeline(5.2, [(0.0, 5.0, 440.0)])
    prepared = prepare_segments([seg(0, 5, confidence=0.1)], audio, EMB, CFG)
    assert prepared.embeddings is None
    assert prepared.quality_tier is None


def test_segments_outside_audio_are_dropped():
    audio = compose_timeline(4.2, [(0.0, 4.0, 440.0)])
    prepared = prepare_segments([seg(0, 3), seg(10, 14)], audio, EMB, CFG)
    assert len(prepared.segments) == 1


def test_snr_attached_and_tier_reported():
    audio = compose_timeline(6.2, [(0.0, 6.0, 440.0)])
    prepared = prepare_segments([seg(0, 6)], audio, EMB, CFG)
    assert prepared.quality_tier == "high"
    assert prepared.segments[0].snr_db is not None and prepared.segments[0].snr_db > 20


def test_unusable_when_no_speech_at_all():
    """Pure noise sits below snr_floor_db — not even the rescue touches it."""
    rng = np.random.default_rng(0)
    audio = (rng.normal(0, 0.1, int(6.2 * 16_000))).astype(np.float32)
    prepared = prepare_segments([seg(0, 6)], audio, EMB, CFG)
    assert prepared.embeddings is None
    assert prepared.best_snr_db is not None  # diagnostics preserved


# -- rescue supplement (telephony-audio starvation fix) ----------------------


def snr_sequence(monkeypatch, values):
    it = iter(values)
    monkeypatch.setattr("impronta.pipeline.wada_snr", lambda clip: next(it))


def test_rescue_supplements_when_tiers_starve(monkeypatch):
    audio = compose_timeline(21.2, [(0.0, 21.0, 440.0)])
    segments = [seg(0, 6), seg(7, 13), seg(14, 20)]
    snr_sequence(monkeypatch, [8.0, 7.0, 5.0])  # all below the 10 dB rung
    prepared = prepare_segments(segments, audio, EMB, CFG)
    assert prepared.embeddings is not None
    assert prepared.quality_tier == "low"
    assert len(prepared.segments) == 2  # 12s of speech clears the 7s target


def test_rescue_respects_absolute_floor(monkeypatch):
    audio = compose_timeline(14.2, [(0.0, 14.0, 440.0)])
    segments = [seg(0, 6), seg(7, 13)]
    snr_sequence(monkeypatch, [8.0, -5.0])  # -5 dB is below snr_floor_db=-2
    prepared = prepare_segments(segments, audio, EMB, CFG)
    assert prepared.embeddings is not None
    assert len(prepared.segments) == 1
    assert prepared.segments[0].start == 0


def test_no_rescue_when_tiers_keep_enough(monkeypatch):
    audio = compose_timeline(27.2, [(0.0, 27.0, 440.0)])
    segments = [seg(0, 5), seg(6, 11), seg(12, 17), seg(18, 23), seg(24, 27)]
    snr_sequence(monkeypatch, [25.0, 26.0, 24.0, 27.0, 5.0])  # 20s of clean speech
    prepared = prepare_segments(segments, audio, EMB, CFG)
    assert prepared.quality_tier == "high"  # the 5 dB segment was NOT pulled in
    assert len(prepared.segments) == 4


def test_rescue_stops_at_speech_target(monkeypatch):
    audio = compose_timeline(40.2, [(0.0, 40.0, 440.0)])
    segments = [seg(i * 8, i * 8 + 7) for i in range(5)]  # 5 x 7s
    snr_sequence(monkeypatch, [9.0, 8.0, 7.0, 6.0, 5.0])
    prepared = prepare_segments(segments, audio, EMB, CFG)
    # the first 7s segment alone meets the 7s target; the rest stay out
    assert len(prepared.segments) == 1
    assert prepared.quality_tier == "low"


def test_cap_keeps_best_quality_segments():
    cfg = ImprontaConfig(max_embeddings_per_enroll=2)
    audio = compose_timeline(16.2, [(0.0, 16.0, 440.0)])
    segments = [seg(0, 1.2), seg(2, 9), seg(10, 16)]  # 1.2s, 7s, 6s
    prepared = prepare_segments(segments, audio, EMB, cfg)
    assert len(prepared.segments) == 2
    assert {round(s.duration) for s in prepared.segments} == {7, 6}
    # time order restored after quality ranking
    assert prepared.segments[0].start < prepared.segments[1].start


def test_embeddings_are_normalized_rows():
    audio = compose_timeline(6.2, [(0.0, 6.0, 440.0)])
    prepared = prepare_segments([seg(0, 6)], audio, EMB, CFG)
    norms = np.linalg.norm(prepared.embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


# -- cohesion ----------------------------------------------------------------


def test_cohesion_identical_rows_is_one():
    rows = np.stack([basis(0), basis(0), basis(0)])
    assert cohesion(rows) == pytest.approx(1.0)


def test_cohesion_orthogonal_rows_is_zero():
    rows = np.stack([basis(0), basis(1)])
    assert cohesion(rows) == pytest.approx(0.0, abs=1e-6)


def test_cohesion_single_row_is_one():
    assert cohesion(np.stack([basis(0)])) == 1.0


def test_cohesion_matches_pairwise_mean():
    a, b, c = basis(0), blend(basis(0), basis(1), 0.8), basis(2)
    rows = np.stack([a, b, c])
    expected = np.mean(
        [float(np.dot(a, b)), float(np.dot(a, c)), float(np.dot(b, c))]
    )
    assert cohesion(rows) == pytest.approx(expected, abs=1e-6)
