"""Duration x similarity weighted voting."""

import numpy as np
import pytest
from conftest import basis, blend

from impronta import ImprontaConfig, InMemoryStore
from impronta.models import UNKNOWN_BUCKET, Segment
from impronta.testing import make_entry
from impronta.vote import run_vote

CFG = ImprontaConfig()


def seg(start: float, end: float) -> Segment:
    return Segment(speaker_id="q", start=start, end=end, confidence=0.9)


def seeded_store() -> InMemoryStore:
    store = InMemoryStore()
    store.add(
        "ns",
        [
            make_entry("a1", "alice", basis(0), display_name="Alice"),
            make_entry("b1", "bob", basis(1), display_name="Bob"),
        ],
    )
    return store


def test_one_long_strong_segment_beats_several_short_weak_ones():
    store = seeded_store()
    strong_alice = blend(basis(0), basis(2), 0.9)
    weak_bob = blend(basis(1), basis(3), 0.55)  # above threshold, but weak
    embeddings = np.stack([strong_alice, weak_bob, weak_bob, weak_bob])
    segments = [seg(0, 8), seg(9, 10.2), seg(11, 12.2), seg(13, 14.2)]
    # plain majority would elect bob 3-1; weighted: alice 7.2 vs bob ~1.98
    outcome = run_vote(embeddings, segments, store, ["ns"], "en", CFG)
    assert outcome.winner_key == "alice"
    assert outcome.winner_display_name == "Alice"


def test_below_threshold_segments_feed_unknown_bucket():
    store = seeded_store()
    barely = blend(basis(0), basis(2), 0.35)  # below the 0.5 threshold
    outcome = run_vote(
        np.stack([barely, barely]), [seg(0, 2), seg(3, 5)], store, ["ns"], "en", CFG
    )
    assert outcome.winner_key == UNKNOWN_BUCKET
    assert outcome.mean_similarity is None


def test_candidate_shares_sum_to_one_and_rank_desc():
    store = seeded_store()
    embeddings = np.stack(
        [blend(basis(0), basis(2), 0.8), blend(basis(1), basis(3), 0.6), basis(4)]
    )
    segments = [seg(0, 5), seg(6, 8), seg(9, 11)]
    outcome = run_vote(embeddings, segments, store, ["ns"], "en", CFG)
    shares = [c.score_share for c in outcome.candidates]
    assert sum(shares) == pytest.approx(1.0)
    assert shares == sorted(shares, reverse=True)
    assert outcome.candidates[0].speaker_key == "alice"
    assert {c.speaker_key for c in outcome.candidates} == {"alice", "bob", UNKNOWN_BUCKET}


def test_empty_store_yields_unknown_with_no_best_score():
    outcome = run_vote(
        np.stack([basis(0)]), [seg(0, 3)], InMemoryStore(), ["ns"], "en", CFG
    )
    assert outcome.winner_key == UNKNOWN_BUCKET
    assert outcome.best_named_score is None
    assert outcome.candidates[0].speaker_key == UNKNOWN_BUCKET


def test_language_filter_none_matches_across_languages():
    store = InMemoryStore()
    store.add("ns", [make_entry("a1", "alice", basis(0), language="bs")])
    embeddings = np.stack([basis(0)])
    assert (
        run_vote(embeddings, [seg(0, 3)], store, ["ns"], "en", CFG).winner_key
        == UNKNOWN_BUCKET
    )
    assert (
        run_vote(embeddings, [seg(0, 3)], store, ["ns"], None, CFG).winner_key == "alice"
    )


def test_near_misses_collected_in_gray_zone_band():
    store = seeded_store()
    grayish = blend(basis(0), basis(2), 0.45)  # in [0.40, 0.50)
    outcome = run_vote(np.stack([grayish]), [seg(0, 3)], store, ["ns"], "en", CFG)
    assert outcome.winner_key == UNKNOWN_BUCKET
    assert outcome.near_misses and outcome.near_misses[0][0] == "alice"
    assert outcome.near_misses[0][1] == pytest.approx(0.45, abs=0.01)


def test_segment_credits_multiple_close_speakers():
    """A near-tie must surface as a near-tie, not a winner-take-all flip."""
    store = seeded_store()
    # one query vector close to BOTH alice (0.65) and bob (0.55)
    both = (0.65 * basis(0) + 0.55 * basis(1) + np.sqrt(1 - 0.65**2 - 0.55**2) * basis(2))
    outcome = run_vote(
        np.stack([both.astype(np.float32)]), [seg(0, 5)], store, ["ns"], "en", CFG
    )
    assert outcome.winner_key == "alice"
    by_key = {c.speaker_key: c for c in outcome.candidates}
    assert "bob" in by_key, "runner-up must be visible in candidates"
    assert by_key["bob"].mean_similarity == pytest.approx(0.55, abs=0.01)
    assert by_key["alice"].score_share > by_key["bob"].score_share


def test_deterministic_tie_break_prefers_higher_mean_similarity():
    store = InMemoryStore()
    store.add(
        "ns",
        [
            make_entry("a1", "alice", basis(0), display_name="Alice"),
            make_entry("b1", "bob", basis(1), display_name="Bob"),
        ],
    )
    # same total weighted score: alice 0.8 x 5s = 4.0, bob 0.5 x 8s = 4.0
    embeddings = np.stack(
        [blend(basis(0), basis(2), 0.8), blend(basis(1), basis(2), 0.5)]
    )
    segments = [seg(0, 5), seg(6, 14)]
    outcome = run_vote(embeddings, segments, store, ["ns"], "en", CFG)
    assert outcome.winner_key == "alice"  # higher mean similarity wins the tie
