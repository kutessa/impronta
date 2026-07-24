"""Eval-harness unit tests: scorer categories, static protocol, chrono sim.

Runs in the hermetic default tier: synthetic caches with basis-vector
embeddings, no torch, no real audio. Every outcome category and the tricky
episode edges are constructed deliberately so the correct answer is known.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from impronta import ImprontaConfig, SpeakerMatch
from tuning.eval_lib.cache import RecordingCache, load_cache, save_cache, seg_knobs
from tuning.eval_lib.dataset import RecordingAnnotation, RecordingRow, SpeakerAnnotation
from tuning.eval_lib.episodes import AccountData, run_chrono, run_static
from tuning.eval_lib.scorer import Metrics, categorize
from tuning.eval_lib.splits import chrono_split

DIM = 8
CFG = ImprontaConfig()


def basis(axis: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[axis] = 1.0
    return v


def blend(a: np.ndarray, b: np.ndarray, sim: float) -> np.ndarray:
    return (sim * a + np.sqrt(1 - sim**2) * b).astype(np.float32)


def make_cache(uid: str, rid: str, speakers: dict[str, list[np.ndarray]],
               empty_speakers: tuple[str, ...] = (), language: str = "eng") -> RecordingCache:
    """speakers: qid -> list of unit embeddings; each becomes a 5s clean segment."""
    embs, starts, ends, snrs, confs, qids = [], [], [], [], [], []
    totals: dict[str, int] = {}
    t = 0.0
    for qid, vecs in speakers.items():
        totals[qid] = len(vecs)
        for v in vecs:
            embs.append(v)
            starts.append(t)
            ends.append(t + 5.0)
            snrs.append(25.0)
            confs.append(0.95)
            qids.append(qid)
            t += 6.0
    for qid in empty_speakers:
        totals[qid] = 2  # structural segments existed but none embeddable
    return RecordingCache(
        uid=uid,
        rid=rid,
        transcription_id=f"tx-{rid}",
        language_code=language,
        embeddings=(np.stack(embs) if embs else np.zeros((0, DIM), dtype=np.float32)),
        start=np.asarray(starts, dtype=np.float32),
        end=np.asarray(ends, dtype=np.float32),
        snr_db=np.asarray(snrs, dtype=np.float32),
        confidence=np.asarray(confs, dtype=np.float32),
        qid=np.asarray(qids, dtype=np.str_),
        segments_total=totals,
        meta={"seg_knobs": seg_knobs(CFG)},
    )


def make_row(uid: str, rid: str, day: int, n_speakers: int = 2) -> RecordingRow:
    return RecordingRow(
        uid=uid, rid=rid, audio_path="", raw_path="",
        timestamp=f"2026-01-{day:02d}T10:00:00Z", timestamp_source="container",
        duration_secs=60.0, language_code="eng", transcription_id=f"tx-{rid}",
        n_speakers=n_speakers, n_words=100, usable=True, exclude_reason=None,
    )


def ann(uid: str, rid: str, speakers: dict[str, tuple[str, str]]) -> RecordingAnnotation:
    return RecordingAnnotation(
        rid=rid, uid=uid,
        speakers={q: SpeakerAnnotation(person=p, quality=ql) for q, (p, ql) in speakers.items()},
    )


def test_trial_params_defaults_mirror_impronta_config():
    """evaluate.py with no overrides must measure the SHIPPED defaults."""
    from tuning.eval_lib.space import TrialParams

    derived = TrialParams().to_config()
    shipped = ImprontaConfig()
    for knob in (
        "similarity_threshold", "merge_threshold", "gray_zone_margin",
        "reinforce_margin", "search_k", "snr_tiers", "snr_floor_db",
        "min_kept_speech_sec", "min_segment_confidence", "cohesion_threshold",
        "min_proposal_segments", "min_proposal_tier",
        "max_embeddings_per_enroll", "max_embeddings_per_speaker",
    ):
        d, s = getattr(derived, knob), getattr(shipped, knob)
        if isinstance(d, float):
            assert abs(d - s) < 1e-9, f"{knob}: TrialParams {d} != config {s}"
        elif knob == "snr_tiers":
            assert [(round(t, 6), lbl) for t, lbl in d] == list(s), knob
        else:
            assert d == s, f"{knob}: TrialParams {d} != config {s}"


# ---------------------------------------------------------------------------
# categorize: all six outcomes
# ---------------------------------------------------------------------------


def match_of(key=None, name=None, identifiable=True):
    return SpeakerMatch(
        query_speaker_id="s0", speaker_key=key, display_name=name,
        namespace="u" if key else None, is_unknown=name is None,
        identifiable=identifiable,
    )


@pytest.mark.parametrize(
    "match,person,enrolled,expected",
    [
        (match_of("alice", "alice"), "alice", True, "correct_name"),
        (match_of("bob", "bob"), "alice", True, "wrong_name"),
        (match_of("bob", "bob"), "alice", False, "false_accept_stranger"),
        (match_of(None, None), "alice", True, "missed_known"),
        (match_of("unknown-x", None), "alice", True, "missed_known"),
        (match_of(None, None), "alice", False, "correct_unknown"),
        (match_of(identifiable=False), "alice", True, "no_usable"),
    ],
)
def test_categorize(match, person, enrolled, expected):
    assert categorize(match, person, enrolled) == expected


def test_metrics_math():
    m = Metrics()
    for _ in range(90):
        m.add("correct_name", enrolled=True)
    m.add("wrong_name", enrolled=True)
    m.add("false_accept_stranger", enrolled=False)
    m.add("missed_known", enrolled=True)
    m.add("correct_unknown", enrolled=False)
    m.add("no_usable", enrolled=True)
    assert m.n == 95
    assert m.emitted == 94
    assert m.wrong_names == 2
    assert m.wrong_rate == pytest.approx(2 / 94)
    # recall: 90 correct / (90 + 1 wrong + 1 missed + 1 no_usable_enrolled)
    assert m.recall_known == pytest.approx(90 / 93)
    lo, hi = m.wrong_rate_interval()
    assert lo < m.wrong_rate < hi


def test_wilson_interval_zero_errors():
    m = Metrics()
    for _ in range(50):
        m.add("correct_name", enrolled=True)
    lo, hi = m.wrong_rate_interval()
    assert lo == 0.0 and 0.0 < hi < 0.12  # zero observed errors is not proof of zero


# ---------------------------------------------------------------------------
# cache round-trip
# ---------------------------------------------------------------------------


def test_cache_roundtrip_and_knob_guard(tmp_path):
    cache = make_cache("u1", "r1", {"s0": [basis(0), basis(1)]}, empty_speakers=("s9",))
    path = tmp_path / "r1.npz"
    save_cache(cache, path)
    loaded = load_cache("u1", "r1", CFG, path=path)
    np.testing.assert_array_equal(loaded.embeddings, cache.embeddings)
    assert loaded.segments_total == {"s0": 2, "s9": 2}
    assert loaded.qids() == ["s0", "s9"]
    segs, embs = loaded.segments_for("s9")
    assert segs == [] and embs.shape == (0, DIM)

    bad_cfg = ImprontaConfig(min_segment_sec=2.0)  # segmentation knob changed
    with pytest.raises(ValueError, match="cache identity"):
        load_cache("u1", "r1", bad_cfg, path=path)


# ---------------------------------------------------------------------------
# static protocol
# ---------------------------------------------------------------------------


def static_account() -> AccountData:
    uid = "acct"
    alice, bob = basis(0), basis(1)
    rows = [make_row(uid, f"r{i}", day=i) for i in range(1, 7)]
    caches = {
        # r1: alice solo -> her enrollment source
        "r1": make_cache(uid, "r1", {"s0": [alice, alice]}),
        # r2: alice recognized; bob's enrollment source
        "r2": make_cache(uid, "r2", {"s0": [alice], "s1": [bob, bob]}),
        # r3: annotated alice but bob's voice -> wrong_name
        "r3": make_cache(uid, "r3", {"s0": [bob]}),
        # r4: dave present but nothing embeddable -> no_usable + enroll fallback
        "r4": make_cache(uid, "r4", {}, empty_speakers=("s0",)),
        # r5: dave's fallback enrollment source
        "r5": make_cache(uid, "r5", {"s0": [basis(3), basis(3)]}),
        # r6 (test bucket): annotated alice, voice matches nobody -> missed_known
        "r6": make_cache(uid, "r6", {"s0": [basis(6)]}),
    }
    annotations = {
        "r1": ann(uid, "r1", {"s0": ("alice", "clean")}),
        "r2": ann(uid, "r2", {"s0": ("alice", "clean"), "s1": ("bob", "clean")}),
        "r3": ann(uid, "r3", {"s0": ("alice", "clean")}),
        "r4": ann(uid, "r4", {"s0": ("dave", "clean")}),
        "r5": ann(uid, "r5", {"s0": ("dave", "clean")}),
        "r6": ann(uid, "r6", {"s0": ("alice", "clean")}),
    }
    return AccountData(uid=uid, rows=rows, annotations=annotations, caches=caches)


def test_static_protocol_end_to_end():
    result = run_static(static_account(), CFG)
    outcomes = {(r.rid, r.qid): r.outcome for r in result.rows}
    assert outcomes == {
        ("r2", "s0"): "correct_name",  # alice recognized from r1 enrollment
        ("r3", "s0"): "wrong_name",  # bob's voice on alice's row
        ("r4", "s0"): "no_usable",  # nothing embeddable
        ("r6", "s0"): "missed_known",  # alice enrolled but unmatched voice
    }
    # r1/s0, r2/s1, r5/s0 are enrollment sources -> never scored
    assert result.enroll_fallbacks == 1  # dave's failed r4 enrollment
    m = result.metrics()
    assert m.wrong_names == 1
    assert m.counts["no_usable_enrolled"] == 0  # dave wasn't enrolled at r4


def test_static_buckets_are_chronological():
    result = run_static(static_account(), CFG)
    buckets = {r.rid: r.bucket for r in result.rows}
    # 6 recordings, ceil(0.7*6)=5 tune -> r6 is test
    assert buckets["r2"] == "tune" and buckets["r6"] == "test"


def test_mixed_rows_never_scored_never_enroll():
    uid = "acct"
    rows = [make_row(uid, "r1", 1), make_row(uid, "r2", 2)]
    caches = {
        "r1": make_cache(uid, "r1", {"s0": [basis(0)]}),
        "r2": make_cache(uid, "r2", {"s0": [basis(0)]}),
    }
    annotations = {
        "r1": ann(uid, "r1", {"s0": ("alice", "mixed")}),  # NOT an enrollment source
        "r2": ann(uid, "r2", {"s0": ("alice", "clean")}),
    }
    result = run_static(AccountData(uid, rows, annotations, caches), CFG)
    # r1 mixed -> unscored; r2 is alice's first CLEAN appearance -> enrollment
    # source -> also unscored
    assert result.rows == []


def test_chrono_split_shapes():
    rows = [make_row("u", f"r{i}", day=i) for i in range(1, 11)]
    split = chrono_split(rows)
    assert sum(1 for v in split.values() if v == "tune") == 7
    assert split["r10"] == "test"


# ---------------------------------------------------------------------------
# chronological simulation
# ---------------------------------------------------------------------------


def test_chrono_promotion_and_recognition():
    uid = "acct"
    frank = basis(5)
    frank_other_take = blend(basis(5), basis(6), 0.9)  # same voice, not identical
    rows = [make_row(uid, "c1", 1, n_speakers=1), make_row(uid, "c2", 2, n_speakers=1)]
    caches = {
        "c1": make_cache(uid, "c1", {"s0": [frank, frank_other_take]}),
        "c2": make_cache(uid, "c2", {"s0": [frank]}),
    }
    annotations = {
        "c1": ann(uid, "c1", {"s0": ("frank", "clean")}),
        "c2": ann(uid, "c2", {"s0": ("frank", "clean")}),
    }
    result = run_chrono(AccountData(uid, rows, annotations, caches), CFG)
    outcomes = {(r.rid): r.outcome for r in result.rows}
    # c1: scored before labeling -> correct_unknown; then proposal committed
    # and promoted to "frank"; c2: recognized
    assert outcomes == {"c1": "correct_unknown", "c2": "correct_name"}
    assert result.final_named_speakers == 1
    assert result.final_unknown_keys == 0
    assert result.true_never_labeled == 0
    assert result.contaminated_keys == {}


def test_chrono_contamination_detected():
    """Two different strangers whose voices dedup-merge -> flagged."""
    uid = "acct"
    gina = basis(6)
    hana = blend(basis(6), basis(7), 0.8)  # 0.8 sim to gina: above merge bar
    rows = [make_row(uid, "c1", 1)]
    caches = {
        "c1": make_cache(
            uid, "c1", {"s0": [gina, gina, gina], "s1": [hana, hana, hana]}
        )
    }
    annotations = {
        "c1": ann(uid, "c1", {"s0": ("gina", "clean"), "s1": ("hana", "clean")})
    }
    result = run_chrono(AccountData(uid, rows, annotations, caches), CFG)
    assert any(
        set(persons) == {"gina", "hana"}
        for persons in result.contaminated_keys.values()
    ), f"expected contamination, got {result.contaminated_keys}"
