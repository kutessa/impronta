"""Serialization round-trips (hypothesis) and model helpers."""

from datetime import datetime, timezone

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from impronta.models import (
    Candidate,
    EnrollResult,
    IdentifyResult,
    SegmentInfo,
    SpeakerMatch,
    SpeakerSummary,
    StoreEntry,
    UnknownProposal,
    composite_quality,
    decode_embedding,
    encode_embedding,
)

DIM = 8

floats = st.floats(-1e6, 1e6, allow_nan=False, width=32)
opt_floats = st.one_of(st.none(), floats)
keys = st.text(st.characters(codec="ascii", categories=["L", "N"]), min_size=1, max_size=12)
names = st.one_of(st.none(), st.text(max_size=20))
dts = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2030, 1, 1),
    timezones=st.just(timezone.utc),
)
vec = st.lists(floats, min_size=DIM, max_size=DIM).map(
    lambda v: np.asarray(v, dtype=np.float32)
)
matrix = st.integers(1, 4).flatmap(
    lambda n: st.lists(vec, min_size=n, max_size=n).map(np.stack)
)

seg_infos = st.builds(
    SegmentInfo,
    start=st.floats(0, 100, allow_nan=False, width=32),
    end=st.floats(100, 200, allow_nan=False, width=32),
    confidence=st.floats(0, 1, allow_nan=False, width=32),
    snr_db=st.floats(-20, 100, allow_nan=False, width=32),
)

entries = st.builds(
    StoreEntry,
    entry_id=keys,
    speaker_key=keys,
    display_name=names,
    language=st.sampled_from(["en", "bs", "de"]),
    confidence=st.floats(0, 1, allow_nan=False, width=32),
    embedding=vec,
    created_at=dts,
    source=st.sampled_from(["scribe_enroll", "direct_enroll", "auto_enroll"]),
    source_transcription_id=st.one_of(st.none(), keys),
    snr_db=st.one_of(st.none(), st.floats(-20, 100, allow_nan=False, width=32)),
    duration_sec=st.one_of(st.none(), st.floats(0, 60, allow_nan=False, width=32)),
    namespace=keys,
)

candidates = st.builds(
    Candidate,
    speaker_key=keys,
    display_name=names,
    score_share=st.floats(0, 1, allow_nan=False, width=32),
    mean_similarity=st.one_of(st.none(), st.floats(-1, 1, allow_nan=False, width=32)),
)

matches = st.builds(
    SpeakerMatch,
    query_speaker_id=keys,
    speaker_key=st.one_of(st.none(), keys),
    display_name=names,
    namespace=st.one_of(st.none(), keys),
    is_unknown=st.booleans(),
    identifiable=st.booleans(),
    candidates=st.tuples(candidates, candidates),
    mean_similarity=st.one_of(st.none(), st.floats(-1, 1, allow_nan=False, width=32)),
    quality_tier=st.one_of(st.none(), st.sampled_from(["high", "medium", "low"])),
    num_segments_used=st.integers(0, 50),
    num_segments_total=st.integers(0, 50),
    near_misses=st.lists(
        st.tuples(keys, st.floats(0, 1, allow_nan=False, width=32)), max_size=3
    ).map(tuple),
    no_proposal_reason=st.one_of(st.none(), st.sampled_from(["gray_zone", "low_cohesion"])),
)

proposals = st.builds(
    UnknownProposal,
    query_speaker_id=keys,
    transcription_id=keys,
    language=st.sampled_from(["en", "bs"]),
    suggested_key=keys,
    embeddings=matrix,
    segments=st.tuples(seg_infos),
    quality_tier=st.sampled_from(["high", "medium", "low"]),
)

enroll_results = st.builds(
    EnrollResult,
    speaker_key=keys,
    display_name=st.text(max_size=20),
    language=st.just("en"),
    segments_total=st.integers(0, 99),
    segments_used=st.integers(0, 99),
    quality_tier=st.sampled_from(["high", "medium", "low"]),
    merged_unknown_keys=st.lists(keys, max_size=2).map(tuple),
    entry_ids=st.lists(keys, max_size=2).map(tuple),
)

summaries = st.builds(
    SpeakerSummary,
    speaker_key=keys,
    display_name=names,
    namespaces=st.tuples(keys),
    languages=st.tuples(st.just("en")),
    num_embeddings=st.integers(0, 500),
    created_at=dts,
)


@given(entries)
def test_store_entry_roundtrip(entry):
    assert StoreEntry.from_dict(entry.to_dict()) == entry


@given(proposals)
def test_proposal_roundtrip(p):
    back = UnknownProposal.from_dict(p.to_dict())
    assert back == p
    assert back.embeddings.dtype == np.float32
    np.testing.assert_array_equal(back.embeddings, p.embeddings.astype(np.float32))


@given(matches)
def test_speaker_match_roundtrip(m):
    assert SpeakerMatch.from_dict(m.to_dict()) == m


@given(enroll_results)
def test_enroll_result_roundtrip(r):
    assert EnrollResult.from_dict(r.to_dict()) == r


@given(summaries)
def test_summary_roundtrip(s):
    assert SpeakerSummary.from_dict(s.to_dict()) == s


@given(st.dictionaries(keys, matches, max_size=3), st.lists(proposals, max_size=2))
def test_identify_result_roundtrip(speakers, props):
    r = IdentifyResult(
        speakers=speakers,
        proposed_unknowns=tuple(props),
        language_code="en",
        transcription_id="t1",
    )
    back = IdentifyResult.from_dict(r.to_dict())
    assert back.speakers == r.speakers
    assert back.proposed_unknowns == r.proposed_unknowns


@given(matrix)
def test_embedding_encoding_is_float32_exact(m):
    decoded = decode_embedding(encode_embedding(m), dim=m.shape[1])
    np.testing.assert_array_equal(decoded, m.astype(np.float32))


def test_identify_result_is_json_serializable():
    import json

    r = IdentifyResult(
        speakers={
            "s0": SpeakerMatch(
                query_speaker_id="s0",
                speaker_key="alice",
                display_name="Alice",
                namespace="ws:1",
                is_unknown=False,
                identifiable=True,
            )
        },
        proposed_unknowns=(
            UnknownProposal(
                query_speaker_id="s1",
                transcription_id="t1",
                language="en",
                suggested_key="unknown-abc",
                embeddings=np.eye(2, dtype=np.float32),
                segments=(SegmentInfo(0.0, 2.0, 0.9, 25.0),),
                quality_tier="high",
            ),
        ),
        language_code="en",
        transcription_id="t1",
    )
    parsed = json.loads(json.dumps(r.to_dict()))
    assert IdentifyResult.from_dict(parsed).speakers["s0"].display_name == "Alice"


def test_composite_quality_saturates():
    assert composite_quality(80.0, 1.0, 60.0) == composite_quality(40.0, 1.0, 10.0)
    assert composite_quality(None, 0.5, None) == 0.5
    assert composite_quality(20.0, 0.5, 5.0) == 0.5 + 0.5 + 0.5


def test_name_map():
    r = IdentifyResult(
        speakers={
            "s0": SpeakerMatch("s0", "alice", "Alice", "ns", False, True),
            "s1": SpeakerMatch("s1", None, None, None, True, True),
        },
        proposed_unknowns=(),
        language_code="en",
        transcription_id="t",
    )
    assert r.name_map() == {"s0": "Alice", "s1": None}
