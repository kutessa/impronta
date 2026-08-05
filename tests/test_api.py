"""End-to-end behavior of the Impronta facade (FakeEmbedder + InMemoryStore).

Synthetic voices (see conftest): 440 Hz -> e0, 880 -> e1, 1320 -> e2,
1760 -> e3; extra blended voices are registered per test to control
similarity outcomes exactly.
"""

from datetime import timedelta

import numpy as np
import pytest
from conftest import (
    FakeEmbedder,
    basis,
    blend,
    compose_timeline,
    make_multichannel_response,
    make_scribe_response,
    make_voice_audio,
    speech_words,
)

from impronta import (
    Impronta,
    ImprontaConfig,
    InMemoryStore,
    NoUsableSegmentsError,
    SpeakerNotFoundError,
    SpeakerNotInTranscriptError,
)
from impronta.models import SegmentInfo, UnknownProposal, utcnow

ALICE, BOB, STRANGER, STRANGER2 = 440.0, 880.0, 1320.0, 1760.0


@pytest.fixture
def voices() -> dict[float, np.ndarray]:
    return {ALICE: basis(0), BOB: basis(1), STRANGER: basis(2), STRANGER2: basis(3)}


def make_app(voices, **kwargs) -> Impronta:
    embedder = FakeEmbedder(voices)
    store = kwargs.pop("store", None) or InMemoryStore()
    return Impronta(store=store, embedder=embedder, **kwargs)


def solo_recording(freq: float, spans=((0.0, 6.0),), speaker_id="speaker_0", tid="tx-1",
                   language="en", logprob=-0.05):
    duration = max(end for _, end in spans) + 0.2
    audio = compose_timeline(duration, [(s, e, freq) for s, e in spans])
    words = []
    for s, e in spans:
        words.extend(speech_words(speaker_id, s, e, logprob=logprob))
    resp = make_scribe_response(words, language=language, transcription_id=tid)
    return resp, audio


def wav_bytes(samples: np.ndarray) -> bytes:
    import io

    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, samples, 16_000, format="WAV", subtype="FLOAT")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# enroll -> identify happy path
# ---------------------------------------------------------------------------


def test_enroll_then_identify_same_voice(voices):
    app = make_app(voices)
    resp, audio = solo_recording(ALICE, tid="tx-enroll")
    result = app.add_speaker(resp, wav_bytes(audio), "speaker_0", "Alice")
    assert result.speaker_key == "alice"
    assert result.quality_tier == "high"
    assert result.segments_used >= 1

    resp2, audio2 = solo_recording(ALICE, spans=((0.0, 4.0),), tid="tx-query")
    identified = app.identify(resp2, wav_bytes(audio2))
    match = identified.speakers["speaker_0"]
    assert match.display_name == "Alice"
    assert match.speaker_key == "alice"
    assert not match.is_unknown
    assert match.identifiable
    assert match.mean_similarity == pytest.approx(1.0, abs=1e-5)
    assert match.candidates[0].speaker_key == "alice"
    assert identified.proposed_unknowns == ()


def test_enroll_unknown_speaker_id_raises(voices):
    app = make_app(voices)
    resp, audio = solo_recording(ALICE)
    with pytest.raises(SpeakerNotInTranscriptError, match="speaker_9"):
        app.add_speaker(resp, wav_bytes(audio), "speaker_9", "Alice")


def test_enroll_all_low_confidence_raises(voices):
    app = make_app(voices)
    resp, audio = solo_recording(ALICE, logprob=-2.0)  # conf ~0.14 < 0.5
    with pytest.raises(NoUsableSegmentsError):
        app.add_speaker(resp, wav_bytes(audio), "speaker_0", "Alice")


def test_duplicate_name_appends_to_same_key(voices):
    app = make_app(voices)
    r1, a1 = solo_recording(ALICE, tid="tx-1")
    r2, a2 = solo_recording(ALICE, spans=((0.0, 5.0),), tid="tx-2")
    app.add_speaker(r1, wav_bytes(a1), "speaker_0", "Alice")
    n_after_first = app.store.count("default")
    app.add_speaker(r2, wav_bytes(a2), "speaker_0", "Alice")
    assert app.store.count("default") > n_after_first
    (summary,) = app.list_speakers()
    assert summary.speaker_key == "alice"


def test_enroll_is_idempotent_per_transcription(voices):
    app = make_app(voices)
    resp, audio = solo_recording(ALICE, tid="tx-same")
    app.add_speaker(resp, wav_bytes(audio), "speaker_0", "Alice")
    count = app.store.count("default")
    app.add_speaker(resp, wav_bytes(audio), "speaker_0", "Alice")
    assert app.store.count("default") == count  # deterministic ids upserted


def test_add_speaker_from_audio(voices):
    app = make_app(voices)
    clip = make_voice_audio(12.0, BOB)
    result = app.add_speaker_from_audio(wav_bytes(clip), "Bob", "en")
    assert result.speaker_key == "bob"
    assert result.segments_used >= 2  # 5s windows

    resp, audio = solo_recording(BOB, spans=((0.0, 4.0),))
    match = app.identify(resp, wav_bytes(audio)).speakers["speaker_0"]
    assert match.display_name == "Bob"


# ---------------------------------------------------------------------------
# unknowns: propose -> commit -> recognize -> label
# ---------------------------------------------------------------------------


def stranger_recording(tid="tx-s", freq=STRANGER, language="en"):
    # three well-separated spans -> three segments -> passes min_proposal_segments
    return solo_recording(
        freq, spans=((0.0, 5.0), (6.0, 11.0), (12.0, 17.0)), tid=tid, language=language
    )


def test_stranger_is_proposed_not_written(voices):
    app = make_app(voices)
    resp, audio = stranger_recording()
    result = app.identify(resp, wav_bytes(audio))
    match = result.speakers["speaker_0"]
    assert match.is_unknown and match.identifiable
    assert match.speaker_key is None
    assert match.no_proposal_reason is None
    assert len(result.proposed_unknowns) == 1
    assert app.store.count("default") == 0  # identify never writes


def test_commit_then_reidentify_recognizes_unknown(voices):
    app = make_app(voices)
    resp, audio = stranger_recording(tid="tx-a")
    proposal = app.identify(resp, wav_bytes(audio)).proposed_unknowns[0]
    (key,) = app.commit_unknowns([proposal])
    assert key == proposal.suggested_key

    resp2, audio2 = stranger_recording(tid="tx-b")
    result = app.identify(resp2, wav_bytes(audio2))
    match = result.speakers["speaker_0"]
    assert match.speaker_key == key
    assert match.is_unknown  # recognized but still unlabeled
    assert result.proposed_unknowns == ()


def test_commit_is_idempotent(voices):
    app = make_app(voices)
    resp, audio = stranger_recording()
    proposal = app.identify(resp, wav_bytes(audio)).proposed_unknowns[0]
    (k1,) = app.commit_unknowns([proposal])
    count = app.store.count("default")
    ids1 = {e.entry_id for e in app.store.get_speaker_entries("default", k1)}
    (k2,) = app.commit_unknowns([proposal])
    assert k1 == k2
    assert app.store.count("default") == count
    assert {e.entry_id for e in app.store.get_speaker_entries("default", k1)} == ids1


def test_commit_dedups_same_stranger_across_recordings(voices):
    app = make_app(voices)
    ra, aa = stranger_recording(tid="tx-a")
    rb, ab = stranger_recording(tid="tx-b")
    # parallel processing: both identified before either commit
    pa = app.identify(ra, wav_bytes(aa)).proposed_unknowns[0]
    pb = app.identify(rb, wav_bytes(ab)).proposed_unknowns[0]
    assert pa.suggested_key != pb.suggested_key
    (ka,) = app.commit_unknowns([pa])
    (kb,) = app.commit_unknowns([pb])
    assert kb == ka  # second commit merged into the first record
    summaries = [s for s in app.list_speakers() if s.is_unknown]
    assert len(summaries) == 1


def make_proposal(embeddings: np.ndarray, tid: str, language="en") -> UnknownProposal:
    return UnknownProposal(
        query_speaker_id="speaker_0",
        transcription_id=tid,
        language=language,
        suggested_key=f"unknown-{tid}",
        embeddings=embeddings.astype(np.float32),
        segments=tuple(
            SegmentInfo(start=float(i), end=float(i) + 2.0, confidence=0.9, snr_db=25.0)
            for i in range(embeddings.shape[0])
        ),
        quality_tier="high",
    )


def test_commit_dedup_uses_merge_threshold_not_similarity_threshold(voices):
    app = make_app(voices)
    app.commit_unknowns([make_proposal(np.stack([basis(2), basis(2)]), tid="t-first")])
    # 0.55 similar: above match threshold (0.5) but below merge bar (0.6)
    nearish = blend(basis(2), basis(4), 0.55)
    keys = app.commit_unknowns([make_proposal(np.stack([nearish, nearish]), tid="t-second")])
    assert keys == ["unknown-t-second"]  # NOT merged
    assert len([s for s in app.list_speakers() if s.is_unknown]) == 2


def test_label_speaker_promotes_unknown(voices):
    app = make_app(voices)
    resp, audio = stranger_recording()
    proposal = app.identify(resp, wav_bytes(audio)).proposed_unknowns[0]
    (key,) = app.commit_unknowns([proposal])
    summary = app.label_speaker(key, "Carol")
    assert summary.speaker_key == "carol"
    assert summary.display_name == "Carol"
    assert not summary.is_unknown

    resp2, audio2 = stranger_recording(tid="tx-later")
    assert app.identify(resp2, wav_bytes(audio2)).speakers["speaker_0"].display_name == "Carol"


def test_label_speaker_merges_into_existing_named(voices):
    app = make_app(voices)
    r, a = solo_recording(ALICE, tid="t-alice")
    app.add_speaker(r, wav_bytes(a), "speaker_0", "Alice")
    app.commit_unknowns([make_proposal(np.stack([basis(2), basis(2)]), tid="t-u")])
    summary = app.label_speaker("unknown-t-u", "Alice")
    assert summary.speaker_key == "alice"
    assert summary.num_embeddings >= 3
    assert app.store.get_speaker_entries("default", "unknown-t-u") == []


def test_label_named_speaker_keeps_key(voices):
    app = make_app(voices)
    r, a = solo_recording(ALICE)
    app.add_speaker(r, wav_bytes(a), "speaker_0", "Alice")
    summary = app.label_speaker("alice", "Alice Liddell")
    assert summary.speaker_key == "alice"
    assert summary.display_name == "Alice Liddell"


def test_label_missing_speaker_raises(voices):
    app = make_app(voices)
    with pytest.raises(SpeakerNotFoundError):
        app.label_speaker("ghost", "Casper")


def test_enroll_triggered_unknown_merge(voices):
    app = make_app(voices)
    app.commit_unknowns([make_proposal(np.stack([basis(0), basis(0)]), tid="t-u")])
    r, a = solo_recording(ALICE, tid="t-alice")
    result = app.add_speaker(r, wav_bytes(a), "speaker_0", "Alice")
    assert result.merged_unknown_keys == ("unknown-t-u",)
    assert [s.speaker_key for s in app.list_speakers()] == ["alice"]
    assert all(
        e.display_name == "Alice"
        for e in app.store.get_speaker_entries("default", "alice")
    )


# ---------------------------------------------------------------------------
# profile reinforcement
# ---------------------------------------------------------------------------

SAME_VOICE_NEW_TAKE = 660.0  # Alice again, different channel: 0.8 sim to e0


def reinforce_setup(voices) -> Impronta:
    voices = dict(voices)
    voices[SAME_VOICE_NEW_TAKE] = blend(basis(0), basis(5), 0.8)
    app = make_app(voices)
    # 3 spans -> 3 entries: a MATURE profile (reinforce_min_profile_entries)
    r, a = solo_recording(
        ALICE, spans=((0.0, 5.0), (6.0, 11.0), (12.0, 17.0)), tid="t-enroll-alice"
    )
    app.add_speaker(r, wav_bytes(a), "speaker_0", "Alice")
    return app


def test_reinforce_happy_path(voices):
    app = reinforce_setup(voices)
    resp, audio = solo_recording(SAME_VOICE_NEW_TAKE, spans=((0.0, 5.0), (6.0, 11.0)),
                                 tid="t-new-take")
    proposals = app.propose_reinforcements(resp, wav_bytes(audio))
    assert len(proposals) == 1
    p = proposals[0]
    assert p.speaker_key == "alice"
    assert p.mean_similarity == pytest.approx(0.8, abs=0.01)
    assert p.embeddings.shape[0] == 2
    before = len(app.store.get_speaker_entries("default", "alice"))

    keys = app.commit_reinforcements(proposals)
    assert keys == ["alice"]
    entries = app.store.get_speaker_entries("default", "alice")
    assert len(entries) == before + 2
    assert {e.source for e in entries} == {"scribe_enroll", "reinforce"}

    # idempotent: committing again changes nothing
    app.commit_reinforcements(proposals)
    assert len(app.store.get_speaker_entries("default", "alice")) == before + 2

    # serialization round-trip survives a task queue
    from impronta import ReinforcementProposal

    assert ReinforcementProposal.from_dict(p.to_dict()) == p


def test_reinforce_skips_weak_match(voices):
    voices = dict(voices)
    voices[660.0] = blend(basis(0), basis(5), 0.55)  # matches (>=0.4), weak (<0.6)
    app = make_app(voices)
    r, a = solo_recording(
        ALICE, spans=((0.0, 5.0), (6.0, 11.0), (12.0, 17.0)), tid="t-enroll"
    )
    app.add_speaker(r, wav_bytes(a), "speaker_0", "Alice")
    resp, audio = solo_recording(660.0, spans=((0.0, 5.0), (6.0, 11.0)), tid="t-weak")
    assert app.identify(resp, wav_bytes(audio)).speakers["speaker_0"].display_name == "Alice"
    assert app.propose_reinforcements(resp, wav_bytes(audio)) == []


def test_reinforce_skips_contested_match(voices):
    voices = dict(voices)
    # voice close to BOTH alice (0.75) and bob (0.65): margin 0.10 < 0.12
    v = 0.75 * basis(0) + 0.65 * basis(1)
    voices[660.0] = (v / np.linalg.norm(v)).astype(np.float32)
    app = make_app(voices)
    ra, aa = solo_recording(ALICE, spans=((0.0, 5.0), (6.0, 11.0), (12.0, 17.0)), tid="t-a")
    rb, ab = solo_recording(BOB, spans=((0.0, 5.0), (6.0, 11.0), (12.0, 17.0)), tid="t-b")
    app.add_speaker(ra, wav_bytes(aa), "speaker_0", "Alice")
    app.add_speaker(rb, wav_bytes(ab), "speaker_0", "Bob")
    resp, audio = solo_recording(660.0, spans=((0.0, 5.0), (6.0, 11.0)), tid="t-contested")
    assert app.propose_reinforcements(resp, wav_bytes(audio)) == []


def test_reinforce_skips_non_novel_segments(voices):
    """Exact-duplicate embeddings (>= novelty ceiling) add nothing."""
    app = make_app(voices)
    r, a = solo_recording(ALICE, tid="t-enroll")
    app.add_speaker(r, wav_bytes(a), "speaker_0", "Alice")
    resp, audio = solo_recording(ALICE, spans=((0.0, 5.0),), tid="t-dup")  # identical vector
    assert app.propose_reinforcements(resp, wav_bytes(audio)) == []


def test_reinforce_commit_skips_deleted_speaker(voices):
    app = reinforce_setup(voices)
    resp, audio = solo_recording(SAME_VOICE_NEW_TAKE, spans=((0.0, 5.0),), tid="t-late")
    proposals = app.propose_reinforcements(resp, wav_bytes(audio))
    assert proposals
    app.remove_speaker("alice")
    assert app.commit_reinforcements(proposals) == [None]
    assert app.store.count("default") == 0


def test_reinforce_uses_current_display_name(voices):
    app = reinforce_setup(voices)
    resp, audio = solo_recording(SAME_VOICE_NEW_TAKE, spans=((0.0, 5.0),), tid="t-rename")
    proposals = app.propose_reinforcements(resp, wav_bytes(audio))
    app.label_speaker("alice", "Alice Liddell")  # renamed after proposing
    app.commit_reinforcements(proposals)
    reinforced = [
        e for e in app.store.get_speaker_entries("default", "alice")
        if e.source == "reinforce"
    ]
    assert reinforced and all(e.display_name == "Alice Liddell" for e in reinforced)


def test_reinforce_skips_immature_profile(voices):
    """A 1-entry profile must never self-train (poisoning anchor)."""
    voices = dict(voices)
    voices[SAME_VOICE_NEW_TAKE] = blend(basis(0), basis(5), 0.8)
    app = make_app(voices)
    r, a = solo_recording(ALICE, tid="t-thin")  # single span -> 1 entry
    app.add_speaker(r, wav_bytes(a), "speaker_0", "Alice")
    assert len(app.store.get_speaker_entries("default", "alice")) < 3
    resp, audio = solo_recording(
        SAME_VOICE_NEW_TAKE, spans=((0.0, 5.0), (6.0, 11.0)), tid="t-take"
    )
    assert app.identify(resp, wav_bytes(audio)).speakers["speaker_0"].display_name == "Alice"
    assert app.propose_reinforcements(resp, wav_bytes(audio)) == []


def test_reinforce_never_proposes_strangers(voices):
    app = reinforce_setup(voices)
    resp, audio = stranger_recording(tid="t-stranger")
    assert app.propose_reinforcements(resp, wav_bytes(audio)) == []


# ---------------------------------------------------------------------------
# proposal gating
# ---------------------------------------------------------------------------


def test_single_segment_stranger_not_proposed(voices):
    app = make_app(voices)
    resp, audio = solo_recording(STRANGER, spans=((0.0, 5.0),))  # one segment only
    result = app.identify(resp, wav_bytes(audio))
    match = result.speakers["speaker_0"]
    assert match.is_unknown
    assert match.no_proposal_reason == "too_few_segments"
    assert result.proposed_unknowns == ()


def test_low_cohesion_mixed_voices_not_proposed(voices):
    voices = dict(voices)
    voices[2200.0] = basis(4)  # a third stranger voice
    app = make_app(voices)
    # one diarized speaker whose segments alternate between THREE voices
    # (pairwise cohesion 1/6 ~ 0.17, below the 0.3 gate)
    duration = 23.2
    audio = compose_timeline(
        duration,
        [(0.0, 5.0, STRANGER), (6.0, 11.0, STRANGER2), (12.0, 17.0, 2200.0),
         (18.0, 23.0, STRANGER)],
    )
    words = []
    for s, e in ((0.0, 5.0), (6.0, 11.0), (12.0, 17.0), (18.0, 23.0)):
        words.extend(speech_words("speaker_0", s, e))
    resp = make_scribe_response(words, transcription_id="tx-mixed")
    result = app.identify(resp, wav_bytes(audio))
    match = result.speakers["speaker_0"]
    assert match.no_proposal_reason == "low_cohesion"
    assert result.proposed_unknowns == ()


def test_gray_zone_near_named_match_not_proposed(voices):
    voices = dict(voices)
    voices[660.0] = blend(basis(0), basis(5), 0.36)  # in the [0.32, 0.40) gray zone vs alice
    app = make_app(voices)
    r, a = solo_recording(ALICE, tid="t-alice")
    app.add_speaker(r, wav_bytes(a), "speaker_0", "Alice")

    resp, audio = stranger_recording(tid="tx-gray", freq=660.0)
    result = app.identify(resp, wav_bytes(audio))
    match = result.speakers["speaker_0"]
    assert match.is_unknown
    assert match.no_proposal_reason == "gray_zone"
    assert match.near_misses and match.near_misses[0][0] == "alice"
    assert result.proposed_unknowns == ()


def test_low_quality_tier_not_proposed(voices):
    cfg = ImprontaConfig(min_proposal_tier="high")
    app = make_app(voices, config=cfg)
    # noisy stranger: passes only a relaxed tier
    duration = 11.2
    audio = compose_timeline(
        duration, [(0.0, 5.0, STRANGER), (6.0, 11.0, STRANGER)], noise_db=13.0
    )
    words = speech_words("speaker_0", 0.0, 5.0) + speech_words("speaker_0", 6.0, 11.0)
    resp = make_scribe_response(words, transcription_id="tx-noisy")
    result = app.identify(resp, wav_bytes(audio))
    match = result.speakers["speaker_0"]
    if match.identifiable:  # WADA is approximate; tier must be sub-high here
        assert match.no_proposal_reason in ("low_quality", "too_few_segments")
        assert result.proposed_unknowns == ()


def test_enroll_drops_diarization_outlier_segments(voices):
    """4 segments of Alice + 1 misattributed segment of Bob's voice, all
    under one speaker_id: the Bob segment must not enter Alice's profile."""
    app = make_app(voices)
    spans = [(0.0, 4.0), (5.0, 9.0), (10.0, 14.0), (15.0, 19.0), (20.0, 24.0)]
    audio = compose_timeline(
        24.4,
        [(s, e, ALICE) for s, e in spans[:4]] + [(spans[4][0], spans[4][1], BOB)],
    )
    words = []
    for s, e in spans:
        words.extend(speech_words("speaker_0", s, e))
    resp = make_scribe_response(words, transcription_id="tx-mixed-enroll")
    result = app.add_speaker(resp, wav_bytes(audio), "speaker_0", "Alice")
    assert result.segments_used == 4  # outlier dropped
    entries = app.store.get_speaker_entries("default", "alice")
    assert all(float(np.dot(e.embedding, basis(0))) > 0.9 for e in entries)


# ---------------------------------------------------------------------------
# quality/filters through the full flow
# ---------------------------------------------------------------------------


def test_unidentifiable_speaker_never_raises(voices):
    app = make_app(voices)
    r, a = solo_recording(ALICE, tid="t-alice")
    app.add_speaker(r, wav_bytes(a), "speaker_0", "Alice")
    # speaker_1 says one 0.3s word — no usable segments
    duration = 6.5
    audio = compose_timeline(duration, [(0.0, 6.0, ALICE)])
    words = speech_words("speaker_0", 0.0, 6.0) + [
        {"text": "ok", "start": 6.1, "end": 6.4, "type": "word",
         "speaker_id": "speaker_1", "logprob": -0.05, "channel_index": None}
    ]
    resp = make_scribe_response(words)
    result = app.identify(resp, wav_bytes(audio))
    assert result.speakers["speaker_0"].display_name == "Alice"
    short = result.speakers["speaker_1"]
    assert not short.identifiable
    assert short.speaker_key is None and short.quality_tier is None


def test_low_confidence_segments_dropped_on_identify(voices):
    app = make_app(voices)
    r, a = solo_recording(ALICE, tid="t-alice")
    app.add_speaker(r, wav_bytes(a), "speaker_0", "Alice")
    resp, audio = solo_recording(ALICE, spans=((0.0, 4.0),), logprob=-2.0, tid="t-q")
    match = app.identify(resp, wav_bytes(audio)).speakers["speaker_0"]
    assert not match.identifiable  # everything gated by confidence


def test_bcms_language_codes_are_one_language(voices):
    """Scribe flips between hrv/srp/bos for the same speech — a speaker
    enrolled under one BCMS code must match under any other."""
    app = make_app(voices)
    r, a = solo_recording(ALICE, language="hrv", tid="t-hrv")
    result = app.add_speaker(r, wav_bytes(a), "speaker_0", "Alice")
    assert result.language == "hbs"  # stored under the canonical macrolanguage

    for code in ("srp", "bos", "cnr", "hr", "sr"):
        resp, audio = solo_recording(ALICE, spans=((0.0, 4.0),), language=code, tid=f"t-{code}")
        match = app.identify(resp, wav_bytes(audio)).speakers["speaker_0"]
        assert match.display_name == "Alice", f"failed for {code}"


def test_bcms_proposals_commit_under_canonical_language(voices):
    app = make_app(voices)
    resp, audio = stranger_recording(tid="tx-s", language="srp")
    proposal = app.identify(resp, wav_bytes(audio)).proposed_unknowns[0]
    assert proposal.language == "hbs"
    (key,) = app.commit_unknowns([proposal])
    # the stranger returns in a recording detected as Croatian this time
    resp2, audio2 = stranger_recording(tid="tx-s2", language="hrv")
    match = app.identify(resp2, wav_bytes(audio2)).speakers["speaker_0"]
    assert match.speaker_key == key


def test_cross_language_hard_filter_and_override(voices):
    app = make_app(voices)
    r, a = solo_recording(ALICE, language="en", tid="t-en")
    app.add_speaker(r, wav_bytes(a), "speaker_0", "Alice")

    resp, audio = solo_recording(ALICE, spans=((0.0, 4.0),), language="bs", tid="t-bs")
    hard = app.identify(resp, wav_bytes(audio)).speakers["speaker_0"]
    assert hard.is_unknown  # enrolled in en, speaking bs -> filtered out

    soft = app.identify(resp, wav_bytes(audio), language_filter=False).speakers["speaker_0"]
    assert soft.display_name == "Alice"


# ---------------------------------------------------------------------------
# namespaces
# ---------------------------------------------------------------------------


def test_hierarchical_namespaces_and_isolation(voices):
    store = InMemoryStore()
    ws_app = make_app(voices, store=store, write_namespace="ws:1")
    r, a = solo_recording(ALICE, tid="t-alice")
    ws_app.add_speaker(r, wav_bytes(a), "speaker_0", "Alice")

    user_app = make_app(
        voices, store=store, write_namespace="user:9", read_namespaces=["ws:1", "user:9"]
    )
    rb, ab = solo_recording(BOB, tid="t-bob")
    user_app.add_speaker(rb, wav_bytes(ab), "speaker_0", "Bob")
    assert store.count("user:9") > 0 and store.count("ws:1") > 0

    # user app sees both; a different tenant sees neither
    resp, audio = solo_recording(ALICE, spans=((0.0, 4.0),), tid="t-q")
    assert user_app.identify(resp, wav_bytes(audio)).speakers["speaker_0"].display_name == "Alice"

    other = make_app(voices, store=store, write_namespace="ws:2")
    stranger_view = other.identify(resp, wav_bytes(audio)).speakers["speaker_0"]
    assert stranger_view.is_unknown

    assert {s.speaker_key for s in user_app.list_speakers()} == {"alice", "bob"}


def test_wipe_namespace(voices):
    store = InMemoryStore()
    app = make_app(voices, store=store, write_namespace="ws:1")
    r, a = solo_recording(ALICE)
    app.add_speaker(r, wav_bytes(a), "speaker_0", "Alice")
    assert app.wipe_namespace("ws:1") > 0
    assert store.count("ws:1") == 0


# ---------------------------------------------------------------------------
# growth control and rollback
# ---------------------------------------------------------------------------


def test_per_speaker_cap_evicts_lowest_quality(voices):
    cfg = ImprontaConfig(max_embeddings_per_speaker=2)
    app = make_app(voices, config=cfg)
    # three segments with distinct durations -> distinct composite quality
    resp, audio = solo_recording(
        ALICE, spans=((0.0, 1.2), (2.0, 8.0), (9.0, 10.5)), tid="t-1"
    )
    app.add_speaker(resp, wav_bytes(audio), "speaker_0", "Alice")
    entries = app.store.get_speaker_entries("default", "alice")
    assert len(entries) == 2
    durations = sorted(e.duration_sec for e in entries)
    assert durations[0] == pytest.approx(1.5, abs=0.1)  # 1.2s segment evicted
    assert durations[1] == pytest.approx(6.0, abs=0.1)


def test_remove_transcription_is_surgical(voices):
    app = make_app(voices)
    r1, a1 = solo_recording(ALICE, tid="tx-1")
    r2, a2 = solo_recording(ALICE, spans=((0.0, 5.0),), tid="tx-2")
    app.add_speaker(r1, wav_bytes(a1), "speaker_0", "Alice")
    app.add_speaker(r2, wav_bytes(a2), "speaker_0", "Alice")
    removed = app.remove_transcription("tx-1")
    assert removed > 0
    remaining = app.store.get_speaker_entries("default", "alice")
    assert remaining and all(e.source_transcription_id == "tx-2" for e in remaining)


def test_prune_unknowns(voices):
    app = make_app(voices)
    app.commit_unknowns([make_proposal(np.stack([basis(2), basis(2)]), tid="t-u")])
    r, a = solo_recording(ALICE)
    app.add_speaker(r, wav_bytes(a), "speaker_0", "Alice")

    assert app.prune_unknowns(older_than=utcnow() - timedelta(days=1)) == 0
    removed = app.prune_unknowns(older_than=utcnow() + timedelta(seconds=5))
    assert removed == 2  # both unknown entries
    assert [s.speaker_key for s in app.list_speakers()] == ["alice"]


def test_remove_speaker(voices):
    app = make_app(voices)
    r, a = solo_recording(ALICE)
    app.add_speaker(r, wav_bytes(a), "speaker_0", "Alice")
    assert app.remove_speaker("alice") > 0
    assert app.list_speakers() == []


# ---------------------------------------------------------------------------
# multichannel
# ---------------------------------------------------------------------------


def test_multichannel_keys_and_per_channel_audio(voices):
    app = make_app(voices)
    ra, aa = solo_recording(ALICE, tid="t-a")
    rb, ab = solo_recording(BOB, tid="t-b")
    app.add_speaker(ra, wav_bytes(aa), "speaker_0", "Alice")
    app.add_speaker(rb, wav_bytes(ab), "speaker_0", "Bob")

    left = compose_timeline(6.2, [(0.0, 6.0, ALICE)])
    right = compose_timeline(6.2, [(0.0, 6.0, BOB)])
    stereo = np.stack([left, right], axis=1)
    resp = make_multichannel_response(
        [speech_words("speaker_0", 0.0, 6.0), speech_words("speaker_0", 0.0, 6.0)]
    )
    result = app.identify(resp, wav_bytes(stereo))
    assert result.speakers["0:speaker_0"].display_name == "Alice"
    assert result.speakers["1:speaker_0"].display_name == "Bob"


# ---------------------------------------------------------------------------
# exposed internals: audio_id threading, segments used, ideal segment
# ---------------------------------------------------------------------------


def test_audio_id_threads_through_enroll(voices):
    app = make_app(voices)
    r, a = solo_recording(ALICE, tid="t-alice")
    result = app.add_speaker(r, wav_bytes(a), "speaker_0", "Alice", audio_id="af-1")
    assert result.segments and all(s.audio_id == "af-1" for s in result.segments)

    clip = make_voice_audio(12.0, BOB)
    direct = app.add_speaker_from_audio(wav_bytes(clip), "Bob", "en", audio_id="af-2")
    assert direct.segments and all(s.audio_id == "af-2" for s in direct.segments)


def test_audio_id_threads_through_identify_and_proposals(voices):
    app = make_app(voices)
    r, a = solo_recording(ALICE, tid="t-alice")
    app.add_speaker(r, wav_bytes(a), "speaker_0", "Alice")

    resp, audio = solo_recording(ALICE, spans=((0.0, 4.0),), tid="t-q")
    match = app.identify(resp, wav_bytes(audio), audio_id="af-q").speakers["speaker_0"]
    assert match.segments and all(s.audio_id == "af-q" for s in match.segments)

    resp2, audio2 = stranger_recording(tid="t-s")
    result = app.identify(resp2, wav_bytes(audio2), audio_id="af-s")
    proposal = result.proposed_unknowns[0]
    assert proposal.segments and all(s.audio_id == "af-s" for s in proposal.segments)
    unknown_match = result.speakers["speaker_0"]
    assert unknown_match.segments and all(s.audio_id == "af-s" for s in unknown_match.segments)


def test_audio_id_defaults_to_none(voices):
    app = make_app(voices)
    r, a = solo_recording(ALICE, tid="t-alice")
    result = app.add_speaker(r, wav_bytes(a), "speaker_0", "Alice")
    assert all(s.audio_id is None for s in result.segments)

    resp, audio = solo_recording(ALICE, spans=((0.0, 4.0),), tid="t-q")
    match = app.identify(resp, wav_bytes(audio)).speakers["speaker_0"]
    assert all(s.audio_id is None for s in match.segments)


def test_enroll_result_segments_and_ideal(voices):
    from impronta.models import composite_quality

    app = make_app(voices)
    r, a = solo_recording(ALICE, spans=((0.0, 5.0), (6.0, 11.0)), tid="t-alice")
    result = app.add_speaker(r, wav_bytes(a), "speaker_0", "Alice")
    assert len(result.segments) == result.segments_used
    assert len(result.segments) == len(result.entry_ids)
    best = max(
        result.segments,
        key=lambda s: composite_quality(s.snr_db, s.confidence, s.duration),
    )
    assert result.ideal_segment == best


def test_identify_match_carries_segments(voices):
    app = make_app(voices)
    r, a = solo_recording(ALICE, tid="t-alice")
    app.add_speaker(r, wav_bytes(a), "speaker_0", "Alice")

    resp, audio = solo_recording(ALICE, spans=((0.0, 4.0),), tid="t-q")
    match = app.identify(resp, wav_bytes(audio)).speakers["speaker_0"]
    assert len(match.segments) == match.num_segments_used
    assert match.ideal_segment_index is not None
    assert 0 <= match.ideal_segment_index < len(match.segments)
    assert match.ideal_segment in match.segments

    # unknown outcome: segments still carried, but no ideal segment
    resp2, audio2 = stranger_recording(tid="t-s")
    unknown = app.identify(resp2, wav_bytes(audio2)).speakers["speaker_0"]
    assert len(unknown.segments) == unknown.num_segments_used
    assert unknown.ideal_segment is None


def test_unidentifiable_match_has_no_segments(voices):
    app = make_app(voices)
    r, a = solo_recording(ALICE, tid="t-alice")
    app.add_speaker(r, wav_bytes(a), "speaker_0", "Alice")
    resp, audio = solo_recording(ALICE, spans=((0.0, 4.0),), logprob=-2.0, tid="t-q")
    match = app.identify(resp, wav_bytes(audio)).speakers["speaker_0"]
    assert not match.identifiable
    assert match.segments == ()
    assert match.ideal_segment is None


def test_ideal_segment_is_best_vote_scorer(voices):
    voices = dict(voices)
    NEAR_ALICE = 550.0
    voices[NEAR_ALICE] = blend(basis(0), basis(5), 0.7)
    app = make_app(voices)
    r, a = solo_recording(ALICE, tid="t-alice")
    app.add_speaker(r, wav_bytes(a), "speaker_0", "Alice")

    # one span is Alice's exact voice (sim 1.0), the other only 0.7 similar
    duration = 11.2
    audio = compose_timeline(duration, [(0.0, 5.0, NEAR_ALICE), (6.0, 11.0, ALICE)])
    words = speech_words("speaker_0", 0.0, 5.0) + speech_words("speaker_0", 6.0, 11.0)
    resp = make_scribe_response(words, language="en", transcription_id="t-mixed")
    match = app.identify(resp, wav_bytes(audio)).speakers["speaker_0"]
    assert match.display_name == "Alice"
    assert match.ideal_segment is not None
    assert match.ideal_segment.start == pytest.approx(6.0, abs=0.5)


def test_reinforcement_segments_carry_audio_id(voices):
    app = reinforce_setup(voices)
    resp, audio = solo_recording(
        SAME_VOICE_NEW_TAKE, spans=((0.0, 5.0), (6.0, 11.0)), tid="t-new-take"
    )
    proposals = app.propose_reinforcements(resp, wav_bytes(audio), audio_id="af-r")
    assert len(proposals) == 1
    assert proposals[0].segments
    assert all(s.audio_id == "af-r" for s in proposals[0].segments)
