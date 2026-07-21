"""apply_names and format_transcript."""

from conftest import make_multichannel_response, make_scribe_response, make_words

from impronta import IdentifyResult, SpeakerMatch, apply_names, format_transcript
from impronta.naming import resolve_label


def result_for(mapping: dict[str, tuple[str | None, str | None]]) -> IdentifyResult:
    """mapping: qid -> (speaker_key, display_name)."""
    return IdentifyResult(
        speakers={
            qid: SpeakerMatch(
                query_speaker_id=qid,
                speaker_key=key,
                display_name=name,
                namespace="ns" if key else None,
                is_unknown=name is None,
                identifiable=True,
            )
            for qid, (key, name) in mapping.items()
        },
        proposed_unknowns=(),
        language_code="en",
        transcription_id="t",
    )


def two_speaker_response():
    words = make_words(
        [
            ("hello", 0.0, 0.5, "speaker_0"),
            (" ", 0.5, 0.55, None),
            ("there", 0.55, 1.0, "speaker_0"),
            ("hi", 1.5, 1.9, "speaker_1"),
            ("back", 2.0, 2.4, "speaker_0"),
        ]
    )
    words[1]["type"] = "spacing"
    return make_scribe_response(words)


def test_apply_names_adds_speaker_name_and_preserves_ids():
    resp = two_speaker_response()
    named = apply_names(resp, result_for({"speaker_0": ("alice", "Alice")}))
    w = named["words"][0]
    assert w["speaker_id"] == "speaker_0"
    assert w["speaker_name"] == "Alice"
    # unmatched speaker gets a fallback label; original response untouched
    assert named["words"][3]["speaker_name"] == "Speaker speaker_1"
    assert "speaker_name" not in resp["words"][0]


def test_apply_names_unknown_with_key():
    named = apply_names(
        two_speaker_response(), result_for({"speaker_0": ("unknown-abc123", None)})
    )
    assert named["words"][0]["speaker_name"] == "Unknown (unknown-abc123)"


def test_format_transcript_groups_turns():
    resp = two_speaker_response()
    text = format_transcript(
        resp, result_for({"speaker_0": ("alice", "Alice"), "speaker_1": ("bob", "Bob")})
    )
    assert text.splitlines() == ["Alice: hello there", "Bob: hi", "Alice: back"]


def test_format_transcript_fallback_labels():
    text = format_transcript(two_speaker_response(), result_for({}))
    assert text.splitlines()[0].startswith("Speaker speaker_0:")


def test_multichannel_naming():
    resp = make_multichannel_response(
        [
            make_words([("left", 0.0, 1.2, "speaker_0")]),
            make_words([("right", 0.0, 1.2, "speaker_0")]),
        ]
    )
    result = result_for({"0:speaker_0": ("alice", "Alice"), "1:speaker_0": ("bob", "Bob")})
    named = apply_names(resp, result)
    assert named["transcripts"][0]["words"][0]["speaker_name"] == "Alice"
    assert named["transcripts"][1]["words"][0]["speaker_name"] == "Bob"
    text = format_transcript(resp, result)
    assert text.splitlines() == ["Alice: left", "Bob: right"]


def test_resolve_label_precedence():
    assert resolve_label(None, "s9") == "Speaker s9"
    named = result_for({"q": ("alice", "Alice")}).speakers["q"]
    assert resolve_label(named, "q") == "Alice"
