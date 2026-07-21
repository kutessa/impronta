"""Scribe v2 response parsing."""

from types import SimpleNamespace

import pytest
from conftest import make_multichannel_response, make_scribe_response, make_words

from impronta import ScribeParseError
from impronta.scribe import derive_transcription_id, parse_scribe_response


def test_mono_parse():
    resp = make_scribe_response(make_words([("hi", 0.0, 0.4, "speaker_0")]))
    (t,) = parse_scribe_response(resp)
    assert t.language_code == "en"
    assert t.transcription_id == "tx-1"
    assert t.channel_index is None
    assert len(t.words) == 1
    assert t.words[0].speaker_id == "speaker_0"


def test_multichannel_parse():
    resp = make_multichannel_response(
        [
            make_words([("a", 0.0, 0.3, "speaker_0")]),
            make_words([("b", 0.0, 0.3, "speaker_0")]),
        ]
    )
    transcripts = parse_scribe_response(resp)
    assert len(transcripts) == 2
    assert [t.channel_index for t in transcripts] == [0, 1]
    assert all(t.transcription_id == "tx-multi" for t in transcripts)


def test_sdk_object_duck_typing_model_dump():
    resp = make_scribe_response(make_words([("hi", 0.0, 0.4, "s0")]))
    obj = SimpleNamespace(model_dump=lambda: resp)
    (t,) = parse_scribe_response(obj)
    assert t.words[0].text == "hi"


def test_sdk_object_duck_typing_dict_method():
    resp = make_scribe_response(make_words([("hi", 0.0, 0.4, "s0")]))
    obj = SimpleNamespace(dict=lambda: resp)
    (t,) = parse_scribe_response(obj)
    assert t.words[0].text == "hi"


def test_garbage_input_raises():
    with pytest.raises(ScribeParseError):
        parse_scribe_response(42)
    with pytest.raises(ScribeParseError):
        parse_scribe_response({"unrelated": True})
    with pytest.raises(ScribeParseError):
        parse_scribe_response({"transcripts": []})


def test_missing_words_is_empty_not_crash():
    (t,) = parse_scribe_response({"language_code": "en", "text": "", "words": None})
    assert t.words == ()


def test_null_fields_preserved():
    words = make_words([("hm", None, None, None)])
    words[0]["logprob"] = None
    (t,) = parse_scribe_response(make_scribe_response(words))
    w = t.words[0]
    assert w.start is None and w.end is None and w.speaker_id is None and w.logprob is None


def test_missing_language_defaults_to_und():
    (t,) = parse_scribe_response({"words": make_words([("x", 0.0, 0.2, "s0")])})
    assert t.language_code == "und"


def test_transcription_id_fallback_is_deterministic():
    words = make_words([("hi", 0.0, 0.4, "s0"), ("yo", 0.5, 0.9, "s1")])
    r1 = make_scribe_response(words, transcription_id=None)
    r2 = make_scribe_response(words, transcription_id=None)
    (t1,) = parse_scribe_response(r1)
    (t2,) = parse_scribe_response(r2)
    assert t1.transcription_id == t2.transcription_id
    assert t1.transcription_id.startswith("derived-")
    # different content -> different id
    r3 = make_scribe_response(make_words([("other", 0.0, 0.4, "s0")]), transcription_id=None)
    (t3,) = parse_scribe_response(r3)
    assert t3.transcription_id != t1.transcription_id


def test_golden_fixture_parses():
    """Pin the parser against a checked-in schema-exact response.

    The live tier (tests/test_live.py) refreshes this fixture from a real
    Scribe v2 call, so schema drift shows up here as a default-tier failure.
    """
    import json
    from pathlib import Path

    fixture = Path(__file__).parent / "fixtures" / "scribe_v2_golden.json"
    (t,) = parse_scribe_response(json.loads(fixture.read_text()))
    assert t.language_code
    assert t.transcription_id
    assert len(t.speaker_ids()) >= 2
    types = {w.type for w in t.words}
    assert "word" in types  # spacing/audio_event tolerated, words required


def test_derive_transcription_id_ignores_word_tail_beyond_50():
    words = make_words([(f"w{i}", i * 0.1, i * 0.1 + 0.05, "s0") for i in range(60)])
    (t,) = parse_scribe_response(make_scribe_response(words, transcription_id=None))
    assert t.transcription_id  # smoke: derivation handles >50 words
    assert derive_transcription_id({"language_code": "en"}, t.words) == t.transcription_id
