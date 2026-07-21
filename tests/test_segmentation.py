"""Word-merge segmentation: explicit cases + hypothesis invariants."""

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from impronta import ImprontaConfig
from impronta.models import Word
from impronta.segmentation import segment_windows, segment_words

CFG = ImprontaConfig()


def word(text, start, end, sid="s0", type="word", logprob=-0.05):
    return Word(text=text, start=start, end=end, type=type, speaker_id=sid, logprob=logprob)


def test_merges_across_small_gaps():
    words = (word("a", 0.0, 0.4), word("b", 0.7, 1.1), word("c", 1.2, 1.6))
    segs = segment_words(words, CFG)["s0"]
    assert len(segs) == 1
    assert segs[0].start == 0.0 and segs[0].end == 1.6


def test_splits_on_large_gap():
    words = (word("a", 0.0, 1.2), word("b", 2.5, 3.7))
    segs = segment_words(words, CFG)["s0"]
    assert len(segs) == 2


def test_splits_at_max_duration():
    words = tuple(word(f"w{i}", i * 0.5, i * 0.5 + 0.45) for i in range(50))  # 25s of speech
    segs = segment_words(words, CFG)["s0"]
    assert len(segs) >= 2
    assert all(s.duration <= CFG.max_segment_sec + 1e-9 for s in segs)


def test_drops_short_segments():
    words = (word("a", 0.0, 0.4),)  # 0.4s < 1s minimum
    assert segment_words(words, CFG) == {}


def test_skips_spacing_and_audio_events():
    words = (
        word("a", 0.0, 1.2),
        word(" ", 1.2, 1.3, type="spacing"),
        word("(laughs)", 1.3, 2.0, type="audio_event"),
        word("b", 1.4, 2.4),
    )
    segs = segment_words(words, CFG)["s0"]
    assert len(segs) == 1  # spacing/event ignored; a..b merged via gap tolerance


def test_null_timed_words_skipped_without_breaking_merge():
    words = (word("a", 0.0, 1.0), word("x", None, None), word("b", 1.3, 2.3))
    segs = segment_words(words, CFG)["s0"]
    assert len(segs) == 1
    assert segs[0].end == 2.3


def test_null_speaker_skipped():
    words = (word("a", 0.0, 2.0, sid=None),)
    assert segment_words(words, CFG) == {}


def test_confidence_is_geometric_mean_probability():
    words = (word("a", 0.0, 1.0, logprob=-0.2), word("b", 1.1, 2.0, logprob=-0.4))
    segs = segment_words(words, CFG)["s0"]
    assert segs[0].confidence == pytest.approx(math.exp(-0.3))


def test_confidence_defaults_to_one_without_logprobs():
    words = (word("a", 0.0, 1.0, logprob=None), word("b", 1.1, 2.0, logprob=None))
    segs = segment_words(words, CFG)["s0"]
    assert segs[0].confidence == 1.0


def test_interleaved_speakers_stay_separate():
    words = (
        word("a", 0.0, 1.2, sid="s0"),
        word("b", 1.3, 2.5, sid="s1"),
        word("c", 2.6, 3.8, sid="s0"),
    )
    result = segment_words(words, CFG)
    assert set(result) == {"s0", "s1"}
    # s0's two words are 1.4s apart -> separate segments
    assert len(result["s0"]) == 2


def test_windows_cover_duration():
    wins = segment_windows(12.5, CFG)
    assert [round(w.duration, 2) for w in wins] == [5.0, 5.0, 2.5]


def test_windows_drop_tiny_tail():
    wins = segment_windows(10.4, CFG)  # tail of 0.4s < min_segment_sec
    assert len(wins) == 2


# ---------------------------------------------------------------------------
# hypothesis invariants
# ---------------------------------------------------------------------------

word_strategy = st.builds(
    word,
    text=st.just("w"),
    start=st.one_of(st.none(), st.floats(0, 60, allow_nan=False)),
    end=st.one_of(st.none(), st.floats(0, 60, allow_nan=False)),
    sid=st.sampled_from(["s0", "s1", None]),
    type=st.sampled_from(["word", "spacing", "audio_event"]),
    logprob=st.one_of(st.none(), st.floats(-5, 0, allow_nan=False)),
)


@given(st.lists(word_strategy, max_size=60))
def test_segmentation_invariants(words):
    result = segment_words(tuple(words), CFG)
    for sid, segs in result.items():
        # time-ordered, non-overlapping, duration within bounds
        for a, b in zip(segs, segs[1:], strict=False):
            assert a.end <= b.start + 1e-9
        for s in segs:
            assert s.speaker_id == sid
            assert CFG.min_segment_sec - 1e-9 <= s.duration <= CFG.max_segment_sec + 1e-9
            assert 0.0 <= s.confidence <= 1.0
