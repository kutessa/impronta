"""Prod-audio tier: real canto recordings (m4a, real voices, real noise).

Run with: uv run pytest -m prodaudio

Requires a local directory of production recordings plus their QC csv
(default: ../reftimbre/prod-audio-over-1min relative to the repo; override
with $IMPRONTA_PROD_AUDIO_DIR). Skips entirely when absent, so CI is
unaffected. Files are selected DYNAMICALLY from audio_qc.csv by quality
criteria — no recording ids are hard-coded here.

Privacy: everything is offline. Audio is read locally, embedded locally,
and never uploaded; no transcription API is called (identify tests use
fabricated word timelines over the real audio).

What this tier buys over the synthetic integration tier:
- the PyAV m4a decode fallback on real aac AND opus files
- WADA-SNR sanity against an independent SNR estimate (the QC pipeline's)
- actual speaker discrimination: same real voice across recordings vs
  different real voices
"""

from __future__ import annotations

import csv
import io
import os
from pathlib import Path

import numpy as np
import pytest

from impronta import EcapaEmbedder, Impronta, ImprontaConfig, InMemoryStore
from impronta.audio import load_audio, select_channel
from impronta.embedder import l2_normalize
from impronta.pipeline import prepare_segments
from impronta.segmentation import segment_windows
from impronta.snr import wada_snr

pytestmark = [pytest.mark.prodaudio, pytest.mark.enable_socket]

PROD_DIR = Path(
    os.environ.get(
        "IMPRONTA_PROD_AUDIO_DIR",
        Path(__file__).parent.parent.parent / "reftimbre" / "prod-audio-over-1min",
    )
)

if not (PROD_DIR / "audio_qc.csv").exists():  # pragma: no cover
    pytest.skip(
        f"no prod audio at {PROD_DIR} (set IMPRONTA_PROD_AUDIO_DIR)",
        allow_module_level=True,
    )

pytest.importorskip("av", reason="prod audio is m4a; install the [m4a] extra / dev group")

SR = 16_000
# short-ish recordings keep the tier fast; speech-rich keeps assertions honest
MAX_DURATION_S = 400.0
MIN_SPEECH_ACTIVITY = 60.0


def qc_rows() -> list[dict]:
    with open(PROD_DIR / "audio_qc.csv") as f:
        return [r for r in csv.DictReader(f) if (PROD_DIR / r["file"]).exists()]


def speech_rich(rows: list[dict], max_duration: float = MAX_DURATION_S) -> list[dict]:
    return sorted(
        (
            r
            for r in rows
            if r["usable"] == "True"
            and float(r["speech_activity_pct"]) > MIN_SPEECH_ACTIVITY
            and float(r["duration_s"]) < max_duration
        ),
        key=lambda r: (r["uid"], float(r["duration_s"])),
    )


def users_with_two_recordings(rows: list[dict]) -> dict[str, list[dict]]:
    by_uid: dict[str, list[dict]] = {}
    for r in speech_rich(rows):
        by_uid.setdefault(r["uid"], []).append(r)
    return {uid: rs for uid, rs in sorted(by_uid.items()) if len(rs) >= 2}


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    rs = qc_rows()
    assert rs, "audio_qc.csv is empty"
    return rs


@pytest.fixture(scope="module")
def embedder() -> EcapaEmbedder:
    return EcapaEmbedder(ImprontaConfig())


def speech_slice(
    row: dict, length_s: float = 60.0, offset_s: float = 10.0
) -> tuple[np.ndarray, float]:
    """Decode a recording; return (mono 16k slice, slice offset in seconds)."""
    audio = load_audio(PROD_DIR / row["file"], SR)
    mono = select_channel(audio, None)
    start = min(offset_s, max(0.0, audio.duration - length_s))
    i = int(start * SR)
    return mono[i : i + int(length_s * SR)], start


class Probe:
    """One recording pushed through the real window pipeline."""

    def __init__(self, row: dict, clip: np.ndarray, offset: float, prepared) -> None:
        self.row = row
        self.clip = clip
        self.offset = offset
        self.prepared = prepared

    @property
    def voiceprint(self) -> np.ndarray:
        return l2_normalize(self.prepared.embeddings.mean(axis=0, keepdims=True))[0]


_PROBE_CACHE: dict[str, Probe | None] = {}


def probe_recording(row: dict, embedder: EcapaEmbedder) -> Probe | None:
    """Qualify a recording with OUR pipeline, not the QC csv's labels.

    The QC 'speech_activity_pct' is a loose VAD — keyboard-typing and
    background-noise recordings score 70%+. This gate only requires >= 3
    embeddable windows (any tier: after the rescue supplement, "low" often
    means real-but-noisy speech, while steady background noise can read
    "high" on WADA). Whether a recording contains the USER'S actual voice
    is decided by mutual-similarity pairing in :func:`two_probed_users` —
    no local heuristic can tell "faint background TV" from "the user
    speaking", but two recordings agreeing with each other can.
    """
    if row["file"] in _PROBE_CACHE:
        return _PROBE_CACHE[row["file"]]
    cfg = ImprontaConfig(max_embeddings_per_enroll=6)
    clip, offset = speech_slice(row)
    windows = segment_windows(clip.shape[0] / SR, cfg)
    prepared = prepare_segments(windows, clip, embedder, cfg)
    result = None
    if prepared.embeddings is not None and len(prepared.segments) >= 3:
        result = Probe(row, clip, offset, prepared)
    _PROBE_CACHE[row["file"]] = result
    return result


MIN_PAIR_SIMILARITY = 0.5  # aligned with the library's similarity_threshold
MAX_PROBES_PER_USER = 8


def best_pair(probes: list[Probe]) -> tuple[Probe, Probe, float] | None:
    """The two recordings of one user that agree most with each other."""
    best: tuple[Probe, Probe, float] | None = None
    for i in range(len(probes)):
        for j in range(i + 1, len(probes)):
            sim = float(np.dot(probes[i].voiceprint, probes[j].voiceprint))
            if best is None or sim > best[2]:
                best = (probes[i], probes[j], sim)
    return best


def two_probed_users(rows, embedder) -> tuple[tuple[Probe, Probe], tuple[Probe, Probe]]:
    """Two users, each represented by their most mutually-similar recording pair.

    Mutual similarity is the ground-truth proxy: a pair of recordings that
    share a voice IS the "same user" signal (prod data contains junk like
    'Empty Recording with Background Noise' under real accounts). Users are
    ranked by pair strength; cross-user similarity is never consulted, so
    the cross-user assertions remain unbiased.
    """
    candidates: list[tuple[Probe, Probe, float]] = []
    for _uid, recs in users_with_two_recordings(rows).items():
        probes = [
            p
            for r in recs[:MAX_PROBES_PER_USER]
            if (p := probe_recording(r, embedder)) is not None
        ]
        if len(probes) < 2:
            continue
        pair = best_pair(probes)
        if pair is not None and pair[2] >= MIN_PAIR_SIMILARITY:
            candidates.append(pair)
    if len(candidates) < 2:
        pytest.skip("need two users with a mutually-consistent recording pair")
    candidates.sort(key=lambda p: -p[2])
    (a1, a2, _), (b1, b2, _) = candidates[0], candidates[1]
    return (a1, a2), (b1, b2)


# ---------------------------------------------------------------------------
# decoding: the PyAV fallback on real prod codecs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("codec", ["aac", "opus"])
def test_decode_real_m4a(rows, codec):
    candidates = [r for r in speech_rich(rows, max_duration=400.0) if r["codec"] == codec]
    if not candidates:
        pytest.skip(f"no short {codec} recording in the prod set")
    row = candidates[0]
    audio = load_audio(PROD_DIR / row["file"], SR)
    assert audio.sample_rate == SR
    assert audio.duration == pytest.approx(float(row["duration_s"]), rel=0.05)
    rms = float(np.sqrt((select_channel(audio, None) ** 2).mean()))
    assert rms > 1e-4, "decoded audio is silent"


def test_decode_m4a_from_bytes(rows):
    row = speech_rich(rows)[0]
    data = (PROD_DIR / row["file"]).read_bytes()
    audio = load_audio(data, SR)
    assert audio.duration == pytest.approx(float(row["duration_s"]), rel=0.05)


# ---------------------------------------------------------------------------
# WADA-SNR vs the QC pipeline's independent estimate
# ---------------------------------------------------------------------------


def test_wada_ordering_matches_independent_qc_estimate(rows):
    usable = [r for r in rows if r["usable"] == "True" and float(r["duration_s"]) > 60]
    best = max(usable, key=lambda r: float(r["est_snr_db"]))
    worst = min(usable, key=lambda r: float(r["est_snr_db"]))
    gap = float(best["est_snr_db"]) - float(worst["est_snr_db"])
    if gap < 15.0:
        pytest.skip(f"QC SNR gap only {gap:.0f} dB — ordering assertion would be noise")
    wada_best = wada_snr(speech_slice(best, length_s=30.0)[0])
    wada_worst = wada_snr(speech_slice(worst, length_s=30.0)[0])
    assert wada_best > wada_worst, (
        f"WADA disagrees with QC: best QC file ({best['file']}, "
        f"qc={best['est_snr_db']}) -> wada {wada_best:.1f}; worst ({worst['file']}, "
        f"qc={worst['est_snr_db']}) -> wada {wada_worst:.1f}"
    )


def test_real_speech_clears_lowest_snr_tier(rows):
    """The 10 dB floor must not reject clean real-world speech."""
    best = max(speech_rich(rows), key=lambda r: float(r["est_snr_db"]))
    assert wada_snr(speech_slice(best, length_s=30.0)[0]) >= 10.0


# ---------------------------------------------------------------------------
# real speaker discrimination
# ---------------------------------------------------------------------------


def test_same_user_beats_cross_user_similarity(rows, embedder):
    (a1, a2), (b1, _b2) = two_probed_users(rows, embedder)
    same = float(np.dot(a1.voiceprint, a2.voiceprint))
    cross = float(np.dot(a1.voiceprint, b1.voiceprint))
    assert same > cross + 0.05, (
        f"same-user similarity {same:.3f} vs cross-user {cross:.3f} (files: "
        f"{a1.row['file']}, {a2.row['file']}, {b1.row['file']})"
    )


def fabricated_response(p: Probe, speaker_id: str, tid: str) -> dict:
    """A word timeline over the probe's SNR-qualified windows.

    Continuous 0.3 s words cover exactly the time ranges our pipeline
    already found to contain clean speech, so identify() re-derives
    comparable segments. (A real Scribe response would align words to
    speech naturally; this keeps the test offline.)
    """
    words = []
    i = 0
    for seg in p.prepared.segments:
        t = p.offset + seg.start
        end = p.offset + seg.end
        while t < end:
            words.append(
                {
                    "text": f"w{i}",
                    "start": round(t, 3),
                    "end": round(min(t + 0.3, end), 3),
                    "type": "word",
                    "speaker_id": speaker_id,
                    "logprob": -0.05,
                }
            )
            t += 0.3
            i += 1
    return {
        "language_code": "eng",
        "text": "<fabricated>",
        "words": words,
        "transcription_id": tid,
    }


def wav_bytes(clip: np.ndarray) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, clip, SR, format="WAV", subtype="FLOAT")
    return buf.getvalue()


def test_enroll_then_identify_across_recordings(rows, embedder):
    """Enroll two real users, identify one of user A's OTHER recordings."""
    (a1, a2), (b1, _b2) = two_probed_users(rows, embedder)
    app = Impronta(store=InMemoryStore(), embedder=embedder)

    app.add_speaker_from_audio(wav_bytes(a1.clip), "User A", "eng")
    app.add_speaker_from_audio(wav_bytes(b1.clip), "User B", "eng")

    resp = fabricated_response(a2, "speaker_0", tid="prod-query-a2")
    result = app.identify(resp, PROD_DIR / a2.row["file"])
    match = result.speakers["speaker_0"]

    assert match.identifiable, f"no usable segments in {a2.row['file']}"
    assert match.display_name == "User A", (
        f"expected User A, got {match.display_name!r} "
        f"(candidates: {[(c.speaker_key, round(c.score_share, 2)) for c in match.candidates]}, "
        f"mean_sim: {match.mean_similarity}, files: {a1.row['file']} -> {a2.row['file']})"
    )
    # and the wrong user must not out-score the right one
    shares = {c.speaker_key: c.score_share for c in match.candidates}
    assert shares.get("user-a", 0) > shares.get("user-b", 0)
