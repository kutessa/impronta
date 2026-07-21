"""Integration tier: the real ECAPA-TDNN embedder.

Run with: uv run pytest -m integration
Downloads the speechbrain model on first run (network required).

Note: the synthetic audio here exercises the full pipeline and embedding
plumbing (shapes, determinism, self-similarity), not real speaker
discrimination — that is validated in the live tier with real recordings.
"""

import numpy as np
import pytest
from conftest import compose_timeline, make_scribe_response, make_voice_audio, speech_words

from impronta import EcapaEmbedder, FaissLocalStore, Impronta, ImprontaConfig

pytestmark = [pytest.mark.integration, pytest.mark.enable_socket]


@pytest.fixture(scope="module")
def embedder() -> EcapaEmbedder:
    return EcapaEmbedder(ImprontaConfig())


def test_embedding_shape_and_normalization(embedder):
    clips = [make_voice_audio(3.0, 440.0), make_voice_audio(2.0, 880.0)]
    rows = embedder.embed_batch(clips, 16_000)
    assert rows.shape == (2, 192)
    assert rows.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(rows, axis=1), 1.0, atol=1e-4)


def test_embedding_is_deterministic(embedder):
    clip = make_voice_audio(3.0, 440.0)
    a = embedder.embed_batch([clip], 16_000)
    b = embedder.embed_batch([clip], 16_000)
    np.testing.assert_array_equal(a, b)


def test_same_signal_more_similar_than_different_signal(embedder):
    a1 = make_voice_audio(4.0, 440.0, seed=1)
    a2 = make_voice_audio(4.0, 440.0, seed=2)  # same "voice", different take
    b = make_voice_audio(4.0, 2200.0, seed=3)  # very different signal
    rows = embedder.embed_batch([a1, a2, b], 16_000)
    same = float(np.dot(rows[0], rows[1]))
    cross = float(np.dot(rows[0], rows[2]))
    assert same > cross


def test_full_enroll_identify_happy_path(tmp_path):
    store = FaissLocalStore()
    app = Impronta(store=store, config=ImprontaConfig())

    audio = compose_timeline(6.2, [(0.0, 6.0, 440.0)])
    words = speech_words("speaker_0", 0.0, 6.0)
    resp = make_scribe_response(words, transcription_id="itx-1")

    import io

    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, audio, 16_000, format="WAV", subtype="FLOAT")
    wav = buf.getvalue()

    result = app.add_speaker(resp, wav, "speaker_0", "Alice")
    assert result.segments_used >= 1
    store.save(tmp_path)

    app2 = Impronta(store=FaissLocalStore.load(tmp_path))
    identified = app2.identify(resp, wav)
    match = identified.speakers["speaker_0"]
    assert match.display_name == "Alice"
    assert match.mean_similarity is not None and match.mean_similarity > 0.9
