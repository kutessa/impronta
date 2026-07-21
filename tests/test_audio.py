"""Audio decoding, resampling, channel handling, slicing."""

import io

import numpy as np
import pytest
import soundfile as sf
from conftest import SR, make_voice_audio

from impronta import AudioDecodeError
from impronta.audio import DecodedAudio, load_audio, select_channel, slice_seconds


def wav_bytes(samples: np.ndarray, sr: int = SR) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="FLOAT")
    return buf.getvalue()


def test_load_from_bytes():
    audio = load_audio(wav_bytes(make_voice_audio(1.0, 440.0)))
    assert audio.sample_rate == SR
    assert audio.num_channels == 1
    assert audio.duration == pytest.approx(1.0, abs=0.01)


def test_load_from_path(tmp_path):
    path = tmp_path / "clip.wav"
    path.write_bytes(wav_bytes(make_voice_audio(0.5, 440.0)))
    audio = load_audio(str(path))
    assert audio.duration == pytest.approx(0.5, abs=0.01)


def test_load_from_pathlike_and_filelike(tmp_path):
    path = tmp_path / "clip.wav"
    data = wav_bytes(make_voice_audio(0.5, 440.0))
    path.write_bytes(data)
    assert load_audio(path).num_channels == 1
    assert load_audio(io.BytesIO(data)).num_channels == 1


def test_resample_44k_to_16k():
    x = make_voice_audio(1.0, 440.0, sr=44_100)
    audio = load_audio(wav_bytes(x, sr=44_100), target_sr=SR)
    assert audio.sample_rate == SR
    assert audio.samples.shape[1] == pytest.approx(SR, rel=0.01)


def test_stereo_kept_as_two_channels_and_downmixed():
    left = make_voice_audio(1.0, 440.0)
    right = make_voice_audio(1.0, 880.0)
    stereo = np.stack([left, right], axis=1)  # (n, 2) for soundfile
    audio = load_audio(wav_bytes(stereo))
    assert audio.num_channels == 2
    mono = select_channel(audio, None)
    assert np.allclose(mono, (left + right) / 2, atol=1e-4)
    assert np.allclose(select_channel(audio, 1), right, atol=1e-4)


def test_out_of_range_channel_falls_back_to_downmix():
    mono = make_voice_audio(1.0, 440.0)
    audio = load_audio(wav_bytes(mono))
    assert np.allclose(select_channel(audio, 3), mono, atol=1e-4)


def test_garbage_bytes_raise_decode_error():
    with pytest.raises(AudioDecodeError):
        load_audio(b"definitely not audio data")


def test_unsupported_input_type():
    with pytest.raises(AudioDecodeError):
        load_audio(12345)  # type: ignore[arg-type]


def test_slice_clamps_past_eof():
    x = make_voice_audio(1.0, 440.0)
    clip = slice_seconds(x, SR, 0.5, 5.0)
    assert clip.shape[0] == SR // 2
    assert slice_seconds(x, SR, 2.0, 3.0).shape[0] == 0
    assert slice_seconds(x, SR, -1.0, 0.5).shape[0] == SR // 2


@pytest.mark.skipif(
    "MPEG" not in sf.available_formats(), reason="libsndfile without mp3 support"
)
def test_mp3_decode():
    buf = io.BytesIO()
    sf.write(buf, make_voice_audio(1.0, 440.0), SR, format="MP3")
    audio = load_audio(buf.getvalue())
    assert audio.duration == pytest.approx(1.0, abs=0.1)


def test_decoded_audio_properties():
    a = DecodedAudio(samples=np.zeros((2, SR), dtype=np.float32), sample_rate=SR)
    assert a.num_channels == 2 and a.duration == 1.0
