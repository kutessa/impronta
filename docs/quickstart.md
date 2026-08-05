# Quickstart

## Install

```bash
uv add impronta            # or: pip install impronta
uv add "impronta[m4a]"     # optional: m4a/aac decoding via PyAV
```

Supported audio out of the box: wav, flac, ogg, mp3 (bundled libsndfile — no system dependencies). Python ≥ 3.10.

:::{tip}
On Linux/cloud, install torch from the CPU wheel index to avoid multi-GB CUDA wheels: add `--index https://download.pytorch.org/whl/cpu` for the torch install, or configure `[tool.uv.sources]` in your app.
:::

## Enroll, identify, name

```python
from impronta import Impronta, FaissLocalStore

app = Impronta(store=FaissLocalStore())

# 1. Enroll: "speaker_0 in this recording is Tarik"
scribe_response = elevenlabs_client.speech_to_text.convert(...)  # scribe_v2, diarize=True
app.add_speaker(scribe_response, "meeting1.wav", speaker_id="speaker_0", name="Tarik")

# 2. Identify speakers in a new recording
result = app.identify(new_scribe_response, "meeting2.wav")
print(result.name_map())          # {"speaker_0": "Tarik", "speaker_1": None}

# 3. Feed real names into your summary prompt
from impronta import format_transcript
print(format_transcript(new_scribe_response, result))
# Tarik: let's ship it on friday
# Speaker speaker_1: sounds good

# 4. Persist the speaker db
app.store.save("./speaker_db")
```

The `examples/` directory in the repository contains runnable scripts covering every workflow.

## Tracing results back to the audio

Every enroll/identify call accepts an opaque `audio_id` — an id from *your* audio storage that impronta never resolves, only carries. Results then expose exactly which spans of that recording were used:

```python
result = app.add_speaker(resp, audio, "speaker_0", "Tarik", audio_id="rec-123")
for seg in result.segments:            # SegmentInfo: start, end, confidence, snr_db, audio_id
    print(seg.audio_id, seg.start, seg.end)
print(result.ideal_segment)            # highest-quality enrolled segment

match = app.identify(resp2, audio2, audio_id="rec-124").speakers["speaker_0"]
print(match.segments)                  # segments used in the vote
print(match.ideal_segment)             # the segment that matched the winner strongest
```

This is the fastest way to audit a decision by ear: fetch `audio_id` from your storage, seek to `ideal_segment.start`, and listen.

## Full example

```{literalinclude} ../examples/01_enroll_and_identify.py
:language: python
```
