# impronta

Speaker identification for [ElevenLabs Scribe v2](https://elevenlabs.io/docs/api-reference/speech-to-text/convert) transcripts. Enroll known voices once, then map every diarized `speaker_id` in future recordings to a real name — backed by ECAPA-TDNN voiceprints and a pluggable vector store.

*Impronta* (it.): fingerprint.

```
scribe v2 response + audio
        │
        ├── merge words → utterance segments (1–10 s)
        ├── drop low-confidence segments (ElevenLabs logprob)
        ├── drop noisy segments (blind WADA-SNR, 20→15→12 dB relaxation)
        ├── ECAPA-TDNN embeddings (192-d, CPU)
        │
   enroll ──→ store (name, embedding, language, confidence)
   identify ─→ nearest-neighbour per segment → duration×similarity vote
        │
        └──→ { speaker_id → name }, ranked candidates, unknown proposals
```

Key design invariants:

- `identify()` **never writes to the store** — strangers come back as serializable proposals your app decides to persist.
- Entry ids are deterministic, so retried jobs upsert instead of duplicating.
- Every result carries its evidence: matches and enrollments expose the segments used, the ideal segment, and an optional caller-supplied `audio_id` for tracing back to source audio.

```{toctree}
:maxdepth: 2
:caption: Usage

quickstart
guides/unknown-speakers
guides/reinforcement
guides/multitenancy
guides/custom-stores
guides/configuration
limitations
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
changelog
```
