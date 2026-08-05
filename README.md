# impronta

Speaker identification for [ElevenLabs Scribe v2](https://elevenlabs.io/docs/api-reference/speech-to-text/convert) transcripts. Enroll known voices once, then map every diarized `speaker_id` in future recordings to a real name — backed by ECAPA-TDNN voiceprints and a pluggable vector store.

*Impronta* (it.): fingerprint.

```
scribe v2 response + audio
        │
        ├── merge words → utterance segments (1–10 s)
        ├── drop low-confidence segments (ElevenLabs logprob)
        ├── drop noisy segments (blind WADA-SNR, 20→15→10 dB relaxation)
        ├── ECAPA-TDNN embeddings (192-d, CPU)
        │
   enroll ──→ store (name, embedding, language, confidence)
   identify ─→ nearest-neighbour per segment → duration×similarity vote
        │
        └──→ { speaker_id → name }, ranked candidates, unknown proposals
```

## Install

```bash
uv add impronta            # or: pip install impronta
uv add "impronta[m4a]"     # optional: m4a/aac decoding via PyAV
```

Supported audio out of the box: wav, flac, ogg, mp3 (bundled libsndfile — no system dependencies). Python ≥ 3.10.

> **Linux/cloud tip:** install torch from the CPU wheel index to avoid multi-GB CUDA wheels: add `--index https://download.pytorch.org/whl/cpu` for the torch install, or configure `[tool.uv.sources]` in your app.

## Quickstart

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

See [`examples/`](examples/) for runnable scripts covering every workflow, and the full documentation (guides, configuration reference, API docs) under [`docs/`](docs/) — build it with `uv run --group docs sphinx-build -b html docs docs/_build/html`.

Every result carries its evidence: pass an opaque `audio_id` into `add_speaker`/`identify` and results expose the exact segments used (`result.segments`, `match.segments`) plus the single best one (`.ideal_segment`), so you can pull the source audio and listen to what a decision was based on.

## Unknown speakers: propose → commit → label

`identify()` **never writes to the store**. Strangers come back as serializable proposals; your app decides what to persist:

```python
result = app.identify(resp, audio)
keys = app.commit_unknowns(result.proposed_unknowns)   # idempotent, dedup-first
# later, once you learn who they are:
app.label_speaker(keys[0], "Sarah")                    # promotes unknown → named
```

Safeguards baked in:

- **Proposal gating** — a stranger is only proposed with ≥ 2 usable segments, medium+ SNR quality, and self-consistent embeddings (a cohesion check catches diarization merging two people into one `speaker_id`). Gated-out voices are reported with a `no_proposal_reason`.
- **Gray zone** — a voice scoring just under the match threshold against a known speaker is never auto-proposed (prevents one person fragmenting into two records); `near_misses` are reported so you can ask the user instead.
- **Dedup at commit** — the same stranger appearing in Monday's and Tuesday's recordings stays one record, even when both are processed in parallel.
- **Idempotency** — entry ids are deterministic in `(transcription_id, speaker_id)`; retried jobs upsert instead of duplicating.
- **Provenance** — every embedding records its source transcription; `app.remove_transcription(tid)` surgically rolls back a bad recording or mislabel, and `app.prune_unknowns(older_than=...)` clears stale never-labeled strangers.

## Profile reinforcement

Profiles never grow from `identify()` — but confidently-matched audio is the best enrollment data there is. The explicit reinforcement pass harvests it safely:

```python
proposals = app.propose_reinforcements(resp, audio)   # read-only, separate pass
app.commit_reinforcements(proposals)                  # idempotent
```

A speaker is only proposed when the match clears `merge_threshold` (0.6, the profile-mutating bar, not the 0.5 match bar), no other stored speaker comes within `reinforce_margin` (0.15) of the winner, and each harvested segment individually scores ≥ 0.6 against the profile while adding novelty (< 0.95 to every existing entry). Entries are tagged `source="reinforce"` with their transcription id, so `remove_transcription()` reverses a bad recording, and the per-speaker cap still bounds growth. Contested or weak matches never feed profiles — one wrong absorption would drag a profile toward the wrong voice and compound.

## Multi-tenancy

Namespaces scope every store operation. Reads merge across a list (e.g. workspace + user); writes target one:

```python
app = Impronta(
    store=store,
    write_namespace="user:42",
    read_namespaces=["ws:acme", "user:42"],   # best score wins across both
)
app.wipe_namespace("user:42")   # full biometric-data deletion for GDPR
```

Voice embeddings are biometric data (GDPR special category) — collect consent in your product and wire `wipe_namespace` into account deletion.

## Late fusion

Every match exposes ranked `candidates` with normalized score shares, so you can combine voice evidence with other signals (calendar attendees, "John, could you…" mentions):

```python
match = result.speakers["speaker_0"]
for c in match.candidates:
    print(c.speaker_key, c.display_name, f"{c.score_share:.0%}", c.mean_similarity)
```

## Writing your own store backend

v1 ships `FaissLocalStore` (local directory, atomic JSON persistence) and `InMemoryStore`. For Firestore/pgvector/Qdrant, implement the `VectorStore` ABC (`impronta/store/base.py`) — all operations are namespace-aware and take a `SearchFilter` (language equality, unknown-only, key exclusion) you can map to native filters. Then prove it with the shipped contract suite:

```python
from impronta.testing import VectorStoreContractSuite

class TestMyFirestoreStore(VectorStoreContractSuite):
    def make_store(self):
        return MyFirestoreStore(client=emulator_client())
```

[`examples/05_custom_store.py`](examples/05_custom_store.py) is a complete ~60-line template.

## Deployment (Cloud Run / workers)

- **Bake the model into your image** — ECAPA weights (~80 MB) download from HuggingFace on first use. In your Dockerfile:
  ```dockerfile
  ENV IMPRONTA_CACHE_DIR=/models
  RUN python -c "from impronta import EcapaEmbedder; EcapaEmbedder().embed_batch([__import__('numpy').zeros(16000, 'float32')], 16000)"
  ```
- **One instance per process** — the model is lazily loaded and process-cached; construct `Impronta` once at startup, not per request.
- CPU is fine: ~50–200 ms per segment embedding.

## Tuning

All knobs live on `ImprontaConfig` (see its docstring for the full reference):

Defaults were tuned 2026-07-24 against 145 annotated production recordings (grid + Optuna search with a wrong-name-rate ≤ 2% constraint, validated on a chronological replay — see `tuning/`). At the shipped defaults: recall ≈ 0.38, wrong-name ≈ 1%, chronological replay 0.39 recall / 0.6% wrong.

| Knob | Default | What it does |
|---|---|---|
| `similarity_threshold` | 0.40 | min cosine for a match; below → unknown. Measured: errors ≈ 0 down to 0.42, cross 2% just below 0.36, explode at 0.34 |
| `merge_threshold` | 0.6 | stricter bar for merges/dedup (destructive ops) |
| `snr_tiers` | 20/15/12 dB | progressive SNR relaxation ladder (quality labels) |
| `snr_floor_db` | −2 | absolute floor — only near-silence is unrescuable |
| `min_kept_speech_sec` | 7 s | rescue target: top up with best remaining segments when the tiers keep less speech than this |
| `min_segment_confidence` | 0.66 | ElevenLabs-confidence gate (hard) |
| `min_segment_sec` / `max_segment_sec` | 1 / 10 s | utterance segment bounds (frozen — embedding-cache identity) |
| `enroll_outlier_margin` | 0.3 | drop enrollment segments far below the batch-median agreement (diarization misattribution guard) |
| `max_embeddings_per_speaker` | 75 | per-speaker cap (quality-based eviction) |

## Caveats

- **Language hard filter**: a person enrolled only in English will *not* match when speaking Bosnian. Enroll per language (same `speaker_key`), or pass `identify(..., language_filter=False)`. `SpeakerSummary.languages` shows coverage.
- **Language equivalence groups**: codes that are phonetically one language are normalized before filtering. By default BCMS (`bos`/`hrv`/`srp`/`cnr`/`bs`/`hr`/`sr`) collapses to the `hbs` macrolanguage — STT language detection flips between those codes recording to recording, and without grouping the hard filter would silently split one speaker db into three. Extend via `ImprontaConfig(language_groups=...)`.
- Two different people sharing a display name need explicit distinct `speaker_key`s (the default key is a slug of the name).
- The ElevenLabs `logprob` gates segments and ranks enrollment quality, but does not weight votes — transcription confidence measures the words, not the voice.
- Deciding which namespace a stranger from a *private* recording is committed into is a privacy decision your app must make consciously.
- `FaissLocalStore` is single-process (internal thread lock, no cross-process coordination) — use a server-backed store for concurrent writers.
- **macOS dev machines**: the pip wheels of faiss-cpu and torch each bundle their own OpenMP runtime, which crashes the process when both are used. Set `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1` when running the real embedder locally (the test suite sets this automatically). Linux wheels — i.e. production — are unaffected.

## Development

```bash
uv sync
uv run pytest                  # hermetic default tier: no network, no torch import, <15 s
uv run pytest -m integration   # real ECAPA (downloads the model)
uv run pytest -m live          # real ElevenLabs API; needs ELEVENLABS_API_KEY + IMPRONTA_LIVE_AUDIO
uv run ruff check src tests && uv run mypy src/impronta
uv run --group docs sphinx-build -b html docs docs/_build/html   # build the docs
```

## License

MIT
