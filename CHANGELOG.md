# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-21

### Added

- `Impronta` facade: `add_speaker`, `add_speaker_from_audio`, `identify`,
  `commit_unknowns`, `label_speaker`, `list_speakers`, `remove_speaker`,
  `remove_transcription`, `prune_unknowns`, `wipe_namespace`.
- ElevenLabs Scribe v2 parsing (mono + multichannel, SDK-object duck typing).
- Utterance segmentation with gap tolerance and min/max duration bounds.
- ElevenLabs-confidence gate and blind WADA-SNR filter with progressive
  relaxation (20/15/10 dB quality tiers).
- ECAPA-TDNN embeddings via speechbrain (lazy-loaded, process-cached).
- Duration x similarity weighted voting with ranked `candidates` output for
  late fusion.
- Read-only identification with gated `UnknownProposal`s (min segments,
  quality tier, cohesion check, gray zone) and idempotent, dedup-first
  `commit_unknowns`.
- Hierarchical multi-tenant namespaces on the `VectorStore` interface.
- `FaissLocalStore` (per-language FAISS indexes, atomic JSON persistence)
  and `InMemoryStore`.
- `impronta.testing.VectorStoreContractSuite` for validating custom store
  backends.
- `apply_names` / `format_transcript` for LLM-ready named transcripts.
- Per-speaker embedding cap with quality-based eviction; provenance
  (`source_transcription_id`) on every entry.
