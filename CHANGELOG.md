# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Vote: unmatched segments now weigh `duration x similarity_threshold` into
  the unknown bucket (was `duration x 1.0`, giving "not sure" ~2x the weight
  of a genuine match — measured cause of most missed identifications).
- Reinforcement hardening from 40-seed randomized paired replay of
  production data: new `reinforce_threshold` (0.7, stricter than
  merge_threshold — wrong-voice absorptions cluster at 0.60-0.70) and
  `reinforce_min_profile_entries` (3 — thin profiles never self-train).
  At the old 0.6 bar, 11% of reinforcement commits harvested the wrong
  person's voice.

- Data-driven default recalibration against 145 annotated production
  recordings (grid + Optuna + chronological replay; wrong-name ≤ 2%
  objective): `similarity_threshold` 0.5 → 0.40, `min_segment_confidence`
  0.5 → 0.66, SNR low rung 10 → 12 dB, `snr_floor_db` 3 → −2,
  `min_kept_speech_sec` 15 → 7, `search_k` 8 → 10, `gray_zone_margin`
  0.1 → 0.08, `reinforce_margin` 0.15 → 0.12, `cohesion_threshold`
  0.2 → 0.3, `min_proposal_segments` 2 → 3, `max_embeddings_per_enroll`
  20 → 30, `max_embeddings_per_speaker` 100 → 75. Measured effect:
  known-speaker recall 0.22 → 0.38 at ~1% wrong-name rate.

### Added

- Segment provenance on results: `add_speaker`, `add_speaker_from_audio`,
  `identify`, and `propose_reinforcements` accept an opaque caller-supplied
  `audio_id` that is stamped on every `Segment`/`SegmentInfo`, so segments
  can be traced back to the source recording in an external audio service.
- `EnrollResult.segments` (aligned 1:1 with `entry_ids`) and
  `EnrollResult.ideal_segment` — the enrolled segment with the best
  composite quality.
- `SpeakerMatch.segments` (the segments used in the vote) and
  `SpeakerMatch.ideal_segment` — the segment whose above-threshold
  similarity to the winner was highest (`VoteOutcome.best_segment_index`).
- `Segment` and `SegmentInfo` are now exported from the package root;
  `SegmentInfo.from_segment()` conversion helper.
- Serialization stays backward compatible: `from_dict` reads all new keys
  with defaults, so pre-refactor payloads (e.g. queued proposals) still load.
- Sphinx documentation site under `docs/` (furo theme, MyST markdown) with a
  `docs` dependency group: `uv run --group docs sphinx-build -b html docs
  docs/_build/html`.
- Embed-free pipeline seams for replay/tuning: `measure_segments` /
  `select_segments` / `assemble_prepared` (pipeline) and `enroll_prepared` /
  `identify_prepared` / `propose_reinforcements_prepared` (API). Behavior
  unchanged; enables threshold tuning to run production code from a
  precomputed embedding cache.
- `import impronta` no longer loads faiss (lazy `FaissLocalStore`) — lighter
  imports, and embedding-only processes escape the macOS faiss/torch
  OpenMP conflict entirely.

### Previous unreleased changes

- Profile reinforcement: `propose_reinforcements()` / `commit_reinforcements()`
  harvest confidently-identified segments back into profiles (gated on
  `merge_threshold`, `reinforce_margin`, per-segment score, and novelty).
- Language equivalence groups (`ImprontaConfig.language_groups`): BCMS codes
  (`bos`/`hrv`/`srp`/`cnr` + 2-letter forms) normalize to `hbs` at enroll and
  query time.
- Speech-duration rescue in the SNR filter (`min_kept_speech_sec`,
  `snr_floor_db`): telephony audio no longer starves profiles.
- Multi-candidate voting (`search_k`): every close speaker is credited per
  segment; near-ties surface in `candidates`.
- Enrollment outlier filter (`enroll_outlier_margin`): diarization
  misattribution guard.

### Changed

- `similarity_threshold` default 0.30 → 0.5, `merge_threshold` 0.45 → 0.6,
  gray zone widened to [0.4, 0.5) — calibrated on production call recordings.
- `cohesion_threshold` default 0.5 → 0.2 (measured on real phone recordings).

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
