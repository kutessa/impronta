# Configuration and tuning

All knobs live on {class}`~impronta.config.ImprontaConfig` — see the [full field reference](../api/config.md) for every tunable and the reasoning behind its default.

Defaults were tuned 2026-07-24 against 145 annotated production recordings (grid + Optuna search with a wrong-name-rate ≤ 2% constraint, validated on a chronological replay — see `tuning/` in the repository). At the shipped defaults: recall ≈ 0.38, wrong-name ≈ 1%, chronological replay 0.39 recall / 0.6% wrong.

The knobs you are most likely to touch:

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

Changing thresholds without measuring on *your* audio is how wrong-name rates quietly grow — the `tuning/` directory in the repository replays production code over annotated recordings so tuned numbers stay honest.
