# impronta threshold tuning

Data-driven tuning of `ImprontaConfig` thresholds against annotated
production recordings. Everything here is repo tooling — none of it ships in
the wheel.

## The pipeline at a glance

```
data/ (bucket dump)          audio + raw Scribe v2 responses (never copied)
     │  tuning/build_manifest.py
dataset/manifest.jsonl       inventory + container timestamps (chronology)
     │  tuning/precompute_cache.py           <- the only torch step
dataset/cache/{uid}/{rid}.npz    embeddings for ALL structural segments
     │  tuning/render_annotation_tasks.py
dataset/ls_media/ + ls_tasks.json            <- per-speaker snippet WAVs
     │  Label Studio (human)  ->  export JSON
     │  tuning/labelstudio_export.py
dataset/annotations.jsonl    ground truth: (rid, speaker) -> person + quality
     │
tuning/evaluate.py           score one config (static / chrono protocols)
tuning/search.py             Stage A grid -> Stage B Optuna
tuning/report.py             report.md + heatmap + Pareto + winner validation
```

Core invariant: **the optimizer replays production code.** Trials call the
real `select_segments` / vote / gates / commit / reinforcement paths via
`Impronta(store=InMemoryStore(), ...)` and the `*_prepared` entry points —
only the embedding step is cached. There is no parallel implementation that
can drift. Segmentation knobs (`min/max_segment_sec`, `gap_tolerance_sec`)
are frozen: they define cache identity and are excluded from the search
(the cache loader enforces this).

## Workflow

```bash
# 1. inventory + timestamp gate
uv run python tuning/build_manifest.py

# 2. embed everything once (resumable; ~30h audio, expect well under an hour)
uv run python -u tuning/precompute_cache.py

# 3. render annotation snippets + tasks
uv run python tuning/render_annotation_tasks.py

# 4. annotate in Label Studio
uv tool install label-studio           # once
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=$PWD/dataset/ls_media \
  label-studio start
# create an access token (Account & Settings), then:
LS_TOKEN=... uv run python tuning/labelstudio_import.py
# ... annotate ... then export the project as JSON and:
uv run python tuning/labelstudio_export.py export.json

# 5. sanity: current defaults on the annotated data
uv run python tuning/evaluate.py            # add --chrono for the simulation

# 6. search
uv run --group tuning python tuning/search.py grid
uv run --group tuning python tuning/search.py optuna --trials 500

# 7. report + winner validation
uv run --group tuning python tuning/report.py --holdout-cluster <cluster>
```

## Annotation rules

- One task per (recording, diarized speaker): listen to the snippet, then
  - **person**: a lowercase-slug key, stable across recordings AND accounts
    (`tarik`, `ahmed`, `guest-cafe-1`). Grow `dataset/persons.json` as you
    go — the export validator rejects unknown keys.
  - **quality**: `clean` = one person's voice · `mixed` = diarization merged
    two people under this speaker id · `garbage` = noise/TV/music/no voice.
- SNR is never annotated — it is always the computed WADA estimate.
- `persons.json` needs `cluster` per person (the ~3 disjoint social groups;
  `guests` for one-offs) and `owner_of` (account uids) for the holdout split.

## Objective and protocols

- **Objective**: wrong-name rate (wrong person named + stranger named)
  ≤ **2%** of emitted judgments; maximize known-speaker recall under it.
  `no_usable` rows count against recall when the person was enrolled — a
  config cannot win by filtering all audio away.
- **Static protocol** (optimizer inner loop): per account, each person is
  enrolled from their chronologically first `clean` appearance; every later
  clean row is scored. Chronological 70/30 tune/test split per account.
- **Chronological simulation** (winner validation only): full canto replay —
  identify → auto-commit unknowns → simulated user labeling at first clean
  appearance → reinforcement with auto-commit. Reports store growth, unknown
  proliferation, and profile contamination (write-time provenance spanning
  ≥2 ground-truth persons).
- Winner must hold up on: test bucket, held-out cluster, chrono simulation —
  and prefer flat regions of the Stage A landscape over sharp optima (2% of
  a few hundred rows is 3–8 errors; the Wilson CIs in every report exist
  for a reason).

## Caveats

- Recording timestamps come from the MP4 container `creation_time`
  (validated per-account by build_manifest); a Firestore export can override
  via `timestamp_source` later.
- The chrono simulation's step order (identify → commit → label → reinforce)
  is an assumption about canto's orchestration — verify against the real
  backend before trusting simulation conclusions.
- Applying a winning config to `ImprontaConfig` defaults is a deliberate,
  reviewed step — never automatic.
