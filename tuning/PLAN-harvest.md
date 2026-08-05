# Plan: use the Bosnian harvest to stress-test impronta at gallery scale

## Context

The reftimbre YouTube harvest has produced, as a by-product of ASR work, something impronta
does not otherwise have: **a large population of distinct Bosnian voices with ECAPA embeddings
already computed**.

As of 2026-07-30, and still growing:

| | |
|---|---|
| ECAPA-embedded segments | **119,242** |
| distinct speaker clusters | **2,361** (projected ~9,000 at full harvest) |
| videos | 412 indexed of 573 annotated, heading to ~3,900 |
| audio | 360 h annotated of ~1,724 h target |
| language | Bosnian — the same BCMS group impronta already special-cases |

Current defaults were tuned 2026-07-24 on **145 annotated production recordings across 33
persons**: `similarity_threshold=0.40`, recall ≈ 0.38, wrong-name ≈ 1%, chronological replay
0.39 / 0.6%. That calibration is careful and this plan does not redo it.

## The gap this addresses

**Open-set identification error grows with gallery size, and impronta has never been measured
beyond 33 enrolled people.**

Every segment is matched against every stored voiceprint and accepted above 0.40 cosine. With
33 people the nearest neighbour has 33 chances to clear that bar spuriously. With 500, it has
500. The wrong-name rate is therefore **not** a property of the threshold alone — it is a
property of the threshold *and* the gallery size, and the ≤2% budget was measured at one end of
that curve.

This matters commercially: the failure appears as customers grow. An account with 20 staff
behaves like the calibration set; an account with 400 may quietly exceed the error budget with
no code change and nothing in the logs to explain it.

The harvest answers this directly, because thousands of distinct voices is exactly what is
needed and exactly what 145 production recordings cannot supply.

## Design: harvest voices as DISTRACTORS, not as ground truth

**The harvest's speaker labels come from agglomerative clustering of ECAPA embeddings.** Using
them as ground truth to evaluate an ECAPA-based identifier would be circular — the same
embedding model produced both the labels and the predictions, so the result would measure
self-consistency rather than accuracy.

So they are never used as ground truth. Instead:

- **Ground truth stays the 145 human-annotated production recordings.** Unchanged.
- **Harvest speakers are injected into the store as distractors.** For that role the cluster
  labels only need to be *different people from the enrolled staff*, which is a far weaker
  claim than "correctly grouped" and is safe even at the measured over-fragmentation.
- Over-fragmentation is conservative here: it inflates the apparent number of distractors
  slightly, which if anything overstates gallery pressure rather than hiding it.

The measurement is then: **hold the evaluation set fixed, grow the gallery, watch the curve.**

## Phase 1 — export a distractor gallery

New `tuning/import_harvest.py`.

Source is already on hasta and needs no recomputation:

```
/mnt/ntfs/yt/spk-emb/<video_id>.npz     192-d ECAPA embeddings, L2-normalised
/mnt/ntfs/yt/speakers.jsonl             {video_id, start, end, speaker, channel}
```

⚠️ **Verify the embeddings are interchangeable before trusting any of this.** Both sides use
`speechbrain/spkrec-ecapa-voxceleb` at 192-d, but reftimbre truncates each segment to an **8 s
centre slice** and impronta uses **1–10 s utterance segments** with its own SNR and confidence
gates. Check cosine agreement on a handful of clips embedded both ways; if they diverge, re-embed
through impronta's own `embedder.py` rather than reusing the cache. Cheap to check, expensive to
get wrong.

Emit distractor speakers at impronta's own `max_embeddings_per_speaker=75` cap, applying the
same quality ordering used for real enrolments so the distractors are not unrealistically clean.

## Phase 2 — the gallery-size curve

New `tuning/gallery_scaling.py`, reusing `evaluate.py`'s scorer unchanged so numbers stay
comparable to the shipped figures.

For N in {0, 50, 100, 250, 500, 1000, 2500}:

1. build the store from the 33 real persons **plus N harvest distractors**
2. run the existing static and chronological evaluations over the 145 annotated recordings
3. record recall, wrong-name rate, unknown rate

N=0 must reproduce the shipped **recall 0.38 / wrong-name 1%** exactly — that is the control,
and if it does not reproduce, the harness is wrong and nothing after it means anything.

Repeat each N over several random distractor draws; with 2,361 available the variance between
draws is itself informative.

**The deliverable is a curve, not a number:** at what gallery size does wrong-name cross 2% at
the current threshold, and what threshold holds the budget at each size?

## Phase 3 — a size-aware threshold, only if the curve demands one

If wrong-name degrades materially with N, the fix is not a new constant but a threshold that
adapts. Options in increasing order of intrusiveness:

- **Document a recommended threshold per gallery size** — zero code, immediately useful, and
  honest about a limit that currently is not stated anywhere.
- **Score normalisation** (z-norm / adaptive s-norm against a cohort). Standard in speaker
  verification precisely because it stabilises thresholds as galleries grow; the harvest is an
  ideal cohort set.
- **Margin-based acceptance** — require the top match to beat the runner-up by a margin, not
  just clear an absolute bar. Directly targets the failure mode, since large galleries produce
  more near-ties.

Test all three through the existing scorer, against the same 145 recordings.

## Phase 4 — recall, the other standing weakness

**Recall 0.38 means nearly two thirds of speakers go unidentified.** That is the number most
visible to a user, and the threshold is held high to protect the wrong-name budget — so recall
and precision are currently traded against each other on a single axis.

If Phase 3 delivers score normalisation or a margin rule, it should buy recall at fixed
wrong-name rate rather than only defending the current operating point. Worth measuring
explicitly, because it is the difference between "we made it safer at scale" and "we made it
better".

## What this plan deliberately does not do

- **No recalibration of `similarity_threshold` on harvest data.** It is Bosnian podcast and
  broadcast audio; production is meetings and calls. The 2026-07-24 calibration is on the right
  distribution and should not be diluted by the wrong one.
- **No claim about harvest identification accuracy.** Circular, as above.
- **No new embedding model.** Out of scope; this is a scaling question, not an architecture one.

## Verification

1. **N=0 reproduces the shipped numbers.** Non-negotiable control.
2. Embedding-compatibility check (Phase 1) passes, or embeddings are regenerated.
3. Distractor identities never overlap real enrolled persons — assert on the store.
4. Multiple random draws per N, with spread reported, so a single unlucky draw is not read as a
   trend.
5. `pytest` stays green in impronta.

## Cost

Phases 1–2 are CPU-only: the embeddings exist, and evaluation is vector search over a store
that already has a FAISS backend. The expensive input — 119k embeddings over 2,361 voices — was
produced for free by ASR work that was happening anyway.

The harvest keeps growing (~3 new speakers per video, ~2,800 speakers now, ~9,000 projected), so
the curve can be extended later without redoing anything.
