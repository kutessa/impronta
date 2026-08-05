# Unknown speakers: propose → commit → label

`identify()` **never writes to the store**. Strangers come back as serializable proposals; your app decides what to persist:

```python
result = app.identify(resp, audio)
keys = app.commit_unknowns(result.proposed_unknowns)   # idempotent, dedup-first
# later, once you learn who they are:
app.label_speaker(keys[0], "Sarah")                    # promotes unknown → named
```

## Safeguards baked in

- **Proposal gating** — a stranger is only proposed with enough usable segments (`min_proposal_segments`), medium+ SNR quality, and self-consistent embeddings (a cohesion check catches diarization merging two people into one `speaker_id`). Gated-out voices are reported with a `no_proposal_reason`.
- **Gray zone** — a voice scoring just under the match threshold against a known speaker is never auto-proposed (prevents one person fragmenting into two records); `near_misses` are reported so you can ask the user instead.
- **Dedup at commit** — the same stranger appearing in Monday's and Tuesday's recordings stays one record, even when both are processed in parallel.
- **Idempotency** — entry ids are deterministic in `(transcription_id, speaker_id)`; retried jobs upsert instead of duplicating.
- **Provenance** — every embedding records its source transcription; `app.remove_transcription(tid)` surgically rolls back a bad recording or mislabel, and `app.prune_unknowns(older_than=...)` clears stale never-labeled strangers. Proposals also carry the segments they were built from (with `audio_id` when supplied), so a human can listen before labeling.

## Full example

```{literalinclude} ../../examples/02_unknowns_workflow.py
:language: python
```
