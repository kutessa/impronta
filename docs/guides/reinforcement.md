# Profile reinforcement

Profiles never grow from `identify()` — but confidently-matched audio is the best enrollment data there is. The explicit reinforcement pass harvests it safely:

```python
proposals = app.propose_reinforcements(resp, audio)   # read-only, separate pass
app.commit_reinforcements(proposals)                  # idempotent
```

A speaker is only proposed when ALL gates hold:

- the match clears `reinforce_threshold` (0.7 — deliberately stricter than the 0.6 `merge_threshold`; measured wrong-voice absorptions cluster at 0.60–0.70 similarity);
- no other stored speaker comes within `reinforce_margin` (0.12) of the winner — contested matches never feed profiles;
- the winner's profile already has at least `reinforce_min_profile_entries` (3) stored embeddings — thin profiles are channel fingerprints, not voiceprints, and never self-train;
- each harvested segment individually scores ≥ `reinforce_threshold` against the profile while adding novelty (< 0.95 to every existing entry).

Entries are tagged `source="reinforce"` with their transcription id, so `remove_transcription()` reverses a bad recording, and the per-speaker cap still bounds growth. One wrong absorption would drag a profile toward the wrong voice and compound — hence the strictness.
