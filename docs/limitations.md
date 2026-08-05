# Limitations and caveats

## Gallery size and the wrong-name rate

**Open-set identification error grows with gallery size, and impronta has never been measured beyond 33 enrolled people.**

Every segment is matched against every stored voiceprint and accepted above the similarity threshold. With 33 enrolled people, the nearest neighbour has 33 chances to clear that bar spuriously; with 500, it has 500. The wrong-name rate is therefore **not** a property of the threshold alone — it is a property of the threshold *and* the gallery size, and the ≤ 2% error budget was measured at one end of that curve.

This matters as deployments grow: an account with 20 staff behaves like the calibration set; an account with 400 may quietly exceed the error budget with no code change and nothing in the logs to explain it. If your gallery grows well past the calibration scale, re-measure on your own audio before trusting the shipped thresholds.

## Language

- **Language hard filter**: a person enrolled only in English will *not* match when speaking Bosnian. Enroll per language (same `speaker_key`), or pass `identify(..., language_filter=False)`. `SpeakerSummary.languages` shows coverage.
- **Language equivalence groups**: codes that are phonetically one language are normalized before filtering. By default BCMS (`bos`/`hrv`/`srp`/`cnr`/`bs`/`hr`/`sr`) collapses to the `hbs` macrolanguage — STT language detection flips between those codes recording to recording, and without grouping the hard filter would silently split one speaker db into three. Extend via `ImprontaConfig(language_groups=...)`.

## Everything else

- Two different people sharing a display name need explicit distinct `speaker_key`s (the default key is a slug of the name).
- The ElevenLabs `logprob` gates segments and ranks enrollment quality, but does not weight votes — transcription confidence measures the words, not the voice.
- Deciding which namespace a stranger from a *private* recording is committed into is a privacy decision your app must make consciously.
- `FaissLocalStore` is single-process (internal thread lock, no cross-process coordination) — use a server-backed store for concurrent writers.
- **macOS dev machines**: the pip wheels of faiss-cpu and torch each bundle their own OpenMP runtime, which crashes the process when both are used. Set `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1` when running the real embedder locally (the test suite sets this automatically). Linux wheels — i.e. production — are unaffected.
