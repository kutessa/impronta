# Multi-tenancy and late fusion

## Namespaces

Namespaces scope every store operation. Reads merge across a list (e.g. workspace + user); writes target one:

```python
app = Impronta(
    store=store,
    write_namespace="user:42",
    read_namespaces=["ws:acme", "user:42"],   # best score wins across both
)
app.wipe_namespace("user:42")   # full biometric-data deletion for GDPR
```

:::{warning}
Voice embeddings are biometric data (GDPR special category) — collect consent in your product and wire `wipe_namespace` into account deletion.
:::

## Late fusion

Every match exposes ranked `candidates` with normalized score shares, so you can combine voice evidence with other signals (calendar attendees, "John, could you…" mentions):

```python
match = result.speakers["speaker_0"]
for c in match.candidates:
    print(c.speaker_key, c.display_name, f"{c.score_share:.0%}", c.mean_similarity)
```

## Full example

```{literalinclude} ../../examples/04_multitenant.py
:language: python
```
