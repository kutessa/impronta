# Writing your own store backend

Impronta ships `FaissLocalStore` (local directory, atomic JSON persistence) and `InMemoryStore`. For Firestore/pgvector/Qdrant, implement the {class}`~impronta.store.base.VectorStore` ABC — all operations are namespace-aware and take a `SearchFilter` (language equality, unknown-only, key exclusion) you can map to native filters. Then prove it with the shipped contract suite:

```python
from impronta.testing import VectorStoreContractSuite

class TestMyFirestoreStore(VectorStoreContractSuite):
    def make_store(self):
        return MyFirestoreStore(client=emulator_client())
```

## Deployment (Cloud Run / workers)

- **Bake the model into your image** — ECAPA weights (~80 MB) download from HuggingFace on first use. In your Dockerfile:

  ```dockerfile
  ENV IMPRONTA_CACHE_DIR=/models
  RUN python -c "from impronta import EcapaEmbedder; EcapaEmbedder().embed_batch([__import__('numpy').zeros(16000, 'float32')], 16000)"
  ```

- **One instance per process** — the model is lazily loaded and process-cached; construct `Impronta` once at startup, not per request.
- CPU is fine: ~50–200 ms per segment embedding.

## Full example

A complete ~60-line template:

```{literalinclude} ../../examples/05_custom_store.py
:language: python
```
