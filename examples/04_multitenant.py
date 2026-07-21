"""Hierarchical multi-tenant namespaces: workspace db + user db.

Every store operation is namespace-scoped. Reads search a LIST of
namespaces (workspace + user layered; best score wins); writes go to
exactly one. Tenants are fully isolated from each other.

This example runs entirely offline against the in-memory store with a fake
embedder shim — it demonstrates the namespace API, not audio processing.

Usage:
    uv run python examples/04_multitenant.py
"""

import numpy as np

from impronta import Impronta, InMemoryStore
from impronta.models import SegmentInfo, UnknownProposal


class OneVectorEmbedder:
    """Toy embedder for the demo (any clip -> the same unit vector)."""

    dim = 4

    def embed_batch(self, clips, sample_rate):
        row = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return np.stack([row] * len(clips))


def proposal(tid: str, vec: list[float]) -> UnknownProposal:
    rows = np.asarray([vec, vec], dtype=np.float32)
    return UnknownProposal(
        query_speaker_id="speaker_0",
        transcription_id=tid,
        language="en",
        suggested_key=f"unknown-{tid}",
        embeddings=rows,
        segments=tuple(
            SegmentInfo(start=i, end=i + 2.0, confidence=0.9, snr_db=25.0) for i in range(2)
        ),
        quality_tier="high",
    )


def main() -> None:
    store = InMemoryStore()  # one shared backend, many tenants

    # workspace-level agent: shared speaker db for the whole team
    ws = Impronta(store=store, embedder=OneVectorEmbedder(), write_namespace="ws:acme")
    ws.commit_unknowns([proposal("meeting-1", [1.0, 0.0, 0.0, 0.0])])
    ws.label_speaker("unknown-meeting-1", "Alice")

    # user-level agent: private additions, reads workspace + own namespace
    me = Impronta(
        store=store,
        embedder=OneVectorEmbedder(),
        write_namespace="user:tarik",
        read_namespaces=["ws:acme", "user:tarik"],
    )
    me.commit_unknowns([proposal("private-call-1", [0.0, 1.0, 0.0, 0.0])])

    print("user sees:", [(s.speaker_key, s.namespaces) for s in me.list_speakers()])
    # -> Alice from ws:acme AND the private unknown from user:tarik

    # a different workspace sees nothing
    other = Impronta(store=store, embedder=OneVectorEmbedder(), write_namespace="ws:rival")
    print("other tenant sees:", other.list_speakers())  # []

    # GDPR: wipe a user's biometric data entirely
    removed = me.wipe_namespace("user:tarik")
    print(f"wiped user:tarik ({removed} embeddings deleted)")


if __name__ == "__main__":
    main()
