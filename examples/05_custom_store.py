"""Writing your own VectorStore backend (template for Firestore/pgvector).

Implement the VectorStore ABC, then prove the implementation with the
shipped contract suite — the exact tests impronta's built-in stores pass.

Usage:
    uv run python examples/05_custom_store.py         # demo run
    uv run pytest examples/05_custom_store.py         # run the contract suite
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from impronta import SearchFilter, SearchHit, StoreEntry, VectorStore
from impronta.models import SpeakerSummary
from impronta.store.base import UNSET
from impronta.store.memory import summarize_speakers
from impronta.testing import VectorStoreContractSuite


class DictStore(VectorStore):
    """A minimal conforming backend: one dict per namespace.

    For a real Firestore backend: one document per entry under
    ``tenants/{namespace}/speaker_entries/{entry_id}``, load a namespace's
    embeddings per request (or use Firestore vector search), and wrap
    update/rename in a transaction.
    """

    def __init__(self) -> None:
        self._ns: dict[str, dict[str, StoreEntry]] = {}

    def add(self, namespace, entries):
        bucket = self._ns.setdefault(namespace, {})
        for e in entries:
            bucket[e.entry_id] = replace(e, namespace=namespace)

    def search(self, embedding, namespaces, k=5, filter=None):
        q = np.asarray(embedding, dtype=np.float32).ravel()
        hits = [
            SearchHit(entry=e, score=float(np.dot(q, e.embedding)))
            for ns in namespaces
            for e in self._ns.get(ns, {}).values()
            if filter is None or filter.matches(e)
        ]
        hits.sort(key=lambda h: (-h.score, h.entry.speaker_key, h.entry.entry_id))
        return hits[:k]

    def get_speaker_entries(self, namespace, speaker_key):
        return [e for e in self._ns.get(namespace, {}).values() if e.speaker_key == speaker_key]

    def delete_entries(self, namespace, entry_ids):
        bucket = self._ns.get(namespace, {})
        return sum(1 for eid in entry_ids if bucket.pop(eid, None) is not None)

    def delete_speaker(self, namespace, speaker_key):
        ids = [e.entry_id for e in self.get_speaker_entries(namespace, speaker_key)]
        return self.delete_entries(namespace, ids)

    def update_speaker(self, namespace, speaker_key, *, new_key=None, display_name=UNSET):
        n = 0
        for e in self._ns.get(namespace, {}).values():
            if e.speaker_key == speaker_key:
                if new_key is not None:
                    e.speaker_key = new_key
                if display_name is not UNSET:
                    e.display_name = display_name
                n += 1
        return n

    def remove_entries(self, namespace, transcription_id):
        ids = [
            e.entry_id
            for e in self._ns.get(namespace, {}).values()
            if e.source_transcription_id == transcription_id
        ]
        return self.delete_entries(namespace, ids)

    def list_speakers(self, namespaces) -> list[SpeakerSummary]:
        entries = [e for ns in namespaces for e in self._ns.get(ns, {}).values()]
        return summarize_speakers(entries)

    def count(self, namespace):
        return len(self._ns.get(namespace, {}))

    def delete_namespace(self, namespace):
        return len(self._ns.pop(namespace, {}))


class TestDictStoreContract(VectorStoreContractSuite):
    """Running `pytest examples/05_custom_store.py` proves conformance."""

    def make_store(self) -> VectorStore:
        return DictStore()


def main() -> None:
    from impronta import Impronta

    app = Impronta(store=DictStore())
    print("custom store wired into Impronta:", app.store.__class__.__name__)
    print("run `uv run pytest examples/05_custom_store.py` for the contract suite")


if __name__ == "__main__":
    main()


# Unused-import guard for the demo: SearchFilter is part of the interface a
# real backend maps to native filters (see class docstring).
_ = SearchFilter
