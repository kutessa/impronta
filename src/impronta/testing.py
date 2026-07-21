"""Test utilities shipped with impronta.

The centerpiece is :class:`VectorStoreContractSuite`: a pytest test class
that any :class:`~impronta.store.base.VectorStore` implementation must pass.
impronta's built-in stores run it in CI; if you write your own backend
(Firestore, pgvector, Qdrant, ...), subclass the suite and implement
``make_store``::

    from impronta.testing import VectorStoreContractSuite

    class TestMyFirestoreStore(VectorStoreContractSuite):
        def make_store(self):
            return MyFirestoreStore(client=fake_firestore())

This module imports only numpy at import time; pytest is required only to
actually run the suite.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from .models import SearchFilter, StoreEntry
from .store.base import VectorStore

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def unit_vector(dim: int, axis: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[axis] = 1.0
    return v


def make_entry(
    entry_id: str,
    speaker_key: str,
    embedding: np.ndarray,
    *,
    display_name: str | None = "?unset?",
    language: str = "en",
    confidence: float = 0.9,
    source: str = "scribe_enroll",
    source_transcription_id: str | None = None,
    created_at: datetime | None = None,
    snr_db: float | None = 25.0,
    duration_sec: float | None = 3.0,
) -> StoreEntry:
    """Build a StoreEntry with sensible defaults for tests."""
    if display_name == "?unset?":
        display_name = speaker_key.replace("-", " ").title()
    return StoreEntry(
        entry_id=entry_id,
        speaker_key=speaker_key,
        display_name=display_name,
        language=language,
        confidence=confidence,
        embedding=np.asarray(embedding, dtype=np.float32),
        created_at=created_at or _T0,
        source=source,
        source_transcription_id=source_transcription_id,
        snr_db=snr_db,
        duration_sec=duration_sec,
    )


class VectorStoreContractSuite:
    """Conformance tests every VectorStore implementation must pass."""

    DIM = 8

    def make_store(self) -> VectorStore:
        raise NotImplementedError("subclass and return a fresh empty store")

    # -- helpers ------------------------------------------------------------

    def _seeded(self) -> VectorStore:
        store = self.make_store()
        e = [unit_vector(self.DIM, i) for i in range(self.DIM)]
        store.add(
            "ws:1",
            [
                make_entry("a1", "alice", e[0], language="en", source_transcription_id="t1"),
                make_entry("a2", "alice", e[1], language="en", source_transcription_id="t2"),
                make_entry("b1", "bob", e[2], language="en"),
                make_entry("b2", "bob", e[3], language="bs"),
                make_entry("u1", "unknown-x", e[4], display_name=None, language="en",
                           source="auto_enroll"),
            ],
        )
        store.add(
            "user:9",
            [
                make_entry("c1", "carol", e[5], language="en"),
                make_entry("u2", "unknown-y", e[6], display_name=None, language="en",
                           source="auto_enroll"),
            ],
        )
        return store

    # -- add / search -------------------------------------------------------

    def test_add_and_count(self):
        store = self._seeded()
        assert store.count("ws:1") == 5
        assert store.count("user:9") == 2
        assert store.count("missing") == 0

    def test_add_stamps_namespace(self):
        store = self.make_store()
        store.add("ws:1", [make_entry("x", "alice", unit_vector(self.DIM, 0))])
        entry = store.get_speaker_entries("ws:1", "alice")[0]
        assert entry.namespace == "ws:1"

    def test_add_is_upsert_by_entry_id(self):
        store = self.make_store()
        v0, v1 = unit_vector(self.DIM, 0), unit_vector(self.DIM, 1)
        store.add("ws:1", [make_entry("x", "alice", v0)])
        store.add("ws:1", [make_entry("x", "alice", v1)])
        assert store.count("ws:1") == 1
        hits = store.search(v1, ["ws:1"], k=1)
        assert hits[0].score > 0.99

    def test_search_topk_ordering(self):
        store = self.make_store()
        base = unit_vector(self.DIM, 0)
        for i, sim in enumerate([0.5, 0.9, 0.7]):
            vec = sim * base + np.sqrt(1 - sim**2) * unit_vector(self.DIM, i + 1)
            store.add("ws:1", [make_entry(f"e{i}", f"spk{i}", vec.astype(np.float32))])
        hits = store.search(base, ["ws:1"], k=3)
        scores = [round(h.score, 2) for h in hits]
        assert scores == [0.9, 0.7, 0.5]

    def test_search_respects_k(self):
        store = self._seeded()
        assert len(store.search(unit_vector(self.DIM, 0), ["ws:1"], k=2)) == 2

    def test_search_empty_store(self):
        store = self.make_store()
        assert store.search(unit_vector(self.DIM, 0), ["ws:1"], k=3) == []

    # -- namespaces ---------------------------------------------------------

    def test_namespace_isolation(self):
        store = self._seeded()
        hits = store.search(unit_vector(self.DIM, 5), ["ws:1"], k=10)
        assert all(h.entry.speaker_key != "carol" for h in hits)

    def test_multi_namespace_merge(self):
        store = self._seeded()
        hits = store.search(unit_vector(self.DIM, 5), ["ws:1", "user:9"], k=1)
        assert hits[0].entry.speaker_key == "carol"
        assert hits[0].entry.namespace == "user:9"

    def test_delete_namespace(self):
        store = self._seeded()
        removed = store.delete_namespace("ws:1")
        assert removed == 5
        assert store.count("ws:1") == 0
        assert store.count("user:9") == 2

    # -- filters ------------------------------------------------------------

    def test_language_hard_filter(self):
        store = self._seeded()
        hits = store.search(
            unit_vector(self.DIM, 3), ["ws:1"], k=10, filter=SearchFilter(language="en")
        )
        assert all(h.entry.language == "en" for h in hits)
        assert all(h.entry.entry_id != "b2" for h in hits)

    def test_unknown_only_filter(self):
        store = self._seeded()
        hits = store.search(
            unit_vector(self.DIM, 0),
            ["ws:1", "user:9"],
            k=10,
            filter=SearchFilter(unknown_only=True),
        )
        assert hits and all(h.entry.display_name is None for h in hits)

    def test_exclude_speaker_keys(self):
        store = self._seeded()
        hits = store.search(
            unit_vector(self.DIM, 0),
            ["ws:1"],
            k=10,
            filter=SearchFilter(exclude_speaker_keys=frozenset({"alice"})),
        )
        assert all(h.entry.speaker_key != "alice" for h in hits)

    # -- speaker ops --------------------------------------------------------

    def test_get_speaker_entries(self):
        store = self._seeded()
        assert {e.entry_id for e in store.get_speaker_entries("ws:1", "alice")} == {"a1", "a2"}

    def test_delete_speaker(self):
        store = self._seeded()
        assert store.delete_speaker("ws:1", "bob") == 2
        assert store.get_speaker_entries("ws:1", "bob") == []
        assert store.count("ws:1") == 3

    def test_delete_entries(self):
        store = self._seeded()
        assert store.delete_entries("ws:1", ["a1", "nope"]) == 1
        assert store.count("ws:1") == 4

    def test_update_display_name(self):
        store = self._seeded()
        n = store.update_speaker("ws:1", "alice", display_name="Alice L.")
        assert n == 2
        assert all(
            e.display_name == "Alice L." for e in store.get_speaker_entries("ws:1", "alice")
        )

    def test_rename_merges_into_existing_speaker(self):
        store = self._seeded()
        n = store.update_speaker("ws:1", "unknown-x", new_key="alice", display_name="Alice")
        assert n == 1
        assert len(store.get_speaker_entries("ws:1", "alice")) == 3
        assert store.get_speaker_entries("ws:1", "unknown-x") == []

    def test_remove_entries_by_transcription(self):
        store = self._seeded()
        assert store.remove_entries("ws:1", "t1") == 1
        assert {e.entry_id for e in store.get_speaker_entries("ws:1", "alice")} == {"a2"}

    # -- listing ------------------------------------------------------------

    def test_list_speakers(self):
        store = self._seeded()
        summaries = {s.speaker_key: s for s in store.list_speakers(["ws:1", "user:9"])}
        assert summaries["alice"].num_embeddings == 2
        assert summaries["bob"].languages == ("bs", "en")
        assert summaries["unknown-x"].is_unknown
        assert summaries["carol"].namespaces == ("user:9",)

    def test_list_speakers_created_at_is_earliest(self):
        store = self.make_store()
        early = _T0
        late = _T0 + timedelta(days=3)
        store.add(
            "ws:1",
            [
                make_entry("x1", "alice", unit_vector(self.DIM, 0), created_at=late),
                make_entry("x2", "alice", unit_vector(self.DIM, 1), created_at=early),
            ],
        )
        (summary,) = store.list_speakers(["ws:1"])
        assert summary.created_at == early
