"""Both built-in stores pass the shipped contract suite, plus
FaissLocalStore-specific persistence and concurrency behavior."""

import json
import threading

import pytest

from impronta import ImprontaError, InMemoryStore
from impronta.store.faiss_local import _SIDECAR, FaissLocalStore
from impronta.testing import VectorStoreContractSuite, make_entry, unit_vector

DIM = VectorStoreContractSuite.DIM


class TestInMemoryStoreContract(VectorStoreContractSuite):
    def make_store(self):
        return InMemoryStore()


class TestFaissLocalStoreContract(VectorStoreContractSuite):
    def make_store(self):
        return FaissLocalStore()


# ---------------------------------------------------------------------------
# FaissLocalStore persistence
# ---------------------------------------------------------------------------


def seeded_faiss() -> FaissLocalStore:
    store = FaissLocalStore()
    store.add(
        "ws:1",
        [
            make_entry("a1", "alice", unit_vector(DIM, 0), source_transcription_id="t1"),
            make_entry("b1", "bob", unit_vector(DIM, 1), language="bs"),
            make_entry("u1", "unknown-x", unit_vector(DIM, 2), display_name=None),
        ],
    )
    return store


def test_save_load_roundtrip(tmp_path):
    store = seeded_faiss()
    store.save(tmp_path)
    loaded = FaissLocalStore.load(tmp_path)
    assert loaded.count("ws:1") == 3
    original = {e.entry_id: e for e in store.get_speaker_entries("ws:1", "alice")}
    restored = {e.entry_id: e for e in loaded.get_speaker_entries("ws:1", "alice")}
    assert restored == original  # includes exact float32 embedding equality


def test_search_results_identical_after_reload(tmp_path):
    store = seeded_faiss()
    q = unit_vector(DIM, 0)
    before = [(h.entry.entry_id, round(h.score, 6)) for h in store.search(q, ["ws:1"], k=3)]
    store.save(tmp_path)
    loaded = FaissLocalStore.load(tmp_path)
    after = [(h.entry.entry_id, round(h.score, 6)) for h in loaded.search(q, ["ws:1"], k=3)]
    assert before == after


def test_atomic_save_preserves_previous_file_on_failure(tmp_path, monkeypatch):
    store = seeded_faiss()
    store.save(tmp_path)

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr("impronta.store.faiss_local.os.replace", boom)
    with pytest.raises(OSError):
        store.save(tmp_path)
    monkeypatch.undo()
    # previous save is intact and loadable; temp file cleaned up
    assert FaissLocalStore.load(tmp_path).count("ws:1") == 3
    assert [p.name for p in tmp_path.iterdir()] == [_SIDECAR]


def test_unknown_format_version_raises_clear_error(tmp_path):
    (tmp_path / _SIDECAR).write_text(json.dumps({"format_version": 99, "entries": {}}))
    with pytest.raises(ImprontaError, match="format_version"):
        FaissLocalStore.load(tmp_path)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(ImprontaError, match="no saved store"):
        FaissLocalStore.load(tmp_path)


def test_index_rebuild_after_mutation():
    store = seeded_faiss()
    q = unit_vector(DIM, 0)
    assert store.search(q, ["ws:1"], k=1)[0].entry.speaker_key == "alice"
    store.delete_speaker("ws:1", "alice")
    hits = store.search(q, ["ws:1"], k=3)
    assert all(h.entry.speaker_key != "alice" for h in hits)


def test_concurrent_add_and_search_smoke():
    store = FaissLocalStore()
    errors: list[Exception] = []

    def writer(offset: int) -> None:
        try:
            for i in range(30):
                axis = (offset + i) % DIM
                store.add(
                    "ws:1",
                    [make_entry(f"e{offset}-{i}", f"spk{axis}", unit_vector(DIM, axis))],
                )
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    def reader() -> None:
        try:
            for _ in range(60):
                store.search(unit_vector(DIM, 0), ["ws:1"], k=3)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(k,)) for k in range(3)] + [
        threading.Thread(target=reader) for _ in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors
    assert store.count("ws:1") == 90
