"""Local FAISS-backed store with a JSON sidecar.

One ``IndexFlatIP`` per (namespace, language) pair makes the language hard
filter exact and free — no over-fetch heuristics for the common case. Raw
embeddings live on the entries themselves, so indexes are cheap to rebuild
and are NOT persisted: ``save`` writes a single JSON file (embeddings as
base64 float32) atomically via ``os.replace``, and ``load`` rebuilds indexes
from it. This removes the index/metadata consistency failure mode entirely.

Thread-safety: an internal lock serializes mutation and search. Cross-process
concurrent writes are out of scope for a local file store — use a
server-backed :class:`~impronta.store.base.VectorStore` implementation for
that.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import faiss
import numpy as np

from ..exceptions import ImprontaError
from ..models import SearchFilter, SearchHit, SpeakerSummary, StoreEntry
from .base import UNSET, VectorStore
from .memory import summarize_speakers

FORMAT_VERSION = 1
_SIDECAR = "impronta_store.json"


class FaissLocalStore(VectorStore):
    def __init__(self) -> None:
        self._entries: dict[str, dict[str, StoreEntry]] = {}
        # (namespace, language) -> (index, entry_ids row-aligned with the index)
        self._indexes: dict[tuple[str, str], tuple[faiss.Index, list[str]]] = {}
        self._dirty: set[tuple[str, str]] = set()
        self._lock = threading.RLock()

    # -- index maintenance --------------------------------------------------

    def _mark_dirty(self, namespace: str, language: str | None = None) -> None:
        if language is not None:
            self._dirty.add((namespace, language))
            return
        for key in list(self._indexes):
            if key[0] == namespace:
                self._dirty.add(key)
        # languages that have entries but no index yet get built on demand
        for e in self._entries.get(namespace, {}).values():
            self._dirty.add((namespace, e.language))

    def _index_for(self, namespace: str, language: str) -> tuple[faiss.Index, list[str]]:
        key = (namespace, language)
        if key in self._indexes and key not in self._dirty:
            return self._indexes[key]
        rows = [
            e for e in self._entries.get(namespace, {}).values() if e.language == language
        ]
        rows.sort(key=lambda e: e.entry_id)
        dim = rows[0].embedding.shape[0] if rows else 1
        index = faiss.IndexFlatIP(dim)
        if rows:
            index.add(np.stack([e.embedding for e in rows]).astype(np.float32))
        pair = (index, [e.entry_id for e in rows])
        self._indexes[key] = pair
        self._dirty.discard(key)
        return pair

    def _languages(self, namespace: str) -> set[str]:
        return {e.language for e in self._entries.get(namespace, {}).values()}

    # -- VectorStore interface ----------------------------------------------

    def add(self, namespace: str, entries: Sequence[StoreEntry]) -> None:
        with self._lock:
            ns = self._entries.setdefault(namespace, {})
            for e in entries:
                stamped = replace(e, namespace=namespace) if e.namespace != namespace else e
                old = ns.get(stamped.entry_id)
                if old is not None:
                    self._mark_dirty(namespace, old.language)
                ns[stamped.entry_id] = stamped
                self._mark_dirty(namespace, stamped.language)

    def search(
        self,
        embedding: np.ndarray,
        namespaces: Sequence[str],
        k: int = 5,
        filter: SearchFilter | None = None,
    ) -> list[SearchHit]:
        q = np.asarray(embedding, dtype=np.float32).ravel()[np.newaxis, :]
        needs_post_filter = filter is not None and (
            filter.unknown_only or filter.exclude_speaker_keys
        )
        hits: list[SearchHit] = []
        with self._lock:
            for namespace in namespaces:
                if filter is not None and filter.language is not None:
                    languages: Sequence[str] = [filter.language]
                else:
                    languages = sorted(self._languages(namespace))
                for language in languages:
                    index, entry_ids = self._index_for(namespace, language)
                    if not entry_ids:
                        continue
                    # metadata post-filters need over-fetch; worst case scan all
                    fetch = len(entry_ids) if needs_post_filter else min(k, len(entry_ids))
                    scores, rows = index.search(q, fetch)
                    ns_entries = self._entries[namespace]
                    for score, row in zip(scores[0], rows[0], strict=True):
                        if row < 0:
                            continue
                        entry = ns_entries[entry_ids[row]]
                        if filter is not None and not filter.matches(entry):
                            continue
                        hits.append(SearchHit(entry=entry, score=float(score)))
        hits.sort(key=lambda h: (-h.score, h.entry.speaker_key, h.entry.entry_id))
        return hits[:k]

    def get_speaker_entries(self, namespace: str, speaker_key: str) -> list[StoreEntry]:
        with self._lock:
            return [
                e
                for e in self._entries.get(namespace, {}).values()
                if e.speaker_key == speaker_key
            ]

    def delete_entries(self, namespace: str, entry_ids: Sequence[str]) -> int:
        with self._lock:
            ns = self._entries.get(namespace, {})
            deleted = 0
            for eid in entry_ids:
                e = ns.pop(eid, None)
                if e is not None:
                    self._mark_dirty(namespace, e.language)
                    deleted += 1
            return deleted

    def delete_speaker(self, namespace: str, speaker_key: str) -> int:
        with self._lock:
            ids = [e.entry_id for e in self.get_speaker_entries(namespace, speaker_key)]
            return self.delete_entries(namespace, ids)

    def update_speaker(
        self,
        namespace: str,
        speaker_key: str,
        *,
        new_key: str | None = None,
        display_name: object = UNSET,
    ) -> int:
        with self._lock:
            updated = 0
            for e in self._entries.get(namespace, {}).values():
                if e.speaker_key != speaker_key:
                    continue
                if new_key is not None:
                    e.speaker_key = new_key
                if display_name is not UNSET:
                    e.display_name = display_name  # type: ignore[assignment]
                updated += 1
            return updated

    def remove_entries(self, namespace: str, transcription_id: str) -> int:
        with self._lock:
            ids = [
                e.entry_id
                for e in self._entries.get(namespace, {}).values()
                if e.source_transcription_id == transcription_id
            ]
            return self.delete_entries(namespace, ids)

    def list_speakers(self, namespaces: Sequence[str]) -> list[SpeakerSummary]:
        with self._lock:
            entries: list[StoreEntry] = []
            for namespace in namespaces:
                entries.extend(self._entries.get(namespace, {}).values())
            return summarize_speakers(entries)

    def count(self, namespace: str) -> int:
        with self._lock:
            return len(self._entries.get(namespace, {}))

    def delete_namespace(self, namespace: str) -> int:
        with self._lock:
            removed = len(self._entries.get(namespace, {}))
            self._entries.pop(namespace, None)
            for key in [k for k in self._indexes if k[0] == namespace]:
                self._indexes.pop(key, None)
                self._dirty.discard(key)
            return removed

    # -- persistence (not part of the VectorStore ABC) ----------------------

    def save(self, directory: str | os.PathLike) -> Path:
        """Atomically write all entries to ``<directory>/impronta_store.json``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / _SIDECAR
        with self._lock:
            payload = {
                "format_version": FORMAT_VERSION,
                "entries": {
                    namespace: [e.to_dict() for e in ns.values()]
                    for namespace, ns in self._entries.items()
                },
            }
        fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".impronta_store_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f)
            os.replace(tmp_path, target)
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
        return target

    @classmethod
    def load(cls, directory: str | os.PathLike) -> FaissLocalStore:
        """Load a previously saved store; indexes are rebuilt on first search."""
        path = Path(directory) / _SIDECAR
        try:
            with open(path) as f:
                payload = json.load(f)
        except FileNotFoundError as exc:
            raise ImprontaError(f"no saved store at {path}") from exc
        version = payload.get("format_version")
        if version != FORMAT_VERSION:
            raise ImprontaError(
                f"saved store at {path} has format_version={version!r}; "
                f"this version of impronta reads format_version={FORMAT_VERSION}. "
                "Upgrade impronta or re-create the store."
            )
        store = cls()
        for namespace, dicts in payload.get("entries", {}).items():
            store.add(namespace, [StoreEntry.from_dict(d) for d in dicts])
        return store
