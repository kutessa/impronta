"""Pure-NumPy reference store: brute-force cosine over normalized rows.

The default store for tests and a legitimate lightweight runtime option at
per-tenant scales (hundreds of embeddings). Single-writer; not thread-safe.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from ..models import SearchFilter, SearchHit, SpeakerSummary, StoreEntry
from .base import UNSET, VectorStore


def summarize_speakers(entries: Sequence[StoreEntry]) -> list[SpeakerSummary]:
    """Group entries by speaker_key into summaries (shared by backends)."""
    grouped: dict[str, list[StoreEntry]] = {}
    for e in entries:
        grouped.setdefault(e.speaker_key, []).append(e)
    out = []
    for key in sorted(grouped):
        group = grouped[key]
        names = [e.display_name for e in group if e.display_name is not None]
        out.append(
            SpeakerSummary(
                speaker_key=key,
                display_name=names[0] if names else None,
                namespaces=tuple(sorted({e.namespace for e in group})),
                languages=tuple(sorted({e.language for e in group})),
                num_embeddings=len(group),
                created_at=min(e.created_at for e in group),
            )
        )
    return out


class InMemoryStore(VectorStore):
    def __init__(self) -> None:
        self._data: dict[str, dict[str, StoreEntry]] = {}

    def add(self, namespace: str, entries: Sequence[StoreEntry]) -> None:
        ns = self._data.setdefault(namespace, {})
        for e in entries:
            stamped = replace(e, namespace=namespace) if e.namespace != namespace else e
            ns[stamped.entry_id] = stamped

    def search(
        self,
        embedding: np.ndarray,
        namespaces: Sequence[str],
        k: int = 5,
        filter: SearchFilter | None = None,
    ) -> list[SearchHit]:
        q = np.asarray(embedding, dtype=np.float32).ravel()
        hits: list[SearchHit] = []
        for namespace in namespaces:
            for e in self._data.get(namespace, {}).values():
                if filter is not None and not filter.matches(e):
                    continue
                score = float(np.dot(q, e.embedding))
                hits.append(SearchHit(entry=e, score=score))
        hits.sort(key=lambda h: (-h.score, h.entry.speaker_key, h.entry.entry_id))
        return hits[:k]

    def get_speaker_entries(self, namespace: str, speaker_key: str) -> list[StoreEntry]:
        return [
            e for e in self._data.get(namespace, {}).values() if e.speaker_key == speaker_key
        ]

    def delete_entries(self, namespace: str, entry_ids: Sequence[str]) -> int:
        ns = self._data.get(namespace, {})
        deleted = 0
        for eid in entry_ids:
            if eid in ns:
                del ns[eid]
                deleted += 1
        return deleted

    def delete_speaker(self, namespace: str, speaker_key: str) -> int:
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
        ns = self._data.get(namespace, {})
        updated = 0
        for e in list(ns.values()):
            if e.speaker_key != speaker_key:
                continue
            if new_key is not None:
                e.speaker_key = new_key
            if display_name is not UNSET:
                e.display_name = display_name  # type: ignore[assignment]
            updated += 1
        return updated

    def remove_entries(self, namespace: str, transcription_id: str) -> int:
        ns = self._data.get(namespace, {})
        ids = [
            e.entry_id for e in ns.values() if e.source_transcription_id == transcription_id
        ]
        return self.delete_entries(namespace, ids)

    def list_speakers(self, namespaces: Sequence[str]) -> list[SpeakerSummary]:
        entries: list[StoreEntry] = []
        for namespace in namespaces:
            entries.extend(self._data.get(namespace, {}).values())
        return summarize_speakers(entries)

    def count(self, namespace: str) -> int:
        return len(self._data.get(namespace, {}))

    def delete_namespace(self, namespace: str) -> int:
        removed = len(self._data.get(namespace, {}))
        self._data.pop(namespace, None)
        return removed
