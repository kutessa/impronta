"""The pluggable vector-store interface.

All operations are namespace-aware: a namespace is an opaque tenant scope
(e.g. ``ws:<workspace_id>`` or ``user:<user_id>``). Reads accept a *list* of
namespaces and merge results across them (hierarchical tenancy: workspace +
user layered, best score wins); writes always target exactly one namespace.

``add`` is an upsert keyed by ``entry_id`` within a namespace — this is what
makes retried commits idempotent. Persistence is deliberately NOT part of the
interface (server-backed stores don't save to disk); ``FaissLocalStore`` adds
``save``/``load`` on top.

Backends should implement metadata filtering (:class:`SearchFilter`)
natively where possible (SQL WHERE, payload filters); the reference
implementations post-filter in memory.

To validate a custom backend, run it through
``impronta.testing.VectorStoreContractSuite``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from ..models import SearchFilter, SearchHit, SpeakerSummary, StoreEntry

#: Sentinel distinguishing "don't change display_name" from "set it to None".
UNSET: object = object()


class VectorStore(ABC):
    """Abstract speaker-embedding store. See module docstring for semantics."""

    @abstractmethod
    def add(self, namespace: str, entries: Sequence[StoreEntry]) -> None:
        """Upsert entries (by ``entry_id``) into ``namespace``.

        Implementations must stamp ``entry.namespace = namespace``.
        """

    @abstractmethod
    def search(
        self,
        embedding: np.ndarray,
        namespaces: Sequence[str],
        k: int = 5,
        filter: SearchFilter | None = None,
    ) -> list[SearchHit]:
        """Top-k cosine-similarity hits merged across ``namespaces``.

        Results are sorted by descending score with a deterministic
        tie-break (speaker_key, entry_id).
        """

    @abstractmethod
    def get_speaker_entries(self, namespace: str, speaker_key: str) -> list[StoreEntry]:
        """All entries for one speaker in one namespace."""

    @abstractmethod
    def delete_entries(self, namespace: str, entry_ids: Sequence[str]) -> int:
        """Delete specific entries; returns how many existed."""

    @abstractmethod
    def delete_speaker(self, namespace: str, speaker_key: str) -> int:
        """Delete all of a speaker's entries; returns how many existed."""

    @abstractmethod
    def update_speaker(
        self,
        namespace: str,
        speaker_key: str,
        *,
        new_key: str | None = None,
        display_name: object = UNSET,
    ) -> int:
        """Rename a speaker and/or set its display name on all entries.

        Renaming onto an existing key merges the two speakers. Returns the
        number of entries updated.
        """

    @abstractmethod
    def remove_entries(self, namespace: str, transcription_id: str) -> int:
        """Delete every entry sourced from one transcription (rollback)."""

    @abstractmethod
    def list_speakers(self, namespaces: Sequence[str]) -> list[SpeakerSummary]:
        """Speaker summaries grouped by key across the given namespaces."""

    @abstractmethod
    def count(self, namespace: str) -> int:
        """Number of entries in a namespace."""

    @abstractmethod
    def delete_namespace(self, namespace: str) -> int:
        """Wipe a namespace entirely (biometric-data deletion path)."""
