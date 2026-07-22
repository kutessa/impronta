"""All data types used across the package.

Every type that crosses the package boundary (store entries, results,
proposals) is a plain dataclass with ``to_dict``/``from_dict`` so callers can
persist them (Firestore documents, task-queue payloads) without knowing about
numpy: embeddings serialize as base64-encoded float32 bytes, datetimes as ISO
strings.

Keeping every dataclass in one module avoids circular imports between the
store, pipeline, and API layers.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

UNKNOWN_BUCKET = "<unknown>"


def encode_embedding(arr: np.ndarray) -> str:
    """float32 array -> base64 string (row-major bytes)."""
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).decode("ascii")


def decode_embedding(data: str, dim: int | None = None) -> np.ndarray:
    """base64 string -> float32 array; 2D of width ``dim`` when given."""
    flat = np.frombuffer(base64.b64decode(data), dtype=np.float32).copy()
    if dim is not None:
        return flat.reshape(-1, dim)
    return flat


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _from_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Transcript-side types (internal to the pipeline)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Word:
    """One entry of the Scribe ``words`` array (nullable fields preserved)."""

    text: str
    start: float | None
    end: float | None
    type: str  # 'word' | 'spacing' | 'audio_event'
    speaker_id: str | None
    logprob: float | None  # <= 0


@dataclass(frozen=True, slots=True)
class ParsedTranscript:
    """One channel's transcript, normalized from the Scribe response."""

    language_code: str
    words: tuple[Word, ...]
    channel_index: int | None
    audio_duration_secs: float | None
    transcription_id: str

    def speaker_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for w in self.words:
            if w.speaker_id is not None:
                seen.setdefault(w.speaker_id, None)
        return list(seen)


@dataclass(frozen=True, slots=True)
class Segment:
    """A merged run of same-speaker words (or a direct-enroll window)."""

    speaker_id: str | None
    start: float
    end: float
    confidence: float  # exp(mean logprob) of member words; 1.0 when unknown
    snr_db: float | None = None  # filled in by the pipeline

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class SegmentInfo:
    """Serializable summary of a segment carried on proposals/entries."""

    start: float
    end: float
    confidence: float
    snr_db: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "snr_db": self.snr_db,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SegmentInfo:
        return cls(
            start=d["start"], end=d["end"], confidence=d["confidence"], snr_db=d["snr_db"]
        )


# ---------------------------------------------------------------------------
# Store-side types
# ---------------------------------------------------------------------------


def composite_quality(
    snr_db: float | None, confidence: float, duration_sec: float | None
) -> float:
    """Deterministic quality score used to rank segments and evict entries.

    Each component is scaled to roughly [0, 1]: SNR over a 0–40 dB useful
    range, confidence as-is, duration saturating at 10 s.
    """
    snr_part = min(max(snr_db or 0.0, 0.0), 40.0) / 40.0
    dur_part = min(max(duration_sec or 0.0, 0.0), 10.0) / 10.0
    return snr_part + confidence + dur_part


@dataclass(slots=True)
class StoreEntry:
    """One embedding plus its metadata as persisted in a vector store.

    ``namespace`` is stamped by ``VectorStore.add``. ``entry_id`` is the
    upsert key within a namespace: adding an entry with an existing id
    replaces it (this is what makes commit retries idempotent).
    """

    entry_id: str
    speaker_key: str
    display_name: str | None  # None => unknown (auto-enrolled) speaker
    language: str
    confidence: float
    embedding: np.ndarray  # float32 (dim,), L2-normalized
    created_at: datetime
    source: str  # 'scribe_enroll' | 'direct_enroll' | 'auto_enroll'
    source_transcription_id: str | None = None
    snr_db: float | None = None
    duration_sec: float | None = None
    namespace: str = ""

    @property
    def is_unknown(self) -> bool:
        return self.display_name is None

    @property
    def quality(self) -> float:
        return composite_quality(self.snr_db, self.confidence, self.duration_sec)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "speaker_key": self.speaker_key,
            "display_name": self.display_name,
            "language": self.language,
            "confidence": self.confidence,
            "embedding": encode_embedding(self.embedding),
            "created_at": _iso(self.created_at),
            "source": self.source,
            "source_transcription_id": self.source_transcription_id,
            "snr_db": self.snr_db,
            "duration_sec": self.duration_sec,
            "namespace": self.namespace,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StoreEntry:
        return cls(
            entry_id=d["entry_id"],
            speaker_key=d["speaker_key"],
            display_name=d["display_name"],
            language=d["language"],
            confidence=d["confidence"],
            embedding=decode_embedding(d["embedding"]),
            created_at=_from_iso(d["created_at"]),
            source=d["source"],
            source_transcription_id=d.get("source_transcription_id"),
            snr_db=d.get("snr_db"),
            duration_sec=d.get("duration_sec"),
            namespace=d.get("namespace", ""),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StoreEntry):
            return NotImplemented
        return self.to_dict() == other.to_dict()


@dataclass(frozen=True, slots=True)
class SearchFilter:
    """Metadata constraints applied by ``VectorStore.search``.

    ``language`` is an exact-equality hard filter; ``unknown_only`` restricts
    to entries with no display name; ``exclude_speaker_keys`` removes specific
    speakers from consideration. Backends map these to their native filters
    (SQL WHERE, Qdrant payload filter, ...).
    """

    language: str | None = None
    unknown_only: bool = False
    exclude_speaker_keys: frozenset[str] = frozenset()

    def matches(self, entry: StoreEntry) -> bool:
        if self.language is not None and entry.language != self.language:
            return False
        if self.unknown_only and not entry.is_unknown:
            return False
        return entry.speaker_key not in self.exclude_speaker_keys


@dataclass(frozen=True, slots=True)
class SearchHit:
    entry: StoreEntry
    score: float  # cosine similarity (inner product of normalized vectors)


@dataclass(frozen=True, slots=True)
class SpeakerSummary:
    speaker_key: str
    display_name: str | None
    namespaces: tuple[str, ...]
    languages: tuple[str, ...]
    num_embeddings: int
    created_at: datetime

    @property
    def is_unknown(self) -> bool:
        return self.display_name is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker_key": self.speaker_key,
            "display_name": self.display_name,
            "namespaces": list(self.namespaces),
            "languages": list(self.languages),
            "num_embeddings": self.num_embeddings,
            "created_at": _iso(self.created_at),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SpeakerSummary:
        return cls(
            speaker_key=d["speaker_key"],
            display_name=d["display_name"],
            namespaces=tuple(d["namespaces"]),
            languages=tuple(d["languages"]),
            num_embeddings=d["num_embeddings"],
            created_at=_from_iso(d["created_at"]),
        )


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EnrollResult:
    speaker_key: str
    display_name: str
    language: str
    segments_total: int
    segments_used: int
    quality_tier: str  # 'high' | 'medium' | 'low'
    merged_unknown_keys: tuple[str, ...] = ()
    entry_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker_key": self.speaker_key,
            "display_name": self.display_name,
            "language": self.language,
            "segments_total": self.segments_total,
            "segments_used": self.segments_used,
            "quality_tier": self.quality_tier,
            "merged_unknown_keys": list(self.merged_unknown_keys),
            "entry_ids": list(self.entry_ids),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EnrollResult:
        return cls(
            speaker_key=d["speaker_key"],
            display_name=d["display_name"],
            language=d["language"],
            segments_total=d["segments_total"],
            segments_used=d["segments_used"],
            quality_tier=d["quality_tier"],
            merged_unknown_keys=tuple(d.get("merged_unknown_keys", ())),
            entry_ids=tuple(d.get("entry_ids", ())),
        )


@dataclass(frozen=True, slots=True)
class Candidate:
    """One ranked identification candidate — the late-fusion hook.

    ``score_share`` is this candidate's fraction of the total weighted vote
    (all shares over a speaker sum to 1.0, including the unknown bucket), so
    the application layer can combine it with outside evidence (attendee
    lists, transcript mentions, historical co-occurrence).
    """

    speaker_key: str  # UNKNOWN_BUCKET for the unknown bucket
    display_name: str | None
    score_share: float
    mean_similarity: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker_key": self.speaker_key,
            "display_name": self.display_name,
            "score_share": self.score_share,
            "mean_similarity": self.mean_similarity,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Candidate:
        return cls(
            speaker_key=d["speaker_key"],
            display_name=d["display_name"],
            score_share=d["score_share"],
            mean_similarity=d["mean_similarity"],
        )


@dataclass(frozen=True, slots=True)
class SpeakerMatch:
    """Identification outcome for one diarized speaker_id."""

    query_speaker_id: str
    speaker_key: str | None  # matched key, or None when unknown/unidentifiable
    display_name: str | None
    namespace: str | None
    is_unknown: bool
    identifiable: bool  # False => zero usable segments, nothing else is meaningful
    candidates: tuple[Candidate, ...] = ()
    mean_similarity: float | None = None
    quality_tier: str | None = None
    num_segments_used: int = 0
    num_segments_total: int = 0
    near_misses: tuple[tuple[str, float], ...] = ()
    no_proposal_reason: str | None = None
    # 'too_few_segments' | 'low_quality' | 'low_cohesion' | 'gray_zone'

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_speaker_id": self.query_speaker_id,
            "speaker_key": self.speaker_key,
            "display_name": self.display_name,
            "namespace": self.namespace,
            "is_unknown": self.is_unknown,
            "identifiable": self.identifiable,
            "candidates": [c.to_dict() for c in self.candidates],
            "mean_similarity": self.mean_similarity,
            "quality_tier": self.quality_tier,
            "num_segments_used": self.num_segments_used,
            "num_segments_total": self.num_segments_total,
            "near_misses": [[k, s] for k, s in self.near_misses],
            "no_proposal_reason": self.no_proposal_reason,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SpeakerMatch:
        return cls(
            query_speaker_id=d["query_speaker_id"],
            speaker_key=d["speaker_key"],
            display_name=d["display_name"],
            namespace=d["namespace"],
            is_unknown=d["is_unknown"],
            identifiable=d["identifiable"],
            candidates=tuple(Candidate.from_dict(c) for c in d.get("candidates", ())),
            mean_similarity=d.get("mean_similarity"),
            quality_tier=d.get("quality_tier"),
            num_segments_used=d.get("num_segments_used", 0),
            num_segments_total=d.get("num_segments_total", 0),
            near_misses=tuple((k, s) for k, s in d.get("near_misses", ())),
            no_proposal_reason=d.get("no_proposal_reason"),
        )


@dataclass(frozen=True, slots=True, eq=False)
class UnknownProposal:
    """A stranger's voiceprint, proposed for enrollment but NOT yet stored.

    Fully serializable so the application can carry it through task queues
    and commit it later (``Impronta.commit_unknowns``). ``suggested_key`` is
    deterministic in (transcription_id, query_speaker_id) so retried jobs
    produce identical proposals.
    """

    query_speaker_id: str
    transcription_id: str
    language: str
    suggested_key: str
    embeddings: np.ndarray  # float32 (n, dim), L2-normalized rows
    segments: tuple[SegmentInfo, ...]
    quality_tier: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_speaker_id": self.query_speaker_id,
            "transcription_id": self.transcription_id,
            "language": self.language,
            "suggested_key": self.suggested_key,
            "embeddings": encode_embedding(self.embeddings),
            "embedding_dim": int(self.embeddings.shape[1]),
            "segments": [s.to_dict() for s in self.segments],
            "quality_tier": self.quality_tier,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> UnknownProposal:
        return cls(
            query_speaker_id=d["query_speaker_id"],
            transcription_id=d["transcription_id"],
            language=d["language"],
            suggested_key=d["suggested_key"],
            embeddings=decode_embedding(d["embeddings"], dim=d["embedding_dim"]),
            segments=tuple(SegmentInfo.from_dict(s) for s in d["segments"]),
            quality_tier=d["quality_tier"],
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UnknownProposal):
            return NotImplemented
        return self.to_dict() == other.to_dict()


@dataclass(frozen=True, slots=True, eq=False)
class ReinforcementProposal:
    """Confidently-identified segments proposed for strengthening a profile.

    Produced by ``Impronta.propose_reinforcements`` (never by ``identify``),
    persisted by ``Impronta.commit_reinforcements``. Fully serializable, and
    deterministic in (transcription_id, query_speaker_id) so retried jobs
    commit identical entries.
    """

    speaker_key: str
    namespace: str
    display_name: str | None  # snapshot; commit uses the CURRENT name
    query_speaker_id: str
    transcription_id: str
    language: str
    mean_similarity: float
    embeddings: np.ndarray  # float32 (n, dim), L2-normalized rows
    segments: tuple[SegmentInfo, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker_key": self.speaker_key,
            "namespace": self.namespace,
            "display_name": self.display_name,
            "query_speaker_id": self.query_speaker_id,
            "transcription_id": self.transcription_id,
            "language": self.language,
            "mean_similarity": self.mean_similarity,
            "embeddings": encode_embedding(self.embeddings),
            "embedding_dim": int(self.embeddings.shape[1]),
            "segments": [s.to_dict() for s in self.segments],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ReinforcementProposal:
        return cls(
            speaker_key=d["speaker_key"],
            namespace=d["namespace"],
            display_name=d["display_name"],
            query_speaker_id=d["query_speaker_id"],
            transcription_id=d["transcription_id"],
            language=d["language"],
            mean_similarity=d["mean_similarity"],
            embeddings=decode_embedding(d["embeddings"], dim=d["embedding_dim"]),
            segments=tuple(SegmentInfo.from_dict(s) for s in d["segments"]),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ReinforcementProposal):
            return NotImplemented
        return self.to_dict() == other.to_dict()


@dataclass(frozen=True, slots=True)
class IdentifyResult:
    """Full identification outcome for one recording."""

    speakers: dict[str, SpeakerMatch]  # keyed by query_speaker_id
    proposed_unknowns: tuple[UnknownProposal, ...]
    language_code: str
    transcription_id: str

    def name_map(self) -> dict[str, str | None]:
        """query_speaker_id -> display name (None for unknown/unidentifiable)."""
        return {sid: m.display_name for sid, m in self.speakers.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "speakers": {sid: m.to_dict() for sid, m in self.speakers.items()},
            "proposed_unknowns": [p.to_dict() for p in self.proposed_unknowns],
            "language_code": self.language_code,
            "transcription_id": self.transcription_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IdentifyResult:
        return cls(
            speakers={
                sid: SpeakerMatch.from_dict(m) for sid, m in d.get("speakers", {}).items()
            },
            proposed_unknowns=tuple(
                UnknownProposal.from_dict(p) for p in d.get("proposed_unknowns", ())
            ),
            language_code=d["language_code"],
            transcription_id=d["transcription_id"],
        )
