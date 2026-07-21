"""The Impronta facade: enroll, identify, and manage speakers.

Design invariants worth knowing before reading the code:

- ``identify()`` never writes to the store. Strangers come back as
  :class:`~impronta.models.UnknownProposal` objects; the application decides
  whether and where to persist them via :meth:`Impronta.commit_unknowns`.
- All write operations target ``write_namespace``; searches read across
  ``read_namespaces`` (hierarchical tenancy, best score wins).
- Entry ids for transcript-derived embeddings are deterministic in
  (transcription_id, speaker, index), so retried jobs upsert instead of
  duplicating.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Sequence
from datetime import datetime

import numpy as np

from .audio import AudioInput, load_audio, select_channel
from .config import ImprontaConfig
from .embedder import EcapaEmbedder, Embedder, l2_normalize
from .exceptions import (
    NoUsableSegmentsError,
    SpeakerNotFoundError,
    SpeakerNotInTranscriptError,
)
from .models import (
    UNKNOWN_BUCKET,
    EnrollResult,
    IdentifyResult,
    ParsedTranscript,
    SearchFilter,
    Segment,
    SegmentInfo,
    SpeakerMatch,
    SpeakerSummary,
    StoreEntry,
    UnknownProposal,
    utcnow,
)
from .pipeline import PreparedSpeaker, cohesion, prepare_segments
from .scribe import parse_scribe_response
from .segmentation import segment_windows, segment_words
from .store.base import VectorStore
from .store.faiss_local import FaissLocalStore
from .vote import run_vote


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "speaker"


def _det_id(*parts: object) -> str:
    return hashlib.sha1(":".join(str(p) for p in parts).encode()).hexdigest()[:24]


def suggested_unknown_key(transcription_id: str, query_speaker_id: str) -> str:
    digest = hashlib.sha1(f"{transcription_id}:{query_speaker_id}".encode()).hexdigest()
    return f"unknown-{digest[:12]}"


class Impronta:
    """Speaker identification on top of ElevenLabs Scribe v2 transcripts.

    Args:
        store: vector store backend; defaults to a fresh
            :class:`~impronta.store.faiss_local.FaissLocalStore`.
        config: pipeline tunables; defaults to :class:`ImprontaConfig`.
        embedder: embedding backend; defaults to a lazily-created
            :class:`~impronta.embedder.EcapaEmbedder` (torch is only imported
            when the first embedding is computed).
        write_namespace: namespace all writes go to.
        read_namespaces: namespaces searched during identification/merging,
            in priority-irrelevant order (best score wins). Defaults to
            ``[write_namespace]``.
    """

    def __init__(
        self,
        store: VectorStore | None = None,
        config: ImprontaConfig | None = None,
        embedder: Embedder | None = None,
        *,
        write_namespace: str = "default",
        read_namespaces: Sequence[str] | None = None,
    ):
        self.config = config or ImprontaConfig()
        self.store = store if store is not None else FaissLocalStore()
        self._embedder = embedder
        self.write_namespace = write_namespace
        self.read_namespaces = (
            list(read_namespaces) if read_namespaces else [write_namespace]
        )

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = EcapaEmbedder(self.config)
        return self._embedder

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------

    def add_speaker(
        self,
        scribe_response: object,
        audio: AudioInput,
        speaker_id: str,
        name: str,
        speaker_key: str | None = None,
    ) -> EnrollResult:
        """Enroll a labeled speaker from a transcribed recording.

        Raises :class:`SpeakerNotInTranscriptError` if ``speaker_id`` is not
        in the transcript and :class:`NoUsableSegmentsError` if nothing
        survives the confidence/SNR filters.
        """
        transcripts = parse_scribe_response(scribe_response)
        decoded = load_audio(audio, self.config.sample_rate)
        multi = len(transcripts) > 1

        transcript, segments = self._find_speaker(transcripts, speaker_id)
        mono = select_channel(decoded, transcript.channel_index if multi else None)
        prepared = prepare_segments(segments, mono, self.embedder, self.config)
        if prepared.embeddings is None:
            raise NoUsableSegmentsError(
                speaker_id, prepared.best_snr_db, prepared.segments_total
            )
        prepared = self._drop_enroll_outliers(prepared)
        assert prepared.embeddings is not None  # the filter never empties a batch

        key = speaker_key or slugify(name)
        language = self.config.canonical_language(transcript.language_code)
        entries = self._make_entries(
            prepared,
            speaker_key=key,
            display_name=name,
            language=language,
            source="scribe_enroll",
            transcription_id=transcript.transcription_id,
            id_seed=("enroll", transcript.transcription_id, speaker_id, key),
        )
        self.store.add(self.write_namespace, entries)

        merged: tuple[str, ...] = ()
        if self.config.merge_unknowns_on_enroll:
            merged = self._merge_matching_unknowns(
                key, name, language, prepared.embeddings
            )
        self._enforce_speaker_cap(self.write_namespace, key)

        return EnrollResult(
            speaker_key=key,
            display_name=name,
            language=language,
            segments_total=prepared.segments_total,
            segments_used=len(prepared.segments),
            quality_tier=prepared.quality_tier or "low",
            merged_unknown_keys=merged,
            entry_ids=tuple(e.entry_id for e in entries),
        )

    def add_speaker_from_audio(
        self,
        audio: AudioInput,
        name: str,
        language: str,
        speaker_key: str | None = None,
    ) -> EnrollResult:
        """Enroll from a raw clip known to contain only this person's voice."""
        decoded = load_audio(audio, self.config.sample_rate)
        mono = select_channel(decoded, None)
        windows = segment_windows(decoded.duration, self.config)
        prepared = prepare_segments(windows, mono, self.embedder, self.config)
        if prepared.embeddings is None:
            raise NoUsableSegmentsError("<direct>", prepared.best_snr_db, len(windows))
        prepared = self._drop_enroll_outliers(prepared)
        assert prepared.embeddings is not None  # the filter never empties a batch

        language = self.config.canonical_language(language)
        key = speaker_key or slugify(name)
        entries = self._make_entries(
            prepared,
            speaker_key=key,
            display_name=name,
            language=language,
            source="direct_enroll",
            transcription_id=None,
            id_seed=None,  # no natural idempotency key for raw clips
        )
        self.store.add(self.write_namespace, entries)

        merged: tuple[str, ...] = ()
        if self.config.merge_unknowns_on_enroll:
            merged = self._merge_matching_unknowns(key, name, language, prepared.embeddings)
        self._enforce_speaker_cap(self.write_namespace, key)

        return EnrollResult(
            speaker_key=key,
            display_name=name,
            language=language,
            segments_total=len(windows),
            segments_used=len(prepared.segments),
            quality_tier=prepared.quality_tier or "low",
            merged_unknown_keys=merged,
            entry_ids=tuple(e.entry_id for e in entries),
        )

    # ------------------------------------------------------------------
    # Identification
    # ------------------------------------------------------------------

    def identify(
        self,
        scribe_response: object,
        audio: AudioInput,
        *,
        language_filter: bool = True,
    ) -> IdentifyResult:
        """Identify every diarized speaker. READ-ONLY — never writes.

        Unknown voices that pass the proposal gates come back in
        ``proposed_unknowns``; persist them with :meth:`commit_unknowns`.
        Set ``language_filter=False`` to match across languages.
        """
        transcripts = parse_scribe_response(scribe_response)
        decoded = load_audio(audio, self.config.sample_rate)
        multi = len(transcripts) > 1

        speakers: dict[str, SpeakerMatch] = {}
        proposals: list[UnknownProposal] = []
        for transcript in transcripts:
            mono = select_channel(decoded, transcript.channel_index if multi else None)
            segment_map = segment_words(transcript.words, self.config)
            for sid in transcript.speaker_ids():
                qid = f"{transcript.channel_index}:{sid}" if multi else sid
                prepared = prepare_segments(
                    segment_map.get(sid, []), mono, self.embedder, self.config
                )
                match, proposal = self._match_speaker(
                    qid, transcript, prepared, language_filter
                )
                speakers[qid] = match
                if proposal is not None:
                    proposals.append(proposal)

        return IdentifyResult(
            speakers=speakers,
            proposed_unknowns=tuple(proposals),
            language_code=transcripts[0].language_code,
            transcription_id=transcripts[0].transcription_id,
        )

    def commit_unknowns(self, proposals: Sequence[UnknownProposal]) -> list[str]:
        """Persist proposed unknowns; returns the speaker key used for each.

        Dedup-first: if an existing unknown matches a proposal's centroid at
        or above ``merge_threshold``, the proposal's embeddings are appended
        to that speaker instead of creating a new one. Entry ids are
        deterministic, so committing the same proposal twice is a no-op.
        """
        committed: list[str] = []
        for p in proposals:
            # normalize defensively: proposals may have been serialized by an
            # older version or built by the caller with a raw STT code
            language = self.config.canonical_language(p.language)
            centroid = l2_normalize(p.embeddings.mean(axis=0, keepdims=True))[0]
            hits = self.store.search(
                centroid,
                self.read_namespaces,
                k=1,
                filter=SearchFilter(language=language, unknown_only=True),
            )
            if hits and hits[0].score >= self.config.merge_threshold:
                namespace = hits[0].entry.namespace
                key = hits[0].entry.speaker_key
            else:
                namespace, key = self.write_namespace, p.suggested_key

            now = utcnow()
            entries = [
                StoreEntry(
                    entry_id=_det_id("commit", p.transcription_id, p.query_speaker_id, i),
                    speaker_key=key,
                    display_name=None,
                    language=language,
                    confidence=seg.confidence,
                    embedding=np.asarray(emb, dtype=np.float32),
                    created_at=now,
                    source="auto_enroll",
                    source_transcription_id=p.transcription_id,
                    snr_db=seg.snr_db,
                    duration_sec=seg.duration,
                )
                for i, (emb, seg) in enumerate(zip(p.embeddings, p.segments, strict=True))
            ]
            self.store.add(namespace, entries)
            self._enforce_speaker_cap(namespace, key)
            committed.append(key)
        return committed

    # ------------------------------------------------------------------
    # Speaker management
    # ------------------------------------------------------------------

    def label_speaker(
        self, speaker_key: str, name: str, namespace: str | None = None
    ) -> SpeakerSummary:
        """Name a speaker. Promotes auto-enrolled unknowns to named speakers.

        For an unknown, the key is renamed to the slug of ``name`` (merging
        into an existing speaker with that key if one exists). For an
        already-named speaker only the display name changes — the key stays
        stable because the caller may have stored it.
        """
        ns = namespace or self.write_namespace
        entries = self.store.get_speaker_entries(ns, speaker_key)
        if not entries:
            raise SpeakerNotFoundError(
                f"no entries for speaker {speaker_key!r} in namespace {ns!r}"
            )
        is_unknown = all(e.display_name is None for e in entries)
        if is_unknown:
            new_key = slugify(name)
            self.store.update_speaker(
                ns, speaker_key, new_key=new_key, display_name=name
            )
            final_key = new_key
            self._enforce_speaker_cap(ns, final_key)
        else:
            self.store.update_speaker(ns, speaker_key, display_name=name)
            final_key = speaker_key
        for summary in self.store.list_speakers([ns]):
            if summary.speaker_key == final_key:
                return summary
        raise SpeakerNotFoundError(final_key)  # pragma: no cover - defensive

    def list_speakers(self) -> list[SpeakerSummary]:
        return self.store.list_speakers(self.read_namespaces)

    def remove_speaker(self, speaker_key: str, namespace: str | None = None) -> int:
        return self.store.delete_speaker(namespace or self.write_namespace, speaker_key)

    def remove_transcription(
        self, transcription_id: str, namespace: str | None = None
    ) -> int:
        """Surgically delete every entry sourced from one recording."""
        namespaces = (
            [namespace]
            if namespace is not None
            else list(dict.fromkeys([*self.read_namespaces, self.write_namespace]))
        )
        return sum(self.store.remove_entries(ns, transcription_id) for ns in namespaces)

    def prune_unknowns(
        self, namespace: str | None = None, *, older_than: datetime
    ) -> int:
        """Delete never-labeled unknowns whose newest entry predates ``older_than``."""
        ns = namespace or self.write_namespace
        removed = 0
        for summary in self.store.list_speakers([ns]):
            if not summary.is_unknown:
                continue
            entries = self.store.get_speaker_entries(ns, summary.speaker_key)
            if entries and max(e.created_at for e in entries) < older_than:
                removed += self.store.delete_speaker(ns, summary.speaker_key)
        return removed

    def wipe_namespace(self, namespace: str) -> int:
        """Delete every entry in a namespace (biometric-data deletion path)."""
        return self.store.delete_namespace(namespace)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _find_speaker(
        self, transcripts: list[ParsedTranscript], speaker_id: str
    ) -> tuple[ParsedTranscript, list[Segment]]:
        available: list[str] = []
        for transcript in transcripts:
            segment_map = segment_words(transcript.words, self.config)
            available.extend(transcript.speaker_ids())
            if speaker_id in segment_map:
                return transcript, segment_map[speaker_id]
        raise SpeakerNotInTranscriptError(speaker_id, available)

    def _drop_enroll_outliers(self, prepared: PreparedSpeaker) -> PreparedSpeaker:
        """Drop enrollment segments that disagree with the batch centroid.

        Diarization occasionally attributes another person's words to the
        labeled speaker; enrolling those segments poisons the profile.
        Only active with >= 4 segments, and never drops below 2.
        """
        embs = prepared.embeddings
        if embs is None or embs.shape[0] < 4:
            return prepared
        centroid = l2_normalize(embs.mean(axis=0, keepdims=True))[0]
        sims = embs @ centroid
        # relative rule: a segment far below the batch's typical agreement is
        # someone else's voice; scale-free across clean and telephony audio
        keep = sims >= (float(np.median(sims)) - self.config.enroll_outlier_margin)
        if keep.all() or int(keep.sum()) < 2:
            return prepared
        return PreparedSpeaker(
            segments_total=prepared.segments_total,
            segments=[s for s, k in zip(prepared.segments, keep, strict=True) if k],
            embeddings=embs[keep],
            quality_tier=prepared.quality_tier,
            best_snr_db=prepared.best_snr_db,
        )

    def _make_entries(
        self,
        prepared: PreparedSpeaker,
        *,
        speaker_key: str,
        display_name: str | None,
        language: str,
        source: str,
        transcription_id: str | None,
        id_seed: tuple[object, ...] | None,
    ) -> list[StoreEntry]:
        assert prepared.embeddings is not None
        now = utcnow()
        entries = []
        for i, (emb, seg) in enumerate(zip(prepared.embeddings, prepared.segments, strict=True)):
            entry_id = _det_id(*id_seed, i) if id_seed else uuid.uuid4().hex[:24]
            entries.append(
                StoreEntry(
                    entry_id=entry_id,
                    speaker_key=speaker_key,
                    display_name=display_name,
                    language=language,
                    confidence=seg.confidence,
                    embedding=np.asarray(emb, dtype=np.float32),
                    created_at=now,
                    source=source,
                    source_transcription_id=transcription_id,
                    snr_db=seg.snr_db,
                    duration_sec=seg.duration,
                )
            )
        return entries

    def _match_speaker(
        self,
        qid: str,
        transcript: ParsedTranscript,
        prepared: PreparedSpeaker,
        language_filter: bool,
    ) -> tuple[SpeakerMatch, UnknownProposal | None]:
        cfg = self.config
        if prepared.embeddings is None:
            return (
                SpeakerMatch(
                    query_speaker_id=qid,
                    speaker_key=None,
                    display_name=None,
                    namespace=None,
                    is_unknown=False,
                    identifiable=False,
                    num_segments_total=prepared.segments_total,
                ),
                None,
            )

        outcome = run_vote(
            prepared.embeddings,
            prepared.segments,
            self.store,
            self.read_namespaces,
            cfg.canonical_language(transcript.language_code) if language_filter else None,
            cfg,
        )

        if outcome.winner_key != UNKNOWN_BUCKET:
            return (
                SpeakerMatch(
                    query_speaker_id=qid,
                    speaker_key=outcome.winner_key,
                    display_name=outcome.winner_display_name,
                    namespace=outcome.winner_namespace,
                    is_unknown=outcome.winner_display_name is None,
                    identifiable=True,
                    candidates=outcome.candidates,
                    mean_similarity=outcome.mean_similarity,
                    quality_tier=prepared.quality_tier,
                    num_segments_used=len(prepared.segments),
                    num_segments_total=prepared.segments_total,
                    near_misses=outcome.near_misses,
                ),
                None,
            )

        reason = self._gate_proposal(prepared, outcome.best_named_score)
        proposal: UnknownProposal | None = None
        if reason is None:
            proposal = UnknownProposal(
                query_speaker_id=qid,
                transcription_id=transcript.transcription_id,
                language=cfg.canonical_language(transcript.language_code),
                suggested_key=suggested_unknown_key(transcript.transcription_id, qid),
                embeddings=prepared.embeddings,
                segments=tuple(
                    SegmentInfo(
                        start=s.start,
                        end=s.end,
                        confidence=s.confidence,
                        snr_db=s.snr_db if s.snr_db is not None else 0.0,
                    )
                    for s in prepared.segments
                ),
                quality_tier=prepared.quality_tier or "low",
            )
        match = SpeakerMatch(
            query_speaker_id=qid,
            speaker_key=None,
            display_name=None,
            namespace=None,
            is_unknown=True,
            identifiable=True,
            candidates=outcome.candidates,
            mean_similarity=None,
            quality_tier=prepared.quality_tier,
            num_segments_used=len(prepared.segments),
            num_segments_total=prepared.segments_total,
            near_misses=outcome.near_misses,
            no_proposal_reason=reason,
        )
        return match, proposal

    def _gate_proposal(
        self, prepared: PreparedSpeaker, best_named_score: float | None
    ) -> str | None:
        """Return why this stranger must NOT be proposed, or None to propose."""
        cfg = self.config
        assert prepared.embeddings is not None
        low = cfg.similarity_threshold - cfg.gray_zone_margin
        if (
            best_named_score is not None
            and low <= best_named_score < cfg.similarity_threshold
        ):
            return "gray_zone"
        if len(prepared.segments) < cfg.min_proposal_segments:
            return "too_few_segments"
        if prepared.quality_tier is None or not cfg.tier_at_least(
            prepared.quality_tier, cfg.min_proposal_tier
        ):
            return "low_quality"
        if cohesion(prepared.embeddings) < cfg.cohesion_threshold:
            return "low_cohesion"
        return None

    def _merge_matching_unknowns(
        self,
        speaker_key: str,
        display_name: str,
        language: str,
        embeddings: np.ndarray,
    ) -> tuple[str, ...]:
        """Absorb existing unknowns that match the new named enrollment."""
        matched: dict[tuple[str, str], None] = {}
        for row in embeddings:
            hits = self.store.search(
                row,
                self.read_namespaces,
                k=3,
                filter=SearchFilter(language=language, unknown_only=True),
            )
            for hit in hits:
                if hit.score >= self.config.merge_threshold:
                    matched.setdefault((hit.entry.namespace, hit.entry.speaker_key), None)
        merged = []
        for namespace, unknown_key in matched:
            self.store.update_speaker(
                namespace, unknown_key, new_key=speaker_key, display_name=display_name
            )
            self._enforce_speaker_cap(namespace, speaker_key)
            merged.append(unknown_key)
        return tuple(sorted(merged))

    def _enforce_speaker_cap(self, namespace: str, speaker_key: str) -> None:
        cap = self.config.max_embeddings_per_speaker
        entries = self.store.get_speaker_entries(namespace, speaker_key)
        if len(entries) <= cap:
            return
        entries.sort(key=lambda e: (e.quality, e.created_at, e.entry_id))
        excess = [e.entry_id for e in entries[: len(entries) - cap]]
        self.store.delete_entries(namespace, excess)
