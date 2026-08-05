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
from dataclasses import replace
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
    ReinforcementProposal,
    SearchFilter,
    Segment,
    SegmentInfo,
    SpeakerMatch,
    SpeakerSummary,
    StoreEntry,
    UnknownProposal,
    composite_quality,
    utcnow,
)
from .pipeline import PreparedSpeaker, cohesion, prepare_segments
from .scribe import parse_scribe_response
from .segmentation import segment_windows, segment_words
from .store.base import VectorStore
from .vote import run_vote


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "speaker"


def _det_id(*parts: object) -> str:
    return hashlib.sha1(":".join(str(p) for p in parts).encode()).hexdigest()[:24]


def suggested_unknown_key(transcription_id: str, query_speaker_id: str) -> str:
    digest = hashlib.sha1(f"{transcription_id}:{query_speaker_id}".encode()).hexdigest()
    return f"unknown-{digest[:12]}"


def _stamp_audio_id(segments: list[Segment], audio_id: str | None) -> list[Segment]:
    if audio_id is None:
        return segments
    return [replace(s, audio_id=audio_id) for s in segments]


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
        if store is None:
            # lazy: keeps faiss (and its OpenMP runtime) out of processes
            # that inject their own store
            from .store.faiss_local import FaissLocalStore

            store = FaissLocalStore()
        self.store = store
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
        *,
        audio_id: str | None = None,
    ) -> EnrollResult:
        """Enroll a labeled speaker from a transcribed recording.

        ``audio_id`` is an opaque caller-supplied id for the source audio
        (impronta never resolves it); it is stamped on every resulting
        segment so results can be traced back to the recording.

        Raises :class:`SpeakerNotInTranscriptError` if ``speaker_id`` is not
        in the transcript and :class:`NoUsableSegmentsError` if nothing
        survives the confidence/SNR filters.
        """
        transcripts = parse_scribe_response(scribe_response)
        decoded = load_audio(audio, self.config.sample_rate)
        multi = len(transcripts) > 1

        transcript, segments = self._find_speaker(transcripts, speaker_id)
        segments = _stamp_audio_id(segments, audio_id)
        mono = select_channel(decoded, transcript.channel_index if multi else None)
        prepared = prepare_segments(segments, mono, self.embedder, self.config)
        return self.enroll_prepared(
            prepared,
            name=name,
            language=transcript.language_code,
            speaker_key=speaker_key,
            source="scribe_enroll",
            transcription_id=transcript.transcription_id,
            speaker_id=speaker_id,
        )

    def enroll_prepared(
        self,
        prepared: PreparedSpeaker,
        *,
        name: str,
        language: str,
        speaker_key: str | None = None,
        source: str = "scribe_enroll",
        transcription_id: str | None = None,
        speaker_id: str | None = None,
    ) -> EnrollResult:
        """Embed-free enrollment: everything after the preparation pipeline.

        Used by the public enroll methods and by the tuning harness (which
        builds ``prepared`` from a precomputed embedding cache via
        :func:`impronta.pipeline.assemble_prepared`). Entry ids are
        deterministic when both ``transcription_id`` and ``speaker_id`` are
        given. Takes no ``audio_id`` — callers building ``prepared``
        directly stamp their own :class:`~impronta.models.Segment` objects.
        """
        if prepared.embeddings is None:
            raise NoUsableSegmentsError(
                speaker_id or "<direct>", prepared.best_snr_db, prepared.segments_total
            )
        prepared = self._drop_enroll_outliers(prepared)
        assert prepared.embeddings is not None  # the filter never empties a batch

        key = speaker_key or slugify(name)
        language = self.config.canonical_language(language)
        id_seed = (
            ("enroll", transcription_id, speaker_id, key)
            if transcription_id is not None and speaker_id is not None
            else None
        )
        entries = self._make_entries(
            prepared,
            speaker_key=key,
            display_name=name,
            language=language,
            source=source,
            transcription_id=transcription_id,
            id_seed=id_seed,
        )
        self.store.add(self.write_namespace, entries)

        merged: tuple[str, ...] = ()
        if self.config.merge_unknowns_on_enroll:
            merged = self._merge_matching_unknowns(
                key, name, language, prepared.embeddings
            )
        self._enforce_speaker_cap(self.write_namespace, key)

        seg_infos = tuple(SegmentInfo.from_segment(s) for s in prepared.segments)
        ideal = (
            max(
                range(len(prepared.segments)),
                key=lambda i: composite_quality(
                    prepared.segments[i].snr_db,
                    prepared.segments[i].confidence,
                    prepared.segments[i].duration,
                ),
            )
            if prepared.segments
            else None
        )
        return EnrollResult(
            speaker_key=key,
            display_name=name,
            language=language,
            segments_total=prepared.segments_total,
            segments_used=len(prepared.segments),
            quality_tier=prepared.quality_tier or "low",
            merged_unknown_keys=merged,
            entry_ids=tuple(e.entry_id for e in entries),
            segments=seg_infos,
            ideal_segment_index=ideal,
        )

    def add_speaker_from_audio(
        self,
        audio: AudioInput,
        name: str,
        language: str,
        speaker_key: str | None = None,
        *,
        audio_id: str | None = None,
    ) -> EnrollResult:
        """Enroll from a raw clip known to contain only this person's voice.

        ``audio_id`` is an opaque caller-supplied id stamped on every
        resulting segment (see :meth:`add_speaker`).
        """
        decoded = load_audio(audio, self.config.sample_rate)
        mono = select_channel(decoded, None)
        windows = _stamp_audio_id(segment_windows(decoded.duration, self.config), audio_id)
        prepared = prepare_segments(windows, mono, self.embedder, self.config)
        return self.enroll_prepared(
            prepared,
            name=name,
            language=language,
            speaker_key=speaker_key,
            source="direct_enroll",
            transcription_id=None,  # no natural idempotency key for raw clips
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
        audio_id: str | None = None,
    ) -> IdentifyResult:
        """Identify every diarized speaker. READ-ONLY — never writes.

        Unknown voices that pass the proposal gates come back in
        ``proposed_unknowns``; persist them with :meth:`commit_unknowns`.
        Set ``language_filter=False`` to match across languages.
        ``audio_id`` is an opaque caller-supplied id stamped on every
        segment carried by matches and proposals (one id per call — a
        multichannel recording is still a single audio file).
        """
        transcripts = parse_scribe_response(scribe_response)
        decoded = load_audio(audio, self.config.sample_rate)
        multi = len(transcripts) > 1

        speakers: dict[str, SpeakerMatch] = {}
        proposals: list[UnknownProposal] = []
        for transcript in transcripts:
            mono = select_channel(decoded, transcript.channel_index if multi else None)
            segment_map = segment_words(transcript.words, self.config)
            prepared_map = {
                (f"{transcript.channel_index}:{sid}" if multi else sid): prepare_segments(
                    _stamp_audio_id(segment_map.get(sid, []), audio_id),
                    mono,
                    self.embedder,
                    self.config,
                )
                for sid in transcript.speaker_ids()
            }
            partial = self.identify_prepared(
                prepared_map,
                language_code=transcript.language_code,
                transcription_id=transcript.transcription_id,
                language_filter=language_filter,
            )
            speakers.update(partial.speakers)
            proposals.extend(partial.proposed_unknowns)

        return IdentifyResult(
            speakers=speakers,
            proposed_unknowns=tuple(proposals),
            language_code=transcripts[0].language_code,
            transcription_id=transcripts[0].transcription_id,
        )

    def identify_prepared(
        self,
        prepared_map: dict[str, PreparedSpeaker],
        *,
        language_code: str,
        transcription_id: str,
        language_filter: bool = True,
    ) -> IdentifyResult:
        """Embed-free identification over already-prepared speakers.

        READ-ONLY like :meth:`identify`. Used by the public method and by
        the tuning harness (cache-built ``PreparedSpeaker`` objects).
        ``language_code`` may be a raw STT code — it is normalized here.
        Takes no ``audio_id`` — callers building ``prepared_map`` directly
        stamp their own :class:`~impronta.models.Segment` objects.
        """
        speakers: dict[str, SpeakerMatch] = {}
        proposals: list[UnknownProposal] = []
        language = self.config.canonical_language(language_code)
        for qid, prepared in prepared_map.items():
            match, proposal = self._match_speaker(
                qid, transcription_id, language, language_filter, prepared
            )
            speakers[qid] = match
            if proposal is not None:
                proposals.append(proposal)
        return IdentifyResult(
            speakers=speakers,
            proposed_unknowns=tuple(proposals),
            language_code=language_code,
            transcription_id=transcription_id,
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
    # Profile reinforcement
    # ------------------------------------------------------------------

    #: Reinforcement segments this similar to an already-stored entry add no
    #: information (channel/condition diversity is the point of reinforcing).
    NOVELTY_CEILING = 0.95

    def propose_reinforcements(
        self,
        scribe_response: object,
        audio: AudioInput,
        *,
        language_filter: bool = True,
        audio_id: str | None = None,
    ) -> list[ReinforcementProposal]:
        """Propose strengthening profiles from confident identifications.

        READ-ONLY, like ``identify``. A speaker's segments are proposed only
        when ALL gates hold:

        - the vote winner is a stored speaker with mean similarity >=
          ``merge_threshold`` (the profile-mutating bar, not the match bar);
        - no other stored speaker comes within ``reinforce_margin`` of the
          winner (contested matches never feed profiles);
        - per segment: the segment itself scores >= ``merge_threshold``
          against the winner's entries (crosstalk stays out) and <
          ``NOVELTY_CEILING`` against all of them (near-duplicates add
          nothing).

        Persist with :meth:`commit_reinforcements`. Note: this re-runs the
        preparation pipeline; it is a separate pass from ``identify``.
        """
        cfg = self.config
        transcripts = parse_scribe_response(scribe_response)
        decoded = load_audio(audio, cfg.sample_rate)
        multi = len(transcripts) > 1

        proposals: list[ReinforcementProposal] = []
        for transcript in transcripts:
            mono = select_channel(decoded, transcript.channel_index if multi else None)
            segment_map = segment_words(transcript.words, cfg)
            prepared_map = {
                (f"{transcript.channel_index}:{sid}" if multi else sid): prepare_segments(
                    _stamp_audio_id(segment_map.get(sid, []), audio_id),
                    mono,
                    self.embedder,
                    cfg,
                )
                for sid in transcript.speaker_ids()
            }
            proposals.extend(
                self.propose_reinforcements_prepared(
                    prepared_map,
                    language_code=transcript.language_code,
                    transcription_id=transcript.transcription_id,
                    language_filter=language_filter,
                )
            )
        return proposals

    def propose_reinforcements_prepared(
        self,
        prepared_map: dict[str, PreparedSpeaker],
        *,
        language_code: str,
        transcription_id: str,
        language_filter: bool = True,
    ) -> list[ReinforcementProposal]:
        """Embed-free reinforcement proposal pass (see gate docs above)."""
        language = self.config.canonical_language(language_code)
        proposals = []
        for qid, prepared in prepared_map.items():
            p = self._reinforce_one(
                qid, prepared, language, transcription_id, language_filter
            )
            if p is not None:
                proposals.append(p)
        return proposals

    def _reinforce_one(
        self,
        qid: str,
        prepared: PreparedSpeaker,
        language: str,  # canonical
        transcription_id: str,
        language_filter: bool,
    ) -> ReinforcementProposal | None:
        cfg = self.config
        if prepared.embeddings is None:
            return None
        outcome = run_vote(
            prepared.embeddings,
            prepared.segments,
            self.store,
            self.read_namespaces,
            language if language_filter else None,
            cfg,
        )
        if (
            outcome.winner_key == UNKNOWN_BUCKET
            or outcome.winner_namespace is None
            or outcome.mean_similarity is None
            or outcome.mean_similarity < cfg.reinforce_threshold
        ):
            return None
        rival_sims = [
            c.mean_similarity
            for c in outcome.candidates
            if c.speaker_key not in (outcome.winner_key, UNKNOWN_BUCKET)
            and c.mean_similarity is not None
        ]
        if rival_sims and (
            outcome.mean_similarity - max(rival_sims) < cfg.reinforce_margin
        ):
            return None
        stored = self.store.get_speaker_entries(
            outcome.winner_namespace, outcome.winner_key
        )
        if len(stored) < cfg.reinforce_min_profile_entries:
            # too thin to be a trustworthy anchor — a 1-2 entry profile is a
            # channel fingerprint that other voices can clear the bar against
            return None
        profile = np.stack([e.embedding for e in stored])
        best_vs_profile = (prepared.embeddings @ profile.T).max(axis=1)
        keep = (best_vs_profile >= cfg.reinforce_threshold) & (
            best_vs_profile < self.NOVELTY_CEILING
        )
        if not keep.any():
            return None
        return ReinforcementProposal(
            speaker_key=outcome.winner_key,
            namespace=outcome.winner_namespace,
            display_name=outcome.winner_display_name,
            query_speaker_id=qid,
            transcription_id=transcription_id,
            language=language,
            mean_similarity=outcome.mean_similarity,
            embeddings=prepared.embeddings[keep],
            segments=tuple(
                SegmentInfo.from_segment(s)
                for s, k in zip(prepared.segments, keep, strict=True)
                if k
            ),
        )

    def commit_reinforcements(
        self, proposals: Sequence[ReinforcementProposal]
    ) -> list[str | None]:
        """Persist reinforcement proposals; returns the key used per proposal.

        A slot is ``None`` when the target speaker no longer exists (deleted
        or renamed since the proposal was made) — nothing is written for it.
        Entry ids are deterministic, so committing twice is a no-op. Entries
        take the speaker's CURRENT display name, not the proposal snapshot.
        """
        committed: list[str | None] = []
        for p in proposals:
            target = self.store.get_speaker_entries(p.namespace, p.speaker_key)
            if not target:
                committed.append(None)
                continue
            display_name = next(
                (e.display_name for e in target if e.display_name is not None), None
            )
            now = utcnow()
            entries = [
                StoreEntry(
                    entry_id=_det_id(
                        "reinforce", p.transcription_id, p.query_speaker_id, i
                    ),
                    speaker_key=p.speaker_key,
                    display_name=display_name,
                    language=self.config.canonical_language(p.language),
                    confidence=seg.confidence,
                    embedding=np.asarray(emb, dtype=np.float32),
                    created_at=now,
                    source="reinforce",
                    source_transcription_id=p.transcription_id,
                    snr_db=seg.snr_db,
                    duration_sec=seg.duration,
                )
                for i, (emb, seg) in enumerate(zip(p.embeddings, p.segments, strict=True))
            ]
            self.store.add(p.namespace, entries)
            self._enforce_speaker_cap(p.namespace, p.speaker_key)
            committed.append(p.speaker_key)
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
        transcription_id: str,
        language: str,  # canonical
        language_filter: bool,
        prepared: PreparedSpeaker,
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
            language if language_filter else None,
            cfg,
        )

        seg_infos = tuple(SegmentInfo.from_segment(s) for s in prepared.segments)

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
                    segments=seg_infos,
                    ideal_segment_index=outcome.best_segment_index,
                ),
                None,
            )

        reason = self._gate_proposal(prepared, outcome.best_named_score)
        proposal: UnknownProposal | None = None
        if reason is None:
            proposal = UnknownProposal(
                query_speaker_id=qid,
                transcription_id=transcription_id,
                language=language,
                suggested_key=suggested_unknown_key(transcription_id, qid),
                embeddings=prepared.embeddings,
                segments=seg_infos,
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
            segments=seg_infos,
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
