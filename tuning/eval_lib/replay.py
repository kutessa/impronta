"""Replay layer: cached embeddings -> production impronta code, no torch.

``ReplaySession`` wraps one ``Impronta`` instance per (account, trial
config), backed by an ``InMemoryStore`` (pure numpy, exact cosine — same
semantics as the FAISS store). All entry points are the ``*_prepared``
methods, so the lazy ECAPA embedder is never constructed and every
downstream decision (selection, vote, gates, commits, reinforcement) runs
the real production code under the trial config.
"""

from __future__ import annotations

from impronta import IdentifyResult, Impronta, ImprontaConfig, InMemoryStore
from impronta.models import EnrollResult, ReinforcementProposal
from impronta.pipeline import PreparedSpeaker, assemble_prepared

from .cache import RecordingCache


def prepared_for(cache: RecordingCache, qid: str, cfg: ImprontaConfig) -> PreparedSpeaker:
    """Re-run the selection stage for one speaker from cached measurements."""
    segments, embeddings = cache.segments_for(qid)
    total = cache.segments_total.get(qid, len(segments))
    return assemble_prepared(segments, embeddings, total, cfg)


class ReplaySession:
    """One account's speaker store under one trial config."""

    def __init__(self, uid: str, cfg: ImprontaConfig):
        self.uid = uid
        self.cfg = cfg
        self.app = Impronta(
            store=InMemoryStore(), config=cfg, write_namespace=uid
        )

    def prepared_map(self, cache: RecordingCache) -> dict[str, PreparedSpeaker]:
        return {qid: prepared_for(cache, qid, self.cfg) for qid in cache.qids()}

    def enroll(self, cache: RecordingCache, qid: str, person: str) -> EnrollResult:
        """Simulate the user labeling ``qid`` in this recording as ``person``."""
        return self.app.enroll_prepared(
            prepared_for(cache, qid, self.cfg),
            name=person,
            language=cache.language_code,
            speaker_key=person,
            source="scribe_enroll",
            transcription_id=cache.transcription_id or cache.rid,
            speaker_id=qid,
        )

    def identify(self, cache: RecordingCache) -> IdentifyResult:
        return self.app.identify_prepared(
            self.prepared_map(cache),
            language_code=cache.language_code,
            transcription_id=cache.transcription_id or cache.rid,
        )

    def reinforce(self, cache: RecordingCache) -> list[ReinforcementProposal]:
        return self.app.propose_reinforcements_prepared(
            self.prepared_map(cache),
            language_code=cache.language_code,
            transcription_id=cache.transcription_id or cache.rid,
        )
