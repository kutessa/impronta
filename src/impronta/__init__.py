"""impronta — speaker identification for ElevenLabs Scribe v2 transcripts.

Enroll known voices, identify diarized speakers in new recordings, and map
``speaker_id`` labels to real names backed by a pluggable vector store.
"""

from .api import Impronta, slugify
from .config import ImprontaConfig
from .embedder import EcapaEmbedder, Embedder
from .exceptions import (
    AudioDecodeError,
    ImprontaError,
    NoUsableSegmentsError,
    ScribeParseError,
    SpeakerExistsError,
    SpeakerNotFoundError,
    SpeakerNotInTranscriptError,
)
from .models import (
    UNKNOWN_BUCKET,
    Candidate,
    EnrollResult,
    IdentifyResult,
    ReinforcementProposal,
    SearchFilter,
    SearchHit,
    Segment,
    SegmentInfo,
    SpeakerMatch,
    SpeakerSummary,
    StoreEntry,
    UnknownProposal,
)
from .naming import apply_names, format_transcript, resolve_label
from .store import InMemoryStore, VectorStore

__version__ = "0.1.0"


def __getattr__(name: str):
    # FaissLocalStore stays lazy so `import impronta` never loads faiss;
    # see impronta/store/__init__.py for why.
    if name == "FaissLocalStore":
        from .store.faiss_local import FaissLocalStore

        return FaissLocalStore
    raise AttributeError(name)

__all__ = [
    "Impronta",
    "ImprontaConfig",
    "Embedder",
    "EcapaEmbedder",
    "VectorStore",
    "FaissLocalStore",
    "InMemoryStore",
    "Candidate",
    "EnrollResult",
    "IdentifyResult",
    "SearchFilter",
    "SearchHit",
    "Segment",
    "SegmentInfo",
    "SpeakerMatch",
    "SpeakerSummary",
    "StoreEntry",
    "UnknownProposal",
    "ReinforcementProposal",
    "UNKNOWN_BUCKET",
    "apply_names",
    "format_transcript",
    "resolve_label",
    "slugify",
    "ImprontaError",
    "AudioDecodeError",
    "ScribeParseError",
    "SpeakerNotInTranscriptError",
    "NoUsableSegmentsError",
    "SpeakerNotFoundError",
    "SpeakerExistsError",
]
