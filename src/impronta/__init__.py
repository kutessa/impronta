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
    SearchFilter,
    SearchHit,
    SpeakerMatch,
    SpeakerSummary,
    StoreEntry,
    UnknownProposal,
)
from .naming import apply_names, format_transcript, resolve_label
from .store import FaissLocalStore, InMemoryStore, VectorStore

__version__ = "0.1.0"

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
    "SpeakerMatch",
    "SpeakerSummary",
    "StoreEntry",
    "UnknownProposal",
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
