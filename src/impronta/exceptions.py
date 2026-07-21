"""Exception hierarchy for impronta.

All package errors inherit from :class:`ImprontaError` so callers can catch
everything with one clause. Identification quality problems are NOT errors —
``identify()`` reports them per speaker (``identifiable=False``) instead of
raising; only enrollment raises when a speaker is unusable.
"""


class ImprontaError(Exception):
    """Base class for all impronta errors."""


class AudioDecodeError(ImprontaError):
    """The audio input could not be decoded (unsupported format, corrupt data)."""


class ScribeParseError(ImprontaError):
    """The scribe response is not a recognizable Scribe v2 payload."""


class SpeakerNotInTranscriptError(ImprontaError):
    """The requested speaker_id does not appear in the transcript."""

    def __init__(self, speaker_id: str, available: list[str]):
        self.speaker_id = speaker_id
        self.available = available
        super().__init__(
            f"speaker_id {speaker_id!r} not found in transcript; "
            f"available speaker ids: {sorted(available)}"
        )


class NoUsableSegmentsError(ImprontaError):
    """Enrollment failed: no segment survived the confidence and SNR filters."""

    def __init__(self, speaker_id: str, best_snr_db: float | None, segments_total: int):
        self.speaker_id = speaker_id
        self.best_snr_db = best_snr_db
        self.segments_total = segments_total
        detail = (
            f"best observed SNR was {best_snr_db:.1f} dB" if best_snr_db is not None
            else "no segments passed the duration/confidence filters"
        )
        super().__init__(
            f"no usable audio segments for speaker {speaker_id!r} "
            f"({segments_total} candidate segments; {detail})"
        )


class SpeakerNotFoundError(ImprontaError):
    """No entries exist for the given speaker key."""


class SpeakerExistsError(ImprontaError):
    """A conflicting speaker already exists for the requested key."""
