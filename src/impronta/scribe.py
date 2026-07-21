"""Tolerant parsing of ElevenLabs Scribe v2 responses.

Accepts a plain dict, or any SDK response object exposing ``model_dump()`` /
``dict()``. Handles both the single-channel shape (top-level ``words``) and
the multichannel shape (``transcripts`` array, one per channel). Missing or
null fields never crash the parser — words that cannot be used are simply
skipped downstream.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any

from .exceptions import ScribeParseError
from .models import ParsedTranscript, Word


def _coerce_to_dict(resp: Any) -> dict[str, Any]:
    if isinstance(resp, dict):
        return resp
    for attr in ("model_dump", "dict"):
        method = getattr(resp, attr, None)
        if callable(method):
            try:
                out = method()
            except TypeError:
                continue
            if isinstance(out, dict):
                return out
    if dataclasses.is_dataclass(resp) and not isinstance(resp, type):
        return dataclasses.asdict(resp)
    raise ScribeParseError(
        f"cannot interpret {type(resp).__name__} as a Scribe response; "
        "pass the response dict or an SDK object with model_dump()"
    )


def _parse_words(raw_words: Any) -> tuple[Word, ...]:
    if not isinstance(raw_words, list):
        return ()
    words = []
    for w in raw_words:
        if not isinstance(w, dict):
            continue
        words.append(
            Word(
                text=str(w.get("text", "")),
                start=w.get("start"),
                end=w.get("end"),
                type=str(w.get("type", "word")),
                speaker_id=w.get("speaker_id"),
                logprob=w.get("logprob"),
            )
        )
    return tuple(words)


def derive_transcription_id(data: dict[str, Any], words: tuple[Word, ...]) -> str:
    """Deterministic fallback id for responses without ``transcription_id``.

    Retried jobs re-derive the same id from the same response content, which
    keeps commit_unknowns idempotent even when ElevenLabs omits the field.
    """
    h = hashlib.sha1()
    h.update(str(data.get("language_code", "")).encode())
    h.update(str(len(words)).encode())
    for w in words[:50]:
        h.update(f"{w.text}|{w.start}|{w.end}|{w.speaker_id};".encode())
    return f"derived-{h.hexdigest()[:16]}"


def _parse_single(data: dict[str, Any], transcription_id: str | None) -> ParsedTranscript:
    words = _parse_words(data.get("words"))
    tid = transcription_id or data.get("transcription_id") or derive_transcription_id(data, words)
    return ParsedTranscript(
        language_code=str(data.get("language_code") or "und"),
        words=words,
        channel_index=data.get("channel_index"),
        audio_duration_secs=data.get("audio_duration_secs"),
        transcription_id=tid,
    )


def parse_scribe_response(resp: Any) -> list[ParsedTranscript]:
    """Parse a Scribe v2 response into one transcript per channel.

    Single-channel responses yield a one-element list. Raises
    :class:`ScribeParseError` only for inputs that are not recognizable as a
    Scribe response at all.
    """
    data = _coerce_to_dict(resp)
    if "transcripts" in data and isinstance(data["transcripts"], list):
        transcripts = [t for t in data["transcripts"] if isinstance(t, dict)]
        if not transcripts:
            raise ScribeParseError("multichannel response has an empty 'transcripts' array")
        shared_tid = data.get("transcription_id")
        parsed = []
        for i, t in enumerate(transcripts):
            p = _parse_single(t, shared_tid)
            if p.channel_index is None:
                p = dataclasses.replace(p, channel_index=i)
            parsed.append(p)
        return parsed
    if "words" not in data and "text" not in data:
        raise ScribeParseError(
            "response has neither 'words', 'text', nor 'transcripts' — not a Scribe payload"
        )
    return [_parse_single(data, None)]
