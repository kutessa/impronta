"""Turning identification results into named transcripts.

This is the payoff step for summary pipelines: replace ``speaker_0`` with
``Tarik`` before the transcript reaches an LLM prompt.
"""

from __future__ import annotations

import copy
from typing import Any

from .models import IdentifyResult, SpeakerMatch
from .scribe import parse_scribe_response


def resolve_label(match: SpeakerMatch | None, query_speaker_id: str) -> str:
    """Human-readable label for one diarized speaker."""
    if match is None:
        return f"Speaker {query_speaker_id}"
    if match.display_name:
        return match.display_name
    if match.speaker_key:  # committed-but-unlabeled unknown
        return f"Unknown ({match.speaker_key})"
    return f"Speaker {query_speaker_id}"


def apply_names(scribe_response: Any, result: IdentifyResult) -> dict[str, Any]:
    """Return a deep copy of the response with ``speaker_name`` on every word.

    ``speaker_id`` is preserved untouched; a new ``speaker_name`` field is
    added next to it (resolved display name, unknown label, or a
    ``Speaker <id>`` fallback). Handles both mono and multichannel shapes.
    """
    parsed = parse_scribe_response(scribe_response)
    multi = len(parsed) > 1
    data: dict[str, Any] = copy.deepcopy(
        scribe_response if isinstance(scribe_response, dict) else scribe_response.model_dump()
    )

    def annotate(words: list[dict[str, Any]], channel: int | None) -> None:
        for w in words:
            sid = w.get("speaker_id")
            if sid is None:
                continue
            qid = f"{channel}:{sid}" if multi else sid
            w["speaker_name"] = resolve_label(result.speakers.get(qid), qid)

    if multi:
        for i, t in enumerate(data.get("transcripts", [])):
            channel = t.get("channel_index", i)
            annotate(t.get("words") or [], channel)
    else:
        annotate(data.get("words") or [], None)
    return data


def format_transcript(scribe_response: Any, result: IdentifyResult) -> str:
    """Render ``Name: text`` lines, one per speaker turn.

    Designed to be dropped straight into a summary prompt. Words merge into
    a turn until the speaker changes; spacing entries are preserved,
    audio_event entries are skipped.
    """
    parsed = parse_scribe_response(scribe_response)
    multi = len(parsed) > 1
    lines: list[str] = []
    for transcript in parsed:
        # collect (speaker_id, texts) turns first, then render
        turns: list[tuple[str, list[str]]] = []
        current_sid: str | None = None
        for w in transcript.words:
            if w.type == "audio_event":
                continue
            if w.type == "word" and w.speaker_id != current_sid:
                current_sid = w.speaker_id
                if current_sid is not None:
                    turns.append((current_sid, []))
            if turns and current_sid is not None:
                turns[-1][1].append(w.text)
        for sid, texts in turns:
            text = "".join(texts).strip()
            if not text:
                continue
            qid = f"{transcript.channel_index}:{sid}" if multi else sid
            lines.append(f"{resolve_label(result.speakers.get(qid), qid)}: {text}")
    return "\n".join(lines)
