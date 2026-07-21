"""Enroll a speaker from one recording, identify them in another.

Usage:
    export ELEVENLABS_API_KEY=...
    uv run python examples/01_enroll_and_identify.py enroll.wav query.wav

Both files should contain speech; the first is used to enroll the first
diarized speaker under a name, the second is then identified against the
store. Requires `pip install requests` (or the elevenlabs SDK — any client
that returns the Scribe v2 response dict works).
"""

import os
import sys

import requests

from impronta import FaissLocalStore, Impronta


def transcribe(path: str) -> dict:
    with open(path, "rb") as f:
        response = requests.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
            data={"model_id": "scribe_v2", "diarize": "true"},
            files={"file": f},
            timeout=300,
        )
    response.raise_for_status()
    return response.json()


def main() -> None:
    enroll_path, query_path = sys.argv[1], sys.argv[2]

    app = Impronta(store=FaissLocalStore())

    enroll_resp = transcribe(enroll_path)
    first_speaker = next(
        w["speaker_id"] for w in enroll_resp["words"] if w.get("speaker_id")
    )
    result = app.add_speaker(enroll_resp, enroll_path, first_speaker, "Hasta")
    print(f"enrolled {result.display_name!r} as {result.speaker_key!r}: "
          f"{result.segments_used}/{result.segments_total} segments, "
          f"quality={result.quality_tier}")

    query_resp = transcribe(query_path)
    identified = app.identify(query_resp, query_path)
    for speaker_id, match in identified.speakers.items():
        if not match.identifiable:
            print(f"{speaker_id}: not enough clean audio to identify")
        elif match.is_unknown:
            print(f"{speaker_id}: unknown voice")
        else:
            print(f"{speaker_id}: {match.display_name} "
                  f"(similarity {match.mean_similarity:.2f}, "
                  f"share {match.candidates[0].score_share:.0%})")

    # persist the speaker db for the next run
    app.store.save("./speaker_db")


if __name__ == "__main__":
    main()
