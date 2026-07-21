"""Live tier: real ElevenLabs Scribe v2 call + real ECAPA.

Never runs in CI. Requires:
  ELEVENLABS_API_KEY            an ElevenLabs API key
  IMPRONTA_LIVE_AUDIO           path to a recording with >= 2 speakers

Run with: uv run pytest -m live -s

Also refreshes tests/fixtures/scribe_v2_golden.json (anonymized: word text
replaced) so the default tier's parser tests track the real schema.
"""

import json
import os
from pathlib import Path

import pytest

from impronta import Impronta, apply_names, format_transcript

pytestmark = [pytest.mark.live, pytest.mark.enable_socket]

FIXTURE = Path(__file__).parent / "fixtures" / "scribe_v2_golden.json"


def _transcribe(path: str) -> dict:
    import requests  # transitive via huggingface_hub; live tier only

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


def _anonymize(resp: dict) -> dict:
    out = json.loads(json.dumps(resp))
    for words in [out.get("words") or []] + [
        t.get("words") or [] for t in out.get("transcripts", [])
    ]:
        for i, w in enumerate(words):
            if w.get("type") == "word":
                w["text"] = f"word{i}"
    out["text"] = "<anonymized>"
    return out


@pytest.mark.skipif(
    not (os.environ.get("ELEVENLABS_API_KEY") and os.environ.get("IMPRONTA_LIVE_AUDIO")),
    reason="set ELEVENLABS_API_KEY and IMPRONTA_LIVE_AUDIO to run the live tier",
)
def test_live_end_to_end(tmp_path):
    audio_path = os.environ["IMPRONTA_LIVE_AUDIO"]
    resp = _transcribe(audio_path)

    # refresh the golden fixture for the default tier
    FIXTURE.parent.mkdir(exist_ok=True)
    FIXTURE.write_text(json.dumps(_anonymize(resp), indent=1))

    app = Impronta()
    result = app.identify(resp, audio_path)
    assert result.speakers, "expected at least one diarized speaker"
    print("\nname map:", result.name_map())
    print("proposals:", [p.suggested_key for p in result.proposed_unknowns])

    # enroll the first identifiable speaker under a test name, re-identify
    first = next(
        (m for m in result.speakers.values() if m.identifiable), None
    )
    if first is not None:
        app.add_speaker(resp, audio_path, first.query_speaker_id.split(":")[-1], "Live Test")
        again = app.identify(resp, audio_path)
        assert again.speakers[first.query_speaker_id].display_name == "Live Test"

    named = apply_names(resp, result)
    assert any("speaker_name" in w for w in named.get("words", []))
    print(format_transcript(resp, result)[:500])
