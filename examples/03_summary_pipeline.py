"""The canto shape: transcribe -> identify -> named transcript -> LLM summary.

apply_names/format_transcript turn "speaker_0 said..." into "Tarik said...",
which is what makes AI summaries actually readable.

Usage:
    export ELEVENLABS_API_KEY=...
    uv run python examples/03_summary_pipeline.py meeting.wav
"""

import sys

from impronta import FaissLocalStore, Impronta, apply_names, format_transcript

sys.path.insert(0, "examples")
from importlib import import_module

transcribe = import_module("01_enroll_and_identify").transcribe

SUMMARY_PROMPT = """\
Summarize this meeting transcript. Attribute action items to people by name.

Transcript:
{transcript}
"""


def main() -> None:
    audio_path = sys.argv[1]
    # load a previously built speaker db (see example 01)
    app = Impronta(store=FaissLocalStore.load("./speaker_db"))

    resp = transcribe(audio_path)
    result = app.identify(resp, audio_path)

    # 1. a prompt-ready "Name: text" transcript
    named_transcript = format_transcript(resp, result)
    prompt = SUMMARY_PROMPT.format(transcript=named_transcript)
    print(prompt[:2000])

    # 2. or the full response dict with speaker_name added to every word,
    #    if your pipeline stores richer structure
    annotated = apply_names(resp, result)
    print({w["speaker_name"] for w in annotated["words"] if "speaker_name" in w})

    # feed `prompt` to your LLM of choice here, e.g.:
    # import anthropic
    # client = anthropic.Anthropic()
    # message = client.messages.create(
    #     model="claude-sonnet-5",
    #     max_tokens=1024,
    #     messages=[{"role": "user", "content": prompt}],
    # )
    # print(message.content[0].text)


if __name__ == "__main__":
    main()
