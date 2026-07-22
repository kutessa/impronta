"""Interactive impronta demo: upload -> Scribe v2 -> speaker ID -> label.

Run with:
    uv run --group demo python demo/app.py

Upload a recording; the demo transcribes it with ElevenLabs Scribe v2,
shows the raw diarized transcript (speaker_0, speaker_1, ...), runs
impronta speaker identification against a persistent local speaker db,
shows the renamed transcript, and lets you label any diarized speaker
("speaker_2 is actually Ahmed") — which enrolls their voice for future
recordings. The db lives in demo/speaker_db/ and survives restarts.

Needs ELEVENLABS_API_KEY (read from the environment or the repo .env).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# macOS: faiss-cpu + torch each bundle libomp; see README caveats
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import logging
import time
from contextlib import contextmanager

import gradio as gr
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
log = logging.getLogger("impronta.demo")
for noisy in ("httpx", "urllib3", "faiss", "faiss.loader"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


@contextmanager
def timed(what: str):
    log.info("%s ...", what)
    t0 = time.perf_counter()
    yield
    log.info("%s done in %.1fs", what, time.perf_counter() - t0)

from impronta import (  # noqa: E402 - must come after the OpenMP env setup
    FaissLocalStore,
    IdentifyResult,
    Impronta,
    ImprontaError,
    format_transcript,
)
from impronta.scribe import parse_scribe_response  # noqa: E402

REPO = Path(__file__).parent.parent
DB_DIR = Path(__file__).parent / "speaker_db"


def _api_key() -> str:
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key and (REPO / ".env").exists():
        for line in (REPO / ".env").read_text().splitlines():
            if line.startswith("ELEVENLABS_API_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        raise RuntimeError("set ELEVENLABS_API_KEY (env or repo .env)")
    return key


API_KEY = _api_key()

store = (
    FaissLocalStore.load(DB_DIR)
    if (DB_DIR / "impronta_store.json").exists()
    else FaissLocalStore()
)
app = Impronta(store=store)


def transcribe(path: str) -> dict:
    size_mb = os.path.getsize(path) / 1e6
    with timed(f"scribe v2 transcription ({Path(path).name}, {size_mb:.1f} MB)"):
        with open(path, "rb") as f:
            r = requests.post(
                "https://api.elevenlabs.io/v1/speech-to-text",
                headers={"xi-api-key": API_KEY},
                data={"model_id": "scribe_v2", "diarize": "true"},
                files={"file": f},
                timeout=300,
            )
        r.raise_for_status()
    resp = r.json()
    words = resp.get("words") or []
    log.info(
        "transcript: %d words, %.0fs audio, language=%s, speakers=%s",
        sum(1 for w in words if w.get("type") == "word"),
        resp.get("audio_duration_secs") or 0,
        resp.get("language_code"),
        sorted({w["speaker_id"] for w in words if w.get("speaker_id")}),
    )
    return resp


# ---------------------------------------------------------------------------
# rendering helpers
# ---------------------------------------------------------------------------


def _empty_result(resp: dict) -> IdentifyResult:
    t = parse_scribe_response(resp)[0]
    return IdentifyResult(
        speakers={}, proposed_unknowns=(), language_code=t.language_code,
        transcription_id=t.transcription_id,
    )


def as_markdown(transcript_text: str) -> str:
    lines = []
    for line in transcript_text.splitlines():
        label, _, text = line.partition(": ")
        lines.append(f"**{label}:**  {text}\n")
    return "\n".join(lines) if lines else "*(no speech found)*"


def match_details(result: IdentifyResult) -> str:
    rows = ["| speaker | resolved | similarity | quality | notes |", "|---|---|---|---|---|"]
    for sid, m in result.speakers.items():
        if not m.identifiable:
            rows.append(f"| {sid} | — | — | — | not enough clean audio |")
            continue
        sim = f"{m.mean_similarity:.2f}" if m.mean_similarity is not None else "—"
        if m.display_name:
            named = [c for c in m.candidates if c.speaker_key not in (m.speaker_key, "<unknown>")
                     and c.mean_similarity is not None]
            runner = named[0] if named else None
            contested = runner is not None and m.mean_similarity is not None and (
                m.mean_similarity - runner.mean_similarity
            ) < 0.10
            weak = m.mean_similarity is not None and m.mean_similarity < 0.55
            if contested:
                resolved = f"**{m.display_name}**? (contested)"
                notes = (f"runner-up: {runner.display_name or runner.speaker_key} "
                         f"@ {runner.mean_similarity:.2f}")
            elif weak:
                resolved = f"**{m.display_name}**? (weak)"
                notes = ""
            else:
                resolved = f"**{m.display_name}**"
                notes = ""
        elif m.speaker_key:
            resolved = "unknown (seen before)"
            notes = f"recognized `{m.speaker_key}`"
        else:
            resolved = "unknown"
            notes = (
                f"gated: {m.no_proposal_reason}" if m.no_proposal_reason else "new voice (proposed)"
            )
        if m.near_misses:
            k, s = m.near_misses[0]
            notes += f" · near miss: {k} @ {s:.2f}"
        rows.append(f"| {sid} | {resolved} | {sim} | {m.quality_tier or '—'} | {notes} |")
    if result.proposed_unknowns:
        keys = ", ".join(f"`{p.suggested_key}`" for p in result.proposed_unknowns)
        rows.append(f"\nProposed new unknown voiceprints (not stored): {keys}")
    return "\n".join(rows)


def speakers_table() -> list[list]:
    return [
        [s.speaker_key, s.display_name or "(unlabeled unknown)", ", ".join(s.languages),
         s.num_embeddings]
        for s in app.list_speakers()
    ]


def _identify_and_render(resp: dict, audio_path: str):
    with timed("speaker identification (first run also loads the ECAPA model)"):
        result = app.identify(resp, audio_path)
    for sid, m in result.speakers.items():
        if not m.identifiable:
            log.info("  %s: unidentifiable (no clean segments)", sid)
        elif m.display_name:
            log.info("  %s: %s (sim %.2f, %d segments, tier %s)", sid, m.display_name,
                     m.mean_similarity or 0, m.num_segments_used, m.quality_tier)
        else:
            log.info("  %s: unknown (key=%s, gate=%s, %d segments)", sid, m.speaker_key,
                     m.no_proposal_reason or "proposed", m.num_segments_used)
    named = as_markdown(format_transcript(resp, result))
    return result, named


# ---------------------------------------------------------------------------
# gradio callbacks
# ---------------------------------------------------------------------------


def run_pipeline(audio_path: str | None, progress=gr.Progress()):  # noqa: B008 - gradio idiom
    if not audio_path:
        raise gr.Error("upload a recording first")
    log.info("=== run: %s", audio_path)
    progress(0.05, desc="uploading to ElevenLabs Scribe v2…")
    try:
        resp = transcribe(audio_path)
    except requests.RequestException as exc:
        log.exception("transcription failed")
        raise gr.Error(f"ElevenLabs transcription failed: {exc}") from exc
    progress(0.5, desc="transcribed — running speaker identification…")
    raw = as_markdown(format_transcript(resp, _empty_result(resp)))
    try:
        result, named = _identify_and_render(resp, audio_path)
    except ImprontaError as exc:
        log.exception("identification failed")
        raise gr.Error(f"speaker identification failed: {exc}") from exc

    # strengthen profiles from confident, uncontested matches
    reinforced_note = ""
    try:
        progress(0.85, desc="reinforcing matched profiles…")
        with timed("profile reinforcement pass"):
            reinforcements = app.propose_reinforcements(resp, audio_path)
        committed = app.commit_reinforcements(reinforcements)
        for p, key in zip(reinforcements, committed, strict=True):
            if key:
                log.info("  reinforced %s (%s) with %d segments (sim %.2f)",
                         p.display_name or key, key, p.embeddings.shape[0],
                         p.mean_similarity)
        if any(committed):
            store.save(DB_DIR)
            reinforced_note = " · reinforced: " + ", ".join(
                f"{p.display_name or k} +{p.embeddings.shape[0]}"
                for p, k in zip(reinforcements, committed, strict=True) if k
            )
    except ImprontaError:
        log.exception("reinforcement failed (identification result unaffected)")
    progress(1.0, desc="done")
    sids = list(result.speakers.keys())
    state = {"resp": resp, "audio_path": audio_path}
    return (
        raw,
        named,
        match_details(result),
        gr.update(choices=sids, value=(sids[0] if sids else None)),
        state,
        speakers_table(),
        f"transcribed ({result.language_code}), {len(sids)} speaker(s) found{reinforced_note}",
    )


def enroll(state: dict | None, speaker_id: str | None, name: str):
    if not state:
        raise gr.Error("run a transcription first")
    if not speaker_id:
        raise gr.Error("pick a speaker")
    name = (name or "").strip()
    if not name:
        raise gr.Error("enter a name")
    resp, audio_path = state["resp"], state["audio_path"]
    try:
        with timed(f"enrolling {speaker_id} as {name!r}"):
            enrolled = app.add_speaker(resp, audio_path, speaker_id.split(":")[-1], name)
    except ImprontaError as exc:
        log.exception("enroll failed")
        raise gr.Error(str(exc)) from exc
    store.save(DB_DIR)
    log.info("db saved: %d entries, %d speakers",
             store.count("default"), len(app.list_speakers()))
    result, named = _identify_and_render(resp, audio_path)
    merged = (
        f", absorbed unknown(s): {', '.join(enrolled.merged_unknown_keys)}"
        if enrolled.merged_unknown_keys
        else ""
    )
    status = (
        f"enrolled {name!r} as `{enrolled.speaker_key}` "
        f"({enrolled.segments_used} segments, quality {enrolled.quality_tier}{merged})"
    )
    if enrolled.quality_tier == "low":
        status += (
            " — ⚠️ noisy/telephone-grade audio: this profile will match weakly; "
            "enroll the same person from a cleaner recording when you can"
        )
    return named, match_details(result), speakers_table(), status, ""


def remove_speaker(speaker_key: str):
    key = (speaker_key or "").strip()
    if not key:
        raise gr.Error("enter a speaker key to remove")
    removed = app.remove_speaker(key)
    if removed == 0:
        raise gr.Error(f"no speaker {key!r} in the db")
    store.save(DB_DIR)
    return speakers_table(), f"removed {key!r} ({removed} embeddings)", ""


with gr.Blocks(title="impronta demo") as demo:
    gr.Markdown(
        "# impronta — speaker identification demo\n"
        "Upload a recording → ElevenLabs Scribe v2 diarization → impronta matches "
        "voices against the local speaker db → label anyone the db doesn't know yet."
    )
    state = gr.State(None)

    with gr.Row():
        audio = gr.Audio(type="filepath", label="recording (wav/mp3/m4a/...)")
        with gr.Column():
            run_btn = gr.Button("Transcribe & identify", variant="primary")
            status = gr.Markdown("")

    with gr.Row():
        raw_md = gr.Markdown(label="1 · ElevenLabs diarization")
        named_md = gr.Markdown(label="2 · after speaker ID")
    details_md = gr.Markdown(label="match details")

    gr.Markdown("## Label a speaker")
    with gr.Row():
        sid_dd = gr.Dropdown(choices=[], label="who?", interactive=True)
        name_tb = gr.Textbox(label="is actually…", placeholder="Ahmed")
        enroll_btn = gr.Button("Enroll into speaker db", variant="secondary")

    gr.Markdown("## Speaker database (demo/speaker_db)")
    speakers_df = gr.Dataframe(
        headers=["key", "name", "languages", "embeddings"],
        value=speakers_table(),
        interactive=False,
    )
    with gr.Row():
        remove_tb = gr.Textbox(label="remove by key", placeholder="unknown-ab12cd34ef56")
        remove_btn = gr.Button("Remove speaker")

    run_btn.click(
        run_pipeline,
        inputs=[audio],
        outputs=[raw_md, named_md, details_md, sid_dd, state, speakers_df, status],
    )
    enroll_btn.click(
        enroll,
        inputs=[state, sid_dd, name_tb],
        outputs=[named_md, details_md, speakers_df, status, name_tb],
    )
    remove_btn.click(
        remove_speaker,
        inputs=[remove_tb],
        outputs=[speakers_df, status, remove_tb],
    )


if __name__ == "__main__":
    log.info(
        "starting: db at %s (%d entries, %d speakers)",
        DB_DIR, store.count("default"), len(app.list_speakers()),
    )
    demo.launch()
