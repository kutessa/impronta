"""Build dataset/manifest.jsonl from the Firebase bucket dump.

Scans data/tessa-canto.firebasestorage.app/users/{uid}/recordings/*.m4a and
raw/*.json, pairs them by recording UUID, extracts the recording timestamp
from the MP4 container `creation_time` tag (file mtimes are all download
time), and classifies exclusions (bot transcripts, no speech, missing
counterpart).

Also validates per-account timestamp plausibility — the chronological eval
protocols depend on ordering, so this is a go/no-go gate:

    uv run python tuning/build_manifest.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tuning.eval_lib.dataset import DATA_ROOT, DATASET_DIR, REPO_ROOT, RecordingRow


def container_timestamp(path: Path) -> str | None:
    import av

    try:
        with av.open(str(path)) as container:
            ts = container.metadata.get("creation_time")
            return ts if ts else None
    except Exception:
        return None


def build() -> list[RecordingRow]:
    rows: list[RecordingRow] = []
    for user_dir in sorted(DATA_ROOT.iterdir()):
        if not user_dir.is_dir():
            continue
        uid = user_dir.name
        audio = {p.stem: p for p in (user_dir / "recordings").glob("*.m4a")}
        raws = {p.stem: p for p in (user_dir / "raw").glob("*.json")}
        for rid in sorted(audio.keys() | raws.keys()):
            audio_path = audio.get(rid)
            raw_path = raws.get(rid)
            resp: dict = {}
            if raw_path is not None:
                try:
                    loaded = json.loads(raw_path.read_text())
                    # bot_*.participants.json are meeting rosters (lists), not
                    # Scribe responses — leave resp empty for anything non-dict
                    resp = loaded if isinstance(loaded, dict) else {}
                except Exception:
                    resp = {}
            words = resp.get("words") or []
            speakers = {
                w.get("speaker_id")
                for w in words
                if w.get("speaker_id") and w.get("start") is not None
            }
            n_words = sum(1 for w in words if w.get("type") == "word")

            exclude: str | None = None
            if rid.startswith("bot_"):
                exclude = "bot"
            elif audio_path is None:
                exclude = "missing_audio"
            elif raw_path is None:
                exclude = "missing_json"
            elif not speakers:
                exclude = "no_speech"

            timestamp = (
                container_timestamp(audio_path) if audio_path is not None else None
            )
            rows.append(
                RecordingRow(
                    uid=uid,
                    rid=rid,
                    audio_path=(
                        str(audio_path.relative_to(REPO_ROOT)) if audio_path else ""
                    ),
                    raw_path=str(raw_path.relative_to(REPO_ROOT)) if raw_path else "",
                    timestamp=timestamp,
                    timestamp_source="container" if timestamp else None,
                    duration_secs=resp.get("audio_duration_secs"),
                    language_code=resp.get("language_code"),
                    transcription_id=resp.get("transcription_id"),
                    n_speakers=len(speakers),
                    n_words=n_words,
                    usable=exclude is None,
                    exclude_reason=exclude,
                )
            )
    return rows


def validate_timestamps(rows: list[RecordingRow]) -> bool:
    """Per-account timestamp sanity for the chronological protocols."""
    ok = True
    print("\n== timestamp validation ==")
    for uid in sorted({r.uid for r in rows}):
        usable = [r for r in rows if r.uid == uid and r.usable]
        missing = [r for r in usable if r.timestamp is None]
        stamped = sorted(
            (r for r in usable if r.timestamp is not None), key=lambda r: r.timestamp
        )
        status = []
        if missing:
            status.append(f"{len(missing)} missing timestamps")
            ok = False
        if len(stamped) >= 2:
            span_days = (
                _iso(stamped[-1].timestamp) - _iso(stamped[0].timestamp)
            ).days
            dupes = len(stamped) - len({r.timestamp for r in stamped})
            status.append(f"span {span_days}d")
            if dupes:
                status.append(f"{dupes} duplicate timestamps")
            if span_days == 0 and len(stamped) > 3:
                status.append("SUSPICIOUS: all same day")
                ok = False
        print(f"  {uid[:10]}: {len(usable)} usable, {'; '.join(status) or 'ok'}")
    return ok


def _iso(ts: str):
    from datetime import datetime

    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def main() -> None:
    rows = build()
    DATASET_DIR.mkdir(exist_ok=True)
    out = DATASET_DIR / "manifest.jsonl"
    out.write_text("".join(json.dumps(r.to_dict()) + "\n" for r in rows))

    usable = [r for r in rows if r.usable]
    reasons = Counter(r.exclude_reason for r in rows if not r.usable)
    multi = sum(1 for r in usable if r.n_speakers >= 2)
    hours = sum(r.duration_secs or 0 for r in usable) / 3600
    print(f"wrote {out} ({len(rows)} rows)")
    print(
        f"usable: {len(usable)} ({multi} multi-speaker, {hours:.1f}h); "
        f"excluded: {dict(reasons)}"
    )
    ts_ok = validate_timestamps(rows)
    print("\ntimestamp gate:", "PASS" if ts_ok else "NEEDS ATTENTION")


if __name__ == "__main__":
    main()
