"""Render Label Studio annotation tasks: one per (recording, speaker).

For every usable recording and diarized speaker: concatenate that speaker's
3 best segments (highest WADA SNR from the embedding cache, longest first
among the top half) into a ~15-20s snippet WAV under dataset/ls_media/,
plus a transcript excerpt, and emit dataset/ls_tasks.json for import.

Run AFTER precompute_cache.py (reads segment bounds + SNR from the cache;
decodes each recording once):

    uv run python tuning/render_annotation_tasks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from impronta import ImprontaConfig
from impronta.audio import load_audio, select_channel, slice_seconds
from tuning.eval_lib.cache import cache_path, load_cache
from tuning.eval_lib.dataset import DATASET_DIR, REPO_ROOT, load_manifest

MEDIA_DIR = DATASET_DIR / "ls_media" / "audio"
TARGET_SNIPPET_SEC = 18.0
MAX_SEGMENTS = 3
SPACER_SEC = 0.4


def speaker_excerpt(resp: dict, qid: str, max_chars: int = 400) -> str:
    sid = qid.split(":")[-1]
    words = [
        w.get("text", "")
        for w in (resp.get("words") or [])
        if w.get("speaker_id") == sid and w.get("type") in ("word", "spacing")
    ]
    text = "".join(words).strip()
    return text[:max_chars] + ("…" if len(text) > max_chars else "")


def pick_segments(cache, qid: str) -> list[tuple[float, float]]:
    """Top-SNR half, then longest first, until the snippet target is met."""
    rows = cache.rows_for(qid)
    if rows.size == 0:
        return []
    snr = cache.snr_db[rows]
    order = rows[np.argsort(-snr)]
    top_half = order[: max(1, len(order) // 2 + 1)]
    durations = cache.end[top_half] - cache.start[top_half]
    by_len = top_half[np.argsort(-durations)]
    picked: list[tuple[float, float]] = []
    total = 0.0
    for i in by_len[:MAX_SEGMENTS]:
        start, end = float(cache.start[i]), float(cache.end[i])
        end = min(end, start + TARGET_SNIPPET_SEC)  # one huge segment is enough
        picked.append((start, end))
        total += end - start
        if total >= TARGET_SNIPPET_SEC:
            break
    picked.sort()
    return picked


def main() -> None:
    cfg = ImprontaConfig()
    sr = cfg.sample_rate
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = [r for r in load_manifest() if r.usable]
    tasks: list[dict] = []
    skipped = 0

    for row in manifest:
        if not cache_path(row.uid, row.rid).exists():
            skipped += 1
            continue
        cache = load_cache(row.uid, row.rid, cfg)
        resp = json.loads((REPO_ROOT / row.raw_path).read_text())
        decoded = None  # decode lazily, once per recording
        for qid in cache.qids():
            segments = pick_segments(cache, qid)
            if not segments:
                continue  # nothing embeddable — nothing to annotate by ear
            wav_name = f"{row.uid}__{row.rid}__{qid.replace(':', '-')}.wav"
            wav_path = MEDIA_DIR / wav_name
            if not wav_path.exists():
                if decoded is None:
                    decoded = load_audio(REPO_ROOT / row.audio_path, sr)
                mono = select_channel(decoded, None)
                spacer = np.zeros(int(SPACER_SEC * sr), dtype=np.float32)
                parts: list[np.ndarray] = []
                for start, end in segments:
                    if parts:
                        parts.append(spacer)
                    parts.append(slice_seconds(mono, sr, start, end))
                sf.write(wav_path, np.concatenate(parts), sr, subtype="PCM_16")
            tasks.append(
                {
                    "data": {
                        "audio": f"/data/local-files/?d=audio/{wav_name}",
                        "uid": row.uid,
                        "rid": row.rid,
                        "speaker_id": qid,
                        "n_speakers": row.n_speakers,
                        "duration": round(row.duration_secs or 0),
                        "timestamp": row.timestamp,
                        "transcript_excerpt": speaker_excerpt(resp, qid),
                    }
                }
            )

    out = DATASET_DIR / "ls_tasks.json"
    out.write_text(json.dumps(tasks, ensure_ascii=False, indent=1))
    print(f"wrote {len(tasks)} tasks to {out} ({skipped} recordings skipped, no cache)")
    print(f"snippet WAVs in {MEDIA_DIR}")


if __name__ == "__main__":
    main()
