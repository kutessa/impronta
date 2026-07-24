"""Precompute the embedding cache for every usable recording.

The ONLY tuning script that imports torch. For each usable manifest row:
decode audio -> segment_words -> measure_segments (structural gate + WADA)
-> embed ALL measured segments -> save dataset/cache/{uid}/{rid}.npz.

Resumable: existing cache files with the current schema/seg-knobs are
skipped. Run:

    uv run python tuning/precompute_cache.py [--limit N] [--uid UID]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from impronta import EcapaEmbedder, ImprontaConfig
from impronta.audio import load_audio, select_channel
from impronta.embedder import l2_normalize
from impronta.pipeline import measure_segments
from impronta.scribe import parse_scribe_response
from impronta.segmentation import segment_words
from tuning.eval_lib.cache import (
    SCHEMA_VERSION,
    RecordingCache,
    cache_path,
    load_cache,
    save_cache,
    seg_knobs,
)
from tuning.eval_lib.dataset import REPO_ROOT, RecordingRow, load_manifest

EMBED_BATCH = 8


def embed_bucketed(embedder: EcapaEmbedder, clips: list, sample_rate: int) -> np.ndarray:
    """Embed clips in near-uniform-length batches, restoring input order.

    Torch on this machine showed a pathological slowdown (100x) on one
    specific large mixed-length batch; small batches bucketed by length
    never exhibited it. Batches group clips whose lengths fall in the same
    2-second bucket, max EMBED_BATCH per call.
    """
    order = sorted(range(len(clips)), key=lambda i: clips[i].shape[0])
    rows: dict[int, np.ndarray] = {}
    batch: list[int] = []

    def flush() -> None:
        if not batch:
            return
        embedded = l2_normalize(
            embedder.embed_batch([clips[i] for i in batch], sample_rate)
        )
        for j, i in enumerate(batch):
            rows[i] = embedded[j]
        batch.clear()

    bucket = None
    for i in order:
        b = clips[i].shape[0] // (2 * sample_rate)
        if batch and (b != bucket or len(batch) >= EMBED_BATCH):
            flush()
        bucket = b
        batch.append(i)
    flush()
    return np.stack([rows[i] for i in range(len(clips))])


def build_one(row: RecordingRow, embedder: EcapaEmbedder, cfg: ImprontaConfig) -> RecordingCache:
    resp = json.loads((REPO_ROOT / row.raw_path).read_text())
    transcripts = parse_scribe_response(resp)
    decoded = load_audio(REPO_ROOT / row.audio_path, cfg.sample_rate)
    multi = len(transcripts) > 1

    embeddings: list[np.ndarray] = []
    starts: list[float] = []
    ends: list[float] = []
    snrs: list[float] = []
    confs: list[float] = []
    qids: list[str] = []
    segments_total: dict[str, int] = {}

    for transcript in transcripts:
        mono = select_channel(decoded, transcript.channel_index if multi else None)
        segment_map = segment_words(transcript.words, cfg)
        for sid in transcript.speaker_ids():
            qid = f"{transcript.channel_index}:{sid}" if multi else sid
            segs = segment_map.get(sid, [])
            measured = measure_segments(segs, mono, cfg)
            segments_total[qid] = measured.segments_total
            if not measured.clips:
                continue
            embeddings.append(
                embed_bucketed(embedder, measured.clips, cfg.sample_rate)
            )
            for seg in measured.segments:
                starts.append(seg.start)
                ends.append(seg.end)
                snrs.append(seg.snr_db if seg.snr_db is not None else -20.0)
                confs.append(seg.confidence)
                qids.append(qid)

    emb = (
        np.concatenate(embeddings)
        if embeddings
        else np.zeros((0, embedder.dim), dtype=np.float32)
    )
    return RecordingCache(
        uid=row.uid,
        rid=row.rid,
        transcription_id=row.transcription_id or "",
        language_code=row.language_code or "und",
        embeddings=emb,
        start=np.asarray(starts, dtype=np.float32),
        end=np.asarray(ends, dtype=np.float32),
        snr_db=np.asarray(snrs, dtype=np.float32),
        confidence=np.asarray(confs, dtype=np.float32),
        qid=np.asarray(qids, dtype=np.str_),
        segments_total=segments_total,
        meta={
            "seg_knobs": seg_knobs(cfg),
            "model_source": cfg.model_source,
            "audio_duration_secs": row.duration_secs,
        },
    )


def is_cached(row: RecordingRow, cfg: ImprontaConfig) -> bool:
    path = cache_path(row.uid, row.rid)
    if not path.exists():
        return False
    try:
        load_cache(row.uid, row.rid, cfg)
        return True
    except (ValueError, KeyError, OSError):
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--uid", default=None)
    args = ap.parse_args()

    # Torch's default all-cores pthreadpool SPINS pathologically on Apple
    # Silicon for these conv shapes (measured: 100x slower than one thread).
    # One thread embeds at ~60x realtime — the whole dataset in <1 hour.
    torch.set_num_threads(1)

    cfg = ImprontaConfig()
    rows = [r for r in load_manifest() if r.usable]
    if args.uid:
        rows = [r for r in rows if r.uid.startswith(args.uid)]
    todo = [r for r in rows if not is_cached(r, cfg)]
    n_cached = len(rows) - len(todo)
    if args.limit:
        todo = todo[: args.limit]
    print(
        f"{len(rows)} usable recordings; {n_cached} cached; "
        f"{len(todo)} to embed (schema v{SCHEMA_VERSION})"
    )
    if not todo:
        return

    embedder = EcapaEmbedder(cfg)
    t0 = time.time()
    done_audio = 0.0
    for i, row in enumerate(todo, 1):
        t_rec = time.time()
        try:
            cache = build_one(row, embedder, cfg)
        except Exception as exc:
            print(f"[{i}/{len(todo)}] {row.uid[:8]}/{row.rid[:8]} FAILED: {exc}")
            continue
        save_cache(cache)
        done_audio += row.duration_secs or 0
        rate = done_audio / max(time.time() - t0, 1)
        print(
            f"[{i}/{len(todo)}] {row.uid[:8]}/{row.rid[:8]}: "
            f"{cache.embeddings.shape[0]:4d} segs, {len(cache.segments_total)} speakers, "
            f"{(row.duration_secs or 0):5.0f}s audio in {time.time() - t_rec:4.1f}s "
            f"(cum {rate:.0f}x realtime)"
        )
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
