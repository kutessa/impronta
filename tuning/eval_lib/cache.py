"""Per-recording embedding cache: measured segments + embeddings for ALL of them.

Built once by ``tuning/precompute_cache.py`` (the only tuning code that
imports torch); consumed by the replay layer, which re-runs
``impronta.pipeline.select_segments`` + everything downstream per trial
config in pure numpy.

Validity: the cache embeds segments produced by ``segment_words`` +
``measure_segments`` under specific SEGMENTATION knobs (min/max_segment_sec,
gap_tolerance_sec) and a specific embedder. Those are stored in the meta and
checked on load — a config that changes them invalidates the cache and must
be rejected, not silently mis-replayed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from impronta.config import ImprontaConfig
from impronta.models import Segment

from .dataset import DATASET_DIR

SCHEMA_VERSION = 1
CACHE_DIR = DATASET_DIR / "cache"


def seg_knobs(cfg: ImprontaConfig) -> dict:
    return {
        "min_segment_sec": cfg.min_segment_sec,
        "max_segment_sec": cfg.max_segment_sec,
        "gap_tolerance_sec": cfg.gap_tolerance_sec,
        "sample_rate": cfg.sample_rate,
    }


@dataclass(slots=True)
class RecordingCache:
    """All measured segments of one recording, embedded, grouped by qid."""

    uid: str
    rid: str
    transcription_id: str
    language_code: str  # raw STT code (canonicalized downstream)
    embeddings: np.ndarray  # (n, dim) f32 L2-normalized
    start: np.ndarray  # (n,) f32
    end: np.ndarray  # (n,) f32
    snr_db: np.ndarray  # (n,) f32
    confidence: np.ndarray  # (n,) f32
    qid: np.ndarray  # (n,) unicode
    segments_total: dict[str, int]  # per qid, PRE-slice counts (parity)
    meta: dict

    def qids(self) -> list[str]:
        """All speakers, including those with zero embeddable segments."""
        return sorted(self.segments_total)

    def rows_for(self, qid: str) -> np.ndarray:
        return np.nonzero(self.qid == qid)[0]

    def segments_for(self, qid: str) -> tuple[list[Segment], np.ndarray]:
        """Measured segments + row-aligned embeddings for one speaker."""
        rows = self.rows_for(qid)
        segs = [
            Segment(
                speaker_id=qid,
                start=float(self.start[i]),
                end=float(self.end[i]),
                confidence=float(self.confidence[i]),
                snr_db=float(self.snr_db[i]),
            )
            for i in rows
        ]
        return segs, self.embeddings[rows]


def cache_path(uid: str, rid: str) -> Path:
    return CACHE_DIR / uid / f"{rid}.npz"


def save_cache(cache: RecordingCache, path: Path | None = None) -> Path:
    path = path or cache_path(cache.uid, cache.rid)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(cache.meta)
    meta.update(
        {
            "schema_version": SCHEMA_VERSION,
            "uid": cache.uid,
            "rid": cache.rid,
            "transcription_id": cache.transcription_id,
            "language_code": cache.language_code,
            "segments_total": cache.segments_total,
        }
    )
    np.savez_compressed(
        path,
        embeddings=cache.embeddings.astype(np.float32),
        start=cache.start.astype(np.float32),
        end=cache.end.astype(np.float32),
        snr_db=cache.snr_db.astype(np.float32),
        confidence=cache.confidence.astype(np.float32),
        qid=cache.qid.astype(np.str_),
        meta=np.array(json.dumps(meta)),
    )
    return path


def load_cache(
    uid: str, rid: str, cfg: ImprontaConfig | None = None, path: Path | None = None
) -> RecordingCache:
    """Load one recording's cache; verify schema and segmentation knobs."""
    path = path or cache_path(uid, rid)
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        if meta.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"{path}: cache schema {meta.get('schema_version')} != {SCHEMA_VERSION}; "
                "re-run tuning/precompute_cache.py"
            )
        if cfg is not None:
            expected = seg_knobs(cfg)
            got = meta.get("seg_knobs", {})
            if got != expected:
                raise ValueError(
                    f"{path}: cache built with segmentation knobs {got}, config has "
                    f"{expected} — segmentation knobs are cache identity and cannot "
                    "be tuned against this cache"
                )
        return RecordingCache(
            uid=meta["uid"],
            rid=meta["rid"],
            transcription_id=meta["transcription_id"],
            language_code=meta["language_code"],
            embeddings=z["embeddings"],
            start=z["start"],
            end=z["end"],
            snr_db=z["snr_db"],
            confidence=z["confidence"],
            qid=z["qid"],
            segments_total={k: int(v) for k, v in meta["segments_total"].items()},
            meta=meta,
        )
