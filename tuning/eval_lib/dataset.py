"""Dataset loaders: manifest, persons registry, annotations.

The dataset lives in ``dataset/`` at the repo root:

- ``manifest.jsonl`` — one row per recording found in the bucket dump,
  including excluded ones (with ``exclude_reason``), so the inventory is
  auditable. Built by ``tuning/build_manifest.py``.
- ``persons.json`` — the global person registry: person key -> {display,
  cluster, owner_of, notes}. People fall into disjoint social clusters
  (used for the generalization holdout); guests use cluster "guests".
- ``annotations.jsonl`` — one row per annotated recording:
  {rid, uid, speakers: {qid: {person, quality}}, annotator, ts} where
  quality is "clean" | "mixed" | "garbage". Produced by
  ``tuning/labelstudio_export.py``.

Audio and raw Scribe responses stay in ``data/`` and are referenced by
repo-relative path — never copied.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
DATA_ROOT = REPO_ROOT / "data" / "tessa-canto.firebasestorage.app" / "users"

QUALITIES = ("clean", "mixed", "garbage")


@dataclass(frozen=True, slots=True)
class RecordingRow:
    uid: str
    rid: str
    audio_path: str  # repo-relative
    raw_path: str  # repo-relative
    timestamp: str | None  # ISO 8601
    timestamp_source: str | None  # "container" | "firestore" | None
    duration_secs: float | None
    language_code: str | None
    transcription_id: str | None
    n_speakers: int
    n_words: int
    usable: bool
    exclude_reason: str | None

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict) -> RecordingRow:
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class SpeakerAnnotation:
    person: str
    quality: str  # clean | mixed | garbage


@dataclass(frozen=True, slots=True)
class RecordingAnnotation:
    rid: str
    uid: str
    speakers: dict[str, SpeakerAnnotation] = field(default_factory=dict)
    annotator: str = ""
    ts: str = ""

    def to_dict(self) -> dict:
        return {
            "rid": self.rid,
            "uid": self.uid,
            "speakers": {
                qid: {"person": a.person, "quality": a.quality}
                for qid, a in self.speakers.items()
            },
            "annotator": self.annotator,
            "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RecordingAnnotation:
        return cls(
            rid=d["rid"],
            uid=d["uid"],
            speakers={
                qid: SpeakerAnnotation(person=s["person"], quality=s["quality"])
                for qid, s in d.get("speakers", {}).items()
            },
            annotator=d.get("annotator", ""),
            ts=d.get("ts", ""),
        )


@dataclass(frozen=True, slots=True)
class Person:
    key: str
    display: str
    cluster: str
    owner_of: tuple[str, ...] = ()
    notes: str = ""


def load_manifest(path: Path | None = None) -> list[RecordingRow]:
    path = path or (DATASET_DIR / "manifest.jsonl")
    rows = [
        RecordingRow.from_dict(json.loads(line))
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    seen: set[tuple[str, str]] = set()
    for r in rows:
        key = (r.uid, r.rid)
        if key in seen:
            raise ValueError(f"duplicate manifest row for {key}")
        seen.add(key)
    return rows


def usable_recordings(manifest: list[RecordingRow]) -> list[RecordingRow]:
    return [r for r in manifest if r.usable]


def load_persons(path: Path | None = None) -> dict[str, Person]:
    path = path or (DATASET_DIR / "persons.json")
    raw = json.loads(path.read_text())
    return {
        key: Person(
            key=key,
            display=v.get("display", key),
            cluster=v.get("cluster", ""),
            owner_of=tuple(v.get("owner_of", ())),
            notes=v.get("notes", ""),
        )
        for key, v in raw.items()
    }


def load_annotations(
    path: Path | None = None,
    persons: dict[str, Person] | None = None,
) -> dict[tuple[str, str], RecordingAnnotation]:
    """Load annotations keyed by (uid, rid); validate against the registry."""
    path = path or (DATASET_DIR / "annotations.jsonl")
    out: dict[tuple[str, str], RecordingAnnotation] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        ann = RecordingAnnotation.from_dict(json.loads(line))
        key = (ann.uid, ann.rid)
        if key in out:
            raise ValueError(f"duplicate annotation for {key}")
        for qid, s in ann.speakers.items():
            if s.quality not in QUALITIES:
                raise ValueError(f"{key}/{qid}: bad quality {s.quality!r}")
            if persons is not None and s.person not in persons:
                raise ValueError(
                    f"{key}/{qid}: person {s.person!r} not in persons.json"
                )
        out[key] = ann
    return out


def account_clusters(persons: dict[str, Person]) -> dict[str, str]:
    """uid -> cluster, via each account owner's person entry."""
    out: dict[str, str] = {}
    for p in persons.values():
        for uid in p.owner_of:
            if uid in out and out[uid] != p.cluster:
                raise ValueError(f"account {uid} claimed by two clusters")
            out[uid] = p.cluster
    return out
