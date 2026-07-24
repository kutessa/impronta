"""Convert a Label Studio export into dataset/annotations.jsonl.

Export from Label Studio (project -> Export -> JSON) or via the API, then:

    uv run python tuning/labelstudio_export.py export.json

Validates hard:
- every person key exists in dataset/persons.json (grow the registry first —
  keys must be lowercase slugs, stable across recordings and accounts)
- person keys are slug-stable (label promotion uses slugify(person))
- no conflicting duplicate annotations for the same (rid, speaker)
- reports recordings with unannotated speakers (annotation still incomplete)
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from impronta import slugify
from tuning.eval_lib.dataset import (
    DATASET_DIR,
    RecordingAnnotation,
    SpeakerAnnotation,
    load_manifest,
    load_persons,
)

QUALITIES = {"clean", "mixed", "garbage"}


def parse_export(path: Path) -> dict[tuple[str, str], dict[str, tuple[str, str, str]]]:
    """-> {(uid, rid): {qid: (person, quality, annotator)}}"""
    data = json.loads(path.read_text())
    out: dict[tuple[str, str], dict[str, tuple[str, str, str]]] = defaultdict(dict)
    problems: list[str] = []
    for task in data:
        d = task.get("data", {})
        uid, rid, qid = d.get("uid"), d.get("rid"), d.get("speaker_id")
        annotations = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if not annotations:
            continue
        person = quality = annotator = None
        for a in annotations:
            for res in a.get("result", []):
                if res.get("from_name") == "quality":
                    quality = (res["value"]["choices"] or [None])[0]
                elif res.get("from_name") == "person":
                    texts = res["value"].get("text") or []
                    person = (texts[0] or "").strip().lower() if texts else None
            annotator = (a.get("completed_by") or {}).get("email", "") if isinstance(
                a.get("completed_by"), dict
            ) else str(a.get("completed_by", ""))
        if person is None or quality is None:
            problems.append(f"{uid}/{rid}/{qid}: incomplete annotation")
            continue
        key = (uid, rid)
        if qid in out[key] and out[key][qid][:2] != (person, quality):
            problems.append(f"{uid}/{rid}/{qid}: conflicting annotations")
            continue
        out[key][qid] = (person, quality, annotator or "")
    if problems:
        raise SystemExit("export problems:\n  " + "\n  ".join(problems))
    return out


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: labelstudio_export.py export1.json [export2.json ...]")
    persons = load_persons()
    parsed: dict[tuple[str, str], dict[str, tuple[str, str, str]]] = {}
    for arg in sys.argv[1:]:
        for key, speakers in parse_export(Path(arg)).items():
            merged = parsed.setdefault(key, {})
            for qid, ann in speakers.items():
                if qid in merged and merged[qid][:2] != ann[:2]:
                    raise SystemExit(
                        f"{key}/{qid}: conflicting annotations across export files"
                    )
                merged[qid] = ann

    errors: list[str] = []
    for (uid, rid), speakers in parsed.items():
        for qid, (person, quality, _) in speakers.items():
            if quality not in QUALITIES:
                errors.append(f"{uid}/{rid}/{qid}: bad quality {quality!r}")
            if person != slugify(person):
                errors.append(
                    f"{uid}/{rid}/{qid}: person {person!r} is not a stable slug "
                    f"(use {slugify(person)!r})"
                )
            elif person not in persons:
                errors.append(
                    f"{uid}/{rid}/{qid}: person {person!r} missing from persons.json"
                )
    if errors:
        raise SystemExit(
            "validation failed — fix persons.json or the annotations:\n  "
            + "\n  ".join(sorted(set(errors))[:40])
        )

    now = datetime.now(timezone.utc).isoformat()
    lines = []
    for (uid, rid), speakers in sorted(parsed.items()):
        ann = RecordingAnnotation(
            rid=rid,
            uid=uid,
            speakers={
                qid: SpeakerAnnotation(person=p, quality=q)
                for qid, (p, q, _) in sorted(speakers.items())
            },
            annotator=next(iter(speakers.values()))[2],
            ts=now,
        )
        lines.append(json.dumps(ann.to_dict(), ensure_ascii=False))
    out = DATASET_DIR / "annotations.jsonl"
    out.write_text("\n".join(lines) + "\n")

    # coverage report
    usable = {(r.uid, r.rid): r for r in load_manifest() if r.usable}
    missing = [k for k in usable if k not in parsed]
    print(f"wrote {out} ({len(lines)} recordings)")
    if missing:
        print(f"NOTE: {len(missing)} usable recordings still unannotated")


if __name__ == "__main__":
    main()
