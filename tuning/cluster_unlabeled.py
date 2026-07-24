"""Group the still-unlabeled annotation tasks by voice similarity.

Fetches the existing project's annotations, computes a voiceprint per
unlabeled (recording, speaker) from the embedding cache, clusters them
(average-linkage agglomerative on cosine), and creates a NEW Label Studio
project where:

- tasks arrive ordered by voice group (same voice = consecutive tasks),
- each task carries a ``voice_group`` field and a prefilled prediction:
  either the best-matching already-labeled person ("sounds like ahmed")
  or a fresh ``guest-NN`` key shared by the whole group,
- so most tasks are listen -> confirm -> next.

    LS_USERNAME=... LS_PASSWORD=... uv run python tuning/cluster_unlabeled.py \\
        [--project 2] [--threshold 0.5]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tuning.eval_lib.cache import cache_path, load_cache
from tuning.eval_lib.dataset import DATASET_DIR
from tuning.labelstudio_import import LABEL_CONFIG, session_login


def task_key(data: dict) -> tuple[str, str, str]:
    return (data["uid"], data["rid"], data["speaker_id"])


def fetch_tasks(s, base: str, project: int) -> list[dict]:
    r = s.get(
        f"{base}/api/projects/{project}/export",
        params={"exportType": "JSON", "download_all_tasks": "true"},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()


def annotation_of(task: dict) -> tuple[str, str] | None:
    """(person, quality) from a completed task, else None."""
    for a in task.get("annotations", []):
        if a.get("was_cancelled"):
            continue
        person = quality = None
        for res in a.get("result", []):
            if res.get("from_name") == "person":
                texts = res["value"].get("text") or []
                person = (texts[0] or "").strip().lower() if texts else None
            elif res.get("from_name") == "quality":
                choices = res["value"].get("choices") or []
                quality = choices[0] if choices else None
        if person and quality:
            return person, quality
    return None


def voiceprint(uid: str, rid: str, qid: str) -> np.ndarray | None:
    if not cache_path(uid, rid).exists():
        return None
    cache = load_cache(uid, rid)
    rows = cache.rows_for(qid)
    if rows.size == 0:
        return None
    c = cache.embeddings[rows].mean(axis=0)
    n = np.linalg.norm(c)
    return c / n if n > 0 else None


def agglomerate(vecs: np.ndarray, threshold: float) -> list[int]:
    """Average-linkage agglomerative clustering on cosine similarity."""
    n = vecs.shape[0]
    clusters: list[list[int]] = [[i] for i in range(n)]
    centroids = [vecs[i] for i in range(n)]
    while len(clusters) > 1:
        best, bi, bj = -1.0, -1, -1
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                sim = float(np.dot(centroids[i], centroids[j]))
                if sim > best:
                    best, bi, bj = sim, i, j
        if best < threshold:
            break
        merged = clusters[bi] + clusters[bj]
        clusters = [c for k, c in enumerate(clusters) if k not in (bi, bj)] + [merged]
        centroids = [c for k, c in enumerate(centroids) if k not in (bi, bj)]
        m = vecs[merged].mean(axis=0)
        centroids.append(m / np.linalg.norm(m))
    labels = [0] * n
    for cid, members in enumerate(clusters):
        for i in members:
            labels[i] = cid
    return labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--projects", type=int, nargs="+", default=[2],
        help="labeled state = union of these; unlabeled = first project minus labels",
    )
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--suggest-labeled-min", type=float, default=0.5)
    args = ap.parse_args()

    base = os.environ.get("LS_URL", "http://localhost:8080").rstrip("/")
    s = session_login(
        base, os.environ["LS_USERNAME"], os.environ["LS_PASSWORD"]
    )
    csrf = {"X-CSRFToken": s.cookies.get("csrftoken", ""), "Referer": base}

    labeled: dict[tuple[str, str, str], tuple[str, str]] = {}
    universe: list[dict] = []
    for k, project in enumerate(args.projects):
        for t in fetch_tasks(s, base, project):
            ann = annotation_of(t)
            if ann:
                labeled[task_key(t["data"])] = ann
            elif k == 0:
                universe.append(t["data"])
    unlabeled = [d for d in universe if task_key(d) not in labeled]
    print(f"projects {args.projects}: {len(labeled)} labeled, {len(unlabeled)} unlabeled")
    if not unlabeled:
        print("nothing left to cluster")
        return

    # voiceprints of labeled persons (clean only) for "sounds like" hints
    person_prints: dict[str, list[np.ndarray]] = {}
    for (uid, rid, qid), (person, quality) in labeled.items():
        if quality != "clean":
            continue
        v = voiceprint(uid, rid, qid)
        if v is not None:
            person_prints.setdefault(person, []).append(v)
    person_centroids = {
        p: (lambda m: m / np.linalg.norm(m))(np.stack(vs).mean(axis=0))
        for p, vs in person_prints.items()
    }

    # voiceprints + clustering of the unlabeled
    items = []
    for data in unlabeled:
        v = voiceprint(*task_key(data))
        if v is not None:
            items.append((data, v))
    skipped = len(unlabeled) - len(items)
    vecs = np.stack([v for _, v in items])
    labels = agglomerate(vecs, args.threshold)

    groups: dict[int, list[int]] = {}
    for i, cid in enumerate(labels):
        groups.setdefault(cid, []).append(i)
    ordered_groups = sorted(groups.values(), key=len, reverse=True)

    # suggestions per group
    new_tasks = []
    guest_counter = 0
    for members in ordered_groups:
        centroid = vecs[members].mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        best_person, best_sim = None, 0.0
        for person, pc in person_centroids.items():
            sim = float(np.dot(centroid, pc))
            if sim > best_sim:
                best_person, best_sim = person, sim
        if best_person is not None and best_sim >= args.suggest_labeled_min:
            suggestion = best_person
            hint = f"sounds like: {best_person} ({best_sim:.2f})"
        else:
            guest_counter += 1
            suggestion = f"guest-{guest_counter:02d}"
            hint = f"new voice group ({len(members)} clips)"
        group_name = f"group-{ordered_groups.index(members) + 1:02d}"
        for i in members:
            data = dict(items[i][0])
            data["voice_group"] = f"{group_name} · {len(members)} clips · {hint}"
            new_tasks.append(
                {
                    "data": data,
                    "predictions": [
                        {
                            "result": [
                                {
                                    "from_name": "person",
                                    "to_name": "audio",
                                    "type": "textarea",
                                    "value": {"text": [suggestion]},
                                },
                                {
                                    "from_name": "quality",
                                    "to_name": "audio",
                                    "type": "choices",
                                    "value": {"choices": ["clean"]},
                                },
                            ]
                        }
                    ],
                }
            )

    config = LABEL_CONFIG.replace(
        '<Header value="$uid / $rid"/>',
        '<Header value="$voice_group"/>\n  <Header value="$uid / $rid"/>',
    )
    r = s.post(
        f"{base}/api/projects",
        headers=csrf,
        json={
            "title": "impronta — voice-grouped remainder",
            "description": "Unlabeled speakers, grouped by voice similarity. "
            "Suggestions prefilled — confirm or correct.",
            "label_config": config,
            "show_collab_predictions": True,
        },
        timeout=30,
    )
    r.raise_for_status()
    pid = r.json()["id"]
    r = s.post(
        f"{base}/api/storages/localfiles",
        headers=csrf,
        json={
            "project": pid,
            "title": "ls_media audio",
            "path": str((DATASET_DIR / "ls_media" / "audio").resolve()),
            "use_blob_urls": False,
        },
        timeout=30,
    )
    r.raise_for_status()
    r = s.post(
        f"{base}/api/projects/{pid}/import", headers=csrf, json=new_tasks, timeout=300
    )
    r.raise_for_status()

    n_groups = len(ordered_groups)
    multi = sum(1 for g in ordered_groups if len(g) > 1)
    print(
        f"created project {pid}: {len(new_tasks)} tasks in {n_groups} voice groups "
        f"({multi} recurring voices, {skipped} skipped without voiceprint)"
    )
    print(f"annotate at {base}/projects/{pid}")


if __name__ == "__main__":
    main()
