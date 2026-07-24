"""Score one impronta config against the annotated dataset.

    uv run python tuning/evaluate.py                         # defaults, static
    uv run python tuning/evaluate.py --chrono                # + chrono simulation
    uv run python tuning/evaluate.py --params '{"similarity_threshold": 0.4}'
    uv run python tuning/evaluate.py --exclude-cluster cluster-b

Loads the manifest, annotations, persons and the embedding cache; runs the
static protocol per annotated account (and optionally the chronological
simulation); prints tune/test metrics overall, the multi-speaker-only
slice, and per-account breakdowns.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tuning.eval_lib.cache import cache_path, load_cache
from tuning.eval_lib.dataset import (
    load_annotations,
    load_manifest,
    load_persons,
)
from tuning.eval_lib.episodes import AccountData, run_chrono, run_static
from tuning.eval_lib.scorer import aggregate
from tuning.eval_lib.space import TrialParams
from tuning.eval_lib.splits import holdout_uids


def load_accounts(cfg, exclude_uids: set[str] | None = None) -> list[AccountData]:
    manifest = [r for r in load_manifest() if r.usable]
    persons = load_persons()
    annotations = load_annotations(persons=persons)
    exclude_uids = exclude_uids or set()

    accounts: list[AccountData] = []
    for uid in sorted({r.uid for r in manifest} - exclude_uids):
        rows = [r for r in manifest if r.uid == uid]
        anns = {rid: a for (u, rid), a in annotations.items() if u == uid}
        if not anns:
            continue
        caches = {}
        for rid in anns:
            if cache_path(uid, rid).exists():
                caches[rid] = load_cache(uid, rid, cfg)
        accounts.append(
            AccountData(uid=uid, rows=rows, annotations=anns, caches=caches)
        )
    return accounts


def evaluate_static(accounts: list[AccountData], cfg) -> dict:
    """The optimizer inner loop: all accounts, aggregated + per-bucket."""
    all_rows = []
    fallbacks = skipped = 0
    per_account = {}
    for data in accounts:
        result = run_static(data, cfg)
        all_rows.extend(result.rows)
        fallbacks += result.enroll_fallbacks
        skipped += result.skipped_recordings
        per_account[data.uid] = aggregate(result.rows).to_dict()

    def bucket(name=None, multi_only=False):
        rows = all_rows
        if name:
            rows = [r for r in rows if r.bucket == name]
        if multi_only:
            rows = [r for r in rows if r.multi_speaker]
        return aggregate(rows)

    return {
        "tune": bucket("tune").to_dict(),
        "test": bucket("test").to_dict(),
        "tune_multi_only": bucket("tune", multi_only=True).to_dict(),
        "test_multi_only": bucket("test", multi_only=True).to_dict(),
        "overall": bucket().to_dict(),
        "enroll_fallbacks": fallbacks,
        "skipped_recordings": skipped,
        "per_account": per_account,
    }


def evaluate_chrono(accounts: list[AccountData], cfg) -> dict:
    out = {}
    total_rows = []
    for data in accounts:
        result = run_chrono(data, cfg)
        total_rows.extend(result.rows)
        out[data.uid] = {
            "metrics": result.metrics().to_dict(),
            "final_entries": result.final_entries,
            "named_speakers": result.final_named_speakers,
            "unknown_keys": result.final_unknown_keys,
            "true_never_labeled": result.true_never_labeled,
            "contaminated_keys": result.contaminated_keys,
            "enroll_fallbacks": result.enroll_fallbacks,
        }
    return {"overall": aggregate(total_rows).to_dict(), "per_account": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="{}", help="JSON overrides for TrialParams")
    ap.add_argument("--chrono", action="store_true")
    ap.add_argument("--exclude-cluster", default=None)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    params = TrialParams(**json.loads(args.params))
    cfg = params.to_config()
    exclude = holdout_uids(load_persons(), args.exclude_cluster)
    accounts = load_accounts(cfg, exclude)
    if not accounts:
        raise SystemExit("no annotated accounts found — run the annotation flow first")

    report = {"params": params.to_dict(), "static": evaluate_static(accounts, cfg)}
    if args.chrono:
        report["chrono"] = evaluate_chrono(accounts, cfg)

    if args.json:
        print(json.dumps(report, indent=1))
        return

    s = report["static"]
    print(f"accounts evaluated: {len(accounts)}"
          f"{f' (cluster {args.exclude_cluster} excluded)' if exclude else ''}")
    for name in ("tune", "test", "tune_multi_only", "test_multi_only"):
        m = s[name]
        print(
            f"  {name:16} n={m['n']:4d} wrong_rate={m['wrong_rate']:.3f} "
            f"CI[{m['wrong_rate_ci95'][0]:.3f},{m['wrong_rate_ci95'][1]:.3f}] "
            f"recall={m['recall_known']:.3f} "
            f"(correct={m['correct_name']}, wrong={m['wrong_names']}, "
            f"missed={m['missed_known']}, no_usable={m['no_usable']})"
        )
    print(f"  enroll_fallbacks={s['enroll_fallbacks']} skipped={s['skipped_recordings']}")
    print("  per-account wrong/recall:")
    for uid, m in s["per_account"].items():
        print(f"    {uid[:10]}: n={m['n']:3d} wrong={m['wrong_rate']:.3f} "
              f"recall={m['recall_known']:.3f}")
    if args.chrono:
        c = report["chrono"]
        m = c["overall"]
        print(f"  chrono: n={m['n']} wrong_rate={m['wrong_rate']:.3f} "
              f"recall={m['recall_known']:.3f}")
        for uid, acct in c["per_account"].items():
            contamination = len(acct["contaminated_keys"])
            print(f"    {uid[:10]}: entries={acct['final_entries']} "
                  f"named={acct['named_speakers']} unknowns={acct['unknown_keys']} "
                  f"contaminated={contamination}")


if __name__ == "__main__":
    main()
