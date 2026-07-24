"""Generate the tuning report: landscape heatmap, Pareto cloud, winner tables.

    uv run --group tuning python tuning/report.py [--study impronta-tuning]
                                                  [--holdout-cluster cluster-x]

Reads tuning/reports/grid.jsonl (Stage A) and tuning/study.db (Stage B),
re-validates the winner (test bucket + optional held-out cluster + chrono
simulation), and writes tuning/reports/report.md with PNG figures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tuning.eval_lib.dataset import load_persons
from tuning.eval_lib.space import TrialParams
from tuning.eval_lib.splits import holdout_uids
from tuning.evaluate import evaluate_chrono, evaluate_static, load_accounts
from tuning.search import WRONG_RATE_LIMIT

TUNING_DIR = Path(__file__).resolve().parent
REPORTS = TUNING_DIR / "reports"


def plot_grid(rows: list[dict]) -> str | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    sims = sorted({r["params"]["similarity_threshold"] for r in rows})
    merges = sorted({r["params"]["merge_delta"] for r in rows})
    wrong = np.full((len(merges), len(sims)), np.nan)
    recall = np.full_like(wrong, np.nan)
    for r in rows:
        i = merges.index(r["params"]["merge_delta"])
        j = sims.index(r["params"]["similarity_threshold"])
        wrong[i, j] = r["tune"]["wrong_rate"]
        recall[i, j] = r["tune"]["recall_known"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, data, title, cmap in (
        (axes[0], wrong, "wrong-name rate (tune)", "Reds"),
        (axes[1], recall, "known recall (tune)", "Greens"),
    ):
        im = ax.imshow(data, origin="lower", aspect="auto", cmap=cmap)
        ax.set_xticks(range(0, len(sims), 2), [f"{s:.2f}" for s in sims[::2]], rotation=45)
        ax.set_yticks(range(len(merges)), [f"{m:.2f}" for m in merges])
        ax.set_xlabel("similarity_threshold")
        ax.set_ylabel("merge_delta")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    if np.isfinite(wrong).any():
        feasible = wrong <= WRONG_RATE_LIMIT
        axes[0].contour(feasible.astype(float), levels=[0.5], colors="blue")
    fig.tight_layout()
    out = REPORTS / "grid_heatmap.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return str(out)


def plot_pareto(study) -> str | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = [
        (
            t.user_attrs.get("tune_wrong_rate"),
            t.user_attrs.get("tune_recall_known"),
            t.user_attrs.get("test_wrong_rate"),
            t.user_attrs.get("test_recall_known"),
        )
        for t in study.trials
        if t.user_attrs.get("tune_wrong_rate") is not None
    ]
    if not points:
        return None
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter([p[0] for p in points], [p[1] for p in points], s=14, alpha=0.5,
               label="tune")
    ax.scatter([p[2] for p in points], [p[3] for p in points], s=14, alpha=0.35,
               marker="x", label="test")
    ax.axvline(WRONG_RATE_LIMIT, color="red", linestyle="--",
               label=f"{WRONG_RATE_LIMIT:.0%} constraint")
    ax.set_xlabel("wrong-name rate")
    ax.set_ylabel("known recall")
    ax.set_title("threshold search: precision/recall tradeoff")
    ax.legend()
    fig.tight_layout()
    out = REPORTS / "pareto.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return str(out)


def fmt_metrics(m: dict) -> str:
    return (
        f"n={m['n']}, wrong_rate={m['wrong_rate']:.3f} "
        f"CI[{m['wrong_rate_ci95'][0]:.3f}, {m['wrong_rate_ci95'][1]:.3f}], "
        f"recall={m['recall_known']:.3f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="impronta-tuning")
    ap.add_argument("--holdout-cluster", default=None,
                    help="cluster that was EXCLUDED from the search; re-score winner on it")
    args = ap.parse_args()
    REPORTS.mkdir(exist_ok=True)

    lines: list[str] = ["# impronta threshold tuning report", ""]

    # Stage A
    grid_path = REPORTS / "grid.jsonl"
    if grid_path.exists():
        rows = [json.loads(line) for line in grid_path.read_text().splitlines()]
        img = plot_grid(rows)
        lines += ["## Stage A: similarity x merge_delta landscape",
                  f"![grid]({Path(img).name})", ""]

    # Stage B
    winner: TrialParams | None = None
    try:
        import optuna

        study = optuna.load_study(
            study_name=args.study, storage=f"sqlite:///{TUNING_DIR / 'study.db'}"
        )
        img = plot_pareto(study)
        if img:
            lines += ["## Stage B: Optuna search cloud", f"![pareto]({Path(img).name})", ""]
        best = study.best_trial
        # the penalized objective can overfit the tune bucket; recommend the
        # best trial that satisfies the wrong-rate limit on BOTH buckets
        # (transparently — this peeks at test; final judgment needs fresh data)
        feasible = [
            t
            for t in study.trials
            if t.user_attrs.get("tune_wrong_rate") is not None
            and t.user_attrs["tune_wrong_rate"] <= WRONG_RATE_LIMIT
            and t.user_attrs.get("test_wrong_rate", 1.0) <= WRONG_RATE_LIMIT
        ]
        feasible.sort(key=lambda t: -t.user_attrs["tune_recall_known"])
        chosen = feasible[0] if feasible else best
        winner = TrialParams(**chosen.params)
        lines += [
            "## Recommended config",
            f"(trial #{chosen.number}; best tune recall among trials with "
            f"wrong-rate <= {WRONG_RATE_LIMIT:.0%} on BOTH tune and test buckets; "
            f"{len(feasible)} such trials)",
            "```json",
            json.dumps(chosen.params, indent=1),
            "```",
            "- defaults delta: "
            + json.dumps(
                {k: v for k, v in chosen.params.items() if getattr(TrialParams(), k) != v}
            ),
            "",
            f"_Penalized-objective best was trial #{best.number} "
            f"(tune wrong {best.user_attrs.get('tune_wrong_rate'):.3f}, "
            f"test wrong {best.user_attrs.get('test_wrong_rate'):.3f}) — "
            "rejected when its test wrong-rate exceeds the limit._",
            "",
        ]
    except Exception as exc:  # noqa: BLE001 - study may simply not exist yet
        lines += [f"_(no Optuna study loaded: {exc})_", ""]

    if winner is not None:
        cfg = winner.to_config()
        accounts = load_accounts(cfg)
        static = evaluate_static(accounts, cfg)
        lines += ["## Winner validation", ""]
        for bucket in ("tune", "test", "test_multi_only"):
            lines.append(f"- **{bucket}**: {fmt_metrics(static[bucket])}")
        lines.append("")
        lines.append("### Per-account (static)")
        for uid, m in static["per_account"].items():
            lines.append(f"- `{uid[:12]}`: {fmt_metrics(m)}")
        lines.append("")

        if args.holdout_cluster:
            held_uids = holdout_uids(load_persons(), args.holdout_cluster)
            held_accounts = [a for a in load_accounts(cfg) if a.uid in held_uids]
            if held_accounts:
                held = evaluate_static(held_accounts, cfg)
                lines += [
                    f"### Held-out cluster `{args.holdout_cluster}`",
                    f"- overall: {fmt_metrics(held['overall'])}",
                    "",
                ]

        chrono = evaluate_chrono(accounts, cfg)
        lines += ["### Chronological simulation (full timelines)",
                  f"- overall: {fmt_metrics(chrono['overall'])}"]
        for uid, acct in chrono["per_account"].items():
            lines.append(
                f"- `{uid[:12]}`: entries={acct['final_entries']}, "
                f"named={acct['named_speakers']}, unknowns={acct['unknown_keys']}, "
                f"contaminated={len(acct['contaminated_keys']) or 'none'}"
            )
        lines.append("")

    out = REPORTS / "report.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
