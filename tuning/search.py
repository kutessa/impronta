"""Threshold search: Stage A coarse grid, Stage B Optuna TPE.

    uv run --group tuning python tuning/search.py grid
    uv run --group tuning python tuning/search.py optuna --trials 500
    uv run --group tuning python tuning/search.py optuna --seed 1 --trials 500

Objective (settled): hard constraint wrong-name rate <= 2% on the tune
bucket; maximize known-speaker recall subject to it. TPE has no native
constraints, so the scalar objective is penalized:

    recall_tune - 50 * max(0, wrong_rate_tune - 0.02)

(1pp of violation costs 0.5 recall — violation always dominates.) All raw
metrics land in trial.user_attrs so report.py can draw the full tradeoff
regardless of the penalty. Storage is sqlite (tuning/study.db): resumable,
inspectable with optuna-dashboard.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tuning.eval_lib.dataset import load_persons
from tuning.eval_lib.space import TrialParams, grid_stage_a, sample_params
from tuning.eval_lib.splits import holdout_uids
from tuning.evaluate import evaluate_static, load_accounts

TUNING_DIR = Path(__file__).resolve().parent
WRONG_RATE_LIMIT = 0.02
PENALTY = 50.0


def score(params: TrialParams, exclude_cluster: str | None) -> dict:
    cfg = params.to_config()
    accounts = load_accounts(cfg, holdout_uids(load_persons(), exclude_cluster))
    return evaluate_static(accounts, cfg)


def objective_value(static: dict) -> float:
    tune = static["tune"]
    return tune["recall_known"] - PENALTY * max(
        0.0, tune["wrong_rate"] - WRONG_RATE_LIMIT
    )


def run_grid(args) -> None:
    params_list = grid_stage_a()
    out_path = TUNING_DIR / "reports" / "grid.jsonl"
    out_path.parent.mkdir(exist_ok=True)
    results = []
    with out_path.open("w") as f:
        for i, params in enumerate(params_list, 1):
            static = score(params, args.exclude_cluster)
            row = {
                "params": params.to_dict(),
                "tune": static["tune"],
                "test": static["test"],
                "objective": objective_value(static),
            }
            results.append(row)
            f.write(json.dumps(row) + "\n")
            t = static["tune"]
            print(
                f"[{i}/{len(params_list)}] sim={params.similarity_threshold:.3f} "
                f"dmerge={params.merge_delta:.2f} -> wrong={t['wrong_rate']:.3f} "
                f"recall={t['recall_known']:.3f}"
            )
    feasible = [r for r in results if r["tune"]["wrong_rate"] <= WRONG_RATE_LIMIT]
    feasible.sort(key=lambda r: -r["tune"]["recall_known"])
    print(f"\nfeasible cells: {len(feasible)}/{len(results)}; top 5:")
    for r in feasible[:5]:
        p = r["params"]
        print(
            f"  sim={p['similarity_threshold']:.3f} dmerge={p['merge_delta']:.2f} "
            f"recall={r['tune']['recall_known']:.3f} wrong={r['tune']['wrong_rate']:.3f} "
            f"(test: recall={r['test']['recall_known']:.3f} wrong={r['test']['wrong_rate']:.3f})"
        )


def run_optuna(args) -> None:
    import optuna

    storage = f"sqlite:///{TUNING_DIR / 'study.db'}"
    study = optuna.create_study(
        study_name=args.study,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            multivariate=True, group=True, seed=args.seed, n_startup_trials=40
        ),
        load_if_exists=True,
    )

    # seed with Stage A winners if the grid has run
    grid_path = TUNING_DIR / "reports" / "grid.jsonl"
    if grid_path.exists() and not study.trials:
        rows = [json.loads(line) for line in grid_path.read_text().splitlines()]
        feasible = [r for r in rows if r["tune"]["wrong_rate"] <= WRONG_RATE_LIMIT]
        feasible.sort(key=lambda r: -r["tune"]["recall_known"])
        for r in feasible[:5]:
            study.enqueue_trial(TrialParams(**r["params"]).to_dict())
        print(f"enqueued {min(5, len(feasible))} grid winners")

    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial)
        static = score(params, args.exclude_cluster)
        for bucket in ("tune", "test"):
            for k, v in static[bucket].items():
                if isinstance(v, (int, float)):
                    trial.set_user_attr(f"{bucket}_{k}", v)
        trial.set_user_attr("enroll_fallbacks", static["enroll_fallbacks"])
        return objective_value(static)

    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    best = study.best_trial
    print(f"\nbest trial #{best.number}: objective={best.value:.4f}")
    print(json.dumps(best.params, indent=1))
    print(
        f"tune: wrong={best.user_attrs['tune_wrong_rate']:.4f} "
        f"recall={best.user_attrs['tune_recall_known']:.4f} | "
        f"test: wrong={best.user_attrs['test_wrong_rate']:.4f} "
        f"recall={best.user_attrs['test_recall_known']:.4f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    g = sub.add_parser("grid")
    g.add_argument("--exclude-cluster", default=None)
    o = sub.add_parser("optuna")
    o.add_argument("--trials", type=int, default=500)
    o.add_argument("--seed", type=int, default=7)
    o.add_argument("--study", default="impronta-tuning")
    o.add_argument("--exclude-cluster", default=None)
    args = ap.parse_args()
    if args.stage == "grid":
        run_grid(args)
    else:
        run_optuna(args)


if __name__ == "__main__":
    main()
