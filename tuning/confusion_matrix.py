"""Person-level confusion matrix for a config (static protocol).

Rows = annotated ground-truth person, columns = predicted person key
(plus "<unknown>" and "<no_usable>"). Cell = scored speaker-rows.

    uv run --group tuning python tuning/confusion_matrix.py [--params JSON]
                                                            [--bucket tune|test|all]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tuning.eval_lib.episodes import run_static
from tuning.eval_lib.space import TrialParams
from tuning.evaluate import load_accounts

OUT = Path(__file__).resolve().parent / "reports"
UNKNOWN, NO_USABLE = "<unknown>", "<no_usable>"


def collect(params: TrialParams, bucket: str, forced: bool = False) -> Counter:
    """forced=True drops the unknown option: best NAMED candidate wins,
    ignoring the threshold — the closed-set identification ceiling.
    Only rows whose true person is enrolled are counted in forced mode
    (a stranger can never be named correctly by construction)."""
    cfg = params.to_config()
    pairs: Counter = Counter()
    for account in load_accounts(cfg):
        for row in run_static(account, cfg).rows:
            if bucket != "all" and row.bucket != bucket:
                continue
            if forced:
                if not row.enrolled:
                    continue
                if row.outcome == "no_usable" or row.match is None:
                    pairs[(row.person, NO_USABLE)] += 1
                    continue
                named = [
                    c for c in row.match.candidates
                    if c.speaker_key != "<unknown>" and c.mean_similarity is not None
                ]
                if not named:
                    pairs[(row.person, "<no-candidate>")] += 1
                    continue
                best = max(named, key=lambda c: c.mean_similarity)
                pairs[(row.person, best.speaker_key)] += 1
                continue
            if row.outcome == "no_usable":
                predicted = NO_USABLE
            elif row.matched_key is None or row.outcome in ("missed_known", "correct_unknown"):
                predicted = row.matched_key or UNKNOWN
            else:
                predicted = row.matched_key
            pairs[(row.person, predicted)] += 1
    return pairs


def render(pairs: Counter, out_png: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    truths = sorted({t for t, _ in pairs})
    predicted_names = sorted(
        {p for _, p in pairs if p not in (UNKNOWN, NO_USABLE)}
    )
    preds = predicted_names + [UNKNOWN, NO_USABLE]
    m = np.zeros((len(truths), len(preds)), dtype=int)
    for (t, p), n in pairs.items():
        m[truths.index(t), preds.index(p)] = n

    fig, ax = plt.subplots(
        figsize=(1.5 + 0.42 * len(preds), 1.2 + 0.36 * len(truths))
    )
    im = ax.imshow(m, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(preds)), preds, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(truths)), truths, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true person")
    for i in range(len(truths)):
        for j in range(len(preds)):
            if m[i, j]:
                on_diag = j < len(predicted_names) and preds[j] == truths[i]
                if m[i, j] > m.max() / 2:
                    color = "white"
                elif on_diag:
                    color = "darkgreen"
                elif j < len(predicted_names):
                    color = "red"
                else:
                    color = "gray"
                ax.text(j, i, str(m[i, j]), ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="{}")
    ap.add_argument("--bucket", default="all", choices=["tune", "test", "all"])
    ap.add_argument("--forced", action="store_true",
                    help="closed-set: drop the unknown option, best named candidate wins")
    args = ap.parse_args()
    params = TrialParams(**json.loads(args.params))
    pairs = collect(params, args.bucket, forced=args.forced)
    OUT.mkdir(exist_ok=True)
    out_png = OUT / ("confusion_forced.png" if args.forced else "confusion.png")
    title = ("impronta closed-set confusion (unknown DROPPED, enrolled rows only)"
             if args.forced else "impronta identification confusion (static protocol)")
    render(pairs, out_png, title)

    # compact text version
    total = sum(pairs.values())
    diag = sum(n for (t, p), n in pairs.items() if t == p)
    unk = sum(n for (_, p), n in pairs.items() if p == UNKNOWN)
    nou = sum(n for (_, p), n in pairs.items() if p == NO_USABLE)
    wrong = total - diag - unk - nou
    print(f"bucket={args.bucket}: {total} scored rows -> "
          f"{diag} correct, {unk} unknown, {wrong} wrong-name, {nou} no-usable")
    offdiag = [
        (t, p, n) for (t, p), n in pairs.items()
        if t != p and p not in (UNKNOWN, NO_USABLE)
    ]
    for t, p, n in sorted(offdiag, key=lambda x: -x[2]):
        print(f"  CONFUSED: true {t} -> predicted {p} x{n}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
