"""Randomized paired simulations isolating the effect of reinforcement.

Each seed produces one scenario per account: a shuffled recording order and
a random labeling moment per person (uniformly among their clean
appearances — real users don't label at first sight). The scenario runs
TWICE with identical randomness: reinforcement on vs off. Deltas are
therefore attributable to reinforcement alone.

Tracked per run: recall / wrong-name over enrolled rows, reinforcement
commits, POISONED reinforcement commits (ground-truth person of the
harvested segments != the profile's person), contaminated profiles, store
growth.

    uv run --group tuning python tuning/reinforce_sim.py [--seeds 40]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from impronta.exceptions import NoUsableSegmentsError, SpeakerNotFoundError
from tuning.eval_lib.episodes import AccountData
from tuning.eval_lib.replay import ReplaySession
from tuning.eval_lib.space import TrialParams
from tuning.evaluate import load_accounts


@dataclass(slots=True)
class SimStats:
    scored: int = 0
    correct: int = 0
    wrong: int = 0
    missed: int = 0
    reinforce_commits: int = 0
    poisoned_reinforcements: int = 0
    contaminated_profiles: int = 0
    final_entries: int = 0

    def add(self, other: SimStats) -> None:
        for f in self.__dataclass_fields__:
            setattr(self, f, getattr(self, f) + getattr(other, f))

    @property
    def recall(self) -> float:
        return self.correct / self.scored if self.scored else 0.0

    @property
    def wrong_rate(self) -> float:
        emitted = self.correct + self.wrong
        return self.wrong / emitted if emitted else 0.0


def clean_map(data: AccountData) -> dict[str, list[tuple[str, str]]]:
    """person -> [(rid, qid), ...] clean appearances (any order)."""
    out: dict[str, list[tuple[str, str]]] = {}
    for rid, ann in data.annotations.items():
        for qid, s in ann.speakers.items():
            if s.quality == "clean":
                out.setdefault(s.person, []).append((rid, qid))
    return out


def run_sim(
    data: AccountData, cfg, rng: np.random.Generator, reinforce: bool
) -> SimStats:
    rids = [r.rid for r in data.rows if r.rid in data.caches and r.rid in data.annotations]
    order = list(rids)
    rng.shuffle(order)
    position = {rid: i for i, rid in enumerate(order)}

    persons = clean_map(data)
    # random labeling moment: one clean appearance chosen uniformly; if
    # enrollment fails there, fall back to later positions in a shuffled list
    schedule: dict[str, list[tuple[str, str]]] = {}
    label_at: dict[str, int] = {}
    for person, occs in persons.items():
        occs = [o for o in occs if o[0] in position]
        if not occs:
            continue
        occs = list(occs)
        rng.shuffle(occs)
        schedule[person] = occs
        label_at[person] = position[occs[0][0]]

    ann_person = {
        (rid, qid): s.person
        for rid, ann in data.annotations.items()
        for qid, s in ann.speakers.items()
        if s.quality == "clean"
    }

    session = ReplaySession(data.uid, cfg)
    stats = SimStats()
    labeled: dict[str, int] = {}  # person -> position labeled at
    contrib: dict[str, set[tuple[str, str]]] = {}

    def move(old: str, new: str) -> None:
        if old != new and old in contrib:
            contrib.setdefault(new, set()).update(contrib.pop(old))

    for pos, rid in enumerate(order):
        cache = data.caches[rid]
        ann = data.annotations[rid]
        identified = session.identify(cache)

        # score enrolled clean rows
        for qid in sorted(ann.speakers):
            s = ann.speakers[qid]
            if s.quality != "clean":
                continue
            if s.person not in labeled or labeled[s.person] >= pos:
                continue
            match = identified.speakers.get(qid)
            if match is None or not match.identifiable:
                continue
            stats.scored += 1
            if match.display_name is not None:
                if match.speaker_key == s.person:
                    stats.correct += 1
                else:
                    stats.wrong += 1
            else:
                stats.missed += 1

        committed = session.app.commit_unknowns(list(identified.proposed_unknowns))
        for proposal, key in zip(identified.proposed_unknowns, committed, strict=True):
            contrib.setdefault(key, set()).add((rid, proposal.query_speaker_id))

        # scheduled labeling
        for person, occs in schedule.items():
            if person in labeled:
                continue
            due = [o for o in occs if o[0] == rid and position[o[0]] >= label_at[person]]
            if not due or position[rid] < label_at[person]:
                continue
            qid = due[0][1]
            match = identified.speakers.get(qid)
            promote_key = None
            if match is not None and match.speaker_key and match.is_unknown:
                promote_key = match.speaker_key
            else:
                for proposal, key in zip(
                    identified.proposed_unknowns, committed, strict=True
                ):
                    if proposal.query_speaker_id == qid:
                        promote_key = key
                        break
            done = False
            if promote_key is not None:
                try:
                    summary = session.app.label_speaker(promote_key, person)
                    move(promote_key, summary.speaker_key)
                    done = True
                except SpeakerNotFoundError:
                    done = False
            if not done:
                try:
                    enrolled = session.enroll(cache, qid, person)
                    for merged in enrolled.merged_unknown_keys:
                        move(merged, person)
                    done = True
                except NoUsableSegmentsError:
                    # postpone to this person's next scheduled occurrence
                    later = [o for o in occs if position[o[0]] > pos]
                    if later:
                        label_at[person] = position[later[0][0]]
                        schedule[person] = later
                    continue
            if done:
                labeled[person] = pos
                contrib.setdefault(person, set()).add((rid, qid))

        if reinforce:
            proposals = session.reinforce(cache)
            keys = session.app.commit_reinforcements(proposals)
            for proposal, key in zip(proposals, keys, strict=True):
                if key is None:
                    continue
                stats.reinforce_commits += 1
                truth = ann_person.get((rid, proposal.query_speaker_id))
                # resolve the TARGET's identity: named keys are themselves;
                # unknown clusters resolve to their dominant ground-truth
                # contributor (reinforcing your own unlabeled cluster is
                # correct behavior, not poisoning)
                if key.startswith("unknown-"):
                    from collections import Counter as _Counter

                    doms = _Counter(
                        ann_person[s]
                        for s in contrib.get(key, set())
                        if s in ann_person
                    )
                    target_identity = doms.most_common(1)[0][0] if doms else None
                else:
                    target_identity = key
                contrib.setdefault(key, set()).add(
                    (rid, proposal.query_speaker_id)
                )
                if (
                    truth is not None
                    and target_identity is not None
                    and truth != target_identity
                ):
                    stats.poisoned_reinforcements += 1
                    if hasattr(run_sim, "poison_log"):
                        profile_n = len(
                            session.app.store.get_speaker_entries(data.uid, key)
                        )
                        truth_labeled = truth in labeled
                        rec_ann = data.annotations[rid].speakers
                        key_in_rec = any(
                            s.person == key for s in rec_ann.values()
                        )
                        run_sim.poison_log.append(dict(
                            uid=data.uid[:8], rid=rid[:8], truth=truth, key=key,
                            sim=round(proposal.mean_similarity, 3),
                            n_segs=int(proposal.embeddings.shape[0]),
                            profile_n=profile_n,
                            truth_labeled=truth_labeled,
                            key_also_in_recording=key_in_rec,
                        ))

    for _key, sources in contrib.items():
        truths = {ann_person[s] for s in sources if s in ann_person}
        if len(truths) >= 2:
            stats.contaminated_profiles += 1
    stats.final_entries = session.app.store.count(data.uid)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--params", default="{}")
    args = ap.parse_args()
    import json

    cfg = TrialParams(**json.loads(args.params)).to_config()
    accounts = load_accounts(cfg)

    deltas_recall, deltas_wrong = [], []
    on_totals, off_totals = SimStats(), SimStats()
    per_seed = []
    for seed in range(args.seeds):
        on, off = SimStats(), SimStats()
        for data in accounts:
            on.add(run_sim(data, cfg, np.random.default_rng(seed * 1000 + 1), True))
            off.add(run_sim(data, cfg, np.random.default_rng(seed * 1000 + 1), False))
        on_totals.add(on)
        off_totals.add(off)
        deltas_recall.append(on.recall - off.recall)
        deltas_wrong.append(on.wrong_rate - off.wrong_rate)
        per_seed.append((seed, on, off))
        print(
            f"seed {seed:3d}: OFF r={off.recall:.3f} w={off.wrong_rate:.3f} | "
            f"ON r={on.recall:.3f} w={on.wrong_rate:.3f} "
            f"(+{on.recall - off.recall:+.3f} recall) "
            f"reinf={on.reinforce_commits} poisoned={on.poisoned_reinforcements} "
            f"contam on/off={on.contaminated_profiles}/{off.contaminated_profiles}"
        )

    n = args.seeds
    dr = np.array(deltas_recall)
    dw = np.array(deltas_wrong)
    print("\n=== paired summary over", n, "randomized scenarios ===")
    print(f"recall:      OFF {off_totals.recall:.3f} -> ON {on_totals.recall:.3f} "
          f"(paired delta {dr.mean():+.3f} ± {dr.std():.3f}, "
          f"improved in {(dr > 0).sum()}/{n}, hurt in {(dr < 0).sum()}/{n})")
    print(f"wrong-rate:  OFF {off_totals.wrong_rate:.3f} -> ON {on_totals.wrong_rate:.3f} "
          f"(paired delta {dw.mean():+.3f} ± {dw.std():.3f})")
    pr = (
        on_totals.poisoned_reinforcements / on_totals.reinforce_commits * 100
        if on_totals.reinforce_commits
        else 0.0
    )
    print(f"reinforcement commits: {on_totals.reinforce_commits} total, "
          f"{on_totals.poisoned_reinforcements} poisoned ({pr:.1f}%)")
    print(f"contaminated profiles: ON {on_totals.contaminated_profiles} vs "
          f"OFF {off_totals.contaminated_profiles} (across all runs)")
    print(f"store size: ON {on_totals.final_entries / n:.0f} vs "
          f"OFF {off_totals.final_entries / n:.0f} entries/scenario avg")


if __name__ == "__main__":
    main()
