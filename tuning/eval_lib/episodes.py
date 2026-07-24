"""Evaluation episodes: the static protocol and the chronological simulation.

Static protocol (optimizer inner loop): per account, enroll each person from
their chronologically first *clean* appearance (simulating the user labeling
them once), identify every recording, and score every clean-annotated
speaker row — except rows of a person in the very recording that enrolled
them (train/test hygiene).

Chronological simulation (winner validation): replay the account timeline
the way canto would run it — identify, auto-commit unknown proposals,
simulate user labeling at first clean appearance (promoting recognized/
committed unknowns when possible), then a reinforcement pass with
auto-commit. Tracks store growth, unknown proliferation, and profile
contamination via write-time provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from impronta import ImprontaConfig
from impronta.exceptions import NoUsableSegmentsError, SpeakerNotFoundError

from .cache import RecordingCache
from .dataset import RecordingAnnotation, RecordingRow
from .replay import ReplaySession
from .scorer import Metrics, ScoredRow, aggregate, categorize
from .splits import chrono_split, order_recordings

CacheLoader = "callable[[str, str], RecordingCache | None]"


@dataclass(slots=True)
class AccountData:
    """Everything the drivers need for one account."""

    uid: str
    rows: list[RecordingRow]  # usable only
    annotations: dict[str, RecordingAnnotation]  # by rid
    caches: dict[str, RecordingCache]  # by rid (annotated recordings)


def clean_occurrences(data: AccountData) -> dict[str, list[tuple[int, str, str]]]:
    """person -> [(order_index, rid, qid), ...] of clean appearances."""
    ordered = order_recordings(data.rows)
    out: dict[str, list[tuple[int, str, str]]] = {}
    for i, r in enumerate(ordered):
        ann = data.annotations.get(r.rid)
        if ann is None:
            continue
        for qid in sorted(ann.speakers):
            s = ann.speakers[qid]
            if s.quality == "clean":
                out.setdefault(s.person, []).append((i, r.rid, qid))
    return out


@dataclass(slots=True)
class StaticResult:
    rows: list[ScoredRow]
    enroll_fallbacks: int = 0
    skipped_recordings: int = 0  # missing cache/annotation

    def metrics(self, bucket: str | None = None, multi_only: bool = False) -> Metrics:
        rows = self.rows
        if bucket is not None:
            rows = [r for r in rows if r.bucket == bucket]
        if multi_only:
            rows = [r for r in rows if r.multi_speaker]
        return aggregate(rows)


def run_static(
    data: AccountData, cfg: ImprontaConfig, tune_frac: float = 0.7
) -> StaticResult:
    ordered = order_recordings(data.rows)
    split = chrono_split(data.rows, tune_frac)
    occurrences = clean_occurrences(data)
    next_try = dict.fromkeys(occurrences, 0)
    enrolled_idx: dict[str, int] = {}

    session = ReplaySession(data.uid, cfg)
    result = StaticResult(rows=[])

    for i, rec in enumerate(ordered):
        ann = data.annotations.get(rec.rid)
        cache = data.caches.get(rec.rid)
        if ann is None or cache is None:
            result.skipped_recordings += 1
            continue

        # scheduled enrollments due at this recording (with fallback)
        for person, occs in occurrences.items():
            if person in enrolled_idx:
                continue
            while next_try[person] < len(occs) and occs[next_try[person]][0] == i:
                _, rid, qid = occs[next_try[person]]
                next_try[person] += 1
                try:
                    session.enroll(cache, qid, person)
                    enrolled_idx[person] = i
                    break  # enrolled — ignore further same-recording occurrences
                except NoUsableSegmentsError:
                    result.enroll_fallbacks += 1

        identified = session.identify(cache)
        multi = rec.n_speakers >= 2
        for qid in sorted(ann.speakers):
            s = ann.speakers[qid]
            if s.quality != "clean":
                continue
            if enrolled_idx.get(s.person) == i:
                continue  # this recording enrolled them — never score it
            match = identified.speakers.get(qid)
            if match is None:
                continue
            enrolled = s.person in enrolled_idx and enrolled_idx[s.person] < i
            result.rows.append(
                ScoredRow(
                    uid=data.uid,
                    rid=rec.rid,
                    qid=qid,
                    person=s.person,
                    outcome=categorize(match, s.person, enrolled),
                    enrolled=enrolled,
                    matched_key=match.speaker_key,
                    mean_similarity=match.mean_similarity,
                    bucket=split[rec.rid],
                    multi_speaker=multi,
                    match=match,
                )
            )
    return result


# ---------------------------------------------------------------------------
# chronological simulation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ChronoResult:
    rows: list[ScoredRow]
    enroll_fallbacks: int = 0
    skipped_recordings: int = 0
    final_entries: int = 0
    final_named_speakers: int = 0
    final_unknown_keys: int = 0
    true_never_labeled: int = 0
    contaminated_keys: dict[str, list[str]] = field(default_factory=dict)

    def metrics(self, multi_only: bool = False) -> Metrics:
        rows = [r for r in self.rows if r.multi_speaker] if multi_only else self.rows
        return aggregate(rows)


def run_chrono(data: AccountData, cfg: ImprontaConfig) -> ChronoResult:
    ordered = order_recordings(data.rows)
    occurrences = clean_occurrences(data)
    first_clean = {p: occs[0] for p, occs in occurrences.items()}
    next_try = dict.fromkeys(occurrences, 0)
    labeled_idx: dict[str, int] = {}

    session = ReplaySession(data.uid, cfg)
    result = ChronoResult(rows=[])
    # write-time provenance: speaker key -> contributing (rid, qid) pairs
    contrib: dict[str, set[tuple[str, str]]] = {}

    def move_contrib(old: str, new: str) -> None:
        if old != new and old in contrib:
            contrib.setdefault(new, set()).update(contrib.pop(old))

    for i, rec in enumerate(ordered):
        ann = data.annotations.get(rec.rid)
        cache = data.caches.get(rec.rid)
        if ann is None or cache is None:
            result.skipped_recordings += 1
            continue

        # 1. identify + score (against what the simulated user has labeled)
        identified = session.identify(cache)
        multi = rec.n_speakers >= 2
        for qid in sorted(ann.speakers):
            s = ann.speakers[qid]
            if s.quality != "clean":
                continue
            enrolled = s.person in labeled_idx and labeled_idx[s.person] < i
            match = identified.speakers.get(qid)
            if match is None:
                continue
            result.rows.append(
                ScoredRow(
                    uid=data.uid,
                    rid=rec.rid,
                    qid=qid,
                    person=s.person,
                    outcome=categorize(match, s.person, enrolled),
                    enrolled=enrolled,
                    matched_key=match.speaker_key,
                    mean_similarity=match.mean_similarity,
                    bucket="all",
                    multi_speaker=multi,
                    match=match,
                )
            )

        # 2. auto-commit unknown proposals
        committed = session.app.commit_unknowns(list(identified.proposed_unknowns))
        for proposal, key in zip(identified.proposed_unknowns, committed, strict=True):
            contrib.setdefault(key, set()).add((rec.rid, proposal.query_speaker_id))

        # 3. simulated user labeling at first clean appearance
        for person, (first_i, _rid, qid) in first_clean.items():
            if person in labeled_idx or first_i != i:
                continue
            match = identified.speakers.get(qid)
            promoted = False
            promote_key: str | None = None
            if match is not None and match.speaker_key and match.is_unknown:
                promote_key = match.speaker_key
            else:
                for proposal, key in zip(
                    identified.proposed_unknowns, committed, strict=True
                ):
                    if proposal.query_speaker_id == qid:
                        promote_key = key
                        break
            if promote_key is not None:
                try:
                    summary = session.app.label_speaker(promote_key, person)
                    move_contrib(promote_key, summary.speaker_key)
                    promoted = True
                except SpeakerNotFoundError:
                    # the unknown was renamed/absorbed since (e.g. two strangers
                    # dedup-merged, the other one got labeled first) — enroll
                    promoted = False
            if promoted:
                labeled_idx[person] = i
                contrib.setdefault(person, set()).add((rec.rid, qid))
            else:
                try:
                    enrolled_result = session.enroll(cache, qid, person)
                    labeled_idx[person] = i
                    contrib.setdefault(person, set()).add((rec.rid, qid))
                    for merged in enrolled_result.merged_unknown_keys:
                        move_contrib(merged, person)
                except NoUsableSegmentsError:
                    result.enroll_fallbacks += 1
                    occs = occurrences[person]
                    next_try[person] += 1
                    if next_try[person] < len(occs):
                        first_clean[person] = occs[next_try[person]]

        # 4. reinforcement pass, auto-committed
        proposals = session.reinforce(cache)
        keys = session.app.commit_reinforcements(proposals)
        for proposal, key in zip(proposals, keys, strict=True):
            if key is not None:
                contrib.setdefault(key, set()).add(
                    (rec.rid, proposal.query_speaker_id)
                )

    # final store stats + contamination
    summaries = session.app.store.list_speakers([data.uid])
    result.final_entries = session.app.store.count(data.uid)
    result.final_named_speakers = sum(1 for s in summaries if not s.is_unknown)
    result.final_unknown_keys = sum(1 for s in summaries if s.is_unknown)
    result.true_never_labeled = sum(1 for p in occurrences if p not in labeled_idx)

    ann_person = {
        (rid, qid): s.person
        for rid, ann in data.annotations.items()
        for qid, s in ann.speakers.items()
        if s.quality == "clean"
    }
    for key, sources in contrib.items():
        persons = sorted({ann_person[src] for src in sources if src in ann_person})
        if len(persons) >= 2:
            result.contaminated_keys[key] = persons
    return result
