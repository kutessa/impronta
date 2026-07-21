"""The full unknown-speaker lifecycle: propose -> commit -> recognize -> label.

identify() never writes to the store. Strangers that pass the quality gates
come back as serializable proposals; your app decides whether to persist
them (commit_unknowns is idempotent and dedups against existing unknowns).

Usage:
    export ELEVENLABS_API_KEY=...
    uv run python examples/02_unknowns_workflow.py meeting1.wav meeting2.wav
"""

import sys

from impronta import FaissLocalStore, Impronta, UnknownProposal

sys.path.insert(0, "examples")
from importlib import import_module

transcribe = import_module("01_enroll_and_identify").transcribe


def main() -> None:
    rec1, rec2 = sys.argv[1], sys.argv[2]
    app = Impronta(store=FaissLocalStore())

    # --- recording 1: everyone is a stranger ---
    resp1 = transcribe(rec1)
    result1 = app.identify(resp1, rec1)
    for sid, match in result1.speakers.items():
        if match.is_unknown and match.no_proposal_reason:
            # gated out: too_few_segments / low_quality / low_cohesion / gray_zone
            print(f"{sid}: unknown, NOT saved ({match.no_proposal_reason})")

    # proposals are plain data — you could park them in Firestore / a task
    # queue and commit later. Round-trip through dicts to prove it:
    payloads = [p.to_dict() for p in result1.proposed_unknowns]
    committed = app.commit_unknowns([UnknownProposal.from_dict(d) for d in payloads])
    print("committed unknown speakers:", committed)

    # committing the same proposals again is a no-op (deterministic ids)
    assert app.commit_unknowns(result1.proposed_unknowns) == committed

    # --- recording 2: the same stranger is now recognized ---
    resp2 = transcribe(rec2)
    result2 = app.identify(resp2, rec2)
    for sid, match in result2.speakers.items():
        if match.speaker_key and match.is_unknown:
            print(f"{sid}: recognized returning unknown {match.speaker_key}")
            # once you learn who they are:
            summary = app.label_speaker(match.speaker_key, "Sara")
            print(f"  promoted to {summary.speaker_key!r} ({summary.display_name})")


if __name__ == "__main__":
    main()
