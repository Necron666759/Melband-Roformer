"""CPU-only tests for reversible candidate quarantine."""
from __future__ import annotations

import pytest

from mseparate.quarantine import QuarantineError, QuarantineLedger


def test_quarantine_preserves_candidate_lineage_and_supports_repromotion(tmp_path):
    ledger = QuarantineLedger(tmp_path / "job" / "quarantine.json")
    parent = ledger.add_candidate(
        "parent-vocals", stem="vocals", path=tmp_path / "parent.wav",
        model_id="bs-roformer-anvuew-vocals",
    )
    candidate = ledger.add_candidate(
        "rescue-vocals", stem="vocals", path=tmp_path / "rescue.wav",
        model_id="zfturbo-bs-roformer-53stem-candidate",
        parent_id=parent["candidate_id"],
        reason="rescue candidate pending comparison",
        analysis={"technical_status": "valid"},
    )
    assert candidate["status"] == "pending"
    assert candidate["parent_id"] == "parent-vocals"

    ledger.transition("rescue-vocals", "quarantined", reason="possible artifact; hold for review")
    ledger.transition("rescue-vocals", "pending", reason="re-evaluation requested")
    promoted = ledger.transition("rescue-vocals", "accepted", reason="comparative review accepted")
    assert promoted["status"] == "accepted"
    assert [event["to"] for event in promoted["history"]] == [
        "pending", "quarantined", "pending", "accepted",
    ]
    assert len(ledger.list_candidates()) == 2


def test_quarantine_rejects_invalid_transitions_and_unknown_parent(tmp_path):
    ledger = QuarantineLedger(tmp_path / "ledger.json")
    with pytest.raises(QuarantineError, match="Unknown parent"):
        ledger.add_candidate(
            "orphan", stem="bass", path="orphan.wav", model_id="test",
            parent_id="missing",
        )
    ledger.add_candidate("candidate", stem="bass", path="bass.wav", model_id="test")
    with pytest.raises(QuarantineError, match="Invalid candidate transition"):
        ledger.transition("candidate", "superseded", reason="not yet accepted")
    with pytest.raises(QuarantineError, match="transition reason"):
        ledger.transition("candidate", "quarantined", reason="")
