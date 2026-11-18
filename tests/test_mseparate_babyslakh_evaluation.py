from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_mseparate_babyslakh import execution_counts, stage_c_progress_message

def test_execution_counts_reads_serialized_task_results():
    payload = {
        "execution": {
            "results": [
                {"task_id": "a", "status": "success"},
                {"task_id": "b", "status": "failed"},
                {"task_id": "c", "status": "skipped"},
            ]
        },
        "plan": {"tasks": [{"task_id": "a"}, {"task_id": "b"}, {"task_id": "c"}]},
    }

    assert execution_counts(payload) == (
        1, 1, payload["execution"]["results"], 3
    )


def test_execution_counts_handles_no_execution_and_no_excerpt_tasks():
    assert execution_counts({"execution": None, "plan": {"tasks": []}}) == (
        0, 0, [], 0
    )


def test_execution_counts_does_not_confuse_plan_with_execution():
    payload = {
        "execution": None,
        "plan": {"tasks": [{"task_id": "planned-but-not-run"}]},
    }

    succeeded, failed, results, excerpt_tasks = execution_counts(payload)

    assert (succeeded, failed, results, excerpt_tasks) == (0, 0, [], 1)

def test_stage_c_progress_message_uses_parsed_counts_without_execution_global():
    assert stage_c_progress_message(
        2, 4, 3, 1, {"status": "completed"}
    ) == "Stage C 2/4: 3 succeeded, 1 failed, rescue=completed"


def test_stage_c_progress_message_handles_missing_rescue_status():
    assert stage_c_progress_message(1, 1, 0, 0, {}) == (
        "Stage C 1/1: 0 succeeded, 0 failed, rescue=unknown"
    )
