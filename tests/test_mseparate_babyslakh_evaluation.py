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



def test_compare_routing_plans_detects_changed_targets_even_when_task_counts_match():
    from evaluate_mseparate_babyslakh import compare_routing_plans

    result = compare_routing_plans(
        {"tasks": [{"model_id": "model-a", "input_stem_id": "mix", "target_stems": ["bass"]}]},
        {"tasks": [{"model_id": "model-b", "input_stem_id": "mix", "target_stems": ["synth"]}]},
    )

    assert result["routing_drift"] is True
    assert result["full_track_only_routes"] == [
        {"model_id": "model-a", "input_stem_id": "mix", "target_stems": ["bass"]}
    ]
    assert result["excerpt_only_routes"] == [
        {"model_id": "model-b", "input_stem_id": "mix", "target_stems": ["synth"]}
    ]


def test_compare_routing_plans_accepts_identical_routes():
    from evaluate_mseparate_babyslakh import compare_routing_plans

    plan = {"tasks": [
        {"model_id": "model-a", "input_stem_id": "mix", "target_stems": ["violin", "cello"]}
    ]}
    result = compare_routing_plans(plan, plan)

    assert result["routing_drift"] is False
    assert result["full_track_only_routes"] == []
    assert result["excerpt_only_routes"] == []


def test_compare_routing_plans_detects_full_track_route_missing_from_excerpt():
    from evaluate_mseparate_babyslakh import compare_routing_plans

    result = compare_routing_plans(
        {"tasks": [{"model_id": "model-a", "target_stems": ["piano"]}]},
        {"tasks": []},
    )

    assert result["routing_drift"] is True
    assert result["full_track_only_routes"][0]["target_stems"] == ["piano"]
