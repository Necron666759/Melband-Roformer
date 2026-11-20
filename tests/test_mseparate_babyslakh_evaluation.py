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


def test_crop_representative_selects_sustained_active_window(tmp_path):
    import numpy as np
    import soundfile as sf
    from evaluate_mseparate_babyslakh import crop_representative

    sample_rate = 1000
    quiet = np.full((3000, 2), 0.02, dtype=np.float32)
    active = np.full((4000, 2), 0.8, dtype=np.float32)
    tail = np.full((3000, 2), 0.15, dtype=np.float32)
    source = tmp_path / "dynamic.wav"
    destination = tmp_path / "excerpt.wav"
    sf.write(source, np.concatenate((quiet, active, tail)), sample_rate)

    result = crop_representative(source, destination, 3.0)
    excerpt, rate = sf.read(destination, dtype="float32", always_2d=True)

    assert result["selection_method"] == "maximum_mean_square_energy"
    assert result["start_seconds"] == 3.0
    assert result["duration_seconds"] == 3.0
    assert result["candidate_windows"] >= 7
    assert rate == sample_rate
    assert float(np.sqrt(np.mean(excerpt ** 2))) > 0.75


def test_routing_drift_details_detects_partial_target_loss():
    from evaluate_mseparate_babyslakh import routing_drift_details

    result = routing_drift_details(
        [
            {"target_stems": ["piano", "organ"]},
            {"target_stems": ["synth"]},
        ],
        [{"target_stems": ["piano", "synth"]}],
    )

    assert result["full_track_target_stems"] == ["organ", "piano", "synth"]
    assert result["excerpt_target_stems"] == ["piano", "synth"]
    assert result["missing_full_track_targets"] == ["organ"]
    assert result["excerpt_only_targets"] == []


def test_routing_drift_details_reports_excerpt_only_targets_without_losing_full_track_targets():
    from evaluate_mseparate_babyslakh import routing_drift_details

    result = routing_drift_details(
        [{"target_stems": ["piano"]}],
        [{"target_stems": ["piano", "organ"]}],
    )

    assert result["missing_full_track_targets"] == []
    assert result["excerpt_only_targets"] == ["organ"]


def test_crop_representative_windows_selects_distinct_regions_without_overlap(tmp_path):
    import numpy as np
    import soundfile as sf
    from evaluate_mseparate_babyslakh import crop_representative_windows

    sample_rate = 1000
    quiet_a = np.full((3000, 2), 0.02, dtype=np.float32)
    active_a = np.full((3000, 2), 0.8, dtype=np.float32)
    quiet_b = np.full((3000, 2), 0.03, dtype=np.float32)
    active_b = np.full((3000, 2), 0.6, dtype=np.float32)
    source = tmp_path / "multi-region.wav"
    sf.write(source, np.concatenate((quiet_a, active_a, quiet_b, active_b)), sample_rate)

    windows = crop_representative_windows(
        source, tmp_path / "windows", 2.0, max_windows=3,
    )

    assert len(windows) == 3
    starts = [item["start_seconds"] for item in windows]
    assert starts == sorted(starts)
    assert all(
        abs(left - right) >= 2.0
        for index, left in enumerate(starts)
        for right in starts[index + 1:]
    )
    assert max(item["window_rms"] for item in windows) > 0.5


def test_multi_window_routing_details_uses_union_and_reports_local_activity():
    from evaluate_mseparate_babyslakh import multi_window_routing_details

    result = multi_window_routing_details(
        [{"target_stems": ["keys", "organ", "synth"]}],
        [{"target_stems": ["keys", "organ", "synth"]}],
        [
            [{"target_stems": ["keys", "synth"]}],
            [{"target_stems": ["organ"]}],
            [],
        ],
    )

    assert result["multi_window_target_stems"] == ["keys", "organ", "synth"]
    assert result["multi_window_missing_full_track_targets"] == []
    assert result["window_target_presence"] == {"keys": 1, "organ": 1, "synth": 1}
    assert result["per_window_target_stems"] == [
        ["keys", "synth"], ["organ"], []
    ]

def test_crop_representative_windows_stratifies_the_entire_arrangement(tmp_path):
    import numpy as np
    import soundfile as sf
    from evaluate_mseparate_babyslakh import crop_representative_windows

    sample_rate = 1000
    # The first third is very loud; later thirds are quieter but contain
    # distinct active passages that global greedy ranking would under-sample.
    loud = np.full((9000, 2), 0.9, dtype=np.float32)
    middle = np.concatenate((
        np.full((3000, 2), 0.03, dtype=np.float32),
        np.full((3000, 2), 0.35, dtype=np.float32),
        np.full((3000, 2), 0.03, dtype=np.float32),
    ))
    quiet_tail = np.concatenate((
        np.full((3000, 2), 0.02, dtype=np.float32),
        np.full((3000, 2), 0.25, dtype=np.float32),
        np.full((3000, 2), 0.02, dtype=np.float32),
    ))
    source = tmp_path / "stratified.wav"
    sf.write(source, np.concatenate((loud, middle, quiet_tail)), sample_rate)

    windows = crop_representative_windows(
        source, tmp_path / "stratified-windows", 2.0, max_windows=3,
    )

    assert len(windows) == 3
    assert [item["selection_method"] for item in windows] == [
        "stratified_time_bins_max_energy"
    ] * 3
    assert [item["window_index"] for item in windows] == [1, 2, 3]
    assert [item["time_bin_start_seconds"] for item in windows] == [0.0, 9.0, 18.0]
    assert [item["time_bin_end_seconds"] for item in windows] == [9.0, 18.0, 27.0]
    assert [item["start_seconds"] for item in windows] == [0.0, 12.0, 21.0]
    assert windows[0]["window_rms"] > windows[1]["window_rms"] > windows[2]["window_rms"]
    assert all(
        left["start_seconds"] + 2.0 <= right["start_seconds"]
        for left, right in zip(windows, windows[1:])
    )


def test_window_source_evidence_details_exposes_temporal_coverage():
    from evaluate_mseparate_babyslakh import window_source_evidence_details

    result = window_source_evidence_details(
        [
            [
                {"kind": "source", "source": "bass", "score": 8.0},
                {"kind": "source", "source": "synth", "score": 7.0},
            ],
            [
                {"kind": "source", "source": "synth", "score": 8.0},
                {"kind": "source", "source": "piano", "score": 7.0},
            ],
            [
                {"kind": "source", "source": "piano", "score": 8.0},
                {"kind": "genre", "source": "electro_industrial", "score": 0.99},
            ],
        ],
        [
            {"window_index": 1, "start_seconds": 0.0, "time_bin_start_seconds": 0.0,
             "time_bin_end_seconds": 20.0, "window_rms": 0.5},
            {"window_index": 2, "start_seconds": 20.0, "time_bin_start_seconds": 20.0,
             "time_bin_end_seconds": 40.0, "window_rms": 0.4},
            {"window_index": 3, "start_seconds": 40.0, "time_bin_start_seconds": 40.0,
             "time_bin_end_seconds": 60.0, "window_rms": 0.3},
        ],
    )

    coverage = {item["source"]: item for item in result["source_temporal_coverage"]}
    assert coverage["bass"]["observed_windows"] == 1
    assert coverage["bass"]["top_5_windows"] == 1
    assert coverage["synth"]["observed_windows"] == 2
    assert coverage["piano"]["observed_windows"] == 2
    assert "electro_industrial" not in coverage
    assert result["per_window_source_predictions"][0]["start_seconds"] == 0.0

