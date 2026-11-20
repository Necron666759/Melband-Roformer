"""Unit tests for metadata-only URMP evaluation diagnostics."""
from __future__ import annotations

import pytest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_mseparate_urmp import (  # noqa: E402
    GIB,
    _ground_truth_ranking,
    _rtx3060ti_low_vram_boundary,
)


def test_ground_truth_ranking_reports_recall_and_repeated_performers():
    result = _ground_truth_ranking(
        [
            {"kind": "source", "source": "cello", "score": 10.0},
            {"kind": "source", "source": "violin", "score": 9.0},
            {"kind": "source", "source": "flute", "score": 8.0},
        ],
        ["violin", "violin", "cello", "oboe"],
    )

    assert result["unique_ground_truth_instruments"] == 3
    assert result["matched_in_top_5"] == 2
    assert result["matched_in_top_12"] == 2
    assert result["top_12_recall"] == round(2 / 3, 4)
    violin = next(item for item in result["instruments"] if item["instrument"] == "violin")
    assert violin["rank"] == 2
    assert violin["mixture_count"] == 2
    assert "cannot count" in result["duplicate_instrument_note"]


def test_ground_truth_ranking_does_not_treat_genre_as_instrument():
    result = _ground_truth_ranking(
        [
            {"kind": "genre", "source": "chamber_music", "score": 99.0},
            {"kind": "source", "source": "oboe", "score": 7.0},
        ],
        ["oboe", "cello"],
    )
    assert result["instruments"][0]["rank"] == 1
    assert result["instruments"][1]["rank"] is None
    assert result["top_12_recall"] == 0.5


def test_rtx3060ti_probe_is_an_explicit_low_vram_boundary_not_live_probe():
    free_bytes, total_bytes = _rtx3060ti_low_vram_boundary()
    assert free_bytes == 6 * GIB
    assert total_bytes == int(7.68 * GIB)


def test_ground_truth_ranking_reports_intermittent_window_recovery():
    result = _ground_truth_ranking(
        [
            {"kind": "source", "source": "guitar", "score": 10.0},
            {"kind": "source", "source": "piano", "score": 9.5},
            {"kind": "source", "source": "bass", "score": 9.0},
            {"kind": "source", "source": "drums", "score": 8.5},
            {
                "kind": "source", "source": "oboe", "score": 7.2,
                "window_best_source_rank": 3,
                "window_mean_source_rank": 9.5,
                "window_top_5_fraction": 0.25,
                "window_top_12_fraction": 0.75,
            },
            {
                "kind": "source", "source": "cello", "score": 8.1,
                "window_best_source_rank": 1,
                "window_mean_source_rank": 2.0,
                "window_top_5_fraction": 1.0,
                "window_top_12_fraction": 1.0,
            },
        ],
        ["oboe", "cello", "flute"],
    )

    oboe = next(row for row in result["instruments"] if row["instrument"] == "oboe")
    assert oboe["rank"] == 6
    assert oboe["in_top_5"] is False
    assert oboe["window_best_source_rank"] == 3
    assert oboe["window_top_5_fraction"] == 0.25
    assert result["window_top_5_any_recall"] == round(2 / 3, 4)
    assert result["window_top_12_any_recall"] == round(2 / 3, 4)
    assert result["window_only_top_5_recoveries"] == ["oboe"]



def test_planner_target_audit_reports_false_positives_and_missed_instruments():
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from evaluate_mseparate_urmp import _planner_target_audit

    result = _planner_target_audit(
        [
            {"target_stems": ["violin", "viola"]},
            {"target_stems": ["brass", "saxophone"]},
        ],
        ["flute", "violin", "viola", "clarinet"],
    )

    assert result["true_positive_targets"] == ["viola", "violin"]
    assert result["false_positive_targets"] == ["brass", "saxophone"]
    assert result["missed_ground_truth_targets"] == ["clarinet", "flute"]
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)


def test_planner_temporal_evidence_recovers_interleaved_woodwinds_without_widening_all_gates():
    from mseparate.planner import _detected_acoustic_targets

    predictions = [
        {"source": "woodwinds", "score": 11.4, "window_best_source_rank": 1,
         "window_top_5_fraction": 0.95},
        {"source": "flute", "score": 10.3, "window_best_source_rank": 1,
         "window_top_5_fraction": 0.66},
        {"source": "oboe", "score": 9.5, "window_best_source_rank": 3,
         "window_top_5_fraction": 0.42},
        {"source": "clarinet", "score": 9.1, "window_best_source_rank": 3,
         "window_top_5_fraction": 0.33},
        {"source": "bassoon", "score": 8.0, "window_best_source_rank": 5,
         "window_top_5_fraction": 0.04},
    ]

    targets = _detected_acoustic_targets(predictions, {"family": "orchestral"})

    assert {"flute", "oboe", "clarinet"} <= set(targets)
    assert "bassoon" not in targets


def test_planner_temporal_gate_rejects_weak_flute_and_broad_brass_false_positives():
    from mseparate.planner import _detected_acoustic_targets

    predictions = [
        {"source": "violin", "score": 10.1, "window_best_source_rank": 1,
         "window_top_5_fraction": 0.83},
        {"source": "woodwinds", "score": 9.9, "window_best_source_rank": 1,
         "window_top_5_fraction": 0.94},
        {"source": "brass", "score": 9.3, "window_best_source_rank": 1,
         "window_top_5_fraction": 0.55},
        {"source": "viola", "score": 8.9, "window_best_source_rank": 3,
         "window_top_5_fraction": 0.44},
        {"source": "saxophone", "score": 8.9, "window_best_source_rank": 3,
         "window_top_5_fraction": 0.22},
        {"source": "flute", "score": 7.6, "window_best_source_rank": 5,
         "window_top_5_fraction": 0.11},
    ]

    targets = _detected_acoustic_targets(predictions, {"family": "orchestral"})

    assert "violin" in targets
    assert "brass" not in targets
    assert "saxophone" not in targets
    assert "flute" not in targets


def test_planner_temporal_audit_suppresses_sparse_sax_horn_and_weak_cello_confusions():
    from mseparate.planner import _detected_acoustic_targets

    predictions = [
        {"source": "violin", "score": 10.1, "window_best_source_rank": 1,
         "window_top_5_fraction": 0.83, "window_mean_source_rank": 6.1},
        {"source": "woodwinds", "score": 9.97, "window_best_source_rank": 1,
         "window_top_5_fraction": 0.94, "window_mean_source_rank": 5.1},
        {"source": "saxophone", "score": 8.95, "window_best_source_rank": 3,
         "window_top_5_fraction": 0.22, "window_mean_source_rank": 8.9},
        {"source": "french_horn", "score": 8.8, "window_best_source_rank": 4,
         "window_top_5_fraction": 0.11, "window_mean_source_rank": 14.1},
        {"source": "cello", "score": 8.82, "window_best_source_rank": 3,
         "window_top_5_fraction": 0.2778, "window_mean_source_rank": 9.556},
        {"source": "viola", "score": 8.96, "window_best_source_rank": 3,
         "window_top_5_fraction": 0.444, "window_mean_source_rank": 9.389},
    ]

    targets = _detected_acoustic_targets(predictions, {"family": "orchestral"})

    assert "violin" in targets
    assert "viola" in targets
    assert "saxophone" not in targets
    assert "french-horn" not in targets
    assert "cello" not in targets


def test_planner_temporal_gate_keeps_sustained_saxophone_and_cello_candidates():
    from mseparate.planner import _detected_acoustic_targets

    predictions = [
        {"source": "violin", "score": 10.0, "window_best_source_rank": 1,
         "window_top_5_fraction": 0.9, "window_mean_source_rank": 2.0},
        {"source": "saxophone", "score": 9.8, "window_best_source_rank": 2,
         "window_top_5_fraction": 0.83, "window_mean_source_rank": 3.0},
        {"source": "cello", "score": 9.7, "window_best_source_rank": 2,
         "window_top_5_fraction": 0.8, "window_mean_source_rank": 3.5},
    ]

    targets = _detected_acoustic_targets(predictions, {"family": "orchestral"})

    assert "saxophone" in targets
    assert "cello" in targets
