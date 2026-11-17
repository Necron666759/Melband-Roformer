"""Unit tests for metadata-only URMP evaluation diagnostics."""
from __future__ import annotations

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
