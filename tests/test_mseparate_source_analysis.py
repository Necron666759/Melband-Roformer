"""Window-level CLAP evidence diagnostics."""
from __future__ import annotations

import numpy as np

from mseparate.source_analysis import _window_rank_diagnostics


def test_window_rank_diagnostics_preserve_intermittent_source_evidence():
    # Columns 0, 1 and 2 are source prompts; column 3 is a genre prompt.
    scores = np.asarray([
        [0.90, 0.70, 0.60, 0.99],
        [0.50, 0.80, 0.70, 0.98],
        [0.85, 0.75, 0.65, 0.97],
    ])
    result = _window_rank_diagnostics(scores, (0, 1, 2), 0)

    assert result["window_score_std"] > 0
    assert result["window_best_source_rank"] == 1
    assert result["window_mean_source_rank"] == 1.667
    assert result["window_top_5_count"] == 3
    assert result["window_top_12_fraction"] == 1.0


def test_window_rank_diagnostics_excludes_genre_columns_from_source_ranks():
    scores = np.asarray([
        [0.70, 0.60, 0.99],
        [0.65, 0.75, 0.98],
    ])
    result = _window_rank_diagnostics(scores, (0, 1), 0)

    assert result["window_best_source_rank"] == 1
    assert result["window_top_5_count"] == 2


def test_window_rank_diagnostics_returns_empty_for_non_source_candidate():
    scores = np.asarray([[0.7, 0.99]])
    assert _window_rank_diagnostics(scores, (0,), 1) == {}
