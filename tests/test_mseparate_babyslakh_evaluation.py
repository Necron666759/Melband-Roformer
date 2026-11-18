from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_mseparate_babyslakh import execution_counts, stage_c_progress_message

def test_stage_c_progress_message_uses_parsed_counts_without_execution_global():
    assert stage_c_progress_message(
        2, 4, 3, 1, {"status": "completed"}
    ) == "Stage C 2/4: 3 succeeded, 1 failed, rescue=completed"


def test_stage_c_progress_message_handles_missing_rescue_status():
    assert stage_c_progress_message(1, 1, 0, 0, {}) == (
        "Stage C 1/1: 0 succeeded, 0 failed, rescue=unknown"
    )
