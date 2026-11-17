"""CPU-only tests for post-inference source analysis and candidate lineage."""
from __future__ import annotations

from types import SimpleNamespace

from mseparate.candidate_analysis import _candidate_record, analyze_execution_candidates


def test_other_stem_flags_only_shared_high_rank_sources_as_residual_hints(tmp_path):
    artifact = SimpleNamespace(stem="other", path=tmp_path / "other.wav")
    predictions = [
        {"kind": "source", "source": "synth_pad", "score": 0.91},
        {"kind": "source", "source": "bass", "score": 0.87},
        {"kind": "source", "source": "vocals", "score": 0.82},
        {"kind": "genre", "source": "techno", "score": 0.99},
    ]
    mix_ranks = {"synth_pad": 2, "bass": 8, "vocals": 12}
    report = SimpleNamespace(status="valid", issues=())
    result = _candidate_record(
        artifact=artifact, task_id="task", model_id="mega53-bass",
        report=report, predictions=predictions, mix_ranks=mix_ranks,
    )
    assert [item["source"] for item in result["residual_source_hints"]] == [
        "synth_pad", "bass",
    ]
    assert "vocals" not in [item["source"] for item in result["residual_source_hints"]]
    assert result["decision"] == "pending_review"


def test_candidate_analysis_registers_pending_candidates_and_lineage(tmp_path, monkeypatch):
    from mseparate import candidate_analysis
    from mseparate.execution import ExecutionSummary, StemArtifact, TaskResult

    output = tmp_path / "bass.wav"
    output.write_bytes(b"placeholder")
    result = TaskResult(
        "task-1", "mega53-bass", "success",
        outputs=(StemArtifact("bass", output),),
        quality_reports=(SimpleNamespace(path=str(output), status="valid", issues=()),),
    )
    monkeypatch.setattr(
        candidate_analysis, "analyze_sources_many",
        lambda paths, **kwargs: {
            str(output): [
                {"kind": "source", "source": "bass", "score": 0.9},
                {"kind": "source", "source": "vocals", "score": 0.4},
            ],
        },
    )
    ledger_path = tmp_path / "quarantine.json"
    report = analyze_execution_candidates(
        ExecutionSummary((result,)), mix_path=tmp_path / "mix.wav",
        mix_predictions=[{"kind": "source", "source": "bass", "score": 0.8}],
        device="cpu", cache_dir=None, no_download=True, ledger_path=ledger_path,
    )
    assert report["candidate_count"] == 1
    candidate = report["candidates"][0]
    assert candidate["target_source_rank"] == 1
    assert candidate["parent_candidate_id"] == report["parent_candidate_id"]
    assert candidate["decision"] == "pending_review"
    from mseparate.quarantine import QuarantineLedger
    records = QuarantineLedger(ledger_path).list_candidates()
    assert len(records) == 2
    assert all(item["status"] == "pending" for item in records)
