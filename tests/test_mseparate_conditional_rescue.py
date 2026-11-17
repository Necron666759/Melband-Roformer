"""CPU-only tests for conservative conditional 53-stem rescue routing."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from mseparate.candidate_analysis import (
    analyze_execution_candidates,
    evaluate_conditional_rescue,
)
from mseparate.execution import ExecutionSummary, StemArtifact, TaskResult
from mseparate.quarantine import QuarantineLedger
from mseparate.registry import parse_manifest
from mseparate.types import AudioMetadata


def _rescue_manifest():
    return parse_manifest({
        "model_id": "zfturbo-bs-roformer-53stem-candidate",
        "name": "ZFTurbo 53-stem",
        "architecture": "BS-RoFormer",
        "backend": "bs-roformer-pymss",
        "source_url": "https://example.invalid/model",
        "license": "MIT",
        "license_status": "verified",
        "status": "available",
        "runtime_validation": "verified",
        "sample_rate": 44100,
        "channels": "stereo",
        "input_formats": ["wav"],
        "category": "mega-multi-stem",
        "native_stems": ["synth", "bass", "vocal", "lead-vocal", "back-vocal"],
        "target_stems": ["synth", "bass", "vocal", "lead-vocal", "back-vocal"],
        "assets": [
            {
                "role": "checkpoint", "url": "https://example.invalid/model.ckpt",
                "relative_path": "model.ckpt", "sha256": "a" * 64,
            },
            {
                "role": "config", "url": "https://example.invalid/model.yaml",
                "relative_path": "model.yaml", "sha256": "b" * 64,
            },
        ],
    })


def _audio(path: Path):
    return AudioMetadata(
        path=str(path), duration_seconds=10.0, sample_rate=44100,
        channels=2, frames=441000, subtype="FLOAT",
    )


def _report(parent_path: Path, *, mix_rank=3, other_rank=2, technical="valid"):
    return {
        "candidates": [{
            "candidate_id": "mvs-other-1",
            "model_id": "mvsep-mega53-bass-specialist-candidate",
            "stem": "other",
            "path": str(parent_path),
            "technical_status": technical,
            "residual_source_hints": [{
                "source": "synth_pad",
                "mix_rank": mix_rank,
                "other_rank": other_rank,
                "clap_score": 0.4,
            }],
        }],
    }


def test_strong_residual_hint_builds_bounded_native_target_request(tmp_path):
    parent = tmp_path / "other.wav"
    parent.write_bytes(b"placeholder")
    request = evaluate_conditional_rescue(
        _report(parent),
        registry={"zfturbo-bs-roformer-53stem-candidate": _rescue_manifest()},
        audio=_audio(tmp_path / "mix.wav"),
    )
    assert request["status"] == "ready"
    assert request["parent_candidate_id"] == "mvs-other-1"
    assert request["target_stems"] == ["synth"]
    assert request["gate"]["mix_rank_max"] == 5
    assert request["gate"]["other_rank_max"] == 3


def test_rescue_is_skipped_when_hint_does_not_meet_both_rank_thresholds(tmp_path):
    parent = tmp_path / "other.wav"
    parent.write_bytes(b"placeholder")
    request = evaluate_conditional_rescue(
        _report(parent, mix_rank=6, other_rank=2),
        registry={"zfturbo-bs-roformer-53stem-candidate": _rescue_manifest()},
        audio=_audio(tmp_path / "mix.wav"),
    )
    assert request["status"] == "skipped"
    assert "conservative rescue gate" in request["reason"]


def test_rescue_requires_technically_valid_mvs_other(tmp_path):
    parent = tmp_path / "other.wav"
    parent.write_bytes(b"placeholder")
    request = evaluate_conditional_rescue(
        _report(parent, technical="warning"),
        registry={"zfturbo-bs-roformer-53stem-candidate": _rescue_manifest()},
        audio=_audio(tmp_path / "mix.wav"),
    )
    assert request["status"] == "skipped"
    assert "technically clean Other" in request["reason"]


def test_rescue_is_skipped_when_device_vram_gate_removed_model(tmp_path):
    parent = tmp_path / "other.wav"
    parent.write_bytes(b"placeholder")
    request = evaluate_conditional_rescue(
        _report(parent), registry={}, audio=_audio(tmp_path / "mix.wav"),
    )
    assert request["status"] == "skipped"
    assert "device-eligible registry" in request["reason"]


def test_rescue_outputs_are_registered_as_children_of_exact_other(tmp_path, monkeypatch):
    from mseparate import candidate_analysis

    ledger_path = tmp_path / "quarantine.json"
    ledger = QuarantineLedger(ledger_path)
    parent_path = tmp_path / "other.wav"
    parent_path.write_bytes(b"parent")
    ledger.add_candidate(
        "mvs-other-1", stem="other", path=parent_path,
        model_id="mvsep-mega53-bass-specialist-candidate",
    )

    output = tmp_path / "synth.wav"
    output.write_bytes(b"candidate")
    result = TaskResult(
        "rescue-1", "zfturbo-bs-roformer-53stem-candidate", "success",
        outputs=(StemArtifact("synth", output),),
        quality_reports=(SimpleNamespace(path=str(output), status="valid", issues=()),),
    )
    monkeypatch.setattr(
        candidate_analysis, "analyze_sources_many",
        lambda paths, **kwargs: {
            str(output): [{"kind": "source", "source": "synthesizer", "score": 0.9}],
        },
    )
    report = analyze_execution_candidates(
        ExecutionSummary((result,)),
        mix_path=parent_path,
        mix_predictions=[{"kind": "source", "source": "synth_pad", "score": 0.8}],
        device="cpu", cache_dir=None, no_download=True,
        ledger_path=ledger_path, parent_candidate_id="mvs-other-1",
    )
    assert report["parent_candidate_id"] == "mvs-other-1"
    records = QuarantineLedger(ledger_path).list_candidates()
    child = next(item for item in records if item["model_id"] == "zfturbo-bs-roformer-53stem-candidate")
    assert child["parent_id"] == "mvs-other-1"
    assert child["status"] == "pending"



def test_rescue_lineage_is_preserved_when_post_rescue_clap_fails(tmp_path, monkeypatch):
    from mseparate import candidate_analysis

    ledger_path = tmp_path / "quarantine.json"
    ledger = QuarantineLedger(ledger_path)
    parent_path = tmp_path / "other.wav"
    parent_path.write_bytes(b"parent")
    ledger.add_candidate(
        "mvs-other-2", stem="other", path=parent_path,
        model_id="mvsep-mega53-synth-specialist-candidate",
    )

    output = tmp_path / "synth.wav"
    output.write_bytes(b"candidate")
    result = TaskResult(
        "rescue-2", "zfturbo-bs-roformer-53stem-candidate", "success",
        outputs=(StemArtifact("synth", output),),
        quality_reports=(SimpleNamespace(path=str(output), status="valid", issues=()),),
    )

    def fail_analysis(*args, **kwargs):
        raise RuntimeError("simulated CLAP failure")

    monkeypatch.setattr(candidate_analysis, "analyze_sources_many", fail_analysis)
    report = analyze_execution_candidates(
        ExecutionSummary((result,)),
        mix_path=parent_path,
        mix_predictions=[],
        device="cuda", cache_dir=None, no_download=True,
        ledger_path=ledger_path, parent_candidate_id="mvs-other-2",
        allow_analysis_failure=True,
    )
    assert report["analysis_status"] == "unavailable"
    assert "simulated CLAP failure" in report["analysis_error"]
    records = QuarantineLedger(ledger_path).list_candidates()
    child = next(item for item in records if item["model_id"] == "zfturbo-bs-roformer-53stem-candidate")
    assert child["parent_id"] == "mvs-other-2"
    assert child["status"] == "pending"
