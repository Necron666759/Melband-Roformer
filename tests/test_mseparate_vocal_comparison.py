"""Tests for opt-in generation of comparable primary vocal candidates."""
from __future__ import annotations

from mseparate.planner import add_vocal_comparison_tasks, build_plan, validate_plan
from mseparate.registry import bundled_registry_path, load_registry
from mseparate.types import AnalysisReport, AudioMetadata


def test_vocal_comparison_schedules_all_eligible_primary_models_sequentially():
    registry = load_registry(bundled_registry_path())
    analysis = AnalysisReport(
        schema_version=1,
        audio=AudioMetadata(
            path="mix.wav", duration_seconds=12.0, sample_rate=44100,
            channels=2, frames=529200, subtype="PCM_16",
        ),
        measurements={"mid_band_energy_ratio": 0.4},
        predictions=[{"kind": "source", "source": "vocals", "score": 0.8}],
    )
    normal = build_plan(analysis=analysis, registry=registry)
    compared = add_vocal_comparison_tasks(
        normal, analysis=analysis, registry=registry,
    )
    validate_plan(compared)

    vocal_models = {
        task.model_id for task in compared.tasks if "vocals" in task.target_stems
    }
    assert vocal_models == {
        "melband-roformer-kim-vocals",
        "bs-roformer-anvuew-vocals",
        "mdx23c-zfturbo-vocals",
    }
    comparison_tasks = [task for task in compared.tasks if task.task_id.startswith("compare-vocals-")]
    assert len(comparison_tasks) == 2
    assert all(task.input_stem_id == "mix" for task in comparison_tasks)
    assert all(task.target_stems == ("vocals",) for task in comparison_tasks)
    assert all(task.depends_on == () for task in comparison_tasks)
    assert any("not measure separation quality" in note for note in compared.notes)


def test_vocal_comparison_does_not_duplicate_a_model_already_in_plan():
    registry = load_registry(bundled_registry_path())
    analysis = AnalysisReport(
        schema_version=1,
        audio=AudioMetadata(
            path="mix.wav", duration_seconds=12.0, sample_rate=44100,
            channels=2, frames=529200, subtype="PCM_16",
        ),
    )
    normal = build_plan(analysis=analysis, registry=registry)
    compared = add_vocal_comparison_tasks(normal, analysis=analysis, registry=registry)
    ids = [task.model_id for task in compared.tasks if "vocals" in task.target_stems]
    assert len(ids) == len(set(ids))
