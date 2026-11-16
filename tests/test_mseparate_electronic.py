"""Regression tests for electronic-music source analysis and model discovery."""
from __future__ import annotations

from dataclasses import replace

from mseparate.planner import _model_specialization, _stem_score, build_plan
from mseparate.registry import bundled_registry_path, eligible_models, load_registry, parse_manifest
from mseparate.source_analysis import GENRES, SOURCES
from mseparate.types import AnalysisReport, AudioMetadata


def test_clap_catalog_distinguishes_electronic_source_families():
    assert {"synth_lead", "synth_pad", "synth_arp", "synth_bass", "synth_fx"} <= set(SOURCES)
    assert {"electronic_drums", "kick", "snare", "hi_hat", "breakbeats", "samples"} <= set(SOURCES)
    assert {"techno", "house", "trance", "drum_and_bass", "dubstep"} <= set(GENRES)


def test_genre_predictions_do_not_drive_stem_routing():
    score, reason = _stem_score("synth_pad", {}, [
        {"kind": "genre", "source": "techno", "score": 0.99},
    ])
    assert score == 0.5
    assert reason == "no stem-specific acoustic heuristic"


def test_synth_pad_uses_its_specific_source_prediction():
    score, reason = _stem_score("synth_pad", {}, [
        {"kind": "source", "source": "synth_pad", "score": 0.91},
        {"kind": "source", "source": "synth_bass", "score": 0.22},
    ])
    assert score == 1.0
    assert "synth_pad" in reason


def test_synth_specialist_category_matches_underscored_stem_names():
    manifest = parse_manifest({
        "model_id": "synth-specialist",
        "name": "Synth specialist",
        "architecture": "test",
        "backend": "test",
        "source_url": "https://example.invalid/model",
        "license": "MIT",
        "license_status": "verified",
        "status": "available",
        "category": "synth-specialist",
        "assets": [{
            "role": "checkpoint",
            "url": "https://example.invalid/model.ckpt",
            "relative_path": "model.ckpt",
            "sha256": "a" * 64,
        }],
        "native_stems": ["synth_pad"],
    })
    assert _model_specialization("synth_pad", manifest) == 1


def test_mega53_specialists_keep_independent_technical_gates():
    registry = load_registry(bundled_registry_path())
    eligible = {model.model_id for model in eligible_models(registry)}
    assert "bs-roformer-sw-6stem-candidate" not in eligible
    assert "zfturbo-bs-roformer-53stem-candidate" in eligible
    assert registry["zfturbo-bs-roformer-53stem-candidate"].auto_route is False
    assert "mvsep-synth-specialist-candidate" not in eligible
    assert registry["mvsep-synth-specialist-candidate"].license_status == "review_required"
    for stem in ("synth", "keys", "bass", "kick", "drums", "organ", "digital-piano", "percussion", "hh"):
        model = registry[f"mvsep-mega53-{stem}-specialist-candidate"]
        assert model.license_status == "verified"
        assert model.license_review["status"] == "accepted_by_maintainer"
        assert model.license_review["training_data_status"] == "unknown"
        assert model.runtime_validation == "verified"
        assert {asset.role for asset in model.assets} == {"checkpoint", "config"}
        assert model.model_id in eligible
        assert model.status == "available"
        assert all(asset.sha256 and len(asset.sha256) == 64 for asset in model.assets)
        assert next(asset for asset in model.assets if asset.role == "checkpoint").size_bytes is not None


def test_planner_prefers_synth_specialist_for_synth_pad_output():
    model = parse_manifest({
        "model_id": "synth-pad-specialist",
        "name": "Synth pad specialist",
        "architecture": "test",
        "backend": "test",
        "source_url": "https://example.invalid/model",
        "license": "MIT",
        "license_status": "verified",
        "status": "available",
        "category": "synth-specialist",
        "sample_rate": 44100,
        "channels": "stereo",
        "input_formats": ["wav"],
        "assets": [{
            "role": "checkpoint",
            "url": "https://example.invalid/model.ckpt",
            "relative_path": "model.ckpt",
            "sha256": "b" * 64,
        }],
        "native_stems": ["synth_pad"],
    })
    analysis = AnalysisReport(
        schema_version=2,
        audio=AudioMetadata("electronic.wav", 1.0, 44100, 2, 44100, "PCM_16"),
        predictions=[
            {"kind": "source", "source": "synth_pad", "score": 0.8},
            {"kind": "genre", "source": "techno", "score": 0.99},
        ],
    )
    plan = build_plan(analysis=analysis, registry={model.model_id: model})
    assert len(plan.tasks) == 1
    assert plan.tasks[0].model_id == model.model_id
    assert plan.tasks[0].target_stems == ("synth_pad",)
    assert "manifest-declared specialist" in plan.tasks[0].rationale


def test_synth_and_keys_are_plannable_after_maintainer_accepts_mit_grant():
    registry = load_registry(bundled_registry_path())
    synth_id = "mvsep-mega53-synth-specialist-candidate"
    keys_id = "mvsep-mega53-keys-specialist-candidate"

    assert registry[synth_id].runtime_validation == "verified"
    assert registry[keys_id].runtime_validation == "verified"
    assert {m.model_id for m in eligible_models(registry)} >= {synth_id, keys_id}
    approved = {
        model_id: replace(model, status="available", license_status="verified", license_review={"status": "accepted_by_maintainer"})
        for model_id, model in registry.items()
        if model_id in {synth_id, keys_id}
    }
    assert {m.model_id for m in eligible_models(approved)} == {synth_id, keys_id}
    analysis = AnalysisReport(
        schema_version=2,
        audio=AudioMetadata("electronic.wav", 20.0, 44100, 2, 882000, "PCM_16"),
        predictions=[
            {"kind": "source", "source": "synth_lead", "score": 0.92},
            {"kind": "source", "source": "synth_keys", "score": 0.88},
        ],
    )
    plan = build_plan(analysis=analysis, registry=approved)
    routes = {task.model_id: task.target_stems for task in plan.tasks}
    assert routes[synth_id] == ("synth",)
    assert routes[keys_id] == ("keys",)


def test_non_vocal_bs_roformer_specialist_without_runtime_validation_is_not_eligible():
    model = parse_manifest({
        "model_id": "unverified-synth", "name": "Unverified synth",
        "architecture": "BS-RoFormer", "backend": "bs-roformer-pymss",
        "source_url": "https://example.invalid/model", "license": "MIT",
        "license_status": "verified", "status": "available",
        "category": "synth-specialist", "native_stems": ["synth"],
        "assets": [{"role": "checkpoint", "url": "https://example.invalid/model",
                    "relative_path": "model.ckpt", "sha256": "a" * 64}],
    })
    assert model.runtime_validation == "unverified"
    assert eligible_models({model.model_id: model}) == []


def test_registry_rejects_unknown_runtime_validation_state():
    import pytest
    from mseparate.registry import RegistryError

    with pytest.raises(RegistryError, match="runtime_validation"):
        parse_manifest({
            "model_id": "bad-runtime-state", "name": "Bad",
            "architecture": "test", "backend": "test",
            "source_url": "https://example.invalid/model",
            "runtime_validation": "probably",
        })
