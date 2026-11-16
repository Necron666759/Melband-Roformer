"""Regression tests for electronic-music source analysis and model discovery."""
from __future__ import annotations

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


def test_unverified_electronic_model_candidates_are_not_executable():
    registry = load_registry(bundled_registry_path())
    eligible = {model.model_id for model in eligible_models(registry)}
    assert "bs-roformer-sw-6stem-candidate" not in eligible
    assert "zfturbo-bs-roformer-53stem-candidate" not in eligible
    assert "mvsep-synth-specialist-candidate" not in eligible
    assert registry["mvsep-synth-specialist-candidate"].license_status == "review_required"
    for stem in ("synth", "keys", "bass", "kick", "drums", "organ", "digital-piano", "percussion", "hh"):
        model = registry[f"mvsep-mega53-{stem}-specialist-candidate"]
        assert model not in eligible
        assert model.license_status == "review_required"
        assert {asset.role for asset in model.assets} == {"checkpoint", "config"}
        if stem in ("synth", "keys"):
            assert all(asset.sha256 and len(asset.sha256) == 64 for asset in model.assets)
            assert all(asset.size_bytes is not None for asset in model.assets if asset.role == "checkpoint")
        else:
            assert all(asset.sha256 is None for asset in model.assets)


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
