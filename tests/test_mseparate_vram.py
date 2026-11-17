"""Deterministic tests for GPU-memory admission and low-VRAM profiles."""
from __future__ import annotations

from dataclasses import replace

from mseparate.registry import parse_manifest
from mseparate.vram import filter_registry_for_device


def _model():
    return parse_manifest({
        "model_id": "mega-53", "name": "Mega 53", "architecture": "BS-RoFormer",
        "backend": "bs-roformer-pymss", "source_url": "https://example.invalid/release",
        "license": "MIT", "license_status": "verified", "status": "available",
        "category": "mega-multi-stem", "cpu_inference": True, "min_vram_gb": 16, "min_free_vram_gb": 14,
        "assets": [
            {"role": "checkpoint", "url": "https://example.invalid/model.ckpt",
             "relative_path": "model.ckpt", "sha256": "a" * 64},
            {"role": "config", "url": "https://example.invalid/model.yaml",
             "relative_path": "model.yaml", "sha256": "b" * 64},
        ],
        "native_stems": ["bass"], "target_stems": ["bass"],
        "low_vram_profile": {
            "status": "unverified", "min_total_vram_gb": 8, "min_free_vram_gb": 7,
            "inference_params": {"batch_size": 1, "chunk_size": 441000},
        },
    })


def test_gpu_memory_requirement_does_not_disable_verified_cpu_runtime():
    model = _model()
    registry, notes = filter_registry_for_device(
        {"mega-53": model}, "cpu",
        memory_probe=lambda: (_ for _ in ()).throw(
            AssertionError("CPU admission must not probe CUDA")
        ),
    )
    assert registry["mega-53"] is model
    assert notes == ()


def test_cpu_inference_is_not_assumed_without_manifest_capability():
    model = replace(_model(), cpu_inference=False)
    registry, notes = filter_registry_for_device({"mega-53": model}, "cpu")
    assert registry == {}
    assert "No CPU fallback" in notes[0]


def test_gpu_gated_model_is_excluded_when_total_vram_is_too_small():
    registry, notes = filter_registry_for_device(
        {"mega-53": _model()}, "cuda",
        memory_probe=lambda: (7 * 1024**3, 8 * 1024**3),
    )
    assert registry == {}
    assert "16 GiB total" in notes[0]


def test_gpu_gated_model_is_excluded_when_free_vram_is_too_small():
    registry, notes = filter_registry_for_device(
        {"mega-53": _model()}, "cuda",
        memory_probe=lambda: (13 * 1024**3, 24 * 1024**3),
    )
    assert registry == {}
    assert "14 GiB free" in notes[0]


def test_gpu_gated_model_is_admitted_only_when_total_and_free_vram_pass():
    model = _model()
    registry, notes = filter_registry_for_device(
        {"mega-53": model}, "cuda",
        memory_probe=lambda: (15 * 1024**3, 24 * 1024**3),
    )
    assert registry["mega-53"] is model
    assert notes == ()


def test_unverified_low_vram_profile_never_bypasses_normal_gate():
    registry, notes = filter_registry_for_device(
        {"mega-53": _model()}, "cuda",
        memory_probe=lambda: (7.5 * 1024**3, 8 * 1024**3),
    )
    assert registry == {}
    assert "skipped before download/inference" in notes[0]


def test_verified_low_vram_profile_can_admit_with_warning_and_overrides():
    model = _model()
    profile = dict(model.low_vram_profile)
    profile["status"] = "verified"
    model = replace(model, low_vram_profile=profile)
    registry, notes = filter_registry_for_device(
        {"mega-53": model}, "cuda",
        memory_probe=lambda: (7.5 * 1024**3, 8 * 1024**3),
    )
    assert registry["mega-53"].inference_params == {
        "batch_size": 1, "chunk_size": 441000,
    }
    assert "verified low-VRAM profile" in notes[0]


def test_vram_probe_failure_fails_closed():
    registry, notes = filter_registry_for_device(
        {"mega-53": _model()}, "cuda",
        memory_probe=lambda: (_ for _ in ()).throw(RuntimeError("probe unavailable")),
    )
    assert registry == {}
    assert "probe failed" in notes[0]


def test_verified_low_vram_profile_does_not_override_normal_profile_when_it_fits():
    model = _model()
    profile = dict(model.low_vram_profile)
    profile["status"] = "verified"
    model = replace(model, low_vram_profile=profile)
    registry, notes = filter_registry_for_device(
        {"mega-53": model}, "cuda",
        memory_probe=lambda: (15 * 1024**3, 24 * 1024**3),
    )
    assert registry["mega-53"] is model
    assert registry["mega-53"].inference_params == model.inference_params
    assert notes == ()
