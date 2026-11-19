"""Edge-case regressions for hybrid electronic/acoustic routing."""
from __future__ import annotations

from mseparate.planner import build_plan
from mseparate.registry import bundled_registry_path, load_registry
from mseparate.types import AnalysisReport, AudioMetadata


def _analysis(predictions):
    return AnalysisReport(
        schema_version=2,
        audio=AudioMetadata("hybrid-edge-case.wav", 20.0, 44100, 2, 882000, "PCM_16"),
        predictions=predictions,
    )


def test_real_violin_and_cello_survive_competing_synth_string_evidence():
    registry = load_registry(bundled_registry_path())
    plan = build_plan(
        analysis=_analysis([
            {"kind": "source", "source": "synth_strings", "score": 7.5},
            {"kind": "source", "source": "violin", "score": 7.8},
            {"kind": "source", "source": "cello", "score": 7.7},
            {"kind": "source", "source": "synth_pad", "score": 7.4},
            {"kind": "source", "source": "electronic_drums", "score": 7.3},
            {"kind": "genre", "source": "electro_industrial", "score": 0.92},
            {"kind": "genre", "source": "orchestral", "score": 0.85},
        ]),
        registry=registry,
    )
    routes = {task.model_id: task.target_stems for task in plan.tasks}
    assert plan.style_profile["family"] == "mixed_or_uncertain"
    assert routes["zfturbo-bs-roformer-53stem-candidate"] == ("cello", "violin")
    assert routes["mvsep-mega53-synth-specialist-candidate"] == ("synth",)


def test_real_trumpet_survives_synth_brass_when_acoustic_evidence_is_stronger():
    registry = load_registry(bundled_registry_path())
    plan = build_plan(
        analysis=_analysis([
            {"kind": "source", "source": "synth_brass", "score": 7.2},
            {"kind": "source", "source": "trumpet", "score": 7.6},
            {"kind": "source", "source": "synth_pad", "score": 7.5},
            {"kind": "source", "source": "electronic_drums", "score": 7.4},
            {"kind": "genre", "source": "cybercore", "score": 0.94},
        ]),
        registry=registry,
    )
    routes = {task.model_id: task.target_stems for task in plan.tasks}
    assert routes["zfturbo-bs-roformer-53stem-candidate"] == ("trumpet",)
    assert routes["mvsep-mega53-synth-specialist-candidate"] == ("synth",)


def test_quiet_acoustic_cello_survives_dense_electronic_percussion_and_textures():
    registry = load_registry(bundled_registry_path())
    plan = build_plan(
        analysis=_analysis([
            {"kind": "source", "source": "electronic_drums", "score": 8.0},
            {"kind": "source", "source": "industrial_noise", "score": 7.95},
            {"kind": "source", "source": "granular_textures", "score": 7.90},
            {"kind": "source", "source": "glitch_fx", "score": 7.85},
            {"kind": "source", "source": "synth_pad", "score": 7.80},
            {"kind": "source", "source": "synth_bass", "score": 7.70},
            {"kind": "source", "source": "cello", "score": 7.45},
            {"kind": "genre", "source": "powernoise", "score": 0.95},
        ]),
        registry=registry,
    )
    routes = {task.model_id: task.target_stems for task in plan.tasks}
    assert plan.style_profile["family"] == "electronic"
    assert routes["mvsep-mega53-drums-specialist-candidate"] == ("drums",)
    assert routes["zfturbo-bs-roformer-53stem-candidate"] == ("cello",)
