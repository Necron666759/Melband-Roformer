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



def _analysis_with_predictions(predictions):
    return AnalysisReport(
        schema_version=2,
        audio=AudioMetadata("style-test.wav", 20.0, 44100, 2, 882000, "PCM_16"),
        predictions=predictions,
    )


def test_style_classifier_identifies_electro_industrial_family():
    from mseparate.source_analysis import classify_style

    profile = classify_style([
        {"kind": "genre", "source": "electro_industrial", "score": 0.91},
        {"kind": "genre", "source": "industrial_electronic", "score": 0.72},
        {"kind": "genre", "source": "classical", "score": 0.21},
    ])
    assert profile["label"] == "electro_industrial"
    assert profile["family"] == "electronic"
    assert profile["status"] == "provisional"


def test_electronic_style_routes_synth_and_keys_without_acoustic_53stem():
    registry = load_registry(bundled_registry_path())
    analysis = _analysis_with_predictions([
        {"kind": "source", "source": "synth_pad", "score": 7.0},
        {"kind": "source", "source": "synth_keys", "score": 6.8},
        {"kind": "source", "source": "electronic_drums", "score": 6.6},
        {"kind": "source", "source": "kick", "score": 6.5},
        {"kind": "source", "source": "synth_brass", "score": 6.4},
        {"kind": "source", "source": "violin", "score": 5.1},
        {"kind": "genre", "source": "electro_industrial", "score": 0.91},
        {"kind": "genre", "source": "industrial_electronic", "score": 0.72},
    ])
    plan = build_plan(analysis=analysis, registry=registry)
    routes = {task.model_id: task.target_stems for task in plan.tasks}
    assert routes["mvsep-mega53-synth-specialist-candidate"] == ("synth",)
    assert routes["mvsep-mega53-keys-specialist-candidate"] == ("keys",)
    assert "mvsep-mega53-kick-specialist-candidate" in routes
    assert "mvsep-mega53-drums-specialist-candidate" in routes
    assert "zfturbo-bs-roformer-53stem-candidate" not in routes
    assert plan.style_profile["family"] == "electronic"


def test_detected_acoustic_instruments_use_only_selected_mega53_heads():
    registry = load_registry(bundled_registry_path())
    analysis = _analysis_with_predictions([
        {"kind": "source", "source": "violin", "score": 7.2},
        {"kind": "source", "source": "cello", "score": 7.0},
        {"kind": "source", "source": "harp", "score": 6.8},
        {"kind": "source", "source": "flute", "score": 6.7},
        {"kind": "source", "source": "trombone", "score": 6.5},
        {"kind": "source", "source": "synth_strings", "score": 5.8},
        {"kind": "genre", "source": "orchestral", "score": 0.94},
        {"kind": "genre", "source": "classical", "score": 0.79},
    ])
    plan = build_plan(analysis=analysis, registry=registry)
    routes = {task.model_id: task.target_stems for task in plan.tasks}
    model_id = "zfturbo-bs-roformer-53stem-candidate"
    assert routes[model_id] == ("cello", "flute", "harp", "trombone", "violin")
    assert len([task for task in plan.tasks if task.model_id == model_id]) == 1
    assert plan.style_profile["family"] == "orchestral"


def test_synthesized_strings_do_not_trigger_acoustic_string_heads():
    from mseparate.planner import _detected_acoustic_targets

    targets = _detected_acoustic_targets([
        {"kind": "source", "source": "synth_strings", "score": 7.0},
        {"kind": "source", "source": "violin", "score": 6.95},
        {"kind": "source", "source": "synthesizer", "score": 6.8},
    ], {"family": "electronic"})
    assert "violin" not in targets
    assert "strings" not in targets


def test_unknown_style_does_not_auto_run_full_53stem_model():
    registry = load_registry(bundled_registry_path())
    analysis = _analysis_with_predictions([
        {"kind": "source", "source": "synth_pad", "score": 7.0},
        {"kind": "source", "source": "synth_keys", "score": 6.8},
    ])
    plan = build_plan(analysis=analysis, registry=registry)
    assert "zfturbo-bs-roformer-53stem-candidate" not in {
        task.model_id for task in plan.tasks
    }



def test_digital_piano_uses_one_specialist_without_duplicate_53stem_heads():
    registry = load_registry(bundled_registry_path())
    analysis = _analysis_with_predictions([
        {"kind": "source", "source": "digital_piano", "score": 7.0},
        {"kind": "source", "source": "synth_keys", "score": 6.2},
        {"kind": "genre", "source": "electro_industrial", "score": 0.9},
        {"kind": "genre", "source": "industrial_electronic", "score": 0.7},
    ])
    plan = build_plan(analysis=analysis, registry=registry)
    routes = {task.model_id: task.target_stems for task in plan.tasks}
    assert routes["mvsep-mega53-digital-piano-specialist-candidate"] == ("digital-piano",)
    assert "mvsep-mega53-keys-specialist-candidate" not in routes
    assert "mvsep-mega53-synth-specialist-candidate" not in routes
    assert "zfturbo-bs-roformer-53stem-candidate" not in routes



def test_synth_keys_only_does_not_also_schedule_generic_synth_specialist():
    registry = load_registry(bundled_registry_path())
    analysis = _analysis_with_predictions([
        {"kind": "source", "source": "synth_keys", "score": 7.0},
        {"kind": "source", "source": "synthesizer", "score": 6.1},
        {"kind": "genre", "source": "electro_industrial", "score": 0.9},
        {"kind": "genre", "source": "industrial_electronic", "score": 0.7},
    ])
    routes = {task.model_id: task.target_stems for task in build_plan(
        analysis=analysis, registry=registry,
    ).tasks}
    assert routes["mvsep-mega53-keys-specialist-candidate"] == ("keys",)
    assert "mvsep-mega53-synth-specialist-candidate" not in routes


def test_live_violin_evidence_suppresses_ambiguous_synth_strings_model():
    registry = load_registry(bundled_registry_path())
    analysis = _analysis_with_predictions([
        {"kind": "source", "source": "synth_strings", "score": 7.0},
        {"kind": "source", "source": "violin", "score": 6.95},
        {"kind": "source", "source": "synthesizer", "score": 6.8},
        {"kind": "genre", "source": "orchestral", "score": 0.9},
        {"kind": "genre", "source": "classical", "score": 0.75},
    ])
    routes = {task.model_id: task.target_stems for task in build_plan(
        analysis=analysis, registry=registry,
    ).tasks}
    assert "mvsep-mega53-synth-specialist-candidate" not in routes
    assert "zfturbo-bs-roformer-53stem-candidate" not in routes



def test_requested_extreme_electronic_genres_are_independently_classified():
    from mseparate.source_analysis import classify_style

    expected = {
        "breakcore", "cybercore", "hardcore", "industrial_hardcore",
        "powernoise", "dark_ambient",
    }
    assert expected <= set(GENRES)
    for genre in sorted(expected):
        profile = classify_style([
            {"kind": "genre", "source": genre, "score": 0.91},
        ])
        assert profile["label"] == genre
        assert profile["family"] == "electronic"
        assert profile["status"] == "provisional"


def test_electronic_genre_context_alone_never_schedules_a_separator():
    from mseparate.source_analysis import classify_style

    for genre in (
        "breakcore", "cybercore", "hardcore",
        "industrial_hardcore", "powernoise", "dark_ambient",
    ):
        predictions = [{"kind": "genre", "source": genre, "score": 0.99}]
        profile = classify_style(predictions)
        score, reason = _stem_score("synth_pad", {}, predictions)
        assert profile["family"] == "electronic"
        assert score == 0.5
        assert reason == "no stem-specific acoustic heuristic"



def test_electronic_texture_taxonomy_distinguishes_noise_drone_granular_and_samples():
    from mseparate.source_analysis import CLAP_CANDIDATES

    expected = {
        "breakbeat_samples", "industrial_noise", "noise_pulses",
        "granular_textures", "drone_layers", "glitch_fx",
        "synth_sequence", "synth_pluck", "synth_texture",
        "distorted_synth_bass", "mechanical_percussion",
        "live_drum_kit", "acoustic_percussion",
    }
    assert expected <= set(SOURCES)
    candidate_sources = {
        name for kind, name, _description in CLAP_CANDIDATES if kind == "source"
    }
    assert expected <= candidate_sources
    descriptions = [SOURCES[name] for name in expected]
    assert len(descriptions) == len(set(descriptions))


def test_texture_only_electronic_evidence_does_not_trigger_instrument_specialists():
    registry = load_registry(bundled_registry_path())
    analysis = _analysis_with_predictions([
        {"kind": "source", "source": "industrial_noise", "score": 7.2},
        {"kind": "source", "source": "granular_textures", "score": 7.0},
        {"kind": "source", "source": "drone_layers", "score": 6.9},
        {"kind": "source", "source": "glitch_fx", "score": 6.8},
        {"kind": "source", "source": "mechanical_percussion", "score": 6.7},
        {"kind": "genre", "source": "powernoise", "score": 0.95},
    ])
    plan = build_plan(analysis=analysis, registry=registry)
    scheduled = {task.model_id for task in plan.tasks}
    assert not any(model_id.startswith("mvsep-mega53-") for model_id in scheduled)
    assert "zfturbo-bs-roformer-53stem-candidate" not in scheduled


def test_sequenced_synth_and_distorted_synth_bass_use_matching_specialists():
    registry = load_registry(bundled_registry_path())
    synth_plan = build_plan(
        analysis=_analysis_with_predictions([
            {"kind": "source", "source": "synth_sequence", "score": 7.2},
            {"kind": "source", "source": "synth_texture", "score": 6.8},
            {"kind": "genre", "source": "cybercore", "score": 0.94},
        ]),
        registry=registry,
    )
    synth_routes = {task.model_id: task.target_stems for task in synth_plan.tasks}
    assert synth_routes["mvsep-mega53-synth-specialist-candidate"] == ("synth",)

    bass_plan = build_plan(
        analysis=_analysis_with_predictions([
            {"kind": "source", "source": "distorted_synth_bass", "score": 7.3},
            {"kind": "source", "source": "synth_bass", "score": 6.9},
            {"kind": "genre", "source": "industrial_hardcore", "score": 0.95},
        ]),
        registry=registry,
    )
    bass_routes = {task.model_id: task.target_stems for task in bass_plan.tasks}
    assert bass_routes["mvsep-mega53-bass-specialist-candidate"] == ("bass",)

def test_live_drums_breakbeat_samples_and_acoustic_percussion_use_matching_specialists():
    registry = load_registry(bundled_registry_path())

    cases = (
        (
            "live drum kit",
            "live_drum_kit",
            "mvsep-mega53-drums-specialist-candidate",
            ("drums",),
        ),
        (
            "sampled breakbeats",
            "breakbeat_samples",
            "mvsep-mega53-drums-specialist-candidate",
            ("drums",),
        ),
        (
            "acoustic hand percussion",
            "acoustic_percussion",
            "mvsep-mega53-percussion-specialist-candidate",
            ("percussion",),
        ),
    )
    for label, source, expected_model, expected_stems in cases:
        plan = build_plan(
            analysis=_analysis_with_predictions([
                {"kind": "source", "source": source, "score": 7.2},
            ]),
            registry=registry,
        )
        routes = {task.model_id: task.target_stems for task in plan.tasks}
        assert routes == {expected_model: expected_stems}, label
        assert len(plan.tasks) == 1, label
        assert "zfturbo-bs-roformer-53stem-candidate" not in routes, label


def test_mechanical_percussion_alone_does_not_trigger_drum_specialists():
    registry = load_registry(bundled_registry_path())
    plan = build_plan(
        analysis=_analysis_with_predictions([
            {"kind": "source", "source": "mechanical_percussion", "score": 7.2},
            {"kind": "source", "source": "industrial_noise", "score": 6.9},
        ]),
        registry=registry,
    )
    scheduled = {task.model_id for task in plan.tasks}
    assert not any(model_id.startswith("mvsep-mega53-") for model_id in scheduled)
    assert "zfturbo-bs-roformer-53stem-candidate" not in scheduled



def test_drum_component_taxonomy_covers_acoustic_and_synthetic_sources():
    assert {
        "acoustic_kick", "electronic_kick", "synth_kick",
        "acoustic_snare", "electronic_snare", "acoustic_toms",
        "acoustic_hi_hat", "electronic_hi_hat",
        "cymbals", "synth_cymbal", "toms", "electronic_toms",
    } <= set(SOURCES)


def test_acoustic_and_synthetic_kicks_route_only_to_declared_kick_output():
    registry = load_registry(bundled_registry_path())
    for source in ("acoustic_kick", "electronic_kick", "synth_kick"):
        plan = build_plan(
            analysis=_analysis_with_predictions([
                {"kind": "source", "source": source, "score": 7.4},
            ]),
            registry=registry,
        )
        routes = {task.model_id: task.target_stems for task in plan.tasks}
        assert routes == {"mvsep-mega53-kick-specialist-candidate": ("kick",)}, source


def test_acoustic_and_synthetic_hats_route_only_to_declared_hh_output():
    registry = load_registry(bundled_registry_path())
    for source in ("acoustic_hi_hat", "electronic_hi_hat"):
        plan = build_plan(
            analysis=_analysis_with_predictions([
                {"kind": "source", "source": source, "score": 7.4},
            ]),
            registry=registry,
        )
        routes = {task.model_id: task.target_stems for task in plan.tasks}
        assert routes == {"mvsep-mega53-hh-specialist-candidate": ("hh",)}, source


def test_acoustic_snare_and_toms_use_verified_53stem_heads_only():
    registry = load_registry(bundled_registry_path())
    manifest = registry["zfturbo-bs-roformer-53stem-candidate"]
    assert {"snare", "toms"} <= set(manifest.native_stems)
    assert manifest.runtime_validation == "verified"
    cases = (
        ("acoustic_snare", "snare"),
        ("toms", "toms"),
        ("acoustic_toms", "toms"),
    )
    for source, expected_stem in cases:
        plan = build_plan(
            analysis=_analysis_with_predictions([
                {"kind": "source", "source": source, "score": 7.4},
            ]),
            registry=registry,
        )
        routes = {task.model_id: task.target_stems for task in plan.tasks}
        assert routes == {"zfturbo-bs-roformer-53stem-candidate": (expected_stem,)}, source


def test_electronic_snare_and_toms_do_not_trigger_acoustic_53stem_heads():
    registry = load_registry(bundled_registry_path())
    for source in ("electronic_snare", "electronic_toms"):
        plan = build_plan(
            analysis=_analysis_with_predictions([
                {"kind": "source", "source": source, "score": 7.4},
            ]),
            registry=registry,
        )
        routes = {task.model_id: task.target_stems for task in plan.tasks}
        assert routes == {"mvsep-mega53-drums-specialist-candidate": ("drums",)}, source
        assert "zfturbo-bs-roformer-53stem-candidate" not in routes


def test_synthetic_snare_counterevidence_blocks_acoustic_snare_head():
    from mseparate.planner import _detected_acoustic_targets

    targets = _detected_acoustic_targets([
        {"kind": "source", "source": "electronic_snare", "score": 7.4},
        {"kind": "source", "source": "acoustic_snare", "score": 7.2},
    ], {"family": "unknown"})
    assert "snare" not in targets


def test_acoustic_snare_tom_routing_suppresses_duplicate_live_drums_route():
    registry = load_registry(bundled_registry_path())
    plan = build_plan(
        analysis=_analysis_with_predictions([
            {"kind": "source", "source": "live_drum_kit", "score": 7.5},
            {"kind": "source", "source": "acoustic_snare", "score": 7.4},
        ]),
        registry=registry,
    )
    routes = {task.model_id: task.target_stems for task in plan.tasks}
    assert routes == {"zfturbo-bs-roformer-53stem-candidate": ("snare",)}


def test_acoustic_snare_can_coexist_with_separate_electronic_drums():
    registry = load_registry(bundled_registry_path())
    plan = build_plan(
        analysis=_analysis_with_predictions([
            {"kind": "source", "source": "electronic_drums", "score": 7.5},
            {"kind": "source", "source": "acoustic_snare", "score": 7.3},
        ]),
        registry=registry,
    )
    routes = {task.model_id: task.target_stems for task in plan.tasks}
    assert routes == {
        "mvsep-mega53-drums-specialist-candidate": ("drums",),
        "zfturbo-bs-roformer-53stem-candidate": ("snare",),
    }


def test_cymbal_components_stay_on_combined_drums_until_a_licensed_head_exists():
    registry = load_registry(bundled_registry_path())
    for source in ("cymbals", "synth_cymbal"):
        plan = build_plan(
            analysis=_analysis_with_predictions([
                {"kind": "source", "source": source, "score": 7.4},
            ]),
            registry=registry,
        )
        routes = {task.model_id: task.target_stems for task in plan.tasks}
        assert routes == {"mvsep-mega53-drums-specialist-candidate": ("drums",)}, source


def test_distinct_drum_components_can_select_distinct_verified_outputs():
    registry = load_registry(bundled_registry_path())
    plan = build_plan(
        analysis=_analysis_with_predictions([
            {"kind": "source", "source": "synth_kick", "score": 7.5},
            {"kind": "source", "source": "electronic_hi_hat", "score": 7.2},
            {"kind": "source", "source": "electronic_snare", "score": 7.0},
        ]),
        registry=registry,
    )
    routes = {task.model_id: task.target_stems for task in plan.tasks}
    assert routes == {
        "mvsep-mega53-drums-specialist-candidate": ("drums",),
        "mvsep-mega53-hh-specialist-candidate": ("hh",),
        "mvsep-mega53-kick-specialist-candidate": ("kick",),
    }
    assert "zfturbo-bs-roformer-53stem-candidate" not in routes


def test_mixed_electronic_orchestral_genre_profile_is_explicitly_uncertain():
    from mseparate.source_analysis import classify_style

    profile = classify_style([
        {"kind": "genre", "source": "electro_industrial", "score": 0.91},
        {"kind": "genre", "source": "orchestral", "score": 0.86},
    ])
    assert profile["label"] == "electro_industrial"
    assert profile["family"] == "mixed_or_uncertain"
    assert profile["status"] == "ambiguous"
    assert profile["alternatives"][1]["label"] == "orchestral"


def test_hybrid_electronic_and_acoustic_sources_route_to_both_specialist_families():
    registry = load_registry(bundled_registry_path())
    analysis = _analysis_with_predictions([
        {"kind": "source", "source": "synth_pad", "score": 7.5},
        {"kind": "source", "source": "violin", "score": 7.3},
        {"kind": "source", "source": "electronic_drums", "score": 7.2},
        {"kind": "source", "source": "synth_bass", "score": 7.0},
        {"kind": "genre", "source": "electro_industrial", "score": 0.91},
        {"kind": "genre", "source": "orchestral", "score": 0.86},
    ])
    plan = build_plan(analysis=analysis, registry=registry)
    routes = {task.model_id: task.target_stems for task in plan.tasks}

    assert plan.style_profile["family"] == "mixed_or_uncertain"
    assert routes["mvsep-mega53-synth-specialist-candidate"] == ("synth",)
    assert routes["mvsep-mega53-drums-specialist-candidate"] == ("drums",)
    assert routes["zfturbo-bs-roformer-53stem-candidate"] == ("violin",)


def test_synth_only_electronic_mix_does_not_promote_ambiguous_acoustic_instrument():
    registry = load_registry(bundled_registry_path())
    analysis = _analysis_with_predictions([
        {"kind": "source", "source": "synth_strings", "score": 7.5},
        {"kind": "source", "source": "synth_pad", "score": 7.3},
        {"kind": "source", "source": "synth_bass", "score": 7.1},
        {"kind": "source", "source": "violin", "score": 6.9},
        {"kind": "genre", "source": "cybercore", "score": 0.94},
    ])
    routes = {task.model_id: task.target_stems for task in build_plan(
        analysis=analysis, registry=registry,
    ).tasks}
    assert "mvsep-mega53-synth-specialist-candidate" in routes
    assert "zfturbo-bs-roformer-53stem-candidate" not in routes


def test_real_instrument_survives_dense_industrial_texture_in_hybrid_profile():
    registry = load_registry(bundled_registry_path())
    analysis = _analysis_with_predictions([
        {"kind": "source", "source": "synth_pad", "score": 7.6},
        {"kind": "source", "source": "industrial_noise", "score": 7.5},
        {"kind": "source", "source": "granular_textures", "score": 7.4},
        {"kind": "source", "source": "glitch_fx", "score": 7.3},
        {"kind": "source", "source": "synth_bass", "score": 7.2},
        {"kind": "source", "source": "violin", "score": 7.1},
        {"kind": "source", "source": "synth_brass", "score": 6.7},
        {"kind": "source", "source": "trombone", "score": 6.9},
        {"kind": "genre", "source": "electro_industrial", "score": 0.91},
        {"kind": "genre", "source": "orchestral", "score": 0.86},
    ])
    plan = build_plan(analysis=analysis, registry=registry)
    routes = {task.model_id: task.target_stems for task in plan.tasks}

    assert plan.style_profile["family"] == "mixed_or_uncertain"
    assert routes["mvsep-mega53-synth-specialist-candidate"] == ("synth",)
    assert routes["zfturbo-bs-roformer-53stem-candidate"] == ("trombone", "violin")


def test_electronic_style_keeps_acoustic_instrument_within_conservative_margin():
    registry = load_registry(bundled_registry_path())
    plan = build_plan(
        analysis=_analysis_with_predictions([
            {"kind": "source", "source": "synth_pad", "score": 7.5},
            {"kind": "source", "source": "industrial_noise", "score": 7.45},
            {"kind": "source", "source": "granular_textures", "score": 7.4},
            {"kind": "source", "source": "glitch_fx", "score": 7.35},
            {"kind": "source", "source": "synth_bass", "score": 7.3},
            {
                "kind": "source", "source": "violin", "score": 7.0,
                "window_top_5_fraction": 0.25, "window_top_12_fraction": 0.75,
                "window_best_source_rank": 4,
            },
            {"kind": "genre", "source": "cybercore", "score": 0.94},
        ]),
        registry=registry,
    )
    routes = {task.model_id: task.target_stems for task in plan.tasks}

    assert plan.style_profile["family"] == "electronic"
    assert routes["mvsep-mega53-synth-specialist-candidate"] == ("synth",)
    assert routes["zfturbo-bs-roformer-53stem-candidate"] == ("violin",)


def test_synthetic_brass_counterevidence_blocks_false_trombone_route():
    from mseparate.planner import _detected_acoustic_targets

    targets = _detected_acoustic_targets([
        {"kind": "source", "source": "synth_brass", "score": 7.4},
        {"kind": "source", "source": "trombone", "score": 7.35},
        {"kind": "source", "source": "synth_pad", "score": 7.2},
    ], {"family": "mixed_or_uncertain"})

    assert "trombone" not in targets
    assert "brass" not in targets


def test_synthetic_flute_counterevidence_blocks_false_flute_route():
    from mseparate.planner import _detected_acoustic_targets

    targets = _detected_acoustic_targets([
        {"kind": "source", "source": "synth_flute", "score": 7.4},
        {"kind": "source", "source": "flute", "score": 7.35},
        {"kind": "source", "source": "synth_pad", "score": 7.2},
    ], {"family": "mixed_or_uncertain"})

    assert "flute" not in targets


def test_electronic_hybrid_keeps_real_violin_when_synth_strings_also_present():
    from mseparate.planner import _detected_acoustic_targets

    predictions = [
        {"kind": "source", "source": "synth_strings", "score": 7.0},
        {
            "kind": "source", "source": "violin", "score": 6.8,
            "window_top_5_fraction": 0.5, "window_top_12_fraction": 1.0,
            "window_best_source_rank": 2,
        },
        {"kind": "source", "source": "synth_pad", "score": 6.5},
        {"kind": "genre", "source": "electro_industrial", "score": 0.96},
    ]

    targets = _detected_acoustic_targets(
        [item for item in predictions if item["kind"] == "source"],
        {"family": "electronic"},
    )

    assert "violin" in targets


def test_non_electronic_style_retains_synth_counterpart_conflict_filter():
    from mseparate.planner import _detected_acoustic_targets

    predictions = [
        {"kind": "source", "source": "synth_strings", "score": 7.0},
        {"kind": "source", "source": "violin", "score": 6.95},
    ]

    targets = _detected_acoustic_targets(
        predictions, {"family": "orchestral"},
    )

    assert "violin" not in targets



def test_electronic_hybrid_keeps_real_flute_alongside_synth_flute_with_window_evidence():
    from mseparate.planner import _detected_acoustic_targets

    predictions = [
        {"kind": "source", "source": "synth_flute", "score": 7.0},
        {
            "kind": "source", "source": "flute", "score": 6.8,
            "window_top_5_fraction": 0.5,
            "window_top_12_fraction": 0.875,
            "window_best_source_rank": 2,
        },
    ]

    targets = _detected_acoustic_targets(predictions, {"family": "electronic"})

    assert "flute" in targets


def test_aggregate_window_predictions_preserves_intermittent_real_instrument_evidence():
    from mseparate.planner import aggregate_window_predictions

    windows = [
        [
            {"kind": "source", "source": "violin", "score": 8.0},
            {"kind": "source", "source": "synth_pad", "score": 7.0},
        ],
        [
            {"kind": "source", "source": "synth_pad", "score": 8.0},
            {"kind": "source", "source": "violin", "score": 6.0},
        ],
        [
            {"kind": "source", "source": "synth_pad", "score": 8.0},
            {"kind": "source", "source": "violin", "score": 5.0},
        ],
        [
            {"kind": "source", "source": "synth_pad", "score": 8.0},
            {"kind": "source", "source": "violin", "score": 4.0},
        ],
    ]

    violin = next(
        item for item in aggregate_window_predictions(windows)
        if item["source"] == "violin"
    )
    assert violin["score"] == 5.75
    assert violin["windows"] == 4
    assert violin["window_top_5_fraction"] == 0.25
    assert violin["window_top_12_fraction"] == 1.0


def test_aggregate_window_predictions_counts_windows_without_local_source_evidence():
    from mseparate.planner import aggregate_window_predictions

    windows = [
        [
            {"kind": "source", "source": "synth_pad", "score": 8.0},
            {"kind": "source", "source": "industrial_noise", "score": 7.0},
            {"kind": "source", "source": "violin", "score": 6.0},
        ],
        [
            {"kind": "source", "source": "synth_pad", "score": 8.0},
            {"kind": "source", "source": "industrial_noise", "score": 7.0},
            {"kind": "source", "source": "violin", "score": 6.0},
        ],
        [
            {"kind": "source", "source": "synth_pad", "score": 8.0},
            {"kind": "source", "source": "industrial_noise", "score": 7.0},
        ],
    ]

    violin = next(
        item for item in aggregate_window_predictions(windows)
        if item["source"] == "violin"
    )
    assert violin["observed_windows"] == 2
    assert violin["window_top_5_fraction"] == round(2 / 3, 6)
