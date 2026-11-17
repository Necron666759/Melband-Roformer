"""Feature-guided, capability-aware conservative plan construction."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from .registry import eligible_models, model_audio_compatibility
from .source_analysis import classify_style
from .types import AnalysisReport, ExecutionPlan, ExecutionTask, ModelManifest


def _stem_score(stem: str, features: dict, predictions: list[dict] | None = None) -> tuple[float, str]:
    """Use semantic source rankings when available, else conservative signal heuristics."""
    name = stem.lower()
    aliases = {
        "vocals": ("vocals", "synth_vocals"), "voice": ("vocals", "synth_vocals"),
        "drums": ("drums", "electronic_drums", "breakbeats", "breakbeat_samples", "live_drum_kit",
                  "acoustic_kick", "electronic_kick", "synth_kick", "acoustic_snare", "electronic_snare",
                  "acoustic_hi_hat", "electronic_hi_hat", "cymbals", "synth_cymbal", "toms", "electronic_toms"),
        "percussion": ("percussion", "acoustic_percussion"),
        "kick": ("kick", "acoustic_kick", "electronic_kick", "synth_kick", "drums", "electronic_drums"),
        "snare": ("snare", "acoustic_snare", "electronic_snare", "drums", "electronic_drums"),
        "hh": ("hi_hat", "acoustic_hi_hat", "electronic_hi_hat", "drums", "electronic_drums"),
        "bass": ("bass", "synth_bass", "distorted_synth_bass"), "guitar": ("guitar",),
        "piano": ("piano", "synth_keys"), "keyboard": ("piano", "synth_keys"),
        "keys": ("synth_keys", "piano"), "organ": ("synth_keys", "piano"),
        "digital-piano": ("piano", "synth_keys"),
        "synth": ("synth_lead", "synth_pad", "synth_arp",
                  "synth_keys", "synth_strings", "synth_brass", "synth_bass", "synth_fx", "synth_sequence", "synth_pluck"),
        "pad": ("synth_pad",), "lead": ("synth_lead",),
        "arp": ("synth_arp",), "arpeggio": ("synth_arp",),
        "effect": ("sound_effects", "synth_fx"), "fx": ("synth_fx",),
        "strings": ("strings", "synth_strings"),
        "brass": ("brass", "synth_brass"), "woodwind": ("woodwinds",),
        "other": ("synthesizer", "synth_lead", "synth_pad", "synth_arp",
                  "synth_keys", "synth_strings", "synth_brass", "synth_bass",
                  "synth_fx", "strings", "brass", "woodwinds", "sound_effects", "samples"),
    }
    if predictions:
        # Genre matches are useful context for reporting, but only source
        # predictions may influence stem-to-model routing.
        lookup = {
            str(item.get("source")): float(item.get("score", 0.0))
            for item in predictions if item.get("kind", "source") == "source"
        }
        candidates = [key for token, values in aliases.items() if token in name
                      for key in values if key in lookup]
        if candidates:
            # Broad stems such as `other` match several source families. Use
            # the strongest CLAP match instead of whichever alias appears first.
            matched = max(candidates, key=lambda key: lookup[key])
            ordered = sorted(lookup.values())
            matched_score = lookup[matched]
            less = sum(value < matched_score for value in ordered)
            equal = sum(value == matched_score for value in ordered)
            rank = (less + 0.5 * max(0, equal - 1)) / max(1, len(ordered) - 1)
            return max(0.0, min(1.0, rank)), (
                f"CLAP relative similarity for {matched}: {lookup[matched]:.4f}; "
                "not a calibrated source-presence probability"
            )

    low = float(features.get("low_band_energy_ratio") or 0.0)
    mid = float(features.get("mid_band_energy_ratio") or 0.0)
    high = float(features.get("high_band_energy_ratio") or 0.0)
    transient = float(features.get("transient_activity") or 0.0)
    flatness = float(features.get("spectral_flatness") or 0.0)
    if any(token in name for token in ("drum", "percussion")):
        score, reason = 0.35 + 0.35 * min(transient / 0.35, 1.0) + 0.2 * high, "transient/high-band heuristic"
    elif any(token in name for token in ("bass", "low")):
        score, reason = 0.35 + 0.45 * low + 0.1 * (1.0 - flatness), "low-band heuristic"
    elif any(token in name for token in ("vocal", "voice")):
        score, reason = 0.35 + 0.3 * mid + 0.15 * (1.0 - abs(0.5 - mid) * 2), "mid-band heuristic"
    elif any(token in name for token in ("guitar", "other", "instrument")):
        score, reason = 0.35 + 0.25 * mid + 0.2 * high, "mid/high-band heuristic"
    else:
        score, reason = 0.5, "no stem-specific acoustic heuristic"
    return max(0.0, min(1.0, score)), reason


def _model_outputs(model: ModelManifest) -> tuple[str, ...]:
    declared = tuple(dict.fromkeys((*model.target_stems, *model.native_stems, *model.derived_stems)))
    return tuple(stem for stem in declared if stem not in model.dummy_stems)


_SOURCE_ALIASES = {
    "vocals": ("vocals", "synth_vocals"),
    "synth": ("synthesizer", "synth_lead", "synth_pad", "synth_arp",
              "synth_strings", "synth_brass", "synth_fx", "synth_sequence", "synth_pluck", "synth_flute",
              "synth_woodwinds", "synth_guitar"),
    "keys": ("synth_keys",),
    "digital-piano": ("digital_piano",),
    "organ": ("organ",),
    "bass": ("bass", "synth_bass", "distorted_synth_bass"),
    "kick": ("kick", "acoustic_kick", "electronic_kick", "synth_kick"),
    "drums": ("drums", "electronic_drums", "breakbeats", "breakbeat_samples", "live_drum_kit",
              "acoustic_kick", "electronic_kick", "synth_kick", "acoustic_snare", "electronic_snare",
              "acoustic_hi_hat", "electronic_hi_hat", "cymbals", "synth_cymbal", "toms", "electronic_toms"),
    "percussion": ("percussion", "acoustic_percussion"),
    "hh": ("hi_hat", "acoustic_hi_hat", "electronic_hi_hat"),
    "snare": ("snare", "acoustic_snare", "electronic_snare"),
    "guitar": ("guitar", "electric_guitar", "acoustic_guitar"),
    "piano": ("acoustic_piano", "digital_piano", "piano", "synth_keys"),
}

_ACOUSTIC_TARGETS = {
    "violin": ("violin", "synth_strings"),
    "viola": ("viola", "synth_strings"),
    "cello": ("cello", "synth_strings"),
    "double_bass": ("double-bass", "synth_bass"),
    "harp": ("harp", None),
    "guitar": ("guitar", "synth_guitar"),
    "piano": ("piano", "synth_keys"),
    "flute": ("flute", "synth_flute"),
    "clarinet": ("clarinet", "synth_woodwinds"),
    "oboe": ("oboe", "synth_woodwinds"),
    "bassoon": ("bassoon", "synth_woodwinds"),
    "saxophone": ("saxophone", "synth_woodwinds"),
    "trombone": ("trombone", "synth_brass"),
    "trumpet": ("trumpet", "synth_brass"),
    "tuba": ("tuba", "synth_brass"),
    "french_horn": ("french-horn", "synth_brass"),
    "strings": ("strings", "synth_strings"),
    "brass": ("brass", "synth_brass"),
    "woodwinds": ("woodwind", "synth_woodwinds"),
    "acoustic_guitar": ("acoustic-guitar", "synth_guitar"),
    "electric_guitar": ("electric-guitar", "synth_guitar"),
    "banjo": ("banjo", None),
    "mandolin": ("mandolin", None),
    "accordion": ("accordion", None),
    "harpsichord": ("harpsichord", None),
    "acoustic_piano": ("piano", "synth_keys"),
    "marimba": ("marimba", None),
    "glockenspiel": ("glockenspiel", None),
    "timpani": ("timpani", None),
    "congas": ("congas", None),
    "tambourine": ("tambourine", None),
    "sitar": ("sitar", None),
    "ukulele": ("ukulele", None),
}


def _source_score(predictions: list[dict], label: str) -> float | None:
    values = [
        float(item.get("score", 0.0)) for item in predictions
        if item.get("source") == label
    ]
    return max(values) if values else None


def _has_synth_evidence(
    predictions: list[dict], *, electronic_style: bool,
) -> bool:
    """Require synth evidence that is not better explained by a live instrument."""
    ranked = sorted(predictions, key=lambda item: float(item.get("score", 0.0)), reverse=True)
    if not ranked:
        return False
    top_score = float(ranked[0].get("score", 0.0))
    max_rank, max_gap = (12, 1.0) if electronic_style else (8, 0.75)
    conflicts = {
        "synthesizer": (
            "strings", "violin", "viola", "cello", "harp",
            "brass", "trombone", "trumpet", "tuba", "french_horn",
            "woodwinds", "flute", "clarinet", "oboe", "bassoon", "saxophone",
            "guitar", "acoustic_guitar", "electric_guitar",
        ),
        "synth_strings": ("strings", "violin", "viola", "cello"),
        "synth_brass": ("brass", "trombone", "trumpet", "tuba", "french_horn"),
        "synth_flute": ("flute",),
        "synth_woodwinds": ("woodwinds", "flute", "clarinet", "oboe", "bassoon", "saxophone"),
        "synth_guitar": ("guitar", "acoustic_guitar", "electric_guitar"),
    }
    specific_synth_sources = tuple(
        label for label in _SOURCE_ALIASES["synth"] if label != "synthesizer"
    ) + ("synth_keys", "synth_bass")
    for rank, item in enumerate(ranked, start=1):
        source = str(item.get("source", ""))
        if source not in _SOURCE_ALIASES["synth"]:
            continue
        score = float(item.get("score", 0.0))
        if rank > max_rank or top_score - score > max_gap:
            continue
        if source == "synthesizer" and any(
            (specific_score := _source_score(predictions, specific)) is not None
            and specific_score >= score - 0.10
            for specific in specific_synth_sources
        ):
            continue
        if any(
            (counterpart_score := _source_score(predictions, counterpart)) is not None
            and counterpart_score >= score - 0.10
            for counterpart in conflicts.get(source, ())
        ):
            continue
        return True
    return False


def _has_source_evidence(
    predictions: list[dict], labels: tuple[str, ...], *,
    max_rank: int = 8, max_score_gap: float = 0.75,
) -> bool:
    ranked = sorted(predictions, key=lambda item: float(item.get("score", 0.0)), reverse=True)
    if not ranked:
        return False
    top_score = float(ranked[0].get("score", 0.0))
    for rank, item in enumerate(ranked, start=1):
        if item.get("source") in labels:
            return rank <= max_rank and top_score - float(item.get("score", 0.0)) <= max_score_gap
    return False


def _detected_acoustic_targets(
    predictions: list[dict], style_profile: dict,
) -> tuple[str, ...]:
    """Select only clearly supported acoustic-instrument heads from Mega 53."""
    ranked = sorted(predictions, key=lambda item: float(item.get("score", 0.0)), reverse=True)
    if not ranked:
        return ()
    top_score = float(ranked[0].get("score", 0.0))
    family = style_profile.get("family")
    if family == "orchestral":
        max_rank, max_gap = 12, 1.20
    elif family == "electronic":
        max_rank, max_gap = 6, 0.50
    else:
        max_rank, max_gap = 8, 0.75
    score_by_source = {str(item.get("source")): float(item.get("score", 0.0)) for item in ranked}
    targets: list[str] = []
    family_for_source = {
        "violin": "strings", "viola": "strings", "cello": "strings",
        "double_bass": "strings", "harp": "strings", "strings": "strings",
        "trombone": "brass", "trumpet": "brass", "tuba": "brass",
        "french_horn": "brass", "brass": "brass",
        "flute": "woodwinds", "clarinet": "woodwinds", "oboe": "woodwinds",
        "bassoon": "woodwinds", "saxophone": "woodwinds", "woodwinds": "woodwinds",
        "guitar": "guitar", "acoustic_guitar": "guitar", "electric_guitar": "guitar",
    }
    accepted = []
    for rank, item in enumerate(ranked, start=1):
        source = str(item.get("source", ""))
        mapping = _ACOUSTIC_TARGETS.get(source)
        if mapping is None:
            continue
        score = float(item.get("score", 0.0))
        if rank > max_rank or top_score - score > max_gap:
            continue
        target, synthetic_counterpart = mapping
        if synthetic_counterpart:
            synthetic_score = score_by_source.get(synthetic_counterpart)
            if synthetic_score is not None and score < synthetic_score + 0.10:
                continue
        accepted.append((source, target, rank, score))
    # Prefer specific instrument heads over a broad family head when both fire.
    specific_families = {
        family_for_source.get(item[0], item[0]) for item in accepted
        if item[0] not in {"strings", "brass", "woodwinds", "guitar"}
    }
    for source, target, rank, score in accepted:
        family_name = family_for_source.get(source)
        if source in {"strings", "brass", "woodwinds", "guitar"} and family_name in specific_families:
            continue
        if target not in targets:
            targets.append(target)
    return tuple(targets[:12])


def _model_specialization(stem: str, model: ModelManifest) -> int:
    """Prefer a declared specialist over a general-purpose model for that stem.

    This is a capability-routing rule, not a claim about perceptual model
    quality. Unknown and multi-stem categories remain neutral.
    """
    category = (model.category or "").strip().lower().replace("_", "-")
    aliases = {
        "vocals": {"vocal", "vocals", "voice"},
        "instrumental": {"instrumental", "accompaniment"},
        "drums": {"drum", "drums", "percussion"},
        "bass": {"bass"},
        "guitar": {"guitar"},
        "piano": {"piano", "keyboard"},
        "synth": {"synth", "synth-specialist", "electronic"},
        "synth-pad": {"synth", "synth-specialist", "electronic"},
        "synth-lead": {"synth", "synth-specialist", "electronic"},
        "synth-arp": {"synth", "synth-specialist", "electronic"},
        "synth-keys": {"synth", "synth-specialist", "electronic"},
        "synth-strings": {"synth", "synth-specialist", "electronic"},
        "synth-brass": {"synth", "synth-specialist", "electronic"},
        "synth-bass": {"synth", "synth-specialist", "electronic"},
        "synth-fx": {"synth", "synth-specialist", "electronic"},
        "synth-vocals": {"synth", "synth-specialist", "electronic"},
        "electronic-drums": {"electronic-drums", "drums", "percussion"},
        "kick": {"drums", "percussion", "electronic-drums"},
        "snare": {"drums", "percussion", "electronic-drums"},
        "hh": {"drums", "percussion", "electronic-drums"},
        "keys": {"keyboard-specialist", "keys", "synth"},
        "organ": {"keyboard-specialist", "organ", "keys"},
        "digital-piano": {"keyboard-specialist", "piano", "keys"},
    }
    stem_key = stem.lower().replace("_", "-")
    accepted = aliases.get(stem_key, {stem_key})
    return 1 if category in accepted else 0


def build_plan(
    max_depth: int = 3,
    analysis: AnalysisReport | None = None,
    registry: dict[str, ModelManifest] | None = None,
    preferred_model_id: str | None = None,
    allow_resample: bool = False,
) -> ExecutionPlan:
    """Choose eligible, input-compatible models using transparent acoustic routing.

    Hard format constraints are applied before heuristic ranking. Routing scores
    are not model-quality estimates and cannot establish that a source exists.
    """
    if max_depth < 0:
        raise ValueError("max_depth must be zero or greater")
    available_registry = registry or {}
    if preferred_model_id is not None:
        preferred_manifest = available_registry.get(preferred_model_id)
        if preferred_manifest is not None and preferred_manifest.backend == "demucs-onnx":
            raise ValueError(
                "HT-Demucs 6-stem is intentionally excluded from normal routing; "
                "use the dedicated 'mseparate-demucs' terminal command"
            )
    source_predictions = [
        item for item in (analysis.predictions if analysis else [])
        if item.get("kind", "source") == "source"
    ]
    style_profile = classify_style(analysis.predictions if analysis else [])
    electronic_style = style_profile.get("family") == "electronic"
    acoustic_targets = _detected_acoustic_targets(source_predictions, style_profile)
    eligible = eligible_models(available_registry)
    candidates = [
        model for model in eligible
        if model.backend != "demucs-onnx"
        and (model.auto_route or model.model_id == preferred_model_id
             or (model.model_id == "zfturbo-bs-roformer-53stem-candidate" and bool(acoustic_targets)))
    ]

    filtered_candidates = []
    for model in candidates:
        category = (model.category or "").strip().lower().replace("_", "-")
        outputs = _model_outputs(model)
        if model.model_id.startswith("mvsep-mega53-"):
            # The complementary residual is not a specialist target; never
            # route "other" through a single-family Mega 53 derivative.
            outputs = tuple(stem for stem in outputs if stem != "other")
        if model.model_id == "zfturbo-bs-roformer-53stem-candidate":
            if preferred_model_id != model.model_id and not acoustic_targets:
                continue
        elif model.model_id.startswith("mvsep-mega53-"):
            # Single-target Mega 53 derivatives are scheduled only when their
            # own source family is detected; unknown audio is not a reason to
            # run every specialist.
            if not source_predictions or not any(
                (
                    _has_synth_evidence(source_predictions, electronic_style=electronic_style)
                    if stem == "synth"
                    else _has_source_evidence(
                        source_predictions, _SOURCE_ALIASES.get(stem, (stem.replace("-", "_"),)),
                        max_rank=12 if electronic_style and stem in {"keys", "bass"} else 8,
                        max_score_gap=1.0 if electronic_style and stem in {"keys", "bass"} else 0.75,
                    )
                )
                for stem in outputs
            ):
                continue
            synth_keys_score = _source_score(source_predictions, "synth_keys")
            digital_piano_score = _source_score(source_predictions, "digital_piano")
            acoustic_piano_score = _source_score(source_predictions, "acoustic_piano")
            broad_piano_score = _source_score(source_predictions, "piano")
            non_synth_keyboard_score = max(
                (value for value in (digital_piano_score, acoustic_piano_score, broad_piano_score)
                 if value is not None),
                default=None,
            )
            if "keys" in outputs and non_synth_keyboard_score is not None and (
                synth_keys_score is None or non_synth_keyboard_score >= synth_keys_score - 0.10
            ):
                continue
            if "digital-piano" in outputs and synth_keys_score is not None and (
                digital_piano_score is None or synth_keys_score >= digital_piano_score - 0.10
            ):
                continue
        elif category == "synth-specialist":
            if not _has_synth_evidence(source_predictions, electronic_style=electronic_style):
                continue
        elif category == "keyboard-specialist":
            if not _has_source_evidence(
                source_predictions, _SOURCE_ALIASES["keys"],
                max_rank=12 if electronic_style else 8,
                max_score_gap=1.0 if electronic_style else 0.75,
            ):
                continue
        elif category in {"vocal", "vocals"} and source_predictions:
            if not _has_source_evidence(source_predictions, _SOURCE_ALIASES["vocals"], max_rank=10):
                continue
        filtered_candidates.append(model)
    candidates = filtered_candidates
    if not candidates:
        return ExecutionPlan(1, (), (
            "No verified, enabled models are available for normal routing; neural inference was not scheduled.",
            "HT-Demucs 6-stem (ONNX) is excluded from normal automatic and preferred-model routing because measured inference is exceptionally slow; use the dedicated mseparate-demucs command for explicit runs.",
        ), max_depth, style_profile)
    features = analysis.measurements if analysis else {}
    audio = analysis.audio if analysis else None
    compatible: list[ModelManifest] = []
    rejected: list[str] = []
    for model in candidates:
        if audio is not None:
            checked_audio = (
                replace(audio, sample_rate=model.sample_rate)
                if allow_resample and model.sample_rate is not None
                else audio
            )
            ok, reason = model_audio_compatibility(model, checked_audio)
            if not ok:
                rejected.append(f"{model.model_id}: {reason}")
                continue
        compatible.append(model)
    if not compatible:
        notes = ["No eligible model accepts the input audio format; no inference was scheduled."]
        notes.extend(rejected)
        return ExecutionPlan(1, (), tuple(notes), max_depth, style_profile)

    if preferred_model_id is not None and preferred_model_id not in {model.model_id for model in compatible}:
        raise ValueError(
            f"Preferred model {preferred_model_id!r} is not eligible or compatible with this input"
        )

    ranked: dict[str, list[tuple[int, int, int, float, ModelManifest, str]]] = defaultdict(list)
    for model in compatible:
        model_stems = (
            acoustic_targets
            if model.model_id == "zfturbo-bs-roformer-53stem-candidate"
            and preferred_model_id != model.model_id
            else _model_outputs(model)
        )
        if model.model_id.startswith("mvsep-mega53-"):
            model_stems = tuple(stem for stem in model_stems if stem != "other")
        for stem in model_stems:
            score, reason = _stem_score(stem, features, analysis.predictions if analysis else None)
            specialization = _model_specialization(stem, model)
            preferred = int(model.model_id == preferred_model_id)
            ranked[stem].append((specialization, preferred, model.planner_priority, score, model, reason))

    selected: dict[str, dict] = {}
    for stem, entries in ranked.items():
        # CLAP/acoustic scores describe source salience, not model quality.
        # Route by declared specialization, then operator-set manifest priority,
        # then source salience; model_id is only the stable final tie-break.
        entries.sort(key=lambda item: (-item[0], -item[1], -item[2], -item[3], item[4].model_id))
        specialization, preferred, priority, score, model, reason = entries[0]
        bucket = selected.setdefault(model.model_id, {"model": model, "stems": [], "scores": [], "reasons": []})
        bucket["stems"].append(stem)
        bucket["scores"].append(score)
        route = "manifest-declared specialist" if specialization else "general-purpose capability"
        if preferred:
            route += " (explicitly preferred)"
        elif priority:
            route += f" (planner priority {priority})"
        bucket["reasons"].append(f"{stem}: {route}; {reason}")

    tasks = []
    for index, model_id in enumerate(sorted(selected), start=1):
        item = selected[model_id]
        score = sum(item["scores"]) / len(item["scores"])
        confidence = min(0.75, max(0.1, score))
        tasks.append(ExecutionTask(
            task_id=f"separate-{index:02d}", model_id=model_id, input_stem_id="mix",
            target_stems=tuple(sorted(item["stems"])), depth=0,
            score=round(score, 4), confidence=round(confidence, 4),
            rationale="; ".join(item["reasons"]),
        ))
    notes = [
        "Hard input-format and channel constraints are checked before model ranking; models without backend-declared sample-rate conversion support require explicit --allow-resample.",
        "Tasks are ordered deterministically and must execute one at a time; parallel model inference is forbidden.",
        "Overlapping routes prefer a manifest-declared specialist, then an explicitly preferred model, then manifest planner_priority and source salience. Priority is an operator routing preference, not a measured quality score; CLAP/acoustic scores do not estimate model quality or calibrated source presence.",
        "CLAP source-family matches are relative similarities, not calibrated source-presence probabilities. Genre classification is provisional and source predictions, not genre labels alone, gate individual stems.",
        "Synth leads, pads, arpeggios, synth bass, vocoder-like vocals and electronic FX are analyzed separately where CLAP can distinguish them; overlapping descriptions remain approximate.",
        "HT-Demucs 6-stem (ONNX) is excluded from normal automatic and preferred-model routing because measured inference is exceptionally slow; use the dedicated mseparate-demucs command for explicit runs.",
        "No refinement tasks are added until parent/child output quality can be compared safely.",
    ]
    if style_profile.get("label") != "unknown":
        notes.append(
            f"Detected music style: {style_profile['label']} "
            f"(family={style_profile['family']}, status={style_profile['status']}); "
            "CLAP ranking is a provisional semantic classification, not ground truth."
        )
    else:
        notes.append("Music style is unclassified; specialist routing remains conservative.")
    if acoustic_targets and preferred_model_id != "zfturbo-bs-roformer-53stem-candidate":
        notes.append(
            "Detected acoustic instrument targets routed through one verified Mega 53 inference "
            "with only these output heads requested: " + ", ".join(acoustic_targets) + "."
        )
    if electronic_style:
        notes.append(
            "Electronic-style profile: verified Synth/Keys specialists are considered when their "
            "source-family evidence is present; unrelated acoustic instrument heads are not scheduled."
        )
    if allow_resample:
        notes.append(
            "Sample-rate conversion is enabled: the execution manager uses a temporary model-rate WAV "
            "for inference; output stems retain the model sample rate."
        )
    notes.extend(rejected)
    if not tasks:
        notes.append("Eligible manifests declared no usable output stems.")
    return ExecutionPlan(1, tuple(tasks), tuple(notes), max_depth, style_profile)



def add_vocal_comparison_tasks(
    plan: ExecutionPlan,
    *,
    analysis: AnalysisReport,
    registry: dict[str, ModelManifest],
    allow_resample: bool = False,
) -> ExecutionPlan:
    """Append explicit alternative runs for all eligible primary vocal models.

    These tasks generate comparable candidate files from the same mix. They
    are opt-in because they increase inference time and do not by themselves
    identify a perceptual winner.
    """
    existing = {
        task.model_id for task in plan.tasks
        if "vocals" in task.target_stems and task.input_stem_id == "mix"
    }
    candidates = []
    rejected = []
    for model in eligible_models(registry):
        category = (model.category or "").strip().lower().replace("_", "-")
        if category not in {"vocal", "vocals"} or not model.auto_route:
            continue
        if model.backend not in {"melband-roformer", "bs-roformer-pymss", "mdx23c-pymss"}:
            continue
        if "vocals" not in _model_outputs(model) or model.model_id in existing:
            continue
        checked_audio = (
            replace(analysis.audio, sample_rate=model.sample_rate)
            if allow_resample and model.sample_rate is not None
            else analysis.audio
        )
        compatible, reason = model_audio_compatibility(model, checked_audio)
        if not compatible:
            rejected.append(f"{model.model_id}: {reason}")
            continue
        candidates.append(model)

    additions = []
    for index, model in enumerate(sorted(candidates, key=lambda item: item.model_id), start=1):
        additions.append(ExecutionTask(
            task_id=f"compare-vocals-{index:02d}",
            model_id=model.model_id,
            input_stem_id="mix",
            target_stems=("vocals",),
            depth=0,
            score=0.0,
            confidence=0.0,
            rationale=(
                "Explicit vocal-model comparison candidate; generated from the same input mix. "
                "This output is an alternative, not an additive stem or a quality verdict."
            ),
        ))

    notes = list(plan.notes)
    if additions:
        notes.append(
            "Explicit vocal comparison is enabled: alternative primary vocal models run sequentially "
            "against the original mix; candidate outputs are kept separate and are never summed."
        )
        notes.append(
            "Post-inference CLAP ranks provide semantic source-alignment hints only; they do not "
            "measure separation quality or automatically select a winner."
        )
    else:
        notes.append(
            "Explicit vocal comparison found no additional eligible and input-compatible primary vocal models."
        )
    notes.extend(rejected)
    return replace(plan, tasks=(*plan.tasks, *additions), notes=tuple(notes))

def validate_plan(plan: ExecutionPlan) -> None:
    ids = [task.task_id for task in plan.tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("Execution plan contains duplicate task IDs")
    known: set[str] = set()
    for task in plan.tasks:
        if task.depth < 0:
            raise ValueError(f"Task {task.task_id} has negative depth")
        missing = set(task.depends_on) - known
        if missing:
            raise ValueError(f"Task {task.task_id} has dependencies not scheduled earlier: {sorted(missing)}")
        if not 0.0 <= task.confidence <= 1.0 or not 0.0 <= task.score <= 1.0:
            raise ValueError(f"Task {task.task_id} has invalid score or confidence")
        known.add(task.task_id)
    if plan.tasks and max(task.depth for task in plan.tasks) > plan.max_depth:
        raise ValueError("Execution plan exceeds its declared max_depth")
