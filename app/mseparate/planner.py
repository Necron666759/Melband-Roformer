"""Feature-guided, capability-aware conservative plan construction."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from .registry import eligible_models, model_audio_compatibility
from .types import AnalysisReport, ExecutionPlan, ExecutionTask, ModelManifest


def _stem_score(stem: str, features: dict, predictions: list[dict] | None = None) -> tuple[float, str]:
    """Use semantic source rankings when available, else conservative signal heuristics."""
    name = stem.lower()
    aliases = {
        "vocals": ("vocals", "synth_vocals"), "voice": ("vocals", "synth_vocals"),
        "drums": ("drums", "electronic_drums", "breakbeats"),
        "percussion": ("drums", "electronic_drums", "breakbeats"),
        "kick": ("kick", "drums", "electronic_drums"),
        "snare": ("snare", "drums", "electronic_drums"),
        "hh": ("hi_hat", "drums", "electronic_drums"),
        "bass": ("bass", "synth_bass"), "guitar": ("guitar",),
        "piano": ("piano", "synth_keys"), "keyboard": ("piano", "synth_keys"),
        "keys": ("synth_keys", "piano"), "organ": ("synth_keys", "piano"),
        "digital-piano": ("piano", "synth_keys"),
        "synth": ("synthesizer", "synth_lead", "synth_pad", "synth_arp",
                  "synth_keys", "synth_strings", "synth_brass", "synth_bass", "synth_fx"),
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
    candidates = [
        model for model in eligible_models(available_registry)
        if model.backend != "demucs-onnx"
        and (model.auto_route or model.model_id == preferred_model_id)
    ]
    source_predictions = [
        item for item in (analysis.predictions if analysis else [])
        if item.get("kind", "source") == "source"
    ]
    # Newly enabled electronic specialists must not become default tasks on
    # unrelated/unknown audio. Require an explicit source-family prediction.
    specialist_evidence = {
        "synth-specialist": any(
            str(item.get("source", "")).startswith("synth_")
            or item.get("source") == "synthesizer"
            for item in source_predictions
        ),
        "keyboard-specialist": any(
            item.get("source") in {"synth_keys", "piano"}
            for item in source_predictions
        ),
    }
    candidates = [
        model for model in candidates
        if (model.category or "").strip().lower().replace("_", "-")
        not in specialist_evidence
        or specialist_evidence[(model.category or "").strip().lower().replace("_", "-")]
    ]
    if not candidates:
        return ExecutionPlan(1, (), (
            "No verified, enabled model manifests are available; neural inference was not scheduled.",
        ), max_depth)
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
        return ExecutionPlan(1, (), tuple(notes), max_depth)

    if preferred_model_id is not None and preferred_model_id not in {model.model_id for model in compatible}:
        raise ValueError(
            f"Preferred model {preferred_model_id!r} is not eligible or compatible with this input"
        )

    ranked: dict[str, list[tuple[int, int, int, float, ModelManifest, str]]] = defaultdict(list)
    for model in compatible:
        for stem in _model_outputs(model):
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
        "CLAP source-family matches are relative similarities, not calibrated source-presence probabilities. Electronic genre matches are contextual hints only and never trigger a stem route by themselves.",
        "Synth leads, pads, arpeggios, synth bass, vocoder-like vocals and electronic FX are analyzed separately where CLAP can distinguish them; overlapping descriptions remain approximate."
        "HT-Demucs 6-stem (ONNX) is excluded from normal automatic and preferred-model routing because measured inference is exceptionally slow; use the dedicated mseparate-demucs command for explicit runs.",
        "No refinement tasks are added until parent/child output quality can be compared safely.",
    ]
    if allow_resample:
        notes.append(
            "Sample-rate conversion is enabled: the execution manager uses a temporary model-rate WAV "
            "for inference; output stems retain the model sample rate."
        )
    notes.extend(rejected)
    if not tasks:
        notes.append("Eligible manifests declared no usable output stems.")
    return ExecutionPlan(1, tuple(tasks), tuple(notes), max_depth)


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
