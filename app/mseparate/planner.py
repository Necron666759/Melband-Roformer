"""Feature-guided, capability-aware conservative plan construction."""
from __future__ import annotations

from collections import defaultdict

from .registry import eligible_models, model_audio_compatibility
from .types import AnalysisReport, ExecutionPlan, ExecutionTask, ModelManifest


def _stem_score(stem: str, features: dict, predictions: list[dict] | None = None) -> tuple[float, str]:
    """Use semantic source rankings when available, else conservative signal heuristics."""
    name = stem.lower()
    aliases = {
        "vocals": ("vocals",), "voice": ("vocals",),
        "drums": ("drums",), "percussion": ("drums",),
        "bass": ("bass",), "guitar": ("guitar",),
        "piano": ("piano",), "keyboard": ("piano",),
        "synth": ("synthesizer",), "strings": ("strings",),
        "brass": ("brass",), "woodwind": ("woodwinds",),
        "effect": ("sound_effects",),
        "other": ("synthesizer", "strings", "brass", "woodwinds", "sound_effects"),
    }
    if predictions:
        lookup = {str(item.get("source")): float(item.get("score", 0.0)) for item in predictions}
        matched = next((key for token, values in aliases.items() if token in name
                        for key in values if key in lookup), None)
        if matched is not None:
            ordered = sorted(lookup.values())
            rank = ordered.index(lookup[matched]) / max(1, len(ordered) - 1)
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


def build_plan(
    max_depth: int = 3,
    analysis: AnalysisReport | None = None,
    registry: dict[str, ModelManifest] | None = None,
) -> ExecutionPlan:
    """Choose eligible, input-compatible models using transparent acoustic routing.

    Hard format constraints are applied before heuristic ranking. Routing scores
    are not model-quality estimates and cannot establish that a source exists.
    """
    if max_depth < 0:
        raise ValueError("max_depth must be zero or greater")
    candidates = eligible_models(registry or {})
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
            ok, reason = model_audio_compatibility(model, audio)
            if not ok:
                rejected.append(f"{model.model_id}: {reason}")
                continue
        compatible.append(model)
    if not compatible:
        notes = ["No eligible model accepts the input audio format; no inference was scheduled."]
        notes.extend(rejected)
        return ExecutionPlan(1, (), tuple(notes), max_depth)

    ranked: dict[str, list[tuple[float, ModelManifest, str]]] = defaultdict(list)
    for model in compatible:
        for stem in _model_outputs(model):
            score, reason = _stem_score(stem, features, analysis.predictions if analysis else None)
            ranked[stem].append((score, model, reason))

    selected: dict[str, dict] = {}
    for stem, entries in ranked.items():
        entries.sort(key=lambda item: (-item[0], item[1].model_id))
        score, model, reason = entries[0]
        margin = score - entries[1][0] if len(entries) > 1 else 0.25
        confidence = max(0.1, min(0.75, 0.35 + margin))
        bucket = selected.setdefault(model.model_id, {"model": model, "stems": [], "scores": [], "reasons": []})
        bucket["stems"].append(stem)
        bucket["scores"].append(score)
        bucket["reasons"].append(f"{stem}: {reason}")

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
        "Hard sample-rate/channel constraints were checked before model ranking.",
        "Tasks are ordered deterministically and must execute one at a time; parallel model inference is forbidden.",
        "Scores rank declared stem routes using CLAP relative similarities when available, otherwise acoustic heuristics; they are not calibrated source-presence probabilities.",
        "CLAP detects broad semantic audio-text matches; it cannot reliably identify every individual synthesizer patch or guarantee clean isolation.",
        "No refinement tasks are added until parent/child output quality can be compared safely.",
    ]
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
