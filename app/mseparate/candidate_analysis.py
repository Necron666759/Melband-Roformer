"""Post-inference source analysis and auditable candidate registration.

CLAP similarities are relative semantic hints, not calibrated probabilities or
perceptual separation scores. This module never automatically rejects,
quarantines or promotes an audio candidate.
"""
from __future__ import annotations

import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from .execution import ExecutionSummary
from .quarantine import QuarantineLedger
from .source_analysis import analyze_sources_many

_TARGETS = {
    "vocals": {"vocals", "synth_vocals"},
    "vocal": {"vocals", "synth_vocals"},
    "bass": {"bass", "synth_bass"},
    "kick": {"kick", "drums", "electronic_drums"},
    "drums": {"drums", "electronic_drums", "breakbeats"},
    "percussion": {"drums", "electronic_drums", "breakbeats", "hi_hat"},
    "hh": {"hi_hat", "electronic_drums", "drums"},
    "snare": {"snare", "drums", "electronic_drums"},
    "guitar": {"guitar"},
    "piano": {"piano", "synth_keys"},
    "digital-piano": {"piano", "synth_keys"},
    "keys": {"synth_keys", "piano"},
    "organ": {"synth_keys", "piano"},
    "synth": {"synthesizer", "synth_lead", "synth_pad", "synth_arp", "synth_keys",
              "synth_strings", "synth_brass", "synth_bass", "synth_fx"},
    "strings": {"strings", "synth_strings"},
    "brass": {"brass", "synth_brass"},
    "woodwind": {"woodwinds"},
}


def _ranked_sources(predictions: list[dict]) -> tuple[list[dict], dict[str, int]]:
    items = sorted(
        (item for item in predictions if item.get("kind", "source") == "source"),
        key=lambda item: float(item.get("score", 0.0)),
        reverse=True,
    )
    return items, {str(item.get("source")): index for index, item in enumerate(items, 1)}


def _candidate_record(
    *,
    artifact,
    task_id: str,
    model_id: str,
    report,
    predictions: list[dict],
    mix_ranks: dict[str, int],
) -> dict[str, Any]:
    ranked, ranks = _ranked_sources(predictions)
    stem_key = artifact.stem.lower().replace("_", "-")
    expected = _TARGETS.get(stem_key)
    top_sources = [
        {"source": str(item["source"]), "score": float(item["score"]), "rank": index}
        for index, item in enumerate(ranked[:8], 1)
    ]
    target_rank = min((ranks[name] for name in expected if name in ranks), default=None) if expected else None
    off_target = []
    if expected:
        off_target = [item for item in top_sources if item["source"] not in expected][:3]

    residual_hints = []
    if stem_key == "other":
        # A source is a residual hint only if it ranked in the top ten on the
        # original mix and remains among the top five semantic matches in Other.
        for item in top_sources:
            source = item["source"]
            original_rank = mix_ranks.get(source)
            if original_rank is not None and original_rank <= 10 and item["rank"] <= 5:
                residual_hints.append({
                    "source": source,
                    "mix_rank": original_rank,
                    "other_rank": item["rank"],
                    "clap_score": item["score"],
                    "interpretation": "possible musical content remaining in Other; verify by listening",
                })

    technical_issues = list(report.issues) if report is not None else []
    flags = []
    if technical_issues:
        flags.append("technical_warning")
    if off_target:
        flags.append("possible_off_target_content")
    if residual_hints:
        flags.append("possible_residual_sources")

    return {
        "task_id": task_id,
        "model_id": model_id,
        "stem": artifact.stem,
        "path": str(artifact.path),
        "technical_status": report.status if report is not None else "not_checked",
        "technical_issues": technical_issues,
        "target_source_rank": target_rank,
        "top_sources": top_sources,
        "possible_off_target_sources": off_target,
        "residual_source_hints": residual_hints,
        "review_flags": flags,
        "decision": "pending_review",
        "interpretation_note": (
            "CLAP scores are relative audio/text similarities, not probabilities or a perceptual quality score. "
            "No automatic acceptance, rejection or quarantine decision was made."
        ),
    }


def analyze_execution_candidates(
    summary: ExecutionSummary,
    *,
    mix_path: str | Path,
    mix_predictions: list[dict],
    device: str,
    cache_dir: str | Path | None,
    no_download: bool,
    ledger_path: str | Path,
    parent_candidate_id: str | None = None,
) -> dict[str, Any]:
    """Analyze outputs and attach them to a root mixture or an explicit parent candidate."""
    artifacts = [
        (result, artifact, report)
        for result in summary.results if result.status == "success"
        for artifact in result.outputs
        for report in (next(
            (item for item in result.quality_reports if Path(item.path) == Path(artifact.path)),
            None,
        ),)
    ]
    if not artifacts:
        return {
            "schema_version": 1, "candidate_count": 0, "candidates": [],
            "comparison_groups": [], "residual_source_hints": [],
            "notes": ["No successful output artifacts were available for deep analysis."],
        }

    prediction_map = analyze_sources_many(
        [artifact.path for _, artifact, _ in artifacts],
        device=device, cache_dir=cache_dir, no_download=no_download,
    )
    _, mix_ranks = _ranked_sources(mix_predictions)
    ledger = QuarantineLedger(ledger_path)
    if parent_candidate_id is None:
        root_id = "mix-" + uuid.uuid4().hex
        ledger.add_candidate(
            root_id, stem="mix", path=mix_path, model_id="input-mixture",
            reason="Root audio for post-inference candidate lineage",
            analysis={"role": "parent_audio", "source_prediction_count": len(mix_predictions)},
        )
    else:
        known_ids = {item["candidate_id"] for item in ledger.list_candidates()}
        if parent_candidate_id not in known_ids:
            raise ValueError(f"Unknown lineage parent candidate: {parent_candidate_id}")
        root_id = parent_candidate_id

    candidates = []
    grouped: dict[str, list[dict]] = defaultdict(list)
    residuals = []
    for result, artifact, quality in artifacts:
        predictions = prediction_map.get(str(Path(artifact.path).expanduser()), [])
        candidate = _candidate_record(
            artifact=artifact, task_id=result.task_id, model_id=result.model_id,
            report=quality, predictions=predictions, mix_ranks=mix_ranks,
        )
        candidate_id = f"{result.task_id}-{artifact.stem}-{uuid.uuid4().hex[:8]}"
        ledger.add_candidate(
            candidate_id, stem=artifact.stem, path=artifact.path,
            model_id=result.model_id, parent_id=root_id,
            reason="Post-inference source analysis completed; awaiting comparative review",
            analysis=candidate,
        )
        candidate["candidate_id"] = candidate_id
        candidate["parent_candidate_id"] = root_id
        candidates.append(candidate)
        grouped[artifact.stem].append(candidate)
        for hint in candidate["residual_source_hints"]:
            residuals.append({
                "candidate_id": candidate_id,
                "model_id": result.model_id,
                "path": str(artifact.path),
                **hint,
            })

    comparisons = []
    for stem, entries in sorted(grouped.items()):
        ordered = sorted(
            entries,
            key=lambda item: (
                item["target_source_rank"] if item["target_source_rank"] is not None else 10**6,
                item["model_id"], item["candidate_id"],
            ),
        )
        comparisons.append({
            "stem": stem,
            "candidate_count": len(ordered),
            "ordered_by_target_source_rank": [
                {
                    "candidate_id": item["candidate_id"],
                    "model_id": item["model_id"],
                    "target_source_rank": item["target_source_rank"],
                }
                for item in ordered
            ],
            "interpretation": (
                "Relative CLAP source-alignment ordering only; not a separation-quality ranking."
                if len(ordered) > 1 else
                "Only one model candidate was generated for this stem; no model-to-model comparison is possible."
            ),
        })

    return {
        "schema_version": 1,
        "candidate_count": len(candidates),
        "parent_candidate_id": root_id,
        "root_candidate_id": root_id if parent_candidate_id is None else None,
        "ledger_path": str(Path(ledger_path).expanduser()),
        "candidates": candidates,
        "comparison_groups": comparisons,
        "residual_source_hints": residuals,
        "notes": [
            "CLAP values are relative similarities, not calibrated probabilities.",
            "A high off-target rank is a review hint, not proof of leakage or an artifact.",
            "Residual hints in an Other stem do not authorize mixture subtraction or automatic rescue execution.",
            "Candidates remain pending; no audio was moved, deleted or overwritten.",
        ],
    }



# CLAP source-family labels are mapped only to exact native labels declared by
# the ZFTurbo 53-stem manifest. These are retrieval targets, not residual math.
_RESCUE_TARGETS = {
    "vocals": ("vocal", "lead-vocal", "back-vocal"),
    "synth_vocals": ("lead-vocal", "back-vocal"),
    "drums": ("drums",),
    "electronic_drums": ("drums",),
    "kick": ("kick",),
    "snare": ("snare",),
    "hi_hat": ("hh",),
    "breakbeats": ("drums", "percussion"),
    "bass": ("bass",),
    "synth_bass": ("bass",),
    "guitar": ("guitar", "electric-guitar", "acoustic-guitar"),
    "piano": ("piano", "digital-piano"),
    "synth_keys": ("keys", "digital-piano"),
    "synthesizer": ("synth",),
    "synth_lead": ("synth",),
    "synth_pad": ("synth",),
    "synth_arp": ("synth",),
    "synth_strings": ("synth",),
    "synth_brass": ("synth",),
    "strings": ("strings", "violin", "viola", "cello"),
    "brass": ("brass",),
    "woodwinds": ("woodwind",),
}


def evaluate_conditional_rescue(
    report: dict[str, Any],
    *,
    registry: dict[str, Any],
    audio,
    model_id: str = "zfturbo-bs-roformer-53stem-candidate",
    max_targets: int = 8,
) -> dict[str, Any]:
    """Build one conservative 53-stem rescue request from a verified MVSep Other.

    A hint must rank in the top five on the mix and top three in Other. Only
    one best parent Other is selected per invocation to bound expensive runs.
    """
    candidates = report.get("candidates", [])
    parents = [
        item for item in candidates
        if item.get("stem", "").strip().lower() == "other"
        and str(item.get("model_id", "")).startswith("mvsep-mega53-")
        and item.get("technical_status") == "valid"
    ]
    if not parents:
        return {
            "status": "skipped",
            "reason": "No technically clean Other output from an MVSep Mega 53 specialist was available.",
        }

    eligible_parents = []
    for candidate in parents:
        strong = [
            hint for hint in candidate.get("residual_source_hints", [])
            if int(hint.get("mix_rank", 10**6)) <= 5
            and int(hint.get("other_rank", 10**6)) <= 3
            and hint.get("source") in _RESCUE_TARGETS
        ]
        if strong:
            eligible_parents.append((candidate, strong))
    if not eligible_parents:
        return {
            "status": "skipped",
            "reason": (
                "No residual hint met the conservative rescue gate "
                "(source rank <= 5 in the mix and <= 3 in MVSep Other)."
            ),
        }

    parent, hints = min(
        eligible_parents,
        key=lambda pair: (
            min(int(h["other_rank"]) for h in pair[1]),
            min(int(h["mix_rank"]) for h in pair[1]),
            pair[0].get("candidate_id", ""),
        ),
    )
    manifest = registry.get(model_id)
    if manifest is None:
        return {
            "status": "skipped",
            "reason": (
                "The ZFTurbo 53-stem model is not present in the device-eligible registry; "
                "its license, runtime and VRAM gates were not satisfied."
            ),
            "parent_candidate_id": parent.get("candidate_id"),
        }
    if (
        manifest.status != "available"
        or manifest.license_status != "verified"
        or manifest.runtime_validation != "verified"
    ):
        return {
            "status": "skipped",
            "reason": "The ZFTurbo 53-stem model did not pass availability, license and runtime gates.",
            "parent_candidate_id": parent.get("candidate_id"),
        }

    from .registry import eligible_models, model_audio_compatibility
    from dataclasses import replace

    eligible_ids = {item.model_id for item in eligible_models(registry)}
    if model_id not in eligible_ids:
        return {
            "status": "skipped",
            "reason": (
                "The ZFTurbo 53-stem model did not pass the full registry eligibility policy "
                "(including license-review and pinned-asset requirements)."
            ),
            "parent_candidate_id": parent.get("candidate_id"),
        }

    parent_path = Path(parent.get("path", "")).expanduser()
    if not parent_path.is_file():
        return {
            "status": "skipped",
            "reason": "The selected MVSep Other artifact is no longer available on disk.",
            "parent_candidate_id": parent.get("candidate_id"),
        }
    compatible, reason = model_audio_compatibility(
        manifest, replace(audio, path=str(parent_path)),
    )
    if not compatible:
        return {
            "status": "skipped",
            "reason": f"The parent Other audio is incompatible with ZFTurbo 53-stem: {reason}",
            "parent_candidate_id": parent.get("candidate_id"),
        }

    declared = set(manifest.native_stems)
    targets = []
    selected_hints = []
    for hint in sorted(hints, key=lambda item: (
        int(item["other_rank"]), int(item["mix_rank"]), str(item["source"])
    )):
        mapped = [stem for stem in _RESCUE_TARGETS[hint["source"]] if stem in declared]
        if not mapped:
            continue
        selected_hints.append(hint)
        for stem in mapped:
            if stem not in targets and len(targets) < max_targets:
                targets.append(stem)
    if not targets:
        return {
            "status": "skipped",
            "reason": "Strong residual hints had no matching native ZFTurbo 53-stem targets.",
            "parent_candidate_id": parent.get("candidate_id"),
        }

    return {
        "status": "ready",
        "model_id": model_id,
        "parent_candidate_id": parent["candidate_id"],
        "parent_model_id": parent["model_id"],
        "input_path": str(parent_path),
        "target_stems": targets,
        "source_hints": selected_hints,
        "gate": {
            "mix_rank_max": 5,
            "other_rank_max": 3,
            "requires_valid_parent_output": True,
            "max_parent_outputs_per_invocation": 1,
            "max_target_stems": max_targets,
        },
        "warning": (
            "MVSep Other is not assumed to be a mathematically exact mixture residual. "
            "ZFTurbo outputs are independent, may overlap, and are never added to or subtracted from other stems."
        ),
    }


def print_candidate_analysis(report: dict[str, Any]) -> None:
    print("Deep output analysis (CLAP relative source similarities)")
    print(f"  Candidates analyzed: {report.get('candidate_count', 0)}")
    for candidate in report.get("candidates", []):
        print(
            f"  {candidate['stem']} [{candidate['model_id']}]: "
            f"target source rank={candidate['target_source_rank'] or 'n/a'}; "
            f"technical={candidate['technical_status']}; "
            f"flags={', '.join(candidate['review_flags']) or 'none'}"
        )
        for hint in candidate["residual_source_hints"]:
            print(
                f"    Possible residual: {hint['source']} "
                f"(mix rank {hint['mix_rank']}, Other rank {hint['other_rank']})"
            )
        for item in candidate["possible_off_target_sources"]:
            print(f"    Possible off-target content: {item['source']} (rank {item['rank']})")
    for note in report.get("notes", []):
        print("  Note: " + note)
