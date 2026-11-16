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
) -> dict[str, Any]:
    """Analyze every successful output using one CLAP load and register its lineage."""
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
    root_id = "mix-" + uuid.uuid4().hex
    ledger.add_candidate(
        root_id, stem="mix", path=mix_path, model_id="input-mixture",
        reason="Root audio for post-inference candidate lineage",
        analysis={"role": "parent_audio", "source_prediction_count": len(mix_predictions)},
    )

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
