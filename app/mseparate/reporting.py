"""Human-readable rendering for analysis and plan reports."""
from __future__ import annotations

from pathlib import Path

from .execution import ExecutionSummary
from .types import AnalysisReport, ExecutionPlan


def print_analysis(report: AnalysisReport) -> None:
    audio = report.audio
    print("Audio analysis")
    print(f"  File:       {audio.path}")
    print(f"  Duration:   {audio.duration_seconds:.3f} s")
    print(f"  Sample rate:{audio.sample_rate:>8} Hz")
    print(f"  Channels:   {audio.channels}")
    print(f"  Frames:     {audio.frames}")
    if audio.subtype:
        print(f"  Subtype:    {audio.subtype}")
    for key, value in report.measurements.items():
        if value is not None:
            print(f"  {key}: {value:.8g}")
    if report.predictions:
        print("  Semantic source ranking (CLAP relative scores; not probabilities):")
        for item in report.predictions[:10]:
            print(f"    {item.get('source')}: {float(item.get('score', 0.0)):.4f}")
    for note in report.notes:
        print(f"  Note: {note}")


def print_plan(plan: ExecutionPlan) -> None:
    print("Execution plan")
    if not plan.tasks:
        print("  No neural inference tasks scheduled.")
    for index, task in enumerate(plan.tasks, start=1):
        print(f"  {index}. {task.task_id}: {task.model_id} -> {', '.join(task.target_stems)}")
        print(f"     Routing score: {task.score:.3f}; confidence: {task.confidence:.3f}")
        if task.rationale:
            print(f"     Rationale: {task.rationale}")
    for note in plan.notes:
        print(f"  Note: {note}")



def print_execution_summary(summary: ExecutionSummary) -> None:
    """Print task outcomes and the technical quality checks for each output."""
    print("Execution results")
    print(
        f"  Succeeded: {summary.succeeded}; "
        f"failed: {summary.failed}; skipped: {summary.skipped}"
    )
    for result in summary.results:
        print(f"  {result.task_id} [{result.model_id}]: {result.status}")
        if result.message:
            print(f"    Message: {result.message}")
        for report in result.quality_reports:
            print(f"    Quality: {report.status} — {Path(report.path).name}")
            if report.duration_seconds is not None:
                print(
                    f"      {report.duration_seconds:.3f}s, "
                    f"{report.sample_rate} Hz, {report.channels} channel(s)"
                )
            if report.issues:
                for issue in report.issues:
                    print(f"      Warning: {issue}")
