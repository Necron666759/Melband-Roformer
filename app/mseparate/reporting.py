"""Human-readable rendering for analysis and plan reports."""
from __future__ import annotations

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
