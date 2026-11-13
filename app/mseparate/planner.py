"""Deterministic plan construction and validation."""
from __future__ import annotations

from .types import ExecutionPlan


def build_plan() -> ExecutionPlan:
    # No production model is enabled in this foundation slice. Returning an
    # explicit empty plan is safer than selecting an unreviewed checkpoint.
    return ExecutionPlan(
        schema_version=1,
        tasks=(),
        notes=("No verified model manifests are enabled; neural inference was not scheduled.",),
        max_depth=0,
    )


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
        known.add(task.task_id)
    if plan.tasks and max(task.depth for task in plan.tasks) > plan.max_depth:
        raise ValueError("Execution plan exceeds its declared max_depth")
