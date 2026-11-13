"""Backend lifecycle and strictly sequential model execution.

Adapters own model-specific loading/inference. The manager owns ordering,
failure isolation, artifact validation, and the one-model-at-a-time invariant.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Protocol

from .types import ExecutionPlan, ExecutionTask, ModelManifest


@dataclass(frozen=True)
class BackendInfo:
    name: str
    version: str | None = None
    device: str | None = None
    details: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class StemArtifact:
    stem: str
    path: Path


class BackendAdapter(Protocol):
    """Contract implemented by each model-family integration."""

    def compatibility(self, manifest: ModelManifest) -> tuple[bool, str]:
        ...

    def load(self, manifest: ModelManifest, model_dir: Path, device: str) -> None:
        ...

    def infer(self, input_audio: Path, output_dir: Path,
              target_stems: tuple[str, ...]) -> list[StemArtifact]:
        ...

    def unload(self) -> None:
        ...

    def info(self) -> BackendInfo:
        ...


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    model_id: str
    status: str
    outputs: tuple[StemArtifact, ...] = ()
    message: str | None = None


@dataclass(frozen=True)
class ExecutionSummary:
    results: tuple[TaskResult, ...]

    @property
    def succeeded(self) -> int:
        return sum(item.status == "success" for item in self.results)

    @property
    def failed(self) -> int:
        return sum(item.status == "failed" for item in self.results)

    @property
    def skipped(self) -> int:
        return sum(item.status == "skipped" for item in self.results)


# Serializes execution across manager instances in this Python process. A
# subprocess adapter also runs one inference process at a time by construction.
_EXECUTION_LOCK = threading.Lock()


class ExecutionManager:
    """Execute a validated plan without overlapping model residency/inference."""

    def __init__(
        self,
        backend_factories: Mapping[str, Callable[[], BackendAdapter]],
        *,
        models_root: str | Path,
        output_root: str | Path,
        device: str = "auto",
        resource_check: Callable[[], None] | None = None,
    ) -> None:
        self.backend_factories = dict(backend_factories)
        self.models_root = Path(models_root).expanduser()
        self.output_root = Path(output_root).expanduser()
        self.device = device
        self.resource_check = resource_check

    def execute(
        self,
        plan: ExecutionPlan,
        manifests: Mapping[str, ModelManifest],
        input_stems: Mapping[str, Path],
    ) -> ExecutionSummary:
        from .planner import validate_plan

        validate_plan(plan)
        self.output_root.mkdir(parents=True, exist_ok=True)
        results: list[TaskResult] = []
        states: dict[str, str] = {}

        with _EXECUTION_LOCK:
            for task in plan.tasks:
                failed_dependencies = [
                    dependency for dependency in task.depends_on
                    if states.get(dependency) != "success"
                ]
                if failed_dependencies:
                    result = TaskResult(
                        task.task_id, task.model_id, "skipped",
                        message="Dependencies did not succeed: " + ", ".join(failed_dependencies),
                    )
                    results.append(result)
                    states[task.task_id] = result.status
                    continue

                manifest = manifests.get(task.model_id)
                input_path = input_stems.get(task.input_stem_id)
                if manifest is None or input_path is None:
                    result = TaskResult(
                        task.task_id, task.model_id, "failed",
                        message="Task references an unknown model manifest or input stem",
                    )
                    results.append(result)
                    states[task.task_id] = result.status
                    continue

                factory = self.backend_factories.get(manifest.backend)
                if factory is None:
                    result = TaskResult(
                        task.task_id, task.model_id, "failed",
                        message=f"No adapter registered for backend {manifest.backend!r}",
                    )
                    results.append(result)
                    states[task.task_id] = result.status
                    continue

                backend = factory()
                outputs: tuple[StemArtifact, ...] = ()
                error: str | None = None
                loaded = False
                try:
                    compatible, reason = backend.compatibility(manifest)
                    if not compatible:
                        raise RuntimeError(reason or "Backend reports model incompatible")
                    if not input_path.is_file():
                        raise FileNotFoundError(f"Input stem does not exist: {input_path}")
                    model_dir = self.models_root / manifest.model_id
                    backend.load(manifest, model_dir, self.device)
                    loaded = True
                    task_output_dir = self.output_root / task.task_id
                    task_output_dir.mkdir(parents=True, exist_ok=True)
                    produced = backend.infer(input_path, task_output_dir, task.target_stems)
                    seen: set[str] = set()
                    validated: list[StemArtifact] = []
                    for artifact in produced:
                        if artifact.stem not in task.target_stems:
                            raise RuntimeError(f"Backend returned undeclared stem {artifact.stem!r}")
                        if artifact.stem in seen:
                            raise RuntimeError(f"Backend returned duplicate stem {artifact.stem!r}")
                        if not artifact.path.is_file():
                            raise RuntimeError(f"Backend output is missing: {artifact.path}")
                        seen.add(artifact.stem)
                        validated.append(artifact)
                    missing = set(task.target_stems) - seen
                    if missing:
                        raise RuntimeError(f"Backend did not produce requested stems: {sorted(missing)}")
                    outputs = tuple(validated)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                finally:
                    if loaded:
                        try:
                            backend.unload()
                        except Exception as exc:
                            suffix = f"Backend unload failed: {type(exc).__name__}: {exc}"
                            error = f"{error}; {suffix}" if error else suffix
                    # The adapter instance and its references go out of scope
                    # before resource checks and before the next model loads.
                    del backend
                    if self.resource_check is not None:
                        try:
                            self.resource_check()
                        except Exception as exc:
                            suffix = f"Resource check failed: {type(exc).__name__}: {exc}"
                            error = f"{error}; {suffix}" if error else suffix

                result = TaskResult(
                    task.task_id, task.model_id,
                    "failed" if error else "success",
                    outputs=() if error else outputs,
                    message=error,
                )
                results.append(result)
                states[task.task_id] = result.status

        return ExecutionSummary(tuple(results))
