"""Backend lifecycle and strictly sequential model execution.

Adapters own model-specific loading/inference. The manager owns ordering,
failure isolation, artifact validation, and the one-model-at-a-time invariant.
"""
from __future__ import annotations

import fcntl
import threading
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Protocol

import soundfile as sf
import soxr

from .quality import StemQualityReport, inspect_stem, inspect_stem_set
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
    quality_reports: tuple[StemQualityReport, ...] = ()
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


# The thread lock serializes manager instances in this process; flock also
# serializes independent mseparate processes sharing the same model store.
_EXECUTION_LOCK = threading.Lock()


class _SystemExecutionLock:
    def __init__(self, path: Path):
        self.path = path
        self._stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a+b")
        fcntl.flock(self._stream.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._stream is not None:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
            self._stream.close()


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

        # Keep one per-user lock regardless of --models-dir so jobs using
        # different model caches still obey the global sequential invariant.
        lock_path = Path.home() / ".local" / "share" / "mseparate" / ".execution.lock"
        with _EXECUTION_LOCK, _SystemExecutionLock(lock_path):
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
                quality_reports: tuple[StemQualityReport, ...] = ()
                error: str | None = None
                load_attempted = False
                try:
                    compatible, reason = backend.compatibility(manifest)
                    if not compatible:
                        raise RuntimeError(reason or "Backend reports model incompatible")
                    if not input_path.is_file():
                        raise FileNotFoundError(f"Input stem does not exist: {input_path}")
                    declared_outputs = set(manifest.native_stems) | set(manifest.derived_stems)
                    unsupported = set(task.target_stems) - declared_outputs
                    if unsupported:
                        raise RuntimeError(
                            "Task requests stems not declared as native or derived by the manifest: "
                            + ", ".join(sorted(unsupported))
                        )
                    model_dir = self.models_root / manifest.model_id
                    load_attempted = True
                    backend.load(manifest, model_dir, self.device)
                    task_output_dir = self.output_root / task.task_id
                    task_output_dir.mkdir(parents=True, exist_ok=True)
                    source_info = sf.info(str(input_path))
                    target_rate = manifest.sample_rate
                    resampled_input = bool(target_rate and source_info.samplerate != target_rate)
                    if resampled_input:
                        # Convert only the model-facing working copy. The
                        # source stays untouched and output stems keep the
                        # model's native sample rate.
                        with tempfile.TemporaryDirectory(prefix="mseparate-rate-") as temp:
                            model_input = Path(temp) / "input.wav"
                            audio, source_rate = sf.read(str(input_path), dtype="float32", always_2d=True)
                            converted = soxr.resample(audio, source_rate, target_rate)
                            sf.write(str(model_input), converted, target_rate, subtype="PCM_24")
                            produced = backend.infer(model_input, task_output_dir, task.target_stems)
                    else:
                        produced = backend.infer(input_path, task_output_dir, task.target_stems)
                    if resampled_input:
                        for artifact in produced:
                            output_rate = sf.info(str(artifact.path)).samplerate
                            if output_rate != target_rate:
                                raise RuntimeError(
                                    f"Model output {artifact.path.name} is {output_rate} Hz; "
                                    f"expected model rate {target_rate} Hz"
                                )
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
                    # Validate every decoded artifact against the exact input
                    # supplied to this task. Warnings (e.g. silence/clipping)
                    # remain visible; structural invalidity fails this task.
                    reports = tuple(
                        inspect_stem(
                            artifact.path,
                            reference=input_path,
                            allow_sample_rate_mismatch=resampled_input,
                        )
                        for artifact in validated
                    )
                    reports = inspect_stem_set(
                        tuple((artifact.stem, artifact.path) for artifact in validated),
                        reports,
                    )
                    quality_reports = reports
                    invalid = [
                        report for report in reports if report.status == "invalid"
                    ]
                    if invalid:
                        details = "; ".join(
                            f"{Path(report.path).name}: {', '.join(report.issues)}"
                            for report in invalid
                        )
                        raise RuntimeError("Output quality validation failed: " + details)
                    outputs = tuple(validated)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                finally:
                    if load_attempted:
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
                    quality_reports=quality_reports,
                    message=error,
                )
                results.append(result)
                states[task.task_id] = result.status

        return ExecutionSummary(tuple(results))
