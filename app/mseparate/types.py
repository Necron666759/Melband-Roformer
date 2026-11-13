"""Typed, backend-neutral data contracts for mseparate."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AudioMetadata:
    path: str
    duration_seconds: float
    sample_rate: int
    channels: int
    frames: int
    subtype: str | None
    peak: float | None = None
    rms: float | None = None
    clipping_fraction: float | None = None


@dataclass(frozen=True)
class AnalysisReport:
    schema_version: int
    audio: AudioMetadata
    measurements: dict[str, Any] = field(default_factory=dict)
    predictions: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ModelAsset:
    role: str
    url: str
    relative_path: str
    sha256: str | None = None
    size_bytes: int | None = None
    required: bool = True


@dataclass(frozen=True)
class ModelManifest:
    model_id: str
    name: str
    architecture: str
    backend: str
    source_url: str
    license: str | None
    license_url: str | None
    license_status: str
    assets: tuple[ModelAsset, ...]
    native_stems: tuple[str, ...] = ()
    derived_stems: tuple[str, ...] = ()
    target_stems: tuple[str, ...] = ()
    dummy_stems: tuple[str, ...] = ()
    refinement_targets: tuple[str, ...] = ()
    sample_rate: int | None = None
    channels: str | None = None
    input_formats: tuple[str, ...] = ()
    category: str | None = None
    planner_priority: int = 0
    upstream_version: str | None = None
    notes: tuple[str, ...] = ()
    status: str = "candidate"


@dataclass(frozen=True)
class ExecutionTask:
    task_id: str
    model_id: str
    input_stem_id: str
    target_stems: tuple[str, ...]
    depends_on: tuple[str, ...] = ()
    depth: int = 0
    score: float = 0.0
    confidence: float = 0.0
    rationale: str = ""


@dataclass(frozen=True)
class ExecutionPlan:
    schema_version: int
    tasks: tuple[ExecutionTask, ...]
    notes: tuple[str, ...] = ()
    max_depth: int = 0


def to_jsonable(value: Any) -> Any:
    """Convert dataclasses and paths to JSON-compatible values."""
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return {k: to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_jsonable(v) for v in value]
    return value
