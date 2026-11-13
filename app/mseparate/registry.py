"""Validated, local model manifest registry.

The initial registry is intentionally empty: no model is enabled until its
checkpoint, configuration, output semantics and license have been reviewed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .types import ModelAsset, ModelManifest

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_LICENSE_STATES = {"verified", "review_required", "rejected"}
_MODEL_STATES = {"candidate", "available", "incompatible", "unavailable", "disabled"}


class RegistryError(ValueError):
    pass


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RegistryError(f"Manifest field {key!r} must be a non-empty string")
    return value


def parse_manifest(data: dict[str, Any]) -> ModelManifest:
    if not isinstance(data, dict):
        raise RegistryError("Each model manifest must be a JSON object")
    assets_data = data.get("assets", [])
    if not isinstance(assets_data, list):
        raise RegistryError("Manifest field 'assets' must be an array")
    assets: list[ModelAsset] = []
    for index, item in enumerate(assets_data):
        if not isinstance(item, dict):
            raise RegistryError(f"Asset #{index + 1} must be an object")
        role = _required_string(item, "role")
        url = _required_string(item, "url")
        relative_path = _required_string(item, "relative_path")
        if relative_path.startswith("/") or ".." in Path(relative_path).parts:
            raise RegistryError(f"Asset path must stay inside the model directory: {relative_path}")
        checksum = item.get("sha256")
        if checksum is not None and (not isinstance(checksum, str) or not _SHA256.fullmatch(checksum)):
            raise RegistryError(f"Asset {role!r} has an invalid SHA-256")
        size = item.get("size_bytes")
        if size is not None and (not isinstance(size, int) or size <= 0):
            raise RegistryError(f"Asset {role!r} has an invalid size_bytes")
        assets.append(ModelAsset(
            role=role, url=url, relative_path=relative_path,
            sha256=checksum, size_bytes=size,
            required=bool(item.get("required", True)),
        ))

    license_status = data.get("license_status", "review_required")
    status = data.get("status", "candidate")
    if license_status not in _LICENSE_STATES:
        raise RegistryError(f"Invalid license_status: {license_status!r}")
    if status not in _MODEL_STATES:
        raise RegistryError(f"Invalid model status: {status!r}")

    def strings(key: str) -> tuple[str, ...]:
        value = data.get(key, [])
        if not isinstance(value, list) or any(not isinstance(x, str) or not x for x in value):
            raise RegistryError(f"Manifest field {key!r} must be an array of non-empty strings")
        return tuple(value)

    sample_rate = data.get("sample_rate")
    if sample_rate is not None and (not isinstance(sample_rate, int) or sample_rate <= 0):
        raise RegistryError("sample_rate must be a positive integer or null")

    model_id = _required_string(data, "model_id")
    if not _MODEL_ID.fullmatch(model_id) or model_id in {".", ".."}:
        raise RegistryError(f"Invalid model_id: {model_id!r}")

    return ModelManifest(
        model_id=model_id,
        name=_required_string(data, "name"),
        architecture=_required_string(data, "architecture"),
        backend=_required_string(data, "backend"),
        source_url=_required_string(data, "source_url"),
        license=data.get("license"),
        license_url=data.get("license_url"),
        license_status=license_status,
        assets=tuple(assets),
        native_stems=strings("native_stems"),
        target_stems=strings("target_stems"),
        dummy_stems=strings("dummy_stems"),
        refinement_targets=strings("refinement_targets"),
        sample_rate=sample_rate,
        channels=data.get("channels"),
        status=status,
    )


def load_registry(path: str | Path) -> dict[str, ModelManifest]:
    registry_path = Path(path).expanduser()
    if not registry_path.exists():
        return {}
    try:
        raw = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(f"Cannot read registry {registry_path}: {exc}") from exc
    if not isinstance(raw, list):
        raise RegistryError("Registry root must be a JSON array")
    result: dict[str, ModelManifest] = {}
    for item in raw:
        manifest = parse_manifest(item)
        if manifest.model_id in result:
            raise RegistryError(f"Duplicate model_id: {manifest.model_id}")
        result[manifest.model_id] = manifest
    return result


def eligible_models(registry: dict[str, ModelManifest]) -> list[ModelManifest]:
    """Return only entries approved for automatic planning.

    Asset verification is a separate step; this function only enforces
    metadata and review gates.
    """
    return [
        model for model in registry.values()
        if model.status == "available"
        and model.license_status == "verified"
        and all((not asset.required) or bool(asset.sha256) for asset in model.assets)
        and bool(model.assets)
    ]
