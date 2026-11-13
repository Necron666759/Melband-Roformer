"""Validated local and bundled model manifest registry."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .types import AudioMetadata, ModelAsset, ModelManifest

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_LICENSE_STATES = {"verified", "review_required", "rejected"}
_MODEL_STATES = {"candidate", "available", "incompatible", "unavailable", "disabled"}
_CHANNEL_POLICIES = {"mono", "stereo", "mono_or_stereo", None}


class RegistryError(ValueError):
    pass


def bundled_registry_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "models.json"


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
    roles: set[str] = set()
    for index, item in enumerate(assets_data):
        if not isinstance(item, dict):
            raise RegistryError(f"Asset #{index + 1} must be an object")
        role = _required_string(item, "role")
        if role in roles:
            raise RegistryError(f"Duplicate asset role: {role!r}")
        roles.add(role)
        url = _required_string(item, "url")
        relative_path = _required_string(item, "relative_path")
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise RegistryError(f"Asset path must stay inside the model directory: {relative_path}")
        checksum = item.get("sha256")
        if checksum is not None and (not isinstance(checksum, str) or not _SHA256.fullmatch(checksum)):
            raise RegistryError(f"Asset {role!r} has an invalid SHA-256")
        size = item.get("size_bytes")
        if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size <= 0):
            raise RegistryError(f"Asset {role!r} has an invalid size_bytes")
        required = item.get("required", True)
        if not isinstance(required, bool):
            raise RegistryError(f"Asset {role!r} field required must be boolean")
        assets.append(ModelAsset(
            role=role, url=url, relative_path=relative_path,
            sha256=checksum, size_bytes=size, required=required,
        ))

    license_value = data.get("license")
    license_url = data.get("license_url")
    if license_value is not None and not isinstance(license_value, str):
        raise RegistryError("license must be a string or null")
    if license_url is not None and not isinstance(license_url, str):
        raise RegistryError("license_url must be a string or null")
    license_status = data.get("license_status", "review_required")
    status = data.get("status", "candidate")
    if license_status not in _LICENSE_STATES:
        raise RegistryError(f"Invalid license_status: {license_status!r}")
    if status not in _MODEL_STATES:
        raise RegistryError(f"Invalid model status: {status!r}")

    def strings(key: str) -> tuple[str, ...]:
        value = data.get(key, [])
        if not isinstance(value, list) or any(not isinstance(x, str) or not x.strip() for x in value):
            raise RegistryError(f"Manifest field {key!r} must be an array of non-empty strings")
        if len(value) != len(set(value)):
            raise RegistryError(f"Manifest field {key!r} must not contain duplicates")
        return tuple(value)

    sample_rate = data.get("sample_rate")
    if sample_rate is not None and (not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0):
        raise RegistryError("sample_rate must be a positive integer or null")
    channels = data.get("channels")
    if channels not in _CHANNEL_POLICIES:
        raise RegistryError(f"channels must be one of {sorted(x for x in _CHANNEL_POLICIES if x)} or null")
    for key in ("category", "upstream_version"):
        if data.get(key) is not None and (not isinstance(data[key], str) or not data[key].strip()):
            raise RegistryError(f"{key} must be a non-empty string or null")

    model_id = _required_string(data, "model_id")
    if not _MODEL_ID.fullmatch(model_id) or model_id in {".", ".."}:
        raise RegistryError(f"Invalid model_id: {model_id!r}")

    notes = strings("notes")
    return ModelManifest(
        model_id=model_id, name=_required_string(data, "name"),
        architecture=_required_string(data, "architecture"),
        backend=_required_string(data, "backend"),
        source_url=_required_string(data, "source_url"),
        license=license_value, license_url=license_url,
        license_status=license_status, assets=tuple(assets),
        native_stems=strings("native_stems"), derived_stems=strings("derived_stems"),
        target_stems=strings("target_stems"), dummy_stems=strings("dummy_stems"),
        refinement_targets=strings("refinement_targets"), sample_rate=sample_rate,
        channels=channels, input_formats=tuple(x.lower().lstrip(".") for x in strings("input_formats")),
        category=data.get("category"),
        upstream_version=data.get("upstream_version"), notes=notes, status=status,
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
    """Return only models that pass metadata, license and pinned-asset gates."""
    return [
        model for model in registry.values()
        if model.status == "available"
        and model.license_status == "verified"
        and all((not asset.required) or bool(asset.sha256) for asset in model.assets)
        and (bool(model.assets) or model.backend == "demucs-onnx")
    ]


def model_audio_compatibility(model: ModelManifest, audio: AudioMetadata) -> tuple[bool, str]:
    """Check hard input constraints before acoustic ranking or inference."""
    if model.input_formats:
        extension = Path(audio.path).suffix.lower().lstrip(".")
        if extension not in model.input_formats:
            allowed = ", ".join(model.input_formats)
            actual = extension or "(no extension)"
            return False, f"requires {allowed} input; got .{actual}"
    if (model.sample_rate is not None and audio.sample_rate != model.sample_rate
            and model.backend != "demucs-onnx"):
        return False, f"requires {model.sample_rate} Hz input; got {audio.sample_rate} Hz (resampling is not supported by this backend)"
    if model.channels == "stereo" and audio.channels != 2:
        return False, f"requires stereo input; got {audio.channels} channel(s)"
    if model.channels == "mono" and audio.channels != 1:
        return False, f"requires mono input; got {audio.channels} channel(s)"
    if model.channels == "mono_or_stereo" and audio.channels not in (1, 2):
        return False, f"supports mono or stereo input; got {audio.channels} channel(s)"
    return True, ""
