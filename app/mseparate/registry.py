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
_RUNTIME_VALIDATION_STATES = {"unverified", "verified", "failed"}
_LICENSE_REVIEW_STATES = {"approved", "accepted_by_maintainer", "blocked", "rejected"}


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
    license_review = data.get("license_review")
    if license_review is not None:
        if not isinstance(license_review, dict):
            raise RegistryError("license_review must be an object or null")
        if license_review.get("status") not in _LICENSE_REVIEW_STATES:
            raise RegistryError(f"Invalid license_review.status: {license_review.get('status')!r}")

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
    runtime_validation = data.get("runtime_validation", "unverified")
    if runtime_validation not in _RUNTIME_VALIDATION_STATES:
        raise RegistryError(f"Invalid runtime_validation: {runtime_validation!r}")
    planner_priority = data.get("planner_priority", 0)
    if not isinstance(planner_priority, int) or isinstance(planner_priority, bool) or not -1000 <= planner_priority <= 1000:
        raise RegistryError("planner_priority must be an integer between -1000 and 1000")
    auto_route = data.get("auto_route", True)
    if not isinstance(auto_route, bool):
        raise RegistryError("auto_route must be a boolean")
    channels = data.get("channels")
    if channels not in _CHANNEL_POLICIES:
        raise RegistryError(f"channels must be one of {sorted(x for x in _CHANNEL_POLICIES if x)} or null")
    cpu_inference = data.get("cpu_inference", False)
    if not isinstance(cpu_inference, bool):
        raise RegistryError("cpu_inference must be a boolean")
    min_vram_gb = data.get("min_vram_gb")
    min_free_vram_gb = data.get("min_free_vram_gb")
    for key, value in (("min_vram_gb", min_vram_gb), ("min_free_vram_gb", min_free_vram_gb)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0):
            raise RegistryError(f"{key} must be a positive number or null")
    inference_params = data.get("inference_params", {})
    if not isinstance(inference_params, dict) or any(not isinstance(key, str) for key in inference_params):
        raise RegistryError("inference_params must be an object with string keys")
    low_vram_profile = data.get("low_vram_profile")
    if low_vram_profile is not None:
        if not isinstance(low_vram_profile, dict):
            raise RegistryError("low_vram_profile must be an object or null")
        if low_vram_profile.get("status") not in {"unverified", "verified", "disabled"}:
            raise RegistryError("low_vram_profile.status must be unverified, verified or disabled")
        if not isinstance(low_vram_profile.get("inference_params", {}), dict):
            raise RegistryError("low_vram_profile.inference_params must be an object")
        for key in ("min_total_vram_gb", "min_free_vram_gb"):
            value = low_vram_profile.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0):
                raise RegistryError(f"low_vram_profile.{key} must be a positive number")
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
        category=data.get("category"), planner_priority=planner_priority,
        auto_route=auto_route, runtime_validation=runtime_validation, license_review=license_review,
        cpu_inference=cpu_inference,
        min_vram_gb=float(min_vram_gb) if min_vram_gb is not None else None,
        min_free_vram_gb=float(min_free_vram_gb) if min_free_vram_gb is not None else None,
        inference_params=dict(inference_params), low_vram_profile=low_vram_profile,
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


def preparable_models(registry: dict[str, ModelManifest]) -> list[ModelManifest]:
    """Return license-cleared models with pinned assets, even before runtime smoke validation."""
    result = []
    for model in registry.values():
        if model.status != "available" or model.license_status != "verified":
            continue
        if model.license_review is not None and model.license_review.get("status") not in {"approved", "accepted_by_maintainer"}:
            continue
        if not all((not asset.required) or bool(asset.sha256) for asset in model.assets):
            continue
        if not model.assets and model.backend != "demucs-onnx":
            continue
        result.append(model)
    return result


def eligible_models(registry: dict[str, ModelManifest]) -> list[ModelManifest]:
    """Return models that pass independent runtime, license and asset gates.

    Non-vocal BS-RoFormer specialists require an explicit successful real-model
    runtime smoke test in addition to the normal availability and license gates.
    """
    eligible = []
    for model in preparable_models(registry):
        category = (model.category or "").strip().lower().replace("_", "-")
        needs_runtime_validation = (
            model.backend == "bs-roformer-pymss"
            and category in {"synth-specialist", "keyboard-specialist", "mega-multi-stem"}
        )
        if needs_runtime_validation and model.runtime_validation != "verified":
            continue
        eligible.append(model)
    return eligible


def model_audio_compatibility(model: ModelManifest, audio: AudioMetadata) -> tuple[bool, str]:
    """Check hard input constraints before acoustic ranking or inference."""
    if model.input_formats:
        extension = Path(audio.path).suffix.lower().lstrip(".")
        if extension not in model.input_formats:
            allowed = ", ".join(model.input_formats)
            actual = extension or "(no extension)"
            return False, f"requires {allowed} input; got .{actual}"
    resampling_backends = {"demucs-onnx", "bs-roformer-pymss", "mdx23c-pymss"}
    if (model.sample_rate is not None and audio.sample_rate != model.sample_rate
            and model.backend not in resampling_backends):
        return False, f"requires {model.sample_rate} Hz input; got {audio.sample_rate} Hz (resampling is not supported by this backend)"
    if model.channels == "stereo" and audio.channels != 2:
        return False, f"requires stereo input; got {audio.channels} channel(s)"
    if model.channels == "mono" and audio.channels != 1:
        return False, f"requires mono input; got {audio.channels} channel(s)"
    if model.channels == "mono_or_stereo" and audio.channels not in (1, 2):
        return False, f"supports mono or stereo input; got {audio.channels} channel(s)"
    return True, ""
