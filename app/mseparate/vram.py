"""Conservative preflight for models with explicit GPU-memory requirements."""
from __future__ import annotations

from dataclasses import replace
from typing import Callable

from .types import ModelManifest


_GIB = 1024 ** 3


def cuda_memory_info() -> tuple[int, int] | None:
    """Return (currently free bytes, physical device bytes) for CUDA device 0."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    free_bytes, _runtime_total = torch.cuda.mem_get_info(0)
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    return int(free_bytes), int(total_bytes)


def filter_registry_for_device(
    registry: dict[str, ModelManifest],
    device: str,
    *,
    memory_probe: Callable[[], tuple[int, int] | None] | None = None,
) -> tuple[dict[str, ModelManifest], tuple[str, ...]]:
    """Apply CUDA memory admission while keeping verified CPU runtimes eligible.

    A verified low-VRAM profile may replace the normal CUDA profile only when both
    its total-capacity and currently-free-memory thresholds are satisfied.
    Unverified profiles are deliberately ignored.
    """
    probe = memory_probe or cuda_memory_info
    memory: tuple[int, int] | None = None
    probed = False
    accepted: dict[str, ModelManifest] = {}
    notes: list[str] = []

    for model in registry.values():
        if model.min_vram_gb is None or device == "cpu":
            # min_vram_gb describes CUDA admission only. A verified CPU runtime
            # remains eligible; throughput is a performance concern, not a gate.
            accepted[model.model_id] = model
            continue
        if device != "cuda":
            notes.append(
                f"{model.model_id} skipped: requires CUDA with at least "
                f"{model.min_vram_gb:g} GiB VRAM; selected device is {device!r}. "
                "No CPU fallback is permitted for this model."
            )
            continue
        if not probed:
            try:
                memory = probe()
            except Exception as exc:
                notes.append(
                    f"GPU-gated models skipped: CUDA VRAM probe failed "
                    f"({type(exc).__name__}: {exc})."
                )
            probed = True
        if memory is None:
            notes.append(
                f"{model.model_id} skipped: CUDA VRAM availability could not be verified."
            )
            continue

        free_bytes, total_bytes = memory
        free_gib = free_bytes / _GIB
        total_gib = total_bytes / _GIB
        required_total = float(model.min_vram_gb)
        required_free = float(model.min_free_vram_gb or model.min_vram_gb)
        # Prefer the upstream/default inference profile whenever it fits.
        # A verified reduced-chunk profile is a fallback, never an upgrade.
        if total_gib >= required_total and free_gib >= required_free:
            accepted[model.model_id] = model
            continue

        profile = model.low_vram_profile or {}
        profile_status = profile.get("status")
        profile_total = float(profile.get("min_total_vram_gb") or 0)
        profile_free = float(profile.get("min_free_vram_gb") or 0)
        if (
            profile_status == "verified"
            and profile_total > 0
            and profile_free > 0
            and total_gib >= profile_total
            and free_gib >= profile_free
        ):
            params = dict(profile.get("inference_params") or {})
            accepted[model.model_id] = replace(model, inference_params=params)
            warning = profile.get("warning") or "Reduced chunk profile may lower throughput."
            notes.append(
                f"{model.model_id} admitted using its verified low-VRAM profile "
                f"({free_gib:.2f} GiB free / {total_gib:.2f} GiB total): {warning}"
            )
            continue

        notes.append(
            f"{model.model_id} skipped before download/inference: "
            f"{free_gib:.2f} GiB free / {total_gib:.2f} GiB total; "
            f"requires at least {required_free:g} GiB free and "
            f"{required_total:g} GiB total VRAM. "
            "No CPU fallback is permitted for this model."
        )

    return accepted, tuple(notes)
