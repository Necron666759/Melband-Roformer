"""GPU / CUDA detection helpers.

Design goal: NEVER let a missing/broken GPU stack surface as a raw Python
traceback to the end user. Every function here degrades to a clear,
human-readable status instead of raising, except where the caller has
explicitly asked to force GPU (`--device cuda`), in which case we raise a
`GPURequiredError` with an actionable message -- the caller decides whether
that's a hard failure or a fallback.
"""
from __future__ import annotations

import dataclasses
import shutil
import subprocess

# Single source of truth for "which CUDA wheel channel does this build use,
# and what NVIDIA driver does that channel require". debian/rules'
# TORCH_INDEX, app/requirements.txt and app/pyproject.toml all pin the
# `cu126` wheel channel; if that ever changes, update the two constants
# below *first* -- everything in this module (and infoblock.py's --info
# warning) derives from them, so there is exactly one number to change
# instead of three that can silently drift out of sync (see debian/changelog
# 0.1.2/0.1.4, where the pins moved cu121 -> cu124 -> torch/torchaudio 2.6.0,
# and 0.1.6, where they moved cu124 -> cu126 -> torch/torchaudio 2.11.0).
TORCH_CUDA_CHANNEL = "cu126"
MIN_DRIVER_VERSION = (560, 28)  # (major, minor) of 560.28.03
MIN_DRIVER_VERSION_STR = "560.28.03"


class GPURequiredError(RuntimeError):
    """Raised when the user forced --device cuda but CUDA is unavailable."""


@dataclasses.dataclass
class GpuStatus:
    nvidia_smi_present: bool
    nvidia_smi_output: str | None
    torch_importable: bool
    torch_version: str | None
    torch_cuda_runtime: str | None  # torch.version.cuda
    cuda_available: bool
    device_name: str | None
    device_total_mem_mib: int | None
    device_free_mem_mib: int | None
    driver_floor_ok: bool | None  # None = unknown (couldn't check)
    error: str | None = None


def _run_nvidia_smi() -> tuple[bool, str | None]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return False, None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=driver_version,name,memory.total,memory.free",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if out.returncode != 0:
            return True, out.stderr.strip() or "nvidia-smi failed"
        return True, out.stdout.strip()
    except Exception as exc:  # pragma: no cover - defensive
        return True, f"nvidia-smi error: {exc}"


def _min_driver_ok(smi_output: str | None) -> bool | None:
    """cu126 wheels (see TORCH_CUDA_CHANNEL) need driver >=
    MIN_DRIVER_VERSION_STR. Best-effort parse; returns None (unknown)
    rather than guessing if the format is unexpected."""
    if not smi_output:
        return None
    try:
        first_line = smi_output.splitlines()[0]
        driver_str = first_line.split(",")[0].strip()
        major, minor = (int(x) for x in driver_str.split(".")[:2])
        return (major, minor) >= MIN_DRIVER_VERSION
    except Exception:
        return None


def detect() -> GpuStatus:
    nvidia_smi_present, smi_output = _run_nvidia_smi()

    torch_importable = False
    torch_version = None
    torch_cuda_runtime = None
    cuda_available = False
    device_name = None
    total_mib = None
    free_mib = None
    error = None

    try:
        import torch  # noqa: WPS433 (intentional local import)
        torch_importable = True
        torch_version = torch.__version__
        torch_cuda_runtime = torch.version.cuda
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            idx = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(idx)
            free_b, total_b = torch.cuda.mem_get_info(idx)
            total_mib = total_b // (1024 * 1024)
            free_mib = free_b // (1024 * 1024)
    except Exception as exc:  # pragma: no cover - defensive
        error = f"{type(exc).__name__}: {exc}"

    return GpuStatus(
        nvidia_smi_present=nvidia_smi_present,
        nvidia_smi_output=smi_output,
        torch_importable=torch_importable,
        torch_version=torch_version,
        torch_cuda_runtime=torch_cuda_runtime,
        cuda_available=cuda_available,
        device_name=device_name,
        device_total_mem_mib=total_mib,
        device_free_mem_mib=free_mib,
        driver_floor_ok=_min_driver_ok(smi_output),
        error=error,
    )


def resolve_device(requested: str, status: GpuStatus | None = None) -> str:
    """requested in {"auto", "cuda", "cpu"}. Returns "cuda" or "cpu".

    - "cpu": always honored.
    - "cuda": honored if available, else raises GPURequiredError (the user
      explicitly asked for GPU; silently downgrading would hide a real
      problem on their system).
    - "auto": cuda if available, else cpu, with a printed notice.
    """
    status = status or detect()
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not status.cuda_available:
            raise GPURequiredError(
                "CUDA was explicitly requested (--device cuda) but is not "
                "available.\n"
                f"  nvidia-smi present: {status.nvidia_smi_present}\n"
                f"  torch importable:   {status.torch_importable}\n"
                f"  torch CUDA build:   {status.torch_cuda_runtime}\n"
                "Run `melband-roformer --info` for full details, or drop "
                "--device to fall back to CPU automatically."
            )
        return "cuda"
    # auto
    if status.cuda_available:
        return "cuda"
    print("CUDA is not available. Falling back to CPU.")
    print("Run `melband-roformer --info` for details.")
    return "cpu"
