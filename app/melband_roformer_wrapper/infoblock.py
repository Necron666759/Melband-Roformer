from __future__ import annotations

import platform
import sys

from . import __version__
from . import gpuinfo


def print_info() -> None:
    status = gpuinfo.detect()

    print(f"Application version: {__version__}")
    print(f"Python:               {platform.python_version()} "
          f"(venv: {sys.prefix})")
    print(f"PyTorch:              {status.torch_version or 'not importable'}")
    print(f"CUDA runtime (torch): {status.torch_cuda_runtime or 'n/a'}")
    print(f"CUDA available:       {'yes' if status.cuda_available else 'no'}")
    if status.nvidia_smi_present:
        print(f"nvidia-smi:           present")
        if status.nvidia_smi_output:
            print(f"  {status.nvidia_smi_output.splitlines()[0]}")
    else:
        print("nvidia-smi:           not found on PATH")
    if status.driver_floor_ok is False:
        print(f"  WARNING: installed NVIDIA driver appears older than the "
              f"{gpuinfo.MIN_DRIVER_VERSION_STR} floor required by the "
              f"bundled CUDA ({gpuinfo.TORCH_CUDA_CHANNEL}) PyTorch build. "
              f"GPU inference may fail even though a GPU is present.")
    print(f"GPU:                  {status.device_name or 'n/a'}")
    if status.device_total_mem_mib:
        print(f"GPU memory:           {status.device_total_mem_mib} MiB total, "
              f"{status.device_free_mem_mib} MiB free")
    print(f"Selected device:      {'cuda:0' if status.cuda_available else 'cpu'}")

    try:
        from mel_band_roformer import DEFAULT_MODEL
        print(f"Model (default):      {DEFAULT_MODEL}")
    except Exception as exc:
        print(f"Model (default):      unavailable ({exc})")

    if status.error:
        print(f"\nNote: torch import raised an error internally: {status.error}")
