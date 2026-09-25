from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from . import __version__
from . import subprocess_env
from . import gpuinfo
from . import modeldl
from .selftest import run_self_test
from .infoblock import print_info


def _venv_bin(name: str) -> str:
    """Locate a console script installed in the same venv as this wrapper,
    regardless of where that venv ends up mounted (build-time path
    rewriting already fixes shebangs; this is a second belt-and-braces
    lookup used when invoking it as a subprocess)."""
    candidate = Path(sys.executable).parent / name
    if candidate.exists():
        return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(
        f"Could not locate '{name}' inside the melband-roformer venv "
        f"({Path(sys.executable).parent}). The package install may be "
        f"broken; try `sudo apt install --reinstall melband-roformer`."
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="melband-roformer",
        description="Separate vocals/speech from music using Mel-Band "
                     "RoFormer (wraps upstream melband-roformer-infer).",
    )
    p.add_argument("inputs", nargs="*", help="Input audio file(s); shell "
                   "globs like ./input/*.wav are supported.")
    p.add_argument("--output-dir", "--store_dir", dest="output_dir",
                   default=".", help="Where to write *_vocals.wav / "
                   "*_instrumental.wav (default: current directory).")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                   help="auto (default): use CUDA if available, else CPU. "
                        "cuda: force GPU, fail loudly if unavailable. "
                        "cpu: force CPU.")
    p.add_argument("--model", default=None,
                   help="Registry model slug (default: "
                        "melband-roformer-kim-vocals). See --list-models.")
    p.add_argument("--config-path", "--config_path", dest="config_path",
                   default=None, help="Explicit model config YAML "
                   "(bypasses registry auto-resolution).")
    p.add_argument("--model-path", "--model_path", dest="model_path",
                   default=None, help="Explicit checkpoint path "
                   "(bypasses registry auto-resolution).")
    p.add_argument("--models-dir", "--models_dir", dest="models_dir",
                   default=None, help="Override model cache directory.")
    p.add_argument("--yes", "-y", action="store_true",
                   help="Assume 'yes' to the model-download confirmation.")

    p.add_argument("--info", action="store_true",
                   help="Print environment/GPU/model diagnostic info and exit.")
    p.add_argument("--self-test", action="store_true",
                   help="Run Python/PyTorch/CUDA/model/inference checks.")
    p.add_argument("--list-models", action="store_true",
                   help="List models in the upstream registry.")
    p.add_argument("--download-model", action="store_true",
                   help="Download (with confirmation + checksum check) and exit.")
    p.add_argument("--version", action="store_true", help="Print version and exit.")
    return p


class _InputsNotFound(Exception):
    """Raised by _expand_inputs when one or more input paths don't exist."""


def _expand_inputs(raw: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in raw:
        matches = glob.glob(item)
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            paths.append(Path(item))
    missing = [p for p in paths if not p.is_file()]
    if missing:
        joined = "\n  ".join(str(m) for m in missing)
        print(f"Input file(s) not found:\n  {joined}", file=sys.stderr)
        raise _InputsNotFound()
    return paths


def _run_separation(args: argparse.Namespace) -> int:
    if not args.inputs:
        print("No input file(s) given. Example:\n"
              "  melband-roformer input.wav\n"
              "  melband-roformer ./input/*.wav --output-dir ./output",
              file=sys.stderr)
        return 2

    try:
        inputs = _expand_inputs(args.inputs)
    except _InputsNotFound:
        return 2

    status = gpuinfo.detect()
    try:
        device = gpuinfo.resolve_device(args.device, status)
    except gpuinfo.GPURequiredError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Device: {status.device_name or ('CPU' if device == 'cpu' else device)}")
    print(f"CUDA: {'available' if status.cuda_available else 'not available'} "
          f"-> using {device}")

    # Resolve / offer to download the model unless explicit paths were given.
    if not (args.config_path and args.model_path):
        rc = modeldl.download_model(
            args.model or "melband-roformer-kim-vocals",
            assume_yes=args.yes,
            models_dir=args.models_dir,
        )
        if rc != 0:
            return rc

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = subprocess_env()
    if device == "cpu":
        # Guarantees CPU-only regardless of internal device-selection
        # logic upstream may use -- we do not rely on an unconfirmed
        # --device flag in the wrapped tool.
        env["CUDA_VISIBLE_DEVICES"] = ""
    if args.models_dir:
        env["MELBAND_ROFORMER_MODELS_PATH"] = args.models_dir

    exe = _venv_bin("melband-roformer-infer")

    # Upstream operates on a folder, not arbitrary file lists, so stage the
    # (possibly non-contiguous) input files into a temp folder of symlinks.
    with tempfile.TemporaryDirectory(prefix="melband-roformer-in-") as tmp:
        tmp_path = Path(tmp)
        for f in inputs:
            (tmp_path / f.name).symlink_to(f.resolve())

        cmd = [exe, "--input_folder", str(tmp_path), "--store_dir", str(out_dir)]
        if args.config_path and args.model_path:
            cmd += ["--config_path", args.config_path, "--model_path", args.model_path]
        elif args.model:
            cmd += ["--model", args.model]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        print("Separation failed (see output above). Run "
              "`melband-roformer --self-test` to check your environment.",
              file=sys.stderr)
        return result.returncode

    print(f"Done. Output written to {out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.version:
        print(f"melband-roformer {__version__}")
        return 0
    if args.list_models:
        return modeldl.list_models()
    if args.download_model:
        return modeldl.download_model(args.model or "melband-roformer-kim-vocals",
                                       assume_yes=args.yes, models_dir=args.models_dir)
    if args.info:
        print_info()
        return 0
    if args.self_test:
        return run_self_test()

    return _run_separation(args)


if __name__ == "__main__":
    sys.exit(main())
