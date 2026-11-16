"""Explicit, opt-in HT-Demucs 6-stem command.

This deliberately slow model is not part of the normal MSeparate planner.
Users must invoke this separate command to run it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .demucs_onnx import DemucsOnnxAdapter
from .execution import ExecutionManager
from .registry import bundled_registry_path, load_registry
from .storage import ensure_layout
from .types import ExecutionPlan, ExecutionTask


MODEL_ID = "demucs-onnx-htdemucs-6s"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mseparate-demucs",
        description=(
            "Explicitly run the slow HT-Demucs 6-stem ONNX model. "
            "This model is never selected by normal automatic routing."
        ),
    )
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("--output-dir", default="./mseparate-demucs-stems",
                        help="Output directory (default: ./mseparate-demucs-stems)")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                        help="Inference device; defaults to strict CUDA with no CPU fallback")
    parser.add_argument("--models-dir", default=None,
                        help="Override persistent model storage directory")
    parser.add_argument("--no-download", action="store_true",
                        help="Require the ONNX model to be available in the local Hugging Face cache")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = Path(args.input).expanduser().resolve()
    if not source.is_file():
        print(f"mseparate-demucs: input file does not exist: {source}", file=sys.stderr)
        return 2

    try:
        paths = ensure_layout()
        if args.models_dir:
            paths["models"] = Path(args.models_dir).expanduser()
            paths["models"].mkdir(parents=True, exist_ok=True)
        registry = load_registry(bundled_registry_path())
        manifest = registry.get(MODEL_ID)
        if manifest is None or manifest.status != "available":
            raise RuntimeError("The reviewed HT-Demucs 6-stem manifest is unavailable")

        print("Explicit HT-Demucs 6-stem run")
        print("  This model is excluded from normal MSeparate routing.")
        print(f"  device: {args.device} (no automatic CPU fallback)")
        print(f"  input: {source}")
        print(f"  output: {Path(args.output_dir).expanduser()}")

        plan = ExecutionPlan(
            schema_version=1,
            tasks=(ExecutionTask(
                task_id="htdemucs-6stem-manual",
                model_id=MODEL_ID,
                input_stem_id="mix",
                target_stems=tuple(manifest.native_stems),
                depth=0,
                rationale="Explicit user invocation of mseparate-demucs",
            ),),
            notes=("Manual HT-Demucs invocation; not part of automatic routing.",),
            max_depth=0,
        )
        manager = ExecutionManager(
            {"demucs-onnx": lambda: DemucsOnnxAdapter(no_download=args.no_download)},
            models_root=paths["models"],
            output_root=Path(args.output_dir).expanduser(),
            device=args.device,
        )
        summary = manager.execute(plan, registry, {"mix": source})
        for result in summary.results:
            print(f"  {result.status}: {result.task_id}")
            if result.message:
                print(f"    {result.message}")
            for artifact in result.outputs:
                print(f"    {artifact.stem}: {artifact.path}")
            for report in result.quality_reports:
                if report.issues:
                    print(f"    quality warning ({Path(report.path).name}): "
                          + "; ".join(report.issues))
        if summary.failed or not summary.succeeded:
            return 1
        print("HT-Demucs separation completed.")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"mseparate-demucs: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
