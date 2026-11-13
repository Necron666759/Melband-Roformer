"""Independent mseparate command line interface."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .audio import analyze_audio
from .planner import build_plan, validate_plan
from .registry import RegistryError, eligible_models, load_registry
from .reporting import print_analysis, print_plan
from .storage import data_root, ensure_layout, write_json_atomic
from .types import to_jsonable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mseparate",
        description="Analyze a music track and prepare a safe multi-model separation plan.",
    )
    parser.add_argument("input", nargs="?", help="Input audio file")
    parser.add_argument("--output-dir", default=".", help="Directory for final stems and reports")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto",
                        help="Target compute device (inference is not enabled in this foundation release)")
    parser.add_argument("--models-dir", default=None, help="Override persistent model storage directory")
    parser.add_argument("--analysis-only", action="store_true", help="Analyze audio without planning inference")
    parser.add_argument("--plan-only", action="store_true", help="Analyze audio and print the execution plan")
    parser.add_argument("--no-download", action="store_true", help="Do not access the network")
    parser.add_argument("--force-download", action="store_true", help="Reserved for a future asset-manager release")
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum refinement depth for future plans")
    parser.add_argument("--verbose", action="store_true", help="Print additional diagnostics")
    parser.add_argument("--report", nargs="?", const="mseparate-report.json",
                        help="Write a JSON report (optional path; default: mseparate-report.json)")
    parser.add_argument("--version", action="version", version=f"mseparate {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.input:
        parser.print_help(sys.stderr)
        return 2
    if args.max_depth < 0:
        parser.error("--max-depth must be zero or greater")
    if args.force_download and args.no_download:
        parser.error("--force-download cannot be combined with --no-download")

    try:
        report = analyze_audio(args.input)
        paths = ensure_layout()
        registry_path = paths["registry"] / "models.json"
        registry = load_registry(registry_path)
        plan = build_plan()
        validate_plan(plan)
    except (OSError, ValueError, RegistryError) as exc:
        print(f"mseparate: {exc}", file=sys.stderr)
        return 2

    print_analysis(report)
    if not args.analysis_only:
        print_plan(plan)
        if not eligible_models(registry):
            print("Model registry: no license-verified, enabled models are currently eligible.")
    if args.report:
        report_path = Path(args.report).expanduser()
        if not report_path.is_absolute():
            report_path = Path(args.output_dir).expanduser() / report_path
        payload = {
            "schema_version": 1,
            "analysis": to_jsonable(report),
            "plan": None if args.analysis_only else to_jsonable(plan),
            "registry_path": str(registry_path),
            "models_root": str(paths["models"]),
            "inference_performed": False,
        }
        try:
            write_json_atomic(report_path, payload)
        except OSError as exc:
            print(f"mseparate: cannot write report: {exc}", file=sys.stderr)
            return 1
        print(f"Report written: {report_path}")
    if args.verbose:
        print(f"Data directory: {data_root()}")
        print(f"Registry entries: {len(registry)}")
        print(f"Device preference: {args.device}")
        print("Inference: disabled in this foundation release")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
