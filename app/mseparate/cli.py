"""Independent mseparate command line interface."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .assets import ensure_model_assets
from .audio import analyze_audio
from .execution import ExecutionManager
from .melband import MelBandCLIAdapter
from .planner import build_plan, validate_plan
from .registry import RegistryError, bundled_registry_path, eligible_models, load_registry
from .reporting import print_analysis, print_execution_summary, print_plan
from .storage import data_root, ensure_layout, write_json_atomic
from .types import to_jsonable


def _registry_path(paths: dict[str, Path], override: str | None) -> Path:
    if override:
        return Path(override).expanduser()
    local = paths["registry"] / "models.json"
    return local if local.exists() else bundled_registry_path()


def _print_models(registry) -> None:
    print("Model registry")
    if not registry:
        print("  No model manifests found.")
        return
    eligible = {model.model_id for model in eligible_models(registry)}
    for model in sorted(registry.values(), key=lambda item: item.model_id):
        state = "eligible" if model.model_id in eligible else "not eligible"
        outputs = ", ".join(model.target_stems or model.native_stems or model.derived_stems) or "none declared"
        print(f"  {model.model_id} [{state}]")
        print(f"    {model.name}")
        print(f"    Architecture/backend: {model.architecture} / {model.backend}")
        print(f"    Category: {model.category or 'unspecified'}; outputs: {outputs}")
        print(f"    License: {model.license or 'unknown'} ({model.license_status}); status: {model.status}")
        if model.sample_rate or model.channels:
            print(f"    Input: {model.sample_rate or 'any'} Hz; {model.channels or 'any channels'}")
        for note in model.notes:
            print(f"    Note: {note}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mseparate",
        description="Analyze a music track and prepare or execute a safe multi-model separation plan.",
    )
    parser.add_argument("input", nargs="?", help="Input audio file")
    parser.add_argument("--output-dir", default=".", help="Directory for reports and separated stems")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto",
                        help="Inference device (auto selects CUDA when available)")
    parser.add_argument("--models-dir", default=None, help="Override persistent model storage directory")
    parser.add_argument("--registry", default=None, help="Use an alternate JSON model registry")
    parser.add_argument("--list-models", action="store_true", help="List bundled/local models and their review eligibility")
    parser.add_argument("--analysis-only", action="store_true", help="Analyze audio without planning or inference")
    parser.add_argument("--plan-only", action="store_true", help="Analyze audio and print the plan without inference")
    parser.add_argument("--execute", action="store_true",
                        help="Explicitly run the planned separation tasks (may use CUDA and download model assets)")
    parser.add_argument("--no-download", action="store_true",
                        help="Never access the network; execution requires all model assets to be verified locally")
    parser.add_argument("--prepare-models", action="store_true",
                        help="Verify or download assets for eligible registry models")
    parser.add_argument("--download-timeout", type=float, default=30.0, help="Per-request timeout in seconds (default: 30)")
    parser.add_argument("--download-retries", type=int, default=3, help="Attempts per asset (default: 3)")
    parser.add_argument("--inference-timeout", type=float, default=0.0,
                        help="Per-task inference timeout in seconds; 0 disables the timeout (default)")
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum refinement depth for future plans")
    parser.add_argument("--verbose", action="store_true", help="Print additional diagnostics")
    parser.add_argument("--report", nargs="?", const="mseparate-report.json",
                        help="Write a JSON report (optional path; default: mseparate-report.json)")
    parser.add_argument("--version", action="version", version=f"mseparate {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_depth < 0:
        parser.error("--max-depth must be zero or greater")
    if args.download_timeout <= 0:
        parser.error("--download-timeout must be greater than zero")
    if args.download_retries < 1:
        parser.error("--download-retries must be at least one")
    if args.inference_timeout < 0:
        parser.error("--inference-timeout must be zero or greater")
    if args.execute and (args.analysis_only or args.plan_only):
        parser.error("--execute cannot be combined with --analysis-only or --plan-only")

    try:
        paths = ensure_layout()
        if args.models_dir:
            paths["models"] = Path(args.models_dir).expanduser()
            paths["models"].mkdir(parents=True, exist_ok=True)
        registry_path = _registry_path(paths, args.registry)
        registry = load_registry(registry_path)
        if args.list_models:
            _print_models(registry)
            print(f"  Registry source: {registry_path}")
            return 0
        if not args.input:
            parser.print_help(sys.stderr)
            return 2

        input_path = Path(args.input).expanduser().resolve()
        report = analyze_audio(input_path)
        plan = build_plan(max_depth=args.max_depth, analysis=report, registry=registry)
        validate_plan(plan)

        asset_results: list[dict] = []
        prepared_models: set[str] = set()
        models_to_prepare = eligible_models(registry) if args.prepare_models else []
        if args.execute:
            by_id = {model.model_id: model for model in eligible_models(registry)}
            for task in plan.tasks:
                model = by_id.get(task.model_id)
                if model is not None and model.model_id not in prepared_models:
                    models_to_prepare.append(model)
                    prepared_models.add(model.model_id)

        # Prepare each selected model at most once, even when --prepare-models
        # and --execute are used together.
        unique_models = {model.model_id: model for model in models_to_prepare}
        model_asset_status: dict[str, bool] = {}
        for model in unique_models.values():
            results = ensure_model_assets(
                model, paths["models"], no_download=args.no_download,
                timeout=args.download_timeout, retries=args.download_retries,
            )
            model_asset_status[model.model_id] = bool(results) and all(
                result.status in {"cached", "recovered", "downloaded"} for result in results
            )
            asset_results.extend({
                "model_id": model.model_id, "role": result.role,
                "status": result.status, "path": result.path,
                "message": result.message, "bytes_downloaded": result.bytes_downloaded,
            } for result in results)

        execution_summary = None
        execution_attempted = False
        execution_blocked = False
        if args.execute and plan.tasks:
            selected_ids = {task.model_id for task in plan.tasks}
            failed_assets = sorted(model_id for model_id in selected_ids
                                   if not model_asset_status.get(model_id, False))
            if failed_assets:
                execution_blocked = True
                print("Execution blocked: model assets are missing or failed verification: "
                      + ", ".join(failed_assets), file=sys.stderr)
            else:
                manager = ExecutionManager(
                    {"melband-roformer": lambda: MelBandCLIAdapter(
                        timeout=args.inference_timeout or None
                    )},
                    models_root=paths["models"],
                    output_root=Path(args.output_dir).expanduser() / "mseparate-stems",
                    device=args.device,
                )
                execution_attempted = True
                execution_summary = manager.execute(
                    plan, registry, {"mix": input_path},
                )

    except (OSError, ValueError, RuntimeError, RegistryError) as exc:
        print(f"mseparate: {exc}", file=sys.stderr)
        return 2

    print_analysis(report)
    if args.prepare_models or args.execute:
        print("Model asset preparation")
        if not asset_results:
            print("  No model assets needed to be prepared.")
        for item in asset_results:
            print("  " + item["model_id"] + " / " + item["role"] + ": " + item["status"])
            if item["message"]:
                print("    " + item["message"])
    if not args.analysis_only:
        print_plan(plan)
        if not eligible_models(registry):
            print("Model registry: no license-verified, enabled models are currently eligible.")
    if execution_summary is not None:
        print_execution_summary(execution_summary)
        if execution_summary.succeeded:
            print("Separated stems: " + str(Path(args.output_dir).expanduser() / "mseparate-stems"))
    elif args.execute and not plan.tasks:
        print("Execution not started: the planner produced no compatible inference tasks.")
    elif execution_blocked:
        print("Inference was not started because required assets are unavailable.")

    inference_performed = execution_attempted
    if args.report:
        report_path = Path(args.report).expanduser()
        if not report_path.is_absolute():
            report_path = Path(args.output_dir).expanduser() / report_path
        payload = {
            "schema_version": 1, "analysis": to_jsonable(report),
            "plan": None if args.analysis_only else to_jsonable(plan),
            "registry_path": str(registry_path), "models_root": str(paths["models"]),
            "inference_performed": inference_performed,
            "execution": to_jsonable(execution_summary) if execution_summary is not None else None,
            "execution_blocked": execution_blocked,
            "asset_results": asset_results,
        }
        try:
            write_json_atomic(report_path, payload)
        except OSError as exc:
            print(f"mseparate: cannot write report: {exc}", file=sys.stderr)
            return 1
        print(f"Report written: {report_path}")
    if args.verbose:
        print(f"Data directory: {data_root()}")
        print("Models directory: " + str(paths["models"]))
        print(f"Registry entries: {len(registry)}")
        print(f"Prepared assets: {len(asset_results)}")
        print(f"Device preference: {args.device}")
        print(f"Inference attempted: {inference_performed}")
    if execution_blocked:
        return 1
    if execution_summary is not None and execution_summary.failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
