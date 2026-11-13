"""Independent mseparate command line interface."""
from __future__ import annotations

import argparse
import os
from dataclasses import replace
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
        print(f"    Planner priority: {model.planner_priority} (routing preference, not quality)")
        print(f"    License: {model.license or 'unknown'} ({model.license_status}); status: {model.status}")
        if model.sample_rate or model.channels:
            print(f"    Input: {model.sample_rate or 'any'} Hz; {model.channels or 'any channels'}")
        for note in model.notes:
            print(f"    Note: {note}")


def _prepare_demucs_onnx(model_dir: Path, device: str, *, no_download: bool) -> dict[str, Path]:
    """Download and prewarm the six-stem ONNX model, then release its session."""
    import onnxruntime as ort
    import demucs_onnx

    preload = getattr(ort, "preload_dlls", None)
    if callable(preload):
        preload(directory="")
    available = set(ort.get_available_providers())
    if device == "cuda" and "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            "CUDAExecutionProvider is unavailable; ONNX Runtime providers: "
            + ", ".join(sorted(available))
        )

    model_dir.mkdir(parents=True, exist_ok=True)
    old_offline = os.environ.get("HF_HUB_OFFLINE")
    if no_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        if device == "cuda":
            from .demucs_onnx import _prewarm_strict_cuda

            return _prewarm_strict_cuda(demucs_onnx, model_dir, "fp16weights")
        if device == "auto" and "CUDAExecutionProvider" in available:
            provider = "cuda"
        else:
            provider = "cpu"
        return demucs_onnx.prewarm(
            models=["htdemucs_6s"], precision="fp16weights",
            providers=provider, cache_dir=str(model_dir),
        )
    finally:
        demucs_onnx.session_pool().clear()
        if no_download:
            if old_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = old_offline


def _run_demucs_diagnostic(input_path: Path, model_dir: Path, device: str,
                           max_chunks: int, *, no_download: bool) -> int:
    """Measure a bounded number of real HT-Demucs ONNX inference chunks."""
    import time

    import demucs_onnx
    import numpy as np
    import onnxruntime as ort
    from demucs_onnx.inference import (
        N_SAMPLES, SAMPLE_RATE,
    )
    from demucs_onnx.providers import resolve_providers

    from .demucs_onnx import _prewarm_strict_cuda

    model_dir.mkdir(parents=True, exist_ok=True)
    old_offline = os.environ.get("HF_HUB_OFFLINE")
    if no_download:
        os.environ["HF_HUB_OFFLINE"] = "1"

    try:
        print("HT-Demucs ONNX diagnostic")
        print(f"  input: {input_path}")
        print(f"  requested device: {device}")
        print(f"  ONNX Runtime: {ort.__version__}")
        print(f"  maximum chunks: {max_chunks}")

        if device == "cuda":
            model_paths = _prewarm_strict_cuda(
                demucs_onnx, model_dir, "fp16weights",
            )
            providers = ["CUDAExecutionProvider"]
        else:
            provider = device
            model_paths = demucs_onnx.prewarm(
                models=["htdemucs_6s"], precision="fp16weights",
                providers=provider, cache_dir=str(model_dir),
            )
            providers = resolve_providers(provider)

        model_path = model_paths.get("htdemucs_6s")
        if model_path is None:
            raise RuntimeError("Could not prepare the htdemucs_6s ONNX model")

        audio, native_rate = demucs_onnx.load_audio(
            str(input_path), target_sr=SAMPLE_RATE,
        )
        session = demucs_onnx.session_pool().get(model_path, providers)
        active_providers = tuple(session.get_providers())
        if not active_providers:
            raise RuntimeError("ONNX Runtime session did not report an execution provider")
        if device == "cuda":
            disable_fallback = getattr(session, "disable_fallback", None)
            if not callable(disable_fallback):
                raise RuntimeError(
                    "ONNX Runtime session does not expose disable_fallback(); "
                    "strict CUDA diagnostic cannot continue"
                )
            disable_fallback()
            if active_providers[0] != "CUDAExecutionProvider":
                raise RuntimeError(
                    "Strict CUDA diagnostic did not select CUDAExecutionProvider first; "
                    f"active providers: {active_providers}"
                )

        total_samples = int(audio.shape[1])
        stride = N_SAMPLES - N_SAMPLES // 4
        total_chunks = max(1, (total_samples + stride - 1) // stride)
        chunks_to_run = min(max_chunks, total_chunks)
        run_times: list[float] = []
        preprocessing_times: list[float] = []

        print(f"  input audio: {total_samples:,} samples ({total_samples / SAMPLE_RATE:.1f} s)")
        print(f"  source sample rate: {native_rate} Hz; model rate: {SAMPLE_RATE} Hz")
        print(f"  chunk size: {N_SAMPLES:,} samples ({N_SAMPLES / SAMPLE_RATE:.1f} s)")
        print(f"  overlap: 25%; stride: {stride:,} samples")
        print(f"  chunks in full track: {total_chunks}")
        print(f"  active providers: {', '.join(active_providers)}")
        print(f"  model: {model_path}")
        print("  running bounded inference (no stem files will be written)")

        diagnostic_started = time.perf_counter()
        for index in range(chunks_to_run):
            prep_started = time.perf_counter()
            start = index * stride
            end = min(start + N_SAMPLES, total_samples)
            chunk = audio[:, start:end]
            if chunk.shape[1] < N_SAMPLES:
                chunk = np.pad(
                    chunk, ((0, 0), (0, N_SAMPLES - chunk.shape[1])),
                    mode="constant",
                )
            model_input = chunk[np.newaxis, ...].astype("float32", copy=False)
            preprocessing_times.append(time.perf_counter() - prep_started)

            run_started = time.perf_counter()
            result = session.run(["stems"], {"mix": model_input})
            run_elapsed = time.perf_counter() - run_started
            run_times.append(run_elapsed)
            del result

            label = " (first run / warm-up)" if index == 0 else ""
            print(
                f"  chunk {index + 1}/{total_chunks}: "
                f"session.run={run_elapsed:.3f}s, "
                f"input prep={preprocessing_times[-1]:.3f}s{label}"
            )

        diagnostic_elapsed = time.perf_counter() - diagnostic_started
        inference_total = sum(run_times)
        print("Diagnostic summary")
        print(f"  chunks measured: {chunks_to_run}/{total_chunks}")
        print(f"  measured ONNX time: {inference_total:.3f}s")
        print(f"  diagnostic wall time: {diagnostic_elapsed:.3f}s")
        if len(run_times) > 1:
            steady_average = sum(run_times[1:]) / (len(run_times) - 1)
            projected = run_times[0] + steady_average * (total_chunks - 1)
            print(f"  average after first run: {steady_average:.3f}s/chunk")
            print("  estimated full-track ONNX time: "
                  f"{projected:.1f}s ({projected / (total_samples / SAMPLE_RATE):.3f} RTF)")
            print("  estimate excludes audio decode, overlap-add and file writing")
        elif total_chunks == 1:
            print(f"  estimated full-track ONNX time: {inference_total:.3f}s")
        else:
            projected = run_times[0] * total_chunks
            print("  estimated full-track ONNX time: "
                  f"{projected:.1f}s (rough estimate from one chunk; low confidence)")
        return 0
    finally:
        demucs_onnx.session_pool().clear()
        if no_download:
            if old_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = old_offline


def _resolve_auto_device() -> tuple[str, str | None]:
    """Probe both CUDA runtimes before selecting a device for the whole plan."""
    failures: list[str] = []
    try:
        import torch

        if not torch.cuda.is_available():
            failures.append("PyTorch reports that CUDA is unavailable")
        else:
            torch.cuda.init()
            probe = torch.empty(1, device="cuda")
            torch.cuda.synchronize()
            del probe
    except Exception as exc:
        failures.append(f"PyTorch CUDA initialization failed: {type(exc).__name__}: {exc}")

    try:
        import onnxruntime as ort

        preload = getattr(ort, "preload_dlls", None)
        if callable(preload):
            preload(directory="")
        providers = set(ort.get_available_providers())
        if "CUDAExecutionProvider" not in providers:
            failures.append(
                "ONNX Runtime CUDAExecutionProvider is unavailable "
                f"(available: {', '.join(sorted(providers)) or 'none'})"
            )
    except Exception as exc:
        failures.append(f"ONNX Runtime CUDA initialization failed: {type(exc).__name__}: {exc}")

    if failures:
        return "cpu", "; ".join(failures) + "; switching to CPU."
    return "cuda", None

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
                        help="Run the planned separation tasks (default unless --analysis-only, --plan-only or --prepare-models is used)")
    parser.add_argument("--diagnose", type=int, metavar="N",
                        help="Run an HT-Demucs ONNX diagnostic on at most N chunks; writes no stems")
    parser.add_argument("--no-download", action="store_true",
                        help="Never access the network; execution requires all model assets to be verified locally")
    parser.add_argument("--prepare-models", action="store_true",
                        help="Verify or download assets for eligible registry models")
    parser.add_argument("--download-timeout", type=float, default=30.0, help="Per-request timeout in seconds (default: 30)")
    parser.add_argument("--download-retries", type=int, default=3, help="Attempts per asset (default: 3)")
    parser.add_argument("--inference-timeout", type=float, default=0.0,
                        help="Per-task inference timeout in seconds; 0 disables the timeout (default)")
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum refinement depth for future plans")
    parser.add_argument("--prefer-model", help="Prefer this eligible model ID for overlapping stem routes (see --list-models)")
    parser.add_argument("--allow-resample", action="store_true",
                        help="Allow temporary WAV sample-rate conversion to satisfy model input rates; outputs keep the model sample rate")
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
    if args.diagnose is not None and args.diagnose < 1:
        parser.error("--diagnose must be at least one chunk")
    if args.diagnose is not None and (args.execute or args.analysis_only or args.plan_only or args.prepare_models or args.report):
        parser.error("--diagnose cannot be combined with --execute, --analysis-only, --plan-only, --prepare-models or --report")

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
        if args.diagnose is not None:
            diagnostic_device = args.device
            if diagnostic_device == "auto":
                diagnostic_device, reason = _resolve_auto_device()
                if reason:
                    print("Warning: " + reason, file=sys.stderr)
            diagnostic_model = next(
                (item for item in registry.values() if item.backend == "demucs-onnx"
                 and item.model_id == "demucs-onnx-htdemucs-6s"),
                None,
            )
            if diagnostic_model is None:
                raise RuntimeError("Registry does not contain the htdemucs_6s ONNX model")
            return _run_demucs_diagnostic(
                input_path, paths["models"] / diagnostic_model.model_id,
                diagnostic_device, args.diagnose, no_download=args.no_download,
            )
        execute_requested = args.execute or not (args.analysis_only or args.plan_only or args.prepare_models)
        effective_device = args.device
        device_fallback_reason = None
        if execute_requested and args.device == "auto":
            effective_device, device_fallback_reason = _resolve_auto_device()
            if device_fallback_reason:
                print("Warning: " + device_fallback_reason, file=sys.stderr)
        report = analyze_audio(input_path)
        if execute_requested:
            # Semantic source recognition is an execution-time routing aid only.
            # Asset preparation must remain independent of the optional CLAP stack.
            from .source_analysis import analyze_sources
            try:
                predictions = analyze_sources(
                    input_path, device=effective_device,
                    cache_dir=paths["models"] / "source-analysis",
                    no_download=args.no_download,
                )
            except Exception as exc:
                if args.device != "auto" or effective_device != "cuda":
                    raise
                device_fallback_reason = (
                    f"CLAP CUDA analysis failed ({type(exc).__name__}: {exc}); switching to CPU."
                )
                print("Warning: " + device_fallback_reason, file=sys.stderr)
                effective_device = "cpu"
                predictions = analyze_sources(
                    input_path, device="cpu",
                    cache_dir=paths["models"] / "source-analysis",
                    no_download=args.no_download,
                )
            report = replace(report, predictions=predictions)
        plan = build_plan(max_depth=args.max_depth, analysis=report, registry=registry,
                          preferred_model_id=args.prefer_model,
                          allow_resample=args.allow_resample)
        validate_plan(plan)

        asset_results: list[dict] = []
        prepared_models: set[str] = set()
        models_to_prepare = eligible_models(registry) if args.prepare_models else []
        if execute_requested:
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
            if model.backend == "demucs-onnx":
                model_dir = paths["models"] / model.model_id
                if args.prepare_models:
                    try:
                        prepare_device = effective_device if execute_requested else args.device
                        prepared = _prepare_demucs_onnx(
                            model_dir, prepare_device, no_download=args.no_download,
                        )
                        model_asset_status[model.model_id] = bool(prepared)
                        asset_results.extend({
                            "model_id": model.model_id, "role": "onnx-model",
                            "status": "prepared", "path": str(path),
                            "message": "Downloaded or reused through Hugging Face Hub and prewarmed.",
                            "bytes_downloaded": 0,
                        } for path in prepared.values())
                    except Exception as exc:
                        model_asset_status[model.model_id] = False
                        asset_results.append({
                            "model_id": model.model_id, "role": "onnx-model",
                            "status": "failed", "path": None,
                            "message": f"{type(exc).__name__}: {exc}",
                            "bytes_downloaded": 0,
                        })
                else:
                    # The backend downloads the pinned model family lazily on
                    # first inference; --no-download is enforced by HF_HUB_OFFLINE.
                    model_asset_status[model.model_id] = True
                    asset_results.append({
                        "model_id": model.model_id, "role": "onnx-model",
                        "status": "managed-by-backend", "path": str(model_dir),
                        "message": "Model download/cache is managed by demucs-onnx.",
                        "bytes_downloaded": 0,
                    })
                continue

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
        if execute_requested and plan.tasks:
            selected_ids = {task.model_id for task in plan.tasks}
            failed_assets = sorted(model_id for model_id in selected_ids
                                   if not model_asset_status.get(model_id, False))
            if failed_assets:
                execution_blocked = True
                print("Execution blocked: model assets are missing or failed verification: "
                      + ", ".join(failed_assets), file=sys.stderr)
            else:
                from .demucs_onnx import DemucsOnnxAdapter
                from .bs_roformer import BSRoformerAdapter
                from .mdx23c import MDX23CAdapter

                manager = ExecutionManager(
                    {
                        "melband-roformer": lambda: MelBandCLIAdapter(
                            timeout=args.inference_timeout or None
                        ),
                        "demucs-onnx": lambda: DemucsOnnxAdapter(
                            no_download=args.no_download,
                        ),
                        "bs-roformer-pymss": BSRoformerAdapter,
                        "mdx23c-pymss": MDX23CAdapter,
                    },
                    models_root=paths["models"],
                    output_root=Path(args.output_dir).expanduser() / "mseparate-stems",
                    device=effective_device,
                )
                execution_attempted = True
                execution_summary = manager.execute(
                    plan, registry, {"mix": input_path},
                )
                # A CUDA provider may pass the initial probe but fail during
                # model loading or inference. ExecutionManager has unloaded
                # every adapter before returning, so a fresh CPU pass is safe.
                # Explicit --device cuda remains strict and never falls back.
                if (args.device == "auto" and effective_device == "cuda"
                        and execution_summary.failed):
                    failed_tasks = [item for item in execution_summary.results
                                    if item.status == "failed"]
                    device_fallback_reason = "; ".join(
                        f"{item.task_id}: {item.message or 'CUDA task failed'}"
                        for item in failed_tasks
                    )
                    print(
                        "Warning: CUDA inference failed; retrying the complete plan on CPU. "
                        + device_fallback_reason,
                        file=sys.stderr,
                    )
                    effective_device = "cpu"
                    cpu_manager = ExecutionManager(
                        {
                            "melband-roformer": lambda: MelBandCLIAdapter(
                                timeout=args.inference_timeout or None
                            ),
                            "demucs-onnx": lambda: DemucsOnnxAdapter(
                                no_download=args.no_download,
                            ),
                            "bs-roformer-pymss": BSRoformerAdapter,
                            "mdx23c-pymss": MDX23CAdapter,
                        },
                        models_root=paths["models"],
                        output_root=Path(args.output_dir).expanduser() / "mseparate-stems",
                        device="cpu",
                    )
                    execution_summary = cpu_manager.execute(
                        plan, registry, {"mix": input_path},
                    )

    except (OSError, ValueError, RuntimeError, RegistryError) as exc:
        print(f"mseparate: {exc}", file=sys.stderr)
        return 2

    print_analysis(report)
    if args.prepare_models or execute_requested:
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
    elif execute_requested and not plan.tasks:
        print("Execution not started: the planner produced no compatible inference tasks.")
    elif execution_blocked:
        print("Inference was not started because required assets are unavailable.")

    inference_attempted = execution_attempted
    inference_performed = bool(execution_summary and execution_summary.succeeded)
    if args.report:
        report_path = Path(args.report).expanduser()
        if not report_path.is_absolute():
            report_path = Path(args.output_dir).expanduser() / report_path
        payload = {
            "schema_version": 1, "analysis": to_jsonable(report),
            "plan": None if args.analysis_only else to_jsonable(plan),
            "registry_path": str(registry_path), "models_root": str(paths["models"]),
            "inference_attempted": inference_attempted,
            "inference_performed": inference_performed,
            "execution": to_jsonable(execution_summary) if execution_summary is not None else None,
            "execution_blocked": execution_blocked,
            "requested_device": args.device,
            "allow_resample": args.allow_resample,
            "effective_device": effective_device,
            "device_fallback_reason": device_fallback_reason,
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
        print(f"Sample-rate conversion allowed: {args.allow_resample}")
        print(f"Effective device: {effective_device}")
        if device_fallback_reason:
            print(f"Device fallback: {device_fallback_reason}")
        print(f"Inference attempted: {inference_attempted}")
        print(f"At least one task succeeded: {inference_performed}")
    if execution_blocked:
        return 1
    if execution_summary is not None and execution_summary.failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
