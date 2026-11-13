"""GPU-backed six-stem separation through the Demucs ONNX runtime.

The ONNX backend is isolated behind the same lifecycle contract as the
existing MelBand subprocess adapter. It uses the CUDA Execution Provider
when requested and clears its process-local session pool after each task.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path

from .execution import BackendInfo, StemArtifact
from .types import ModelManifest


def _prewarm_strict_cuda(runtime, cache_dir: Path, precision: str) -> dict[str, Path]:
    """Prewarm HT-Demucs and disable ONNX Runtime's CPU retry fallback."""
    paths = runtime.prewarm(
        models=["htdemucs_6s"], precision=precision,
        providers="CUDAExecutionProvider", cache_dir=str(cache_dir),
    )
    model_path = paths.get("htdemucs_6s")
    if model_path is None:
        raise RuntimeError("demucs-onnx prewarm did not return the htdemucs_6s model path")
    pool = runtime.session_pool()
    session = pool.get(model_path, ["CUDAExecutionProvider"])
    active = tuple(session.get_providers())
    if active != ("CUDAExecutionProvider",):
        raise RuntimeError(
            "Strict CUDA session did not register CUDAExecutionProvider exclusively; "
            f"active providers: {active}"
        )
    disable_fallback = getattr(session, "disable_fallback", None)
    if not callable(disable_fallback):
        raise RuntimeError(
            "ONNX Runtime session does not expose disable_fallback(); "
            "cannot guarantee strict CUDA execution"
        )
    disable_fallback()
    return paths


class DemucsOnnxAdapter:
    """Run the htdemucs_6s ONNX model and materialize six WAV stems."""

    MODEL_NAME = "htdemucs_6s"
    STEMS = ("drums", "bass", "other", "vocals", "guitar", "piano")

    def __init__(self, *, no_download: bool = False, precision: str = "fp16weights"):
        self.no_download = no_download
        self.precision = precision
        self._provider = "auto"
        self._strict_cuda = False
        self._cache_dir: Path | None = None
        self._runtime = None
        self._model_id: str | None = None

    def compatibility(self, manifest: ModelManifest) -> tuple[bool, str]:
        if manifest.backend != "demucs-onnx":
            return False, "Manifest backend must be 'demucs-onnx'"
        missing = set(self.STEMS) - set(manifest.native_stems)
        if missing:
            return False, "Manifest is missing native stems: " + ", ".join(sorted(missing))
        if set(manifest.target_stems) - set(self.STEMS):
            return False, "Manifest declares stems unsupported by htdemucs_6s"
        return True, ""

    def load(self, manifest: ModelManifest, model_dir: Path, device: str) -> None:
        compatible, reason = self.compatibility(manifest)
        if not compatible:
            raise RuntimeError(reason)
        if device not in {"auto", "cuda", "cpu"}:
            raise ValueError(f"Unsupported device: {device!r}")
        try:
            import onnxruntime as ort
            import demucs_onnx
        except ImportError as exc:
            raise RuntimeError(
                "Six-stem backend is not installed. Rebuild the AppImage with "
                "demucs-onnx and onnxruntime-gpu CUDA dependencies."
            ) from exc

        # CUDA/cuDNN are installed as NVIDIA Python packages inside the
        # AppImage. ORT 1.27+ can preload these libraries from site-packages.
        preload = getattr(ort, "preload_dlls", None)
        if callable(preload):
            try:
                preload(directory="")
            except Exception as exc:
                if device == "cuda":
                    raise RuntimeError(
                        f"Could not preload CUDA/cuDNN libraries for ONNX Runtime: {exc}"
                    ) from exc

        available = set(ort.get_available_providers())
        if device == "cuda" and "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is unavailable in ONNX Runtime. "
                "Check the NVIDIA driver and bundled CUDA/cuDNN runtime; "
                f"available providers: {sorted(available)}"
            )
        self._strict_cuda = device == "cuda"
        if device == "cuda":
            # The upstream "cuda" alias appends CPUExecutionProvider. Strict
            # mode uses the explicit provider and disables ORT's retry fallback.
            self._provider = "CUDAExecutionProvider"
        elif device == "auto" and "CUDAExecutionProvider" in available:
            self._provider = "cuda"
        else:
            self._provider = "cpu"
        self._cache_dir = model_dir
        self._runtime = demucs_onnx
        self._model_id = manifest.model_id
        model_dir.mkdir(parents=True, exist_ok=True)

    def infer(self, input_audio: Path, output_dir: Path,
              target_stems: tuple[str, ...]) -> list[StemArtifact]:
        if self._runtime is None or self._cache_dir is None:
            raise RuntimeError("Demucs ONNX adapter is not loaded")
        if not input_audio.is_file():
            raise FileNotFoundError(input_audio)
        unsupported = set(target_stems) - set(self.STEMS)
        if unsupported:
            raise RuntimeError("Unsupported htdemucs_6s stems: " + ", ".join(sorted(unsupported)))

        output_dir.mkdir(parents=True, exist_ok=True)
        old_offline = os.environ.get("HF_HUB_OFFLINE")
        if self.no_download:
            os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            if self._strict_cuda:
                _prewarm_strict_cuda(
                    self._runtime, self._cache_dir, self.precision,
                )
            self._runtime.separate(
                str(input_audio), str(output_dir),
                model=self.MODEL_NAME, stems=target_stems,
                providers=self._provider, precision=self.precision,
                cache_dir=str(self._cache_dir), verbose=True,
                progress=True, output_format="wav",
            )
        finally:
            if self.no_download:
                if old_offline is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = old_offline

        return [
            StemArtifact(stem, output_dir / f"{stem}.wav")
            for stem in target_stems
            if (output_dir / f"{stem}.wav").is_file()
        ]

    def unload(self) -> None:
        runtime = self._runtime
        self._runtime = None
        self._cache_dir = None
        self._model_id = None
        self._strict_cuda = False
        if runtime is not None:
            try:
                runtime.session_pool().clear()
            finally:
                del runtime
        gc.collect()

    def info(self) -> BackendInfo:
        return BackendInfo(
            name="demucs-onnx", device=self._provider,
            details={"model": self.MODEL_NAME, "precision": self.precision,
                     "model_id": self._model_id or ""},
        )
