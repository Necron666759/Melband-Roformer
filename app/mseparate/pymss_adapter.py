"""Shared adapter for MSST-format models through the pinned pymss runtime."""
from __future__ import annotations

import gc
import sys
from contextlib import redirect_stderr
from pathlib import Path

import numpy as np
import soundfile as sf

from .assets import verify_asset
from .execution import BackendInfo, StemArtifact
from .types import ModelManifest


class _TqdmLineStream:
    """Make tqdm's carriage-return updates visible in captured GUI logs."""

    def __init__(self, stream):
        self._stream = stream

    def write(self, value: str) -> int:
        normalized = value.replace("\\r\\n", "\\n").replace("\\r", "\\n")
        written = self._stream.write(normalized)
        if "\\r" in value:
            self._stream.flush()
        return len(value)

    def flush(self) -> None:
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


class PymssAdapter:
    """Run one MSST-format model through pymss with explicit local assets."""

    BACKEND = ""
    MODEL_TYPE = ""

    def __init__(self) -> None:
        self._separator = None
        self._model_id: str | None = None
        self._device: str | None = None
        self._native_stems: tuple[str, ...] = ()
        self._derived_stems: tuple[str, ...] = ()

    def compatibility(self, manifest: ModelManifest) -> tuple[bool, str]:
        if manifest.backend != self.BACKEND:
            return False, f"Manifest backend must be {self.BACKEND!r}"
        if "vocals" not in manifest.native_stems:
            return False, "MSST vocal adapter requires vocals as a native output"
        supported = set(manifest.native_stems) | set(manifest.derived_stems)
        if set(manifest.target_stems) - supported:
            return False, "MSST adapter target stems must be native or explicitly derived"
        if set(manifest.derived_stems) - {"instrumental"}:
            return False, "MSST adapter currently supports only the mixture-minus-vocals instrumental residual"
        assets = {asset.role: asset for asset in manifest.assets}
        missing = {"checkpoint", "config"} - set(assets)
        if missing:
            return False, "Manifest is missing required assets: " + ", ".join(sorted(missing))
        if any(not assets[role].sha256 for role in ("checkpoint", "config")):
            return False, "Checkpoint and config must both have pinned SHA-256 digests"
        return True, ""

    def load(self, manifest: ModelManifest, model_dir: Path, device: str) -> None:
        compatible, reason = self.compatibility(manifest)
        if not compatible:
            raise RuntimeError(reason)
        if device not in {"auto", "cuda", "cpu"}:
            raise ValueError(f"Unsupported device: {device!r}")

        assets = {asset.role: model_dir / asset.relative_path for asset in manifest.assets}
        for role in ("checkpoint", "config"):
            asset = next(item for item in manifest.assets if item.role == role)
            if not verify_asset(assets[role], asset):
                raise RuntimeError(
                    f"{role} asset is missing or failed SHA-256/size verification: {assets[role]}"
                )

        try:
            from pymss import MSSeparator
        except ImportError as exc:
            raise RuntimeError(
                "The MSST model runtime is unavailable. Install pymss==2.3.5 "
                "in the mseparate environment."
            ) from exc

        # Explicit local paths prevent pymss from consulting or downloading
        # another model catalog entry. Model bytes are verified by mseparate
        # before the runtime is allowed to deserialize them.
        self._separator = MSSeparator(
            model_type=self.MODEL_TYPE,
            model_path=str(assets["checkpoint"]),
            config_path=str(assets["config"]),
            device=device,
            device_ids=[0],
            output_format="wav",
            use_tta=False,
            store_dirs={},
            debug=True,
            inference_params={"batch_size": 1, "normalize": False},
        )
        self._model_id = manifest.model_id
        self._device = str(getattr(self._separator, "device", device))
        self._native_stems = manifest.native_stems
        self._derived_stems = manifest.derived_stems

    def infer(
        self, input_audio: Path, output_dir: Path, target_stems: tuple[str, ...]
    ) -> list[StemArtifact]:
        if self._separator is None:
            raise RuntimeError("MSST adapter is not loaded")
        if not input_audio.is_file():
            raise FileNotFoundError(input_audio)

        mixture, sample_rate = sf.read(
            str(input_audio), dtype="float32", always_2d=True
        )
        # pymss expects channel-major input and returns sample-major outputs.
        native_targets = [stem for stem in target_stems if stem in self._native_stems]
        # Instrumental is deliberately a derived residual, never presented as
        # a native model output: mixture - vocals gives an exactly complementary
        # pair for the selected model's vocal estimate.
        if "instrumental" in target_stems and "vocals" not in native_targets:
            native_targets.append("vocals")
        # pymss uses tqdm's carriage-return updates on stderr. A captured
        # GUI log treats those updates as one line and can leave the initial
        # 0% snapshot on screen. Convert each update to a flushed log line.
        progress_stream = _TqdmLineStream(sys.stderr)
        with redirect_stderr(progress_stream):
            separated = self._separator.separate(
                mixture.T, pbar=True, stems=native_targets
            )
            progress_stream.flush()
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts: list[StemArtifact] = []
        for stem in target_stems:
            if stem == "instrumental" and stem in self._derived_stems:
                if "vocals" not in separated:
                    raise RuntimeError(
                        f"{self.MODEL_TYPE} did not return vocals needed for the instrumental residual"
                    )
                vocals = np.asarray(separated["vocals"], dtype=np.float32)
                if vocals.ndim == 1:
                    vocals = vocals[:, None]
                elif (
                    vocals.ndim == 2
                    and vocals.shape[0] in (1, 2)
                    and vocals.shape[1] == mixture.shape[0]
                ):
                    vocals = vocals.T
                if vocals.shape != mixture.shape:
                    raise RuntimeError(
                        f"{self.MODEL_TYPE} vocals shape {vocals.shape} does not match mixture {mixture.shape}"
                    )
                samples = mixture - vocals
            else:
                if stem not in separated:
                    raise RuntimeError(f"{self.MODEL_TYPE} did not return requested stem {stem!r}")
                samples = np.asarray(separated[stem], dtype=np.float32)
                if samples.ndim == 1:
                    samples = samples[:, None]
                elif (
                    samples.ndim == 2
                    and samples.shape[0] in (1, 2)
                    and samples.shape[1] == mixture.shape[0]
                ):
                    samples = samples.T
            if samples.ndim != 2 or samples.shape[1] not in (1, 2):
                raise RuntimeError(
                    f"{self.MODEL_TYPE} returned an invalid {stem!r} array shape {samples.shape}"
                )
            if samples.shape != mixture.shape:
                raise RuntimeError(
                    f"{self.MODEL_TYPE} output {stem!r} has shape {samples.shape}; "
                    f"expected {mixture.shape}"
                )
            path = output_dir / f"{stem}.wav"
            sf.write(str(path), samples, sample_rate, subtype="FLOAT")
            artifacts.append(StemArtifact(stem, path))
        return artifacts

    def unload(self) -> None:
        separator = self._separator
        self._separator = None
        self._model_id = None
        self._device = None
        self._native_stems = ()
        self._derived_stems = ()
        if separator is not None:
            separator.close()
        del separator
        gc.collect()

    def info(self) -> BackendInfo:
        return BackendInfo(
            name=f"pymss-{self.MODEL_TYPE}",
            device=self._device,
            details={"model_id": self._model_id or ""},
        )
