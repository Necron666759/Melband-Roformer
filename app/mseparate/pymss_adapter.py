"""Shared adapter for MSST-format models through the pinned pymss runtime."""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import soundfile as sf

from .assets import verify_asset
from .execution import BackendInfo, StemArtifact
from .types import ModelManifest


class PymssAdapter:
    """Run one MSST-format model through pymss with explicit local assets."""

    BACKEND = ""
    MODEL_TYPE = ""

    def __init__(self) -> None:
        self._separator = None
        self._model_id: str | None = None
        self._device: str | None = None

    def compatibility(self, manifest: ModelManifest) -> tuple[bool, str]:
        if manifest.backend != self.BACKEND:
            return False, f"Manifest backend must be {self.BACKEND!r}"
        if "vocals" not in manifest.native_stems:
            return False, "MSST vocal adapter requires vocals as a native output"
        if set(manifest.target_stems) - set(manifest.native_stems):
            return False, "MSST adapter only supports manifest-declared native stems"
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
        separated = self._separator.separate(
            mixture.T, pbar=True, stems=list(target_stems)
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts: list[StemArtifact] = []
        for stem in target_stems:
            if stem not in separated:
                raise RuntimeError(f"{self.MODEL_TYPE} did not return requested stem {stem!r}")
            samples = np.asarray(separated[stem], dtype=np.float32)
            if samples.ndim == 1:
                pass
            elif samples.ndim == 2 and samples.shape[0] == mixture.shape[0]:
                # Accept channel-major output defensively; the documented
                # pymss API returns sample-major arrays.
                samples = samples.T
            elif samples.ndim != 2 or samples.shape[1] not in (1, 2):
                raise RuntimeError(
                    f"{self.MODEL_TYPE} returned an invalid {stem!r} array shape {samples.shape}"
                )
            if samples.shape[0] != mixture.shape[0]:
                raise RuntimeError(
                    f"{self.MODEL_TYPE} output {stem!r} has {samples.shape[0]} frames; "
                    f"expected {mixture.shape[0]}"
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
