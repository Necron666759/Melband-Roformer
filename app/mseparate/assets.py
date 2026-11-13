"""Verified, resumable model asset downloads.

This module uses only the Python standard library. It never executes
downloaded content and only installs assets with a pinned SHA-256 digest.
"""
from __future__ import annotations

import hashlib
import http.client
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .types import ModelAsset, ModelManifest

_CHUNK_SIZE = 1024 * 1024
_CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+|\*)$", re.IGNORECASE)


class AssetError(RuntimeError):
    """An asset could not be safely verified or installed."""


@dataclass(frozen=True)
class AssetResult:
    role: str
    status: str
    path: str | None = None
    message: str | None = None
    bytes_downloaded: int = 0


def _safe_asset_path(model_dir: Path, relative_path: str) -> Path:
    rel = Path(relative_path)
    if rel.is_absolute() or not rel.parts or any(part in ("", ".", "..") for part in rel.parts):
        raise AssetError(f"Unsafe asset path: {relative_path!r}")
    root = model_dir.resolve()
    target = model_dir / rel
    # Resolve existing parents too: a symlink inside the model directory
    # must not redirect writes outside the asset store.
    parent = target.parent.resolve()
    try:
        parent.relative_to(root)
    except ValueError as exc:
        raise AssetError(f"Asset path escapes model directory: {relative_path!r}") from exc
    if target.is_symlink():
        raise AssetError(f"Refusing to replace symlink asset path: {target}")
    return target


def _digest(path: Path) -> tuple[str, int]:
    sha = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            block = stream.read(_CHUNK_SIZE)
            if not block:
                break
            sha.update(block)
            size += len(block)
    return sha.hexdigest(), size


def verify_asset(path: str | Path, asset: ModelAsset) -> bool:
    """Return True only when the local file matches all pinned metadata."""
    file_path = Path(path)
    if not file_path.is_file() or file_path.is_symlink() or not asset.sha256:
        return False
    if asset.size_bytes is not None and file_path.stat().st_size != asset.size_bytes:
        return False
    actual, size = _digest(file_path)
    return size > 0 and actual == asset.sha256


class _AssetLock:
    """Small cross-process lock based on exclusive file creation."""

    def __init__(self, path: Path, timeout: float = 120.0):
        self.path = path
        self.timeout = timeout
        self.fd: int | None = None

    def __enter__(self):
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.write(self.fd, f"{os.getpid()}\n".encode())
                return self
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise AssetError(f"Timed out waiting for asset lock: {self.path}")
                time.sleep(0.2)

    def __exit__(self, exc_type, exc, tb):
        if self.fd is not None:
            os.close(self.fd)
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def _request(url: str, offset: int, timeout: float):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("https", "http") or not parsed.netloc:
        raise AssetError(f"Unsupported asset URL (HTTP(S) required): {url!r}")
    headers = {"User-Agent": "mseparate/0.1", "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(request, timeout=timeout)


def _download_once(asset: ModelAsset, part: Path, timeout: float,
                   expected_size: int | None) -> int:
    offset = part.stat().st_size if part.exists() else 0
    with _request(asset.url, offset, timeout) as response:
        status = getattr(response, "status", response.getcode())
        content_range = response.headers.get("Content-Range")
        if offset and status == 206:
            match = _CONTENT_RANGE.match(content_range or "")
            if not match or int(match.group(1)) != offset:
                raise AssetError("Server returned an invalid Content-Range; refusing unsafe resume")
            mode = "ab"
        elif status == 200:
            # Server ignored Range (or this is a fresh transfer): restart
            # from byte zero rather than appending a full response.
            mode = "wb"
            offset = 0
        elif status == 206 and not offset:
            match = _CONTENT_RANGE.match(content_range or "")
            if not match or int(match.group(1)) != 0:
                raise AssetError("Unexpected partial response for a fresh download")
            mode = "wb"
        else:
            raise AssetError(f"Unexpected HTTP status {status}")

        received = 0
        with part.open(mode) as output:
            while True:
                block = response.read(_CHUNK_SIZE)
                if not block:
                    break
                received += len(block)
                if expected_size is not None and offset + received > expected_size:
                    raise AssetError("Download exceeds the manifest's expected file size")
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
    return received


def ensure_asset(model_dir: str | Path, asset: ModelAsset, *,
                 timeout: float = 30.0, retries: int = 3,
                 lock_timeout: float = 120.0) -> AssetResult:
    """Ensure one asset is present, complete and SHA-256 verified.

    A valid file is reused. An invalid existing file is preserved under a
    unique .corrupt-* name; it is never silently trusted or deleted.
    """
    if not asset.sha256:
        raise AssetError(f"Asset {asset.role!r} has no pinned SHA-256; refusing download")
    if not re.fullmatch(r"[0-9a-f]{64}", asset.sha256):
        raise AssetError(f"Asset {asset.role!r} has an invalid SHA-256")
    if retries < 1:
        raise ValueError("retries must be at least 1")
    root = Path(model_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    target = _safe_asset_path(root, asset.relative_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    lock = target.with_name(target.name + ".lock")

    with _AssetLock(lock, timeout=lock_timeout):
        if verify_asset(target, asset):
            return AssetResult(asset.role, "cached", str(target))

        if target.exists():
            preserved = target.with_name(target.name + ".corrupt-" + uuid.uuid4().hex[:8])
            target.replace(preserved)

        downloaded = 0
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                downloaded += _download_once(asset, part, timeout, asset.size_bytes)
                actual_sha, actual_size = _digest(part)
                if asset.size_bytes is not None and actual_size != asset.size_bytes:
                    raise AssetError(
                        f"Size mismatch for {asset.role}: expected {asset.size_bytes}, got {actual_size}"
                    )
                if actual_sha != asset.sha256:
                    raise AssetError(
                        f"SHA-256 mismatch for {asset.role}: expected {asset.sha256}, got {actual_sha}"
                    )
                os.replace(part, target)
                return AssetResult(asset.role, "downloaded", str(target),
                                   bytes_downloaded=downloaded)
            except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError, AssetError) as exc:
                last_error = exc
                # Keep a resumable partial file for transient failures. A
                # checksum/size failure cannot be repaired by appending;
                # discard only the incomplete .part, never a prior asset.
                if isinstance(exc, AssetError) and ("SHA-256 mismatch" in str(exc)
                                                     or "Size mismatch" in str(exc)
                                                     or "exceeds the manifest" in str(exc)):
                    try:
                        part.unlink()
                    except FileNotFoundError:
                        pass
                if attempt + 1 < retries:
                    time.sleep(min(2 ** attempt, 8))
        raise AssetError(f"Could not install asset {asset.role!r} after {retries} attempt(s): {last_error}") from last_error


def ensure_model_assets(manifest: ModelManifest, models_root: str | Path, *,
                        no_download: bool = False, timeout: float = 30.0,
                        retries: int = 3) -> list[AssetResult]:
    """Check/download all assets independently; one failure never skips peers."""
    model_dir = Path(models_root).expanduser() / manifest.model_id
    results: list[AssetResult] = []
    for asset in manifest.assets:
        try:
            target = _safe_asset_path(model_dir, asset.relative_path)
            if verify_asset(target, asset):
                results.append(AssetResult(asset.role, "cached", str(target)))
            elif no_download:
                results.append(AssetResult(asset.role, "missing", str(target),
                                           "Offline mode: asset is not verified locally"))
            else:
                results.append(ensure_asset(model_dir, asset, timeout=timeout, retries=retries))
        except Exception as exc:
            results.append(AssetResult(asset.role, "failed", message=str(exc)))
    return results
