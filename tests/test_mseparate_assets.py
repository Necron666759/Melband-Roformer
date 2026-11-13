"""CPU-only tests for the mseparate asset manager; no model weights or GPU required."""
from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from mseparate.assets import AssetError, ensure_asset, ensure_model_assets, verify_asset
from mseparate.registry import RegistryError, eligible_models, load_registry, parse_manifest
from mseparate.storage import ensure_layout, write_json_atomic
from mseparate.types import ModelAsset


class _AssetHandler(BaseHTTPRequestHandler):
    payload = b"mseparate-test-checkpoint-bytes"
    status_code = 200
    range_requests: list[str] = []

    def do_GET(self):
        type(self).range_requests.append(self.headers.get("Range", ""))
        if type(self).status_code != 200:
            self.send_error(type(self).status_code)
            return
        range_header = self.headers.get("Range")
        if range_header:
            start = int(range_header.removeprefix("bytes=").split("-", 1)[0])
            body = type(self).payload[start:]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(type(self).payload)-1}/{len(type(self).payload)}")
        else:
            body = type(self).payload
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


@pytest.fixture
def asset_server():
    _AssetHandler.status_code = 200
    _AssetHandler.range_requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _AssetHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/checkpoint"
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def _asset(url: str, payload: bytes | None = None, relative_path="weights/model.ckpt") -> ModelAsset:
    body = payload if payload is not None else _AssetHandler.payload
    return ModelAsset(
        role="checkpoint", url=url, relative_path=relative_path,
        sha256=hashlib.sha256(body).hexdigest(), size_bytes=len(body),
    )


def test_verified_asset_downloaded_and_reused(tmp_path, asset_server):
    asset = _asset(asset_server)
    model_dir = tmp_path / "models" / "model-a"
    first = ensure_asset(model_dir, asset)
    assert first.status == "downloaded"
    assert verify_asset(first.path, asset)
    second = ensure_asset(model_dir, asset)
    assert second.status == "cached"
    assert len(_AssetHandler.range_requests) == 1


def test_partial_file_resumes_with_http_range(tmp_path, asset_server):
    asset = _asset(asset_server)
    model_dir = tmp_path / "model-a"
    target = model_dir / asset.relative_path
    target.parent.mkdir(parents=True)
    part = target.with_name(target.name + ".part")
    split = len(_AssetHandler.payload) // 2
    part.write_bytes(_AssetHandler.payload[:split])

    result = ensure_asset(model_dir, asset)
    assert result.status == "downloaded"
    assert verify_asset(result.path, asset)
    assert _AssetHandler.range_requests == [f"bytes={split}-"]


def test_server_ignoring_range_restarts_instead_of_appending(tmp_path, asset_server):
    asset = _asset(asset_server)
    model_dir = tmp_path / "model-a"
    target = model_dir / asset.relative_path
    target.parent.mkdir(parents=True)
    part = target.with_name(target.name + ".part")
    part.write_bytes(b"partial-garbage")

    # A handler without Range support answers with 200 and the complete body.
    class IgnoreRange(_AssetHandler):
        def do_GET(self):
            type(self).range_requests.append(self.headers.get("Range", ""))
            body = type(self).payload
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), IgnoreRange)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/checkpoint"
        result = ensure_asset(model_dir, _asset(url))
        assert result.status == "downloaded"
        assert verify_asset(result.path, asset)
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_checksum_mismatch_refuses_install_and_preserves_no_bad_final(tmp_path, asset_server):
    wrong = ModelAsset(
        role="checkpoint", url=asset_server, relative_path="model.ckpt",
        sha256="0" * 64, size_bytes=len(_AssetHandler.payload),
    )
    with pytest.raises(AssetError, match="SHA-256 mismatch"):
        ensure_asset(tmp_path / "model-a", wrong, retries=1)
    assert not (tmp_path / "model-a" / "model.ckpt").exists()


def test_corrupt_existing_asset_is_preserved_before_replacement(tmp_path, asset_server):
    asset = _asset(asset_server)
    model_dir = tmp_path / "model-a"
    target = model_dir / asset.relative_path
    target.parent.mkdir(parents=True)
    target.write_bytes(b"corrupt old file")

    result = ensure_asset(model_dir, asset)
    assert result.status == "downloaded"
    assert verify_asset(result.path, asset)
    preserved = list(target.parent.glob(target.name + ".corrupt-*"))
    assert len(preserved) == 1
    assert preserved[0].read_bytes() == b"corrupt old file"


def test_no_download_reports_missing_asset_without_network(tmp_path, asset_server):
    from mseparate.types import ModelManifest

    manifest = ModelManifest(
        model_id="model-a", name="Model A", architecture="test", backend="test",
        source_url="https://example.invalid", license="MIT",
        license_url="https://example.invalid/license", license_status="verified",
        assets=(_asset(asset_server),), status="available",
    )
    results = ensure_model_assets(manifest, tmp_path / "models", no_download=True)
    assert results[0].status == "missing"
    assert _AssetHandler.range_requests == []


def test_asset_failures_do_not_abort_other_assets(tmp_path, asset_server):
    from mseparate.types import ModelManifest

    good = _asset(asset_server)
    bad = ModelAsset(
        role="config", url="file:///etc/passwd", relative_path="config.yaml",
        sha256=hashlib.sha256(b"x").hexdigest(), size_bytes=1,
    )
    manifest = ModelManifest(
        model_id="model-a", name="Model A", architecture="test", backend="test",
        source_url="https://example.invalid", license="MIT",
        license_url="https://example.invalid/license", license_status="verified",
        assets=(bad, good), status="available",
    )
    results = ensure_model_assets(manifest, tmp_path / "models", retries=1)
    assert [item.status for item in results] == ["failed", "downloaded"]


def test_registry_rejects_model_id_path_traversal():
    with pytest.raises(RegistryError):
        parse_manifest({
            "model_id":"../outside","name":"Bad","architecture":"x","backend":"x",
            "source_url":"https://example.invalid","assets":[],
        })


def test_eligible_registry_requires_pinned_required_asset():
    raw = {
        "model_id":"model-a","name":"Model A","architecture":"test","backend":"test",
        "source_url":"https://example.invalid","license":"MIT",
        "license_url":"https://example.invalid/license","license_status":"verified",
        "status":"available","assets":[{"role":"checkpoint","url":"https://example.invalid/a","relative_path":"a.ckpt"}],
    }
    manifest = parse_manifest(raw)
    assert eligible_models({manifest.model_id: manifest}) == []



def test_verified_partial_file_is_promoted_without_network(tmp_path, asset_server):
    asset = _asset(asset_server)
    model_dir = tmp_path / "model-a"
    target = model_dir / asset.relative_path
    target.parent.mkdir(parents=True)
    part = target.with_name(target.name + ".part")
    part.write_bytes(_AssetHandler.payload)

    result = ensure_asset(model_dir, asset)
    assert result.status == "recovered"
    assert verify_asset(target, asset)
    assert _AssetHandler.range_requests == []


@pytest.mark.parametrize("status_code", [403, 404, 500])
def test_http_errors_are_reported_without_installing_bad_asset(tmp_path, asset_server, status_code):
    _AssetHandler.status_code = status_code
    asset = _asset(asset_server)
    with pytest.raises(AssetError, match="Could not install asset"):
        ensure_asset(tmp_path / "model-a", asset, retries=1)
    assert not (tmp_path / "model-a" / asset.relative_path).exists()
