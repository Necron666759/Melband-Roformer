from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from inspect_mseparate_stemgmd_archive import ARCHIVE_URL, _open_remote_archive


def test_archive_inventory_retries_and_uses_zenodo_content_endpoint(monkeypatch):
    import requests

    calls = []
    archive = object()

    def remote_zip(url, *, session):
        calls.append((url, session))
        assert isinstance(session, requests.Session)
        if len(calls) < 3:
            raise OSError("temporary Zenodo TLS/proxy failure")
        return archive

    monkeypatch.setitem(
        sys.modules, "remotezip", types.SimpleNamespace(RemoteZip=remote_zip)
    )

    result = _open_remote_archive(
        sys.modules["remotezip"], ARCHIVE_URL
    )

    assert result is archive
    assert [item[0] for item in calls] == [
        ARCHIVE_URL,
        ARCHIVE_URL.replace("?download=1", ""),
        "https://zenodo.org/api/records/7882857/files/StemGMD_eval_session.zip/content",
    ]
    assert len({id(item[1]) for item in calls}) == 1
