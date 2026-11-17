from __future__ import annotations

import csv
import io
import sys
import types
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import download_mseparate_stemgmd_subset as subset


def _wav_bytes(value: float, frames: int = 256) -> bytes:
    buffer = io.BytesIO()
    audio = np.full((frames, 2), value, dtype=np.float32)
    sf.write(buffer, audio, subset.SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _members(drummer: str, kit: str, groove: str, *, silent_toms: bool = False) -> dict[str, bytes]:
    return {
        f"StemGMD/audio/{drummer}/eval_session/{kit}/{instrument}/{groove}.wav":
            _wav_bytes(
                0.0 if silent_toms and instrument in subset.TOM_INSTRUMENTS
                else (index + 1) / 1000
            )
        for index, instrument in enumerate(subset.INSTRUMENTS)
    }


def test_index_archive_requires_all_nine_aligned_source_members():
    complete = _members("drummer1", "bluebird", "1_funk-groove1_138_beat_4-4")
    incomplete = _members("drummer2", "detroit_garage", "2_funk-groove2_105_beat_4-4")
    incomplete.pop(next(iter(incomplete)))
    infos = [
        types.SimpleNamespace(filename=name, file_size=len(data))
        for name, data in {**complete, **incomplete}.items()
    ]

    groups = subset.index_archive_members(infos)

    assert len(groups) == 1
    assert groups[0]["drummer"] == "drummer1"
    assert groups[0]["kit"] == "bluebird"
    assert set(groups[0]["members"]) == set(subset.INSTRUMENTS)


def test_track_selection_prefers_distinct_kits_and_drummers():
    groups = []
    for index in range(4):
        groups.append({
            "drummer": f"drummer{index + 1}",
            "kit": f"kit{index + 1}",
            "groove": f"{index + 1}_groove.wav",
            "members": {name: f"{index}/{name}.wav" for name in subset.INSTRUMENTS},
        })

    selected = subset.select_tracks(groups, 4)

    assert len(selected) == 4
    assert len({row["kit"] for row in selected}) == 4
    assert len({row["drummer"] for row in selected}) == 4


def test_download_builds_mixture_and_manifest_from_selected_members(tmp_path, monkeypatch):
    members = _members("drummer1", "bluebird", "1_funk-groove1_138_beat_4-4")

    class FakeArchive:
        opened: list[str] = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def infolist(self):
            return [
                types.SimpleNamespace(filename=name, file_size=len(data))
                for name, data in members.items()
            ]

        def open(self, name, _mode="r"):
            self.opened.append(name)
            return io.BytesIO(members[name])

    archive = FakeArchive()
    monkeypatch.setitem(
        sys.modules, "remotezip",
        types.SimpleNamespace(RemoteZip=lambda _url: archive),
    )

    output = tmp_path / "audio"
    manifest = tmp_path / "reports" / "stemgmd.csv"
    report = tmp_path / "reports" / "subset.json"
    result = subset.download_subset(output, manifest, report, max_tracks=1)

    assert result["selected_tracks"] == 1
    assert result["full_archive_downloaded"] is False
    assert result["extracted_audio_files"] == 10
    assert len(archive.opened) == 9
    assert len(list(output.rglob("*.wav"))) == 10

    with manifest.open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert set(row) == {"track_id", "mixture", "snare", "toms"}
    assert len(row["toms"].split(";")) == 3

    mixture_path = (manifest.parent / row["mixture"]).resolve()
    mixture, rate = sf.read(str(mixture_path), dtype="float32", always_2d=True)
    expected = sum(
        sf.read(str(output / row["track_id"] / "references" / f"{name}.wav"),
                dtype="float32", always_2d=True)[0]
        for name in subset.INSTRUMENTS
    )
    assert rate == subset.SAMPLE_RATE
    assert mixture.shape == expected.shape
    np.testing.assert_allclose(mixture, expected, atol=2e-6)

    assert report.is_file()
    assert result["quality_evaluation"]["status"] == "prepared_not_run"


def test_select_tracks_rejects_more_than_eight():
    try:
        subset.select_tracks([], 9)
    except ValueError as exc:
        assert "between 1 and 8" in str(exc)
    else:
        raise AssertionError("Expected max_tracks validation")


def test_download_skips_tracks_with_silent_toms_and_avoids_other_stems(tmp_path, monkeypatch):
    members = {
        **_members("drummer1", "z_kit", "1_silent-groove.wav", silent_toms=True),
        **_members("drummer2", "a_kit", "2_active-groove.wav"),
    }

    class FakeArchive:
        opened: list[str] = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def infolist(self):
            return [
                types.SimpleNamespace(filename=name, file_size=len(data))
                for name, data in members.items()
            ]

        def open(self, name, _mode="r"):
            self.opened.append(name)
            return io.BytesIO(members[name])

    archive = FakeArchive()
    monkeypatch.setitem(
        sys.modules, "remotezip",
        types.SimpleNamespace(RemoteZip=lambda _url: archive),
    )

    output = tmp_path / "audio"
    manifest = tmp_path / "reports" / "stemgmd.csv"
    report = tmp_path / "reports" / "subset.json"
    result = subset.download_subset(output, manifest, report, max_tracks=1)

    assert result["selected_tracks"] == 1
    assert result["candidate_groups_checked"] == 2
    assert result["selected_track_metadata"][0]["kit"] == "a_kit"
    assert result["rejected_silent_target_tracks"] == [{
        "track_id": "drummer1__z_kit__1_silent-groove",
        "reason": "silent_toms_reference",
    }]
    # Four target references for the rejected track; all nine for the accepted one.
    assert len(archive.opened) == 13
    assert len(list(output.rglob("*.wav"))) == 10
