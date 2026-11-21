from __future__ import annotations

import io
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import download_mseparate_babyslakh_subset as subset


def test_track_metadata_members_are_sorted_and_track_scoped():
    names = [
        "Track00002/metadata.yaml",
        "Track00001/mix.wav",
        "Track00001/metadata.yaml",
        "other/readme.txt",
    ]
    assert subset._track_metadata_members(names) == {
        "Track00001": "Track00001/metadata.yaml",
        "Track00002": "Track00002/metadata.yaml",
    }


def test_drum_stem_ids_require_explicit_boolean_ground_truth():
    metadata = {
        "stems": {
            "S01": {"is_drum": True},
            "S02": {"is_drum": False},
            "S03": {"is_drum": "true"},
            "S04": None,
        }
    }
    assert subset._drum_stem_ids(metadata) == ["S01"]


def test_resolve_mix_and_drum_stem_members():
    names = {
        "Track00001/mix.wav",
        "Track00001/stems/S01.wav",
        "Track00001/stems/S02.flac",
        "Track00002/mix.wav",
    }
    assert subset._find_mix_member("Track00001", names) == "Track00001/mix.wav"
    assert subset._find_stem_member(
        "Track00001", {"audio_dir": "stems"}, "S01", names
    ) == "Track00001/stems/S01.wav"
    assert subset._find_stem_member(
        "root/Track00001", {"audio_dir": "Track00001/stems"}, "S01",
        {"root/Track00001/stems/S01.wav"},
    ) == "root/Track00001/stems/S01.wav"


def test_safe_destination_rejects_parent_traversal(tmp_path):
    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        subset._safe_destination(tmp_path, "../outside.wav")



def test_midi_program_family_uses_general_midi_groups():
    assert subset._midi_program_family(0) == "piano"
    assert subset._midi_program_family(40) == "strings"
    assert subset._midi_program_family(80) == "synth_lead"
    assert subset._midi_program_family(127) == "sound_effects"
    assert subset._midi_program_family(-1) is None
    assert subset._midi_program_family(128) is None
    assert subset._midi_program_family("unknown") is None


def test_instrument_program_families_ignore_drum_stems():
    metadata = {"stems": {
        "S01": {"is_drum": True, "program_num": 0},
        "S02": {"is_drum": False, "program_num": 40},
        "S03": {"is_drum": False, "program_num": 80},
        "S04": {"is_drum": False, "program_num": 81},
    }}
    assert subset._instrument_program_families(metadata) == ["strings", "synth_lead"]


def test_diverse_candidate_selection_prioritizes_new_program_families():
    candidates = [
        {"track": "first", "instrument_program_families": ["strings"]},
        {"track": "second", "instrument_program_families": ["strings"]},
        {"track": "third", "instrument_program_families": ["synth_lead"]},
        {"track": "fourth", "instrument_program_families": ["brass", "reed"]},
    ]
    selected = subset._select_diverse_candidates(candidates, 3)
    assert [item["track"] for item in selected] == ["fourth", "first", "third"]



def test_download_subset_fetches_only_selected_members(tmp_path, monkeypatch):
    class FakeArchive:
        read_members = []
        names = [
            f"Track{i:05d}/metadata.yaml" for i in range(1, 7)
        ] + [
            f"Track{i:05d}/mix.wav" for i in range(1, 7)
        ] + [
            f"Track{i:05d}/stems/S01.wav" for i in range(1, 7)
        ]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def namelist(self):
            return list(self.names)

        def read(self, member):
            self.read_members.append(member)
            return (
                "audio_dir: stems\nstems:\n  S01:\n    is_drum: true\n"
            ).encode("utf-8")

        def open(self, member, _mode="r"):
            return io.BytesIO(b"test-audio" if member.endswith(".wav") else b"")

    archive = FakeArchive()
    monkeypatch.setitem(
        sys.modules, "remotezip",
        types.SimpleNamespace(RemoteZip=lambda _url: archive),
    )
    monkeypatch.setitem(
        sys.modules, "yaml",
        types.SimpleNamespace(safe_load=lambda raw: {
            "audio_dir": "stems",
            "stems": {"S01": {"is_drum": True}},
        }),
    )

    output = tmp_path / "audio"
    manifest = tmp_path / "reports" / "manifest.json"
    result = subset.download_subset(output, manifest, max_tracks=4)

    assert result["selected_tracks"] == 4
    assert result["metadata_files_scanned"] == 6
    assert result["candidate_tracks_found"] == 6
    assert result["selection_strategy"] == "greedy_gm_program_family_coverage_then_archive_order"
    assert len(archive.read_members) == 4
    assert result["full_archive_downloaded"] is False
    assert result["download_method"].startswith("HTTP range")
    assert len(result["tracks"]) == 4
    assert all(row["ground_truth"]["drums_present"] for row in result["tracks"])
    assert all((output / row["mix"]).is_file() for row in result["tracks"])
    assert all(
        (output / stem).is_file()
        for row in result["tracks"] for stem in row["drum_stems"]
    )
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    assert saved["selected_tracks"] == 4
