"""Regression tests for the restricted AppImage command shell."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SHELL_PATH = ROOT / "appimage" / "melband-roformer-shell.py"


def _load_shell():
    spec = importlib.util.spec_from_file_location("melband_roformer_appimage_shell", SHELL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mseparate_help_docstring_is_valid_and_visible():
    shell_module = _load_shell()
    doc = shell_module.MelbandRoformerShell.do_mseparate.__doc__
    assert doc is not None
    assert "mseparate <file>" in doc
    assert "--analysis-only" in doc
    assert "--plan-only" in doc
    assert "--report [PATH]" in doc


def test_mseparate_command_dispatches_parsed_arguments(monkeypatch):
    calls = []
    fake_cli = types.ModuleType("mseparate.cli")
    fake_cli.main = lambda argv: calls.append(argv) or 0
    fake_package = types.ModuleType("mseparate")
    fake_package.__path__ = []
    monkeypatch.setitem(sys.modules, "mseparate", fake_package)
    monkeypatch.setitem(sys.modules, "mseparate.cli", fake_cli)

    shell_module = _load_shell()
    shell = shell_module.MelbandRoformerShell()
    shell.default('mseparate "input file.wav" --analysis-only')

    assert calls == [["input file.wav", "--analysis-only"]]


def test_mseparate_argument_parse_error_stays_inside_shell(monkeypatch, capsys):
    calls = []
    fake_cli = types.ModuleType("mseparate.cli")
    fake_cli.main = lambda argv: calls.append(argv) or 0
    fake_package = types.ModuleType("mseparate")
    fake_package.__path__ = []
    monkeypatch.setitem(sys.modules, "mseparate", fake_package)
    monkeypatch.setitem(sys.modules, "mseparate.cli", fake_cli)

    shell_module = _load_shell()
    shell = shell_module.MelbandRoformerShell()
    shell.default('mseparate "unterminated')

    assert calls == []
    assert "Could not parse arguments" in capsys.readouterr().out


def test_unknown_shell_command_is_refused(capsys):
    shell_module = _load_shell()
    shell = shell_module.MelbandRoformerShell()
    shell.default("ls -la")

    output = capsys.readouterr().out
    assert "unrecognized input" in output
    assert "does not run shell commands" not in output
