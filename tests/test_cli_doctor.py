"""Unit tests for catml doctor command."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from automl.interfaces.cli.doctor_cli import run_doctor_cli, run_doctor_diagnostics


def test_doctor_diagnostics_structure(tmp_path: Path):
    diag = run_doctor_diagnostics(workspace_dir=tmp_path)
    assert "platform_version" in diag
    assert "python" in diag
    assert "core_stack" in diag
    assert "model_backends" in diag
    assert "hardware" in diag
    assert "workspace" in diag
    assert diag["workspace"]["writable"] is True
    assert isinstance(diag["ready"], bool)


def test_doctor_cli_human_output(tmp_path: Path, capsys):
    args = argparse.Namespace(workspace=str(tmp_path), json=False)
    exit_code = run_doctor_cli(args)
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "CATML Environment & System Doctor" in captured
    assert "Núcleo Tabular ML" in captured
    assert "NumPy" in captured


def test_doctor_cli_json_output(tmp_path: Path, capsys):
    args = argparse.Namespace(workspace=str(tmp_path), json=True)
    exit_code = run_doctor_cli(args)
    assert exit_code == 0
    captured = capsys.readouterr().out
    data = json.loads(captured)
    assert data["python"]["compatible"] is True
    assert "core_stack" in data
