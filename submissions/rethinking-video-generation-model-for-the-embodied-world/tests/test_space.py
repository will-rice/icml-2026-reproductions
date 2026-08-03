"""Tests for the read-only RBench evidence Space."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import socket

import pytest


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_app_module(project_root: Path):
    spec = importlib.util.spec_from_file_location(
        "rbench_space_app", project_root / "app.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_space_loads_only_committed_evidence(monkeypatch, project_root):
    def fail_network(*_args, **_kwargs):
        raise AssertionError("Space import attempted network access")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    module = load_app_module(project_root)
    assert module.EVIDENCE["paper_id"] == "p5QSlnwume"
    assert (
        module.EVIDENCE["attempt_id"]
        == "8c21f2dc-a357-422e-9c1b-79a4d417e3dc"
    )


def test_readme_and_poster_state_unreproduced_limits(project_root):
    text = (
        (project_root / "README.md").read_text()
        + (project_root / "POSTER.md").read_text()
    ).lower()
    assert "video generation was not rerun" in text
    assert "human correlation was not reproduced" in text


def test_space_metadata_targets_exact_challenge_paper(project_root):
    readme = (project_root / "README.md").read_text()
    assert "sdk: gradio" in readme
    assert "sdk_version: 6.20.0" in readme
    assert "app_file: app.py" in readme
    assert "- paper-p5QSlnwume" in readme
    assert "- icml2026-repro" in readme
