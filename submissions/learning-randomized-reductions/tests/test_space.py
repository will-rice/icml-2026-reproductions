import importlib.util
from pathlib import Path
import socket
import pytest


def test_space_is_offline_and_evidence_backed(monkeypatch):
    def fail_network(*args, **kwargs):
        raise RuntimeError("Network access disabled during space load")

    monkeypatch.setattr(socket, "create_connection", fail_network)

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    spec = importlib.util.spec_from_file_location("app_module", app_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.EVIDENCE["paper_id"] == "hCAEcqig2C"
    assert len(module.PAGE_TEXT) == 7


def test_readme_has_exact_space_metadata(project_root):
    readme = (project_root / "README.md").read_text(encoding="utf-8")
    assert "sdk_version: 6.20.0" in readme
    assert "- paper-hCAEcqig2C" in readme
    assert "- icml2026-repro" in readme
