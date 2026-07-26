"""Global isolation guard for the PostTrainBench test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from posttrainbench_repro.constants import CANONICAL_OUTPUTS
from posttrainbench_repro.pipeline import PROJECT_ROOT


_REAL_OUTPUT_PATHS = {
    (PROJECT_ROOT / relative_path).resolve()
    for relative_path in CANONICAL_OUTPUTS
}


@pytest.fixture(scope="session", autouse=True)
def canonical_project_bytes_remain_unchanged():
    """The complete test session must preserve every real canonical output."""
    before = {
        path: path.read_bytes() if path.exists() else None
        for path in _REAL_OUTPUT_PATHS
    }
    yield
    after = {
        path: path.read_bytes() if path.exists() else None
        for path in _REAL_OUTPUT_PATHS
    }
    assert after == before


@pytest.fixture(autouse=True)
def canonical_project_outputs_are_read_only(monkeypatch):
    """Fail immediately if a test tries to mutate a real canonical output."""
    original_write_text = Path.write_text
    original_write_bytes = Path.write_bytes
    original_unlink = Path.unlink

    def require_isolated(path: Path) -> None:
        if path.resolve() in _REAL_OUTPUT_PATHS:
            pytest.fail(
                f"test attempted to mutate real canonical output: {path}"
            )

    def guarded_write_text(path: Path, *args, **kwargs):
        require_isolated(path)
        return original_write_text(path, *args, **kwargs)

    def guarded_write_bytes(path: Path, *args, **kwargs):
        require_isolated(path)
        return original_write_bytes(path, *args, **kwargs)

    def guarded_unlink(path: Path, *args, **kwargs):
        require_isolated(path)
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", guarded_write_text)
    monkeypatch.setattr(Path, "write_bytes", guarded_write_bytes)
    monkeypatch.setattr(Path, "unlink", guarded_unlink)
