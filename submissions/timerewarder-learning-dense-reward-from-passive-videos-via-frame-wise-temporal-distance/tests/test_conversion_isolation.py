import json
from pathlib import Path
import tomllib

import pytest

from timerewarder_repro.conversion import (
    ConversionRejected,
    _sandbox_command,
    convert_checkpoint,
)


def test_parent_rejects_when_bwrap_is_unavailable(tmp_path: Path, monkeypatch) -> None:
    request = tmp_path / "request.json"
    request.write_text(json.dumps({}), encoding="utf-8")
    monkeypatch.setattr("timerewarder_repro.conversion.shutil.which", lambda _: None)

    with pytest.raises(ConversionRejected, match="sandbox_unavailable"):
        convert_checkpoint(request)


def test_sandbox_command_binds_audited_source_and_interpreter(tmp_path: Path) -> None:
    command = _sandbox_command(
        checkpoint=tmp_path / "checkpoint.pth",
        schema_path=tmp_path / "model-schema.json",
        output_dir=tmp_path,
        runtime=tmp_path / "venv",
        package_root=tmp_path / "package",
        python_root=tmp_path / "python",
    )

    assert "--unshare-all" in command
    assert ["--ro-bind", str(tmp_path / "package"), "/package"] == command[
        command.index("--ro-bind", command.index("/runtime") + 1) : command.index(
            "--ro-bind", command.index("/runtime") + 1
        )
        + 3
    ]
    assert "PYTHONPATH" in command
    assert "/package:/runtime/lib/python3.12/site-packages" in command
    assert command[-7:] == [
        "/python/bin/python3.12",
        "-m",
        "timerewarder_repro.conversion",
        "child",
        "/input/checkpoint.pth",
        "/input/model-schema.json",
        "/output/model.safetensors",
    ]


def test_child_runtime_declares_safetensors_dependencies() -> None:
    pyproject = Path(__file__).parents[1] / "conversion" / "pyproject.toml"
    dependencies = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"][
        "dependencies"
    ]

    assert {"numpy==2.3.2", "packaging==25.0"} <= set(dependencies)
