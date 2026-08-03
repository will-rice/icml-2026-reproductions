import importlib.util
from pathlib import Path

import gradio as gr
from gradio_client import Client


ROOT = Path(__file__).resolve().parents[1]


def _load_app():
    spec = importlib.util.spec_from_file_location(
        "timerewarder_space_app", ROOT / "app.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_app_exports_small_artifact_api_functions() -> None:
    app = _load_app()
    assert isinstance(app.demo, gr.Blocks)
    claims = app.claim_records()
    summary = app.evidence_summary()
    first = app.rerun_fixture()
    second = app.rerun_fixture()

    assert len(claims["claims"]) == 6
    assert claims["measurement_sha256"] == summary["measurement_sha256"]
    assert first["diagnostic_only"] is True
    assert first["measurement_sha256"] == second["measurement_sha256"]
    assert "checkpoint" not in first
    assert "video" not in first


def test_named_endpoints_work_through_local_gradio_client() -> None:
    app = _load_app()
    launch = app.demo.launch(
        prevent_thread_lock=True,
        quiet=True,
        show_error=True,
    )
    try:
        client = Client(launch[1], verbose=False)
        claims = client.predict(api_name="/claim_records")
        summary = client.predict(api_name="/evidence_summary")
        first = client.predict(api_name="/rerun_fixture")
        second = client.predict(api_name="/rerun_fixture")
        assert len(claims["claims"]) == 6
        assert claims["measurement_sha256"] == summary["measurement_sha256"]
        assert first["measurement_sha256"] == second["measurement_sha256"]
    finally:
        app.demo.close()


def test_app_has_no_payload_or_conversion_operations() -> None:
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    forbidden = (
        "torch.load",
        "safetensors",
        "decord",
        "checkpoint",
        "video",
        "convert",
        "reviewer",
        "requests.",
        "http://",
        "https://",
    )
    assert not any(token in source for token in forbidden)


def test_space_adds_src_before_importing_project_package() -> None:
    source = (ROOT / "app.py").read_text(encoding="utf-8")

    assert source.index("sys.path.insert") < source.index(
        "from timerewarder_repro"
    )
