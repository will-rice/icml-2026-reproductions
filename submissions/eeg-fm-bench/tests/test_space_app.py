from __future__ import annotations

import importlib.util
import inspect
import json
import socket
import sys
import tomllib
from html.parser import HTMLParser
from pathlib import Path
from types import ModuleType
from typing import get_type_hints

import gradio as gr
from gradio_client import Client
from huggingface_hub.repocard import metadata_load


PROJECT_ROOT = Path(__file__).parents[1]
APP_PATH = PROJECT_ROOT / "app.py"
POSTER_PATH = PROJECT_ROOT / "poster_embed.html"
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"
README_PATH = PROJECT_ROOT / "README.md"

EXPECTED_STATUSES = {
    "fourteen-dataset-ten-paradigm-curation": "partial",
    "standardized-preprocessing-reproducibility": "verified",
    "three-strategy-evaluation-harness": "partial",
}
EXPECTED_TAGS = [
    "icml2026-repro",
    "open-experiment",
    "paper-vGeNaFHdET",
]


class IframeParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.iframes: list[dict[str, str | None]] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag == "iframe":
            self.iframes.append(dict(attrs))


def load_space_app() -> ModuleType:
    spec = importlib.util.spec_from_file_location("eeg_fm_bench_space_app", APP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_space_card_selects_the_exact_platform_runtime() -> None:
    metadata = metadata_load(README_PATH)

    assert metadata is not None
    assert metadata["sdk"] == "gradio"
    assert metadata["sdk_version"] == "6.20.0"
    assert metadata["python_version"] == "3.12"
    assert metadata["app_file"] == "app.py"
    assert metadata["tags"] == EXPECTED_TAGS
    assert 0 < len(metadata["short_description"]) <= 60
    assert not (PROJECT_ROOT / "requirements.txt").exists()


def test_project_declares_the_exact_gradio_runtime_dependency() -> None:
    metadata = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    dependencies = metadata["project"]["dependencies"]

    assert dependencies.count("gradio==6.20.0") == 1


def test_import_is_offline_and_does_not_launch(
    monkeypatch,
) -> None:
    def reject_outbound_connect(
        _socket: socket.socket,
        address: object,
    ) -> None:
        raise AssertionError(f"unexpected network connection during import: {address}")

    monkeypatch.setattr(socket.socket, "connect", reject_outbound_connect)
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    module = load_space_app()

    assert gr.__version__ == "6.20.0"
    assert isinstance(module.demo, gr.Blocks)
    assert module.demo.local_url is None


def test_summary_api_is_typed_documented_and_derived_from_committed_evidence() -> None:
    module = load_space_app()

    signature = inspect.signature(module.evidence_summary)
    assert not signature.parameters
    assert get_type_hints(module.evidence_summary)["return"] == dict[str, object]
    assert inspect.getdoc(module.evidence_summary)

    summary = module.evidence_summary()
    assert summary["scope"] == "released_artifact_audit_not_leaderboard_reproduction"
    assert {
        claim["claim_id"]: claim["status"]
        for claim in summary["claims"]
    } == EXPECTED_STATUSES
    assert {
        item["status"] for item in summary["unavailable_claims"]
    } == {"unavailable"}


def test_demo_renders_committed_results_and_poster_and_registers_public_api() -> None:
    module = load_space_app()
    config = module.demo.get_config_file()
    rendered_config = json.dumps(config, sort_keys=True)

    assert "EEG-FM-Bench released-artifact audit" in rendered_config
    assert all(claim_id in rendered_config for claim_id in EXPECTED_STATUSES)
    assert "GPU leaderboard: unavailable." in rendered_config

    dependency = next(
        item
        for item in config["dependencies"]
        if item["api_name"] == "evidence_summary"
    )
    assert dependency["inputs"] == []
    assert dependency["outputs"]
    assert dependency["api_visibility"] == "public"
    assert dependency["api_description"] == inspect.getdoc(
        module.evidence_summary
    )


def test_poster_document_is_isolated_in_a_scriptless_sandbox() -> None:
    module = load_space_app()
    config = module.demo.get_config_file()
    poster_component = next(
        component
        for component in config["components"]
        if component["type"] == "html"
    )
    rendered_html = poster_component["props"]["value"]
    parser = IframeParser()
    parser.feed(rendered_html)

    assert len(parser.iframes) == 1
    iframe = parser.iframes[0]
    assert "sandbox" in iframe
    assert "allow-scripts" not in (iframe["sandbox"] or "").split()
    assert iframe["srcdoc"] == POSTER_PATH.read_text(encoding="utf-8")
    assert "<style>" not in rendered_html
    assert "<main>" not in rendered_html


def test_gradio_620_launch_exposes_stable_summary_endpoint(
    monkeypatch,
) -> None:
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    module = load_space_app()

    _, local_url, _ = module.demo.launch(
        prevent_thread_lock=True,
        server_name="127.0.0.1",
        share=False,
        quiet=True,
    )
    try:
        response = Client(local_url, verbose=False).predict(
            api_name="/evidence_summary"
        )
        assert {
            claim["claim_id"]: claim["status"]
            for claim in response["claims"]
        } == EXPECTED_STATUSES
    finally:
        module.demo.close()
