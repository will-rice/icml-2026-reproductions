import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "upstream_minimal"


def test_evidence_bundle_binds_selected_claims_and_hashes(monkeypatch):
    monkeypatch.setenv("PIPESD_SOURCE_DIR", str(FIXTURE_ROOT))
    monkeypatch.setenv("ICML_REPRO_GENERATED_AT", "2026-07-29T13:10:00+00:00")

    from pipesd_repro.evidence import build_evidence_bundle

    bundle = build_evidence_bundle()

    assert bundle["paper_id"] == "1ebAvNphi7"
    assert bundle["attempt_id"] == "987c1913-c83b-4be0-acc6-7dae2e0a97e1"
    assert bundle["upstream_revision"] == (
        "arxiv:2605.13319+github:Ghanyunhe/PipeSD@dac0f52eae7ba55e5def2d82003d8413cb58340c"
    )
    assert [claim["challenge_claim_sha256"] for claim in bundle["claims"]] == [
        "19a23e4cc1dfac8ce0e14f60a54e7e0936077f6a149691b73bd82ab72db5672e",
        "ef7fc12943849e199ae46aa7e3e68b0392afb74ad1d465c887bcb3a9b645066b",
    ]
    assert all(claim["status"] == "verified" for claim in bundle["claims"])
    assert not any("reported_tpt_speedup" in claim for claim in bundle["claims"])


def test_evidence_detects_dp_scheduling_and_dual_threshold_nav(monkeypatch, tmp_path):
    monkeypatch.setenv("PIPESD_SOURCE_DIR", str(FIXTURE_ROOT))

    from pipesd_repro.evidence import build_evidence_bundle, write_evidence_bundle

    bundle = build_evidence_bundle()
    summary = bundle["implementation_summary"]

    assert summary["dynamic_token_scheduling_dp"] is True
    assert summary["engine_uses_dp_merge_plan"] is True
    assert summary["hybrid_strategy_sets_dual_thresholds"] is True
    assert summary["single_and_multi_threshold_args"] is True

    output = tmp_path / "bundle.json"
    persisted = write_evidence_bundle(output)
    assert json.loads(output.read_text(encoding="utf-8")) == persisted


def test_app_loads_persisted_bundle_without_runtime_checkout(monkeypatch):
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / "src"))

    from pipesd_repro import evidence

    def fail_runtime_checkout(*args, **kwargs):
        raise AssertionError("app should not rebuild evidence during display")

    monkeypatch.setattr(evidence, "build_evidence_bundle", fail_runtime_checkout)

    spec = importlib.util.spec_from_file_location("pipesd_app", PROJECT_ROOT / "app.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    bundle = module.load_bundle()

    assert bundle["paper_id"] == "1ebAvNphi7"
    assert bundle["attempt_id"] == "987c1913-c83b-4be0-acc6-7dae2e0a97e1"


def test_space_metadata_and_scoring_pages_are_present():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    pages = sorted((PROJECT_ROOT / "pages").glob("*.md"))
    substantive_characters = sum(
        len(path.read_text(encoding="utf-8").strip()) for path in pages
    )

    assert "sdk: gradio" in readme
    assert "icml2026-repro" in readme
    assert "paper-1ebAvNphi7" in readme
    assert substantive_characters >= 200
