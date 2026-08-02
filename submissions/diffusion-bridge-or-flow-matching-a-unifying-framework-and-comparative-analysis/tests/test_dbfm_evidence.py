import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dbfm_repro.evidence import (  # noqa: E402
    ATTEMPT_ID,
    PAPER_ID,
    UPSTREAM_COMMIT,
    brownian_bridge_proxy,
    flow_matching_interpolate,
    generate_evidence_bundle,
    repository_audit,
)


def test_flow_matching_interpolation_matches_released_formula():
    x0 = [0.0, 2.0, -2.0]
    x1 = [10.0, 6.0, 2.0]
    x_t, velocity = flow_matching_interpolate(x0, x1, 0.25)

    assert x_t == [2.5, 3.0, -1.0]
    assert velocity == [10.0, 4.0, 4.0]


def test_bridge_proxy_pins_endpoints_and_reduces_endpoint_error():
    result = brownian_bridge_proxy(samples=128, steps=33, seed=7)

    assert result["bridge_endpoint_abs_error"] == 0.0
    assert result["flow_noisy_endpoint_abs_error"] > result["bridge_endpoint_abs_error"]
    assert result["bridge_action"] < result["flow_noisy_action"]


def test_repository_audit_identifies_cpu_limitations():
    audit = repository_audit(Path("/tmp/dbfm-upstream-daf2"))

    assert audit["commit"] == UPSTREAM_COMMIT
    assert audit["has_flow_matching_code"] is True
    assert audit["has_diffusion_bridge_code"] is True
    assert audit["requires_cuda_configs"] is True
    assert audit["has_released_checkpoints"] is False


def test_evidence_bundle_contains_claim_statuses(tmp_path):
    bundle_path = generate_evidence_bundle(tmp_path, upstream_root=Path("/tmp/dbfm-upstream-daf2"))
    data = json.loads(bundle_path.read_text(encoding="utf-8"))

    assert data["paper_id"] == PAPER_ID
    assert data["attempt_id"] == ATTEMPT_ID
    assert data["upstream_pins"]["code_commit"] == UPSTREAM_COMMIT
    assert len(data["claims"]) == 6
    assert {claim["status"] for claim in data["claims"]} == {"toy", "unavailable"}


def test_generate_evidence_script_runs_from_project_root():
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "generate_evidence.py")],
        cwd=PROJECT_ROOT.parent.parent,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Wrote evidence bundle" in result.stdout


def test_generate_evidence_script_writes_computed_numbers_page():
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "generate_evidence.py")],
        cwd=PROJECT_ROOT.parent.parent,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    page = PROJECT_ROOT / "pages" / "01-computed-evidence.md"
    text = page.read_text(encoding="utf-8")
    assert "bridge_action: 0.1502805366" in text
    assert "flow_noisy_action: 0.8254394387" in text
    assert "flow_noisy_endpoint_abs_error: 0.0614979566" in text
    assert "x_t: [2.5, 3.0, -1.0]" in text
