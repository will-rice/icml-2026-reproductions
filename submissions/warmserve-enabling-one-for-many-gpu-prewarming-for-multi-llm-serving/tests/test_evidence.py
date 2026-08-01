from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate_evidence import build_evidence, main


def test_bundle_provenance_is_pinned():
    bundle = build_evidence(
        source_root="/tmp/warmserve-upstream-codex03",
        arxiv_source="/tmp/warmserve-artifacts-codex03/2512.09472v2.tar",
        arxiv_pdf="/tmp/warmserve-artifacts-codex03/2512.09472v2.pdf",
    )
    assert bundle["paper_id"] == "DVHpvumD60"
    assert bundle["snapshot_id"] == "98fe583a0a55974d2d28e1beba12e398eb7e8b7f05fadb6d33fdd243b1988644"
    assert bundle["upstream"]["github"] == "LLMServe/WarmServe@a60121519e077d2f128b597cbabc947e3e618aaf"
    assert bundle["upstream"]["code_license"] == "Apache-2.0"
    assert bundle["upstream"]["arxiv_source_sha256"] == "c945e654f3309f6de207c29c6151305a8ae4e53163846946958befc30a9d05d5"


def test_warmserve_source_indicators_are_detected():
    bundle = build_evidence(source_root="/tmp/warmserve-upstream-codex03")
    indicators = bundle["observations"]["source_indicators"]
    assert indicators["scheduler"]["present"] is True
    assert indicators["prewarm_manager"]["present"] is True
    assert indicators["vmm"]["present"] is True
    assert indicators["worker_hooks"]["present"] is True
    assert indicators["trace_generator"]["present"] is True
    assert indicators["model_config"]["present"] is True


def test_trace_generator_metadata_is_parseable():
    bundle = build_evidence(source_root="/tmp/warmserve-upstream-codex03")
    trace = bundle["observations"]["trace_generator"]
    assert trace["interval_seconds"] == 300
    assert trace["model_yaml_models"] == 4
    assert trace["cluster"] == {"nnodes": 2, "ngpus_per_node": 8}


def test_performance_claims_are_not_promoted_without_gpu_logs():
    bundle = build_evidence(source_root="/tmp/warmserve-upstream-codex03")
    assert bundle["claim_results"]["claim-1"]["status"] == "verified"
    assert bundle["claim_results"]["claim-2"]["status"] in {"toy", "inconclusive"}
    for claim_id in ("claim-3", "claim-4", "claim-5", "claim-6"):
        assert bundle["claim_results"][claim_id]["status"] == "unavailable"
    assert any("512-GPU" in item for item in bundle["unreplicated"])


def test_bundle_file_round_trips(tmp_path):
    output = tmp_path / "bundle.json"
    main(["--source-root", "/tmp/warmserve-upstream-codex03", "--output", str(output)])
    data = json.loads(output.read_text(encoding="utf-8"))
    assert sorted(data["claim_results"]) == [
        "claim-1",
        "claim-2",
        "claim-3",
        "claim-4",
        "claim-5",
        "claim-6",
    ]
    assert data["observations"]["source_tree"]["commit"] == "a60121519e077d2f128b597cbabc947e3e618aaf"
