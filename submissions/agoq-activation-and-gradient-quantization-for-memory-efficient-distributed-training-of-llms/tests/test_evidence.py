import subprocess
import sys

from agoq_repro.evidence import (
    LIVE_CLAIMS,
    build_evidence,
    canonical_json_bytes,
)


def test_bundle_has_all_live_claims_in_order(project_root):
    evidence = build_evidence(project_root)
    assert evidence["schema_version"] == 3
    assert evidence["identity"]["attempt_id"] == (
        "2fc3b006-3307-4fc3-8df6-c000379298c4"
    )
    assert [claim["claim_id"] for claim in evidence["claims"]] == [
        "claim-1",
        "claim-2",
        "claim-3",
        "claim-4",
        "claim-5",
        "claim-6",
    ]
    assert [claim["claim"] for claim in evidence["claims"]] == list(LIVE_CLAIMS)
    assert [claim["challenge_claim_sha256"] for claim in evidence["claims"]] == [
        "0b198b87a5abf16409a547a6f5277a41a62eac4a791b71cada94b054c65a1a13",
        "89292ed940125355f402bc04bc847acbed65f01bd0718124cceb88416ec24228",
        "a5a088563e0ab1a912f212da4246d90e8df679e6312e494ec486f0c38953b5bf",
        "88c789000f385b4692435064cb66b427ecbfd05b92c0632adde7681cb7b69eaa",
        "a513e6751344f810d77db2b7cd9a2fac9cf9ceab94f2a583a0247f917e64145d",
        "7391424029d3da524d5b5dfe17c88119ee6b0b7d1808ec6d0bc80366630efd1a",
    ]
    assert [claim["status"] for claim in evidence["claims"]] == [
        "partial",
        "partial",
        "partial",
        "partial",
        "unavailable",
        "unavailable",
    ]


def test_bundle_separates_context_observations_and_limitations(project_root):
    evidence = build_evidence(project_root)
    assert evidence["paper_context"]["table_1"]["agoq_total_u"] == "31/4"
    assert evidence["reproduced_observations"]["table_1"]["agoq_total_u"] == "31/4"
    assert (
        evidence["reproduced_observations"]["pipeline"][
            "maximum_reported_overshoot_units"
        ]
        == 1
    )
    assert evidence["limitations"]["single_gpu_fused_kernel_body"] == (
        "Call sites are present, but a fused GPU kernel implementation body "
        "is not present in the pinned selected source."
    )
    assert "64 GPUs" in evidence["limitations"]["claim-5"]
    assert "16 NVIDIA Blackwell GPUs" in evidence["limitations"]["claim-6"]


def test_bundle_is_deterministic_and_contains_no_synthetic_fields(project_root):
    first = canonical_json_bytes(build_evidence(project_root))
    second = canonical_json_bytes(build_evidence(project_root))
    assert first == second
    text = first.decode()
    for forbidden in (
        "generated_at",
        "timestamp",
        "hostname",
        "git_head",
        "random_seed",
        "quantization_error",
        "proxy_score",
    ):
        assert forbidden not in text


def test_generator_cli_writes_requested_canonical_output(project_root, tmp_path):
    output = tmp_path / "bundle.json"
    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "generate_evidence.py"),
            "--output",
            str(output),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    expected = canonical_json_bytes(build_evidence(project_root))
    assert output.read_bytes() == expected
    assert result.stdout.strip().endswith(
        __import__("hashlib").sha256(expected).hexdigest()
    )
