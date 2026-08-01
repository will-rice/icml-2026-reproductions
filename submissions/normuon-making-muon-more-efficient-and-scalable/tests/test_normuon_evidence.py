import json
import math
import subprocess
import sys
from pathlib import Path

import torch


def test_reference_update_matches_official_normuon_update():
    from normuon_repro import evidence

    source_root = evidence.ensure_upstream_checkout()
    official = evidence.load_official_normuon(source_root)

    grad = torch.tensor(
        [
            [3.0, -2.0, 0.5, 1.0],
            [0.25, 4.0, -1.0, 2.0],
            [1.5, 0.0, -3.5, 0.75],
        ],
        dtype=torch.float32,
    )
    momentum = torch.zeros_like(grad)
    second_momentum = torch.zeros(grad.shape[0], 1)

    reference_update, reference_second = evidence.reference_normuon_update(
        grad,
        torch.zeros_like(grad),
        torch.zeros(grad.shape[0], 1),
        beta=0.95,
        beta2=0.95,
    )
    official_update = official.normuon_update(
        grad.clone(),
        momentum,
        second_momentum,
        beta=0.95,
        beta2=0.95,
    )

    assert torch.allclose(official_update, reference_update, atol=1e-5, rtol=1e-5)
    assert torch.allclose(second_momentum, reference_second, atol=1e-6, rtol=1e-6)


def test_neuron_normalization_reduces_row_norm_dispersion():
    from normuon_repro import evidence

    update = torch.tensor(
        [
            [8.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.25, 0.25, 0.25, 0.25],
        ],
        dtype=torch.float32,
    )
    normalized, second = evidence.apply_neuron_adaptive_normalization(
        update,
        torch.zeros(update.shape[0], 1),
        beta2=0.95,
    )

    assert second.shape == (3, 1)
    assert evidence.row_norm_cv(normalized) < evidence.row_norm_cv(update) * 0.1
    assert math.isclose(
        float(torch.linalg.vector_norm(normalized)),
        float(torch.linalg.vector_norm(update)),
        rel_tol=1e-6,
    )


def test_bundle_binds_attempt_claims_and_unreplicated_large_scale_claims(tmp_path):
    from normuon_repro import evidence

    bundle = evidence.build_evidence_bundle()
    output = tmp_path / "bundle.json"

    assert evidence.write_evidence_bundle(output) == bundle
    assert json.loads(output.read_text()) == bundle
    assert bundle["attempt_id"] == "daee6151-3f6d-429d-a01b-c6d91b72dd1c"
    assert bundle["paper_id"] == "m1IRWFAMsa"
    assert bundle["snapshot_id"] == (
        "0692f289e0163260c616a30969f5e5d5db781c4d463ff6b123403db29926e574"
    )
    assert bundle["upstream_revision"] == (
        "arxiv:2510.05491+github:zichongli5/NorMuon@"
        "c6989a8354730695d9f5a9faa6c55eeb24865209"
    )
    assert [claim["challenge_claim_sha256"] for claim in bundle["claims"][:2]] == [
        "c2b6b1756f6922c77ecc6915c14be6996d0b2111cc654462c685bebfce5d1b32",
        "ed5bd5aaea367741a42e41f1b0dc573013283967c88896699d9893b32942e08e",
    ]
    assert all(
        claim["status"] in {"verified", "toy", "falsified", "inconclusive", "unavailable"}
        for claim in bundle["claims"]
    )
    assert all(
        claim["status"] == "unavailable"
        for claim in bundle["claims"][2:]
    )
    assert any("GPU-scale LLM pretraining" in item for item in bundle["limitations"])


def test_generate_evidence_script_writes_bundle():
    script = Path(__file__).resolve().parent.parent / "generate_evidence.py"

    subprocess.run([sys.executable, str(script)], check=True)

    output = script.parent / "evidence" / "bundle.json"
    assert output.is_file()
    bundle = json.loads(output.read_text())
    assert bundle["paper_id"] == "m1IRWFAMsa"
    assert bundle["claims"][0]["status"] == "verified"
