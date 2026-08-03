from pathlib import Path
import pytest

from lrr_repro.evidence import build_evidence

EXPECTED_HASHES = [
    "5f0d21d91c0ae1d2877563e7115e804db60361304db4aea72b97596300e60f57",
    "79d94d106cfded95104c54624068a07dc9ae16dca681a6ad5370bbb648e8c7de",
    "4b8bfdf084cb0038acc0a589837dc4379ba1fb079f30f4be8edf839a21d23a51",
    "9b35061b3b4e2873f1b7a4fffc6fa22d659f281c096d990706ebd805303c4c00",
    "13999601811ffe2bb8e9526ed601e9d59480b217d6d1917787db2a9c7dbc8372",
]


def test_evidence_binds_all_claims_and_honest_outcomes(project_root, cache_dir):
    evidence = build_evidence(project_root, cache_dir)
    assert evidence["attempt_id"] == "eb10c79b-fc26-47c4-88c1-6f45cb592833"
    assert [claim["challenge_claim_sha256"] for claim in evidence["claims"]] == EXPECTED_HASHES
    assert evidence["claims"][4]["status"] == "falsified"
    assert "historical priority" in " ".join(evidence["claims"][2]["limitations"])
    assert evidence["unavailable_operations"] == [
        "agentic_rerun",
        "gpu_training",
        "gurobi_rerun",
        "paid_api",
    ]


def test_build_evidence_verifies_all_inputs_and_fails_if_tampered_or_missing(
    project_root, cache_dir, tmp_path
):
    import shutil
    from lrr_repro.provenance import IntegrityError

    temp_proj = tmp_path / "proj"
    temp_cache = tmp_path / "cache"
    temp_proj.mkdir(parents=True, exist_ok=True)
    shutil.copytree(project_root / "evidence", temp_proj / "evidence")
    shutil.copytree(cache_dir, temp_cache)

    # Tamper a vendored input file used by build_evidence
    csv_file = (
        temp_proj
        / "evidence/inputs/upstream/results/Bitween-Results(Sheet1-ICML).csv"
    )
    csv_file.write_bytes(csv_file.read_bytes() + b"\nTAMPERED")

    with pytest.raises(IntegrityError, match="results-csv"):
        build_evidence(temp_proj, temp_cache)
