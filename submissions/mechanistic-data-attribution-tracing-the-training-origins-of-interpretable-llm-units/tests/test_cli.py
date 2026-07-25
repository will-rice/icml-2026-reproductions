import json
import os
import pytest
from mechanistic_data_attribution_repro.cli import main

def test_cli_execution(tmp_path):
    output_dir = tmp_path / "evidence"
    exit_code = main(["--output-dir", str(output_dir), "--num-samples", "20", "--seed", "42"])
    assert exit_code == 0
    assert (output_dir / "results.json").exists()
    assert (output_dir / "measurements.csv").exists()
    assert (output_dir / "provenance.json").exists()
    assert (output_dir / "repro-bundle.tar.gz").exists()

    with open(output_dir / "results.json") as f:
        res = json.load(f)
        assert "target_claims" in res
        assert len(res["target_claims"]) == 3
        assert res["paper_id"] == "PQaxfoEcRc"
