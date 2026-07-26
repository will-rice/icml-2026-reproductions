from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

import pytest

from graph_pruning_repro.evidence import build_evidence, canonical_json_bytes


PROJECT_ROOT = Path(__file__).parents[1]
SOURCE_REVISION = "0" * 40


def _cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "graph_pruning_repro.cli",
            *arguments,
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value))


def _search(evidence: dict[str, object], search_id: str) -> dict[str, object]:
    return next(
        search
        for search in evidence["searches"]
        if search["id"] == search_id
    )


@pytest.fixture(scope="session")
def cli_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("cli-evidence")
    shutil.copytree(PROJECT_ROOT / "paper_transcriptions", root / "paper_transcriptions")
    (root / "evidence").mkdir()
    shutil.copy2(PROJECT_ROOT / "evidence" / "schema.json", root / "evidence")
    build_evidence(root / "evidence", source_revision=SOURCE_REVISION)
    return root


def _mutated_cli_tree(source: Path, destination: Path) -> tuple[Path, Path]:
    shutil.copytree(source / "paper_transcriptions", destination / "paper_transcriptions")
    shutil.copytree(source / "evidence", destination / "evidence")
    return (
        destination / "evidence" / "evidence.json",
        destination / "evidence" / "schema.json",
    )


def _mutate_candidate(evidence: dict[str, object], mutation: str) -> None:
    greedy = _search(evidence, "greedy")
    appendix = next(
        witness
        for witness in evidence["witnesses"]
        if witness["property"] == "appendix_inline_shift_diminishing_returns"
    )
    if mutation == "altered_claim_text":
        evidence["target_claims"][0] += " altered"
    elif mutation == "undeclared_domain":
        evidence["searches"].append(dict(evidence["searches"][0], id="undeclared"))
    elif mutation == "actual_above_ceiling":
        component = evidence["searches"][0]["components"][0]
        component["actual"] = component["declared_ceiling"] + 1
    elif mutation == "missing_completion_status":
        del evidence["commands"][0]["status"]
    elif mutation == "merged_variants":
        greedy["result"]["model_variants"][1] = greedy["result"]["model_variants"][0]
    elif mutation == "non_fraction_rational":
        appendix["inputs"]["alpha"] = "1.0"
    elif mutation == "incomplete_search_as_pass":
        evidence["searches"][0]["completed"] = False
        evidence["searches"][0]["status"] = "pass"
    elif mutation == "ineligible_in_guarantee":
        evidence["guarantee_violations"].append(
            dict(evidence["out_of_premise_diagnostics"][0])
        )
    elif mutation == "out_of_premise_counterexample":
        evidence["out_of_premise_diagnostics"][0][
            "classification"
        ] = "counterexample"
    elif mutation == "missing_classification":
        evidence["claim_results"].pop()
    elif mutation == "reordered_classification":
        evidence["claim_results"].reverse()
    elif mutation == "altered_canonical_parameters":
        greedy["result"]["domain_manifest"]["parameterized_examples"][
            "modular_shift_candidate"
        ]["eta"] = "1/1"
    elif mutation == "altered_parameterized_id":
        greedy["result"]["domain_manifest"]["parameterized_examples"][
            "appendix_inline_shift_literal"
        ]["instance_id"] += "::candidate"
    elif mutation == "measured_clock":
        evidence["environment"]["elapsed_seconds"] = 1
    elif mutation == "path_traversal":
        evidence["witnesses"][0]["artifact_path"] = "../outside.json"
    elif mutation == "duplicate_ids":
        evidence["claim_results"][1]["id"] = evidence["claim_results"][0]["id"]
    elif mutation == "pseudo_pointer":
        evidence["artifacts"][0]["render_pointers"][0] = "/witnesses/{id}"
    else:
        raise AssertionError(f"unknown CLI mutation {mutation}")


def test_cli_help_exposes_all_three_commands() -> None:
    result = _cli("--help")
    assert result.returncode == 0
    assert "{recompute,validate,render}" in result.stdout


def test_cli_recompute_and_validate_clean_tree(tmp_path: Path) -> None:
    output = tmp_path / "output"
    recompute = _cli(
        "recompute",
        str(output),
        "--source-revision",
        SOURCE_REVISION,
    )
    assert recompute.returncode == 0, recompute.stderr
    assert "completed actual=" in recompute.stdout
    assert "ceiling=13833860" in recompute.stdout
    validate = _cli("validate", str(output / "evidence.json"))
    assert validate.returncode == 0, validate.stderr
    assert "schema and full-replay semantic acceptance: PASS" in validate.stdout


@pytest.mark.parametrize(
    "mutation",
    [
        "altered_claim_text",
        "undeclared_domain",
        "actual_above_ceiling",
        "missing_completion_status",
        "merged_variants",
        "non_fraction_rational",
        "incomplete_search_as_pass",
        "ineligible_in_guarantee",
        "out_of_premise_counterexample",
        "missing_classification",
        "reordered_classification",
        "altered_canonical_parameters",
        "altered_parameterized_id",
        "measured_clock",
        "path_traversal",
        "duplicate_ids",
        "pseudo_pointer",
    ],
)
def test_cli_subprocess_rejects_candidate_controlled_acceptance(
    tmp_path: Path,
    cli_tree: Path,
    mutation: str,
) -> None:
    evidence_path, _ = _mutated_cli_tree(cli_tree, tmp_path)
    evidence = json.loads(evidence_path.read_text())
    _mutate_candidate(evidence, mutation)
    _write_json(evidence_path, evidence)
    result = _cli("validate", str(evidence_path))
    assert result.returncode != 0
    assert "acceptance failed:" in result.stderr
