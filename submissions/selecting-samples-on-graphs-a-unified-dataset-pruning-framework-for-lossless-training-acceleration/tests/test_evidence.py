from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Callable

import pytest

from graph_pruning_repro.evidence import (
    build_evidence,
    canonical_json_bytes,
    validate_evidence,
)


PROJECT_ROOT = Path(__file__).parents[1]
SOURCE_REVISION = "0" * 40


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value))


def _search(evidence: dict[str, object], search_id: str) -> dict[str, object]:
    return next(
        search
        for search in evidence["searches"]
        if search["id"] == search_id
    )


def _copy_fixture_tree(source: Path, destination: Path) -> None:
    shutil.copytree(source / "paper_transcriptions", destination / "paper_transcriptions")
    shutil.copytree(source / "evidence", destination / "evidence")


@pytest.fixture(scope="session")
def accepted_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("accepted-evidence")
    shutil.copytree(PROJECT_ROOT / "paper_transcriptions", root / "paper_transcriptions")
    (root / "evidence").mkdir()
    shutil.copy2(PROJECT_ROOT / "evidence" / "schema.json", root / "evidence")
    build_evidence(root / "evidence", source_revision=SOURCE_REVISION)
    validate_evidence(
        root / "evidence" / "evidence.json",
        root / "evidence" / "schema.json",
        root,
    )
    return root


def build_fixture_tree(tmp_path: Path, accepted_tree: Path) -> tuple[Path, Path]:
    _copy_fixture_tree(accepted_tree, tmp_path)
    return (
        tmp_path / "evidence" / "evidence.json",
        tmp_path / "evidence" / "schema.json",
    )


def _repair_domain_manifest(search: dict[str, object]) -> None:
    manifest = search["result"]["domain_manifest"]
    payload = {
        "graphs": manifest["graphs"],
        "subsets": manifest["subsets"],
        "parameterized_examples": manifest["parameterized_examples"],
    }
    manifest["record_count"] = len(manifest["graphs"]) + len(manifest["subsets"])
    manifest["sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _proof_conclusion(
    evidence: dict[str, object],
    row_id: str,
    conclusion_id: str,
) -> dict[str, object]:
    rows = evidence["proof_ledger"]["symbolic"]["ledgers"][
        "paper_samplewise_literal"
    ]
    row = next(candidate for candidate in rows if candidate["row_id"] == row_id)
    return next(
        candidate
        for candidate in row["conclusions"]
        if candidate["conclusion_id"] == conclusion_id
    )


def apply_adversarial_mutation_and_repair_candidate_metadata(
    root: Path,
    mutation: str,
) -> None:
    evidence_path = root / "evidence" / "evidence.json"
    evidence = json.loads(evidence_path.read_text())
    witnesses = evidence["witnesses"]
    first_witness = witnesses[0]
    first_witness_path = root / first_witness["artifact_path"]
    greedy = _search(evidence, "greedy")
    manifest = greedy["result"]["domain_manifest"]

    if mutation == "stored_status":
        evidence["claim_results"][0]["status"] = "supported"
    elif mutation == "transcription_text":
        evidence["transcriptions"]["records"][0][
            "normalized_expression"
        ] = "candidate-authored replacement"
    elif mutation == "witness_id":
        first_witness["id"] = "candidate-replaced-witness"
    elif mutation == "witness_bytes":
        first_witness_path.write_bytes(first_witness_path.read_bytes() + b" ")
    elif mutation == "witness_hash":
        first_witness["artifact_sha256"] = "0" * 64
    elif mutation == "witness_link":
        linked = next(result for result in evidence["claim_results"] if result["witness_ids"])
        linked["witness_ids"] = ["candidate-replaced-link"]
    elif mutation == "missing_witness_file":
        first_witness_path.unlink()
    elif mutation == "extra_witness_file":
        (root / "evidence" / "witnesses" / "extra.json").write_text("{}\n")
    elif mutation == "replaced_witness_file":
        replacement = dict(json.loads(first_witness_path.read_text()))
        replacement["property"] = "candidate-replaced-property"
        first_witness_path.write_bytes(canonical_json_bytes(replacement))
        first_witness["artifact_sha256"] = hashlib.sha256(
            first_witness_path.read_bytes()
        ).hexdigest()
    elif mutation == "missing_middle_graph":
        manifest["graphs"].pop(len(manifest["graphs"]) // 2)
        _repair_domain_manifest(greedy)
    elif mutation == "duplicate_middle_graph_same_length":
        manifest["graphs"][len(manifest["graphs"]) // 2] = dict(
            manifest["graphs"][0]
        )
        _repair_domain_manifest(greedy)
    elif mutation == "replace_middle_graph_under_original_id":
        graph = manifest["graphs"][len(manifest["graphs"]) // 2]
        graph["vertex_weights"][0] = "99/1"
        _repair_domain_manifest(greedy)
    elif mutation == "missing_middle_subset":
        manifest["subsets"].pop(len(manifest["subsets"]) // 2)
        _repair_domain_manifest(greedy)
    elif mutation == "duplicate_middle_subset_same_length":
        manifest["subsets"][len(manifest["subsets"]) // 2] = dict(
            manifest["subsets"][0]
        )
        _repair_domain_manifest(greedy)
    elif mutation == "replace_middle_subset_under_original_id":
        subset = manifest["subsets"][len(manifest["subsets"]) // 2]
        subset["selected"] = ["candidate-replacement"]
        _repair_domain_manifest(greedy)
    elif mutation in {
        "wrong_alpha",
        "wrong_unshifted_eta",
        "wrong_appendix_eta",
        "wrong_modular_eta",
        "replaced_parameterized_case_id",
    }:
        examples = manifest["parameterized_examples"]
        if mutation == "wrong_alpha":
            examples["paper_mwcp"]["alpha"] = "2/1"
        elif mutation == "wrong_unshifted_eta":
            examples["paper_samplewise_literal"]["eta"] = "1/1"
        elif mutation == "wrong_appendix_eta":
            examples["appendix_inline_shift_literal"]["eta"] = "2/1"
        elif mutation == "wrong_modular_eta":
            examples["modular_shift_candidate"]["eta"] = "1/1"
        else:
            examples["modular_shift_candidate"]["instance_id"] += "::replaced"
        _repair_domain_manifest(greedy)
    elif mutation == "missing_guarantee_violations":
        del evidence["guarantee_violations"]
    elif mutation == "missing_out_of_premise_diagnostics":
        del evidence["out_of_premise_diagnostics"]
    elif mutation == "remove_out_of_premise_diagnostic":
        evidence["out_of_premise_diagnostics"].pop()
    elif mutation == "move_ineligible_result_to_guarantee_violations":
        moved = dict(evidence["out_of_premise_diagnostics"].pop())
        evidence["guarantee_violations"].append(moved)
    elif mutation == "insert_measured_runtime_field":
        evidence["commands"][0]["runtime_seconds"] = 0
    elif mutation == "nested_eq28_conclusion":
        rows = evidence["proof_ledger"]["symbolic"]["ledgers"][
            "paper_samplewise_literal"
        ]
        eq28 = next(row for row in rows if row["row_id"] == "eq28")
        eq28["conclusions"] = [{"nested": eq28["conclusions"]}]
    elif mutation in {
        "missing_prerequisite_edge",
        "duplicate_prerequisite_edge",
        "redirected_prerequisite_edge",
        "unknown_prerequisite_edge",
        "cyclic_prerequisite_edge",
    }:
        conclusion = _proof_conclusion(evidence, "eq30", "eq30_gap_to_next_gain")
        references = conclusion["prerequisite_conclusion_refs"]
        if mutation == "missing_prerequisite_edge":
            references.pop()
        elif mutation == "duplicate_prerequisite_edge":
            references.append(references[0])
        elif mutation == "redirected_prerequisite_edge":
            references[0] = references[-1]
        elif mutation == "unknown_prerequisite_edge":
            references[0] = "unknown/symbolic/eq28/unknown"
        else:
            eq28 = _proof_conclusion(
                evidence,
                "eq28",
                "eq28_union_submodular_bound",
            )
            eq28["prerequisite_conclusion_refs"].append(
                "paper_samplewise_literal/symbolic/eq30/eq30_gap_to_next_gain"
            )
    elif mutation in {"manifest_missing_key", "manifest_replaced_key"}:
        manifest_path = root / "paper_transcriptions" / "manifest.json"
        source_manifest = json.loads(manifest_path.read_text())
        if mutation == "manifest_missing_key":
            del source_manifest[0]["section"]
        else:
            source_manifest[0]["equation"] = "candidate-replaced-equation"
        _write_json(manifest_path, source_manifest)
    elif mutation == "excerpt_bytes":
        excerpt = root / "paper_transcriptions" / "excerpts" / "eq-02.txt"
        excerpt.write_bytes(excerpt.read_bytes() + b"candidate")
    else:
        raise AssertionError(f"unknown mutation {mutation}")

    _write_json(evidence_path, evidence)


def test_required_appendix_witness_has_three_audit_linkages(
    tmp_path: Path,
) -> None:
    evidence = build_evidence(tmp_path, source_revision=SOURCE_REVISION)
    witness = next(
        witness
        for witness in evidence["witnesses"]
        if witness["property"] == "appendix_inline_shift_diminishing_returns"
    )
    assert witness["model_variant"] == "appendix_inline_shift_literal"
    assert witness["inputs"]["alpha"] == "1/1"
    assert witness["inputs"]["eta"] == "1/1"
    assert witness["intermediate_values"]["marginal_empty"] == "1/1"
    assert witness["intermediate_values"]["marginal_y"] == "3/1"
    links = {
        result["audit"]
        for result in evidence["claim_results"]
        if witness["id"] in result["witness_ids"]
    }
    assert {
        "diminishing_returns",
        "greedy_guarantee_premise",
        "appendix_f_proof_ledger",
    } <= links


def test_repaired_variants_cannot_claim_literal_witness(tmp_path: Path) -> None:
    evidence = build_evidence(tmp_path, source_revision=SOURCE_REVISION)
    literal_id = next(
        witness["id"]
        for witness in evidence["witnesses"]
        if witness.get("model_variant") == "appendix_inline_shift_literal"
    )
    assert not any(
        literal_id in result["witness_ids"]
        for result in evidence["claim_results"]
        if result["model_variant"] == "modular_shift_candidate"
    )


def test_canonical_build_is_byte_identical(tmp_path: Path) -> None:
    first = build_evidence(tmp_path / "first", source_revision=SOURCE_REVISION)
    second = build_evidence(tmp_path / "second", source_revision=SOURCE_REVISION)
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert (tmp_path / "first" / "evidence.json").read_bytes() == (
        tmp_path / "second" / "evidence.json"
    ).read_bytes()


def test_required_top_level_classification_arrays_are_present(
    tmp_path: Path,
) -> None:
    evidence = build_evidence(tmp_path, source_revision=SOURCE_REVISION)
    assert "guarantee_violations" in evidence
    assert "out_of_premise_diagnostics" in evidence
    assert isinstance(evidence["guarantee_violations"], list)
    assert isinstance(evidence["out_of_premise_diagnostics"], list)


def test_canonical_evidence_contains_no_measured_clock_fields(
    tmp_path: Path,
) -> None:
    evidence = build_evidence(tmp_path, source_revision=SOURCE_REVISION)
    forbidden = {
        "runtime",
        "runtime_seconds",
        "wall_time",
        "wall_time_seconds",
        "duration",
        "duration_seconds",
        "elapsed",
        "elapsed_seconds",
        "started_at",
        "finished_at",
    }

    def assert_no_clock_keys(value: object) -> None:
        if isinstance(value, dict):
            assert forbidden.isdisjoint(value)
            for child in value.values():
                assert_no_clock_keys(child)
        elif isinstance(value, list):
            for child in value:
                assert_no_clock_keys(child)

    assert_no_clock_keys(evidence)


def test_generation_components_and_aggregate_are_bounded(tmp_path: Path) -> None:
    evidence = build_evidence(tmp_path, source_revision=SOURCE_REVISION)
    components = [
        component
        for search in evidence["searches"]
        for component in search["components"]
    ]
    assert len(components) == 18
    assert len({component["id"] for component in components}) == 18
    assert all(
        0 <= component["actual"] <= component["declared_ceiling"]
        and component["completed"]
        for component in components
    )
    assert sum(component["declared_ceiling"] for component in components) == 13_833_860
    command = next(command for command in evidence["commands"] if command["id"] == "recompute")
    assert command["actual"] == sum(component["actual"] for component in components)
    assert command["ceiling"] == 13_833_860
    assert command["status"] == "completed"


def test_clean_fixture_passes_schema_and_full_replay(
    accepted_tree: Path,
) -> None:
    validate_evidence(
        accepted_tree / "evidence" / "evidence.json",
        accepted_tree / "evidence" / "schema.json",
        accepted_tree,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "stored_status",
        "transcription_text",
        "witness_id",
        "witness_bytes",
        "witness_hash",
        "witness_link",
        "missing_witness_file",
        "extra_witness_file",
        "replaced_witness_file",
        "missing_middle_graph",
        "duplicate_middle_graph_same_length",
        "replace_middle_graph_under_original_id",
        "missing_middle_subset",
        "duplicate_middle_subset_same_length",
        "replace_middle_subset_under_original_id",
        "wrong_alpha",
        "wrong_unshifted_eta",
        "wrong_appendix_eta",
        "wrong_modular_eta",
        "replaced_parameterized_case_id",
        "missing_guarantee_violations",
        "missing_out_of_premise_diagnostics",
        "remove_out_of_premise_diagnostic",
        "move_ineligible_result_to_guarantee_violations",
        "insert_measured_runtime_field",
        "nested_eq28_conclusion",
        "missing_prerequisite_edge",
        "duplicate_prerequisite_edge",
        "redirected_prerequisite_edge",
        "unknown_prerequisite_edge",
        "cyclic_prerequisite_edge",
        "manifest_missing_key",
        "manifest_replaced_key",
        "excerpt_bytes",
    ],
)
def test_full_replay_rejects_self_consistent_tampering(
    tmp_path: Path,
    accepted_tree: Path,
    mutation: str,
) -> None:
    evidence_path, schema_path = build_fixture_tree(tmp_path, accepted_tree)
    apply_adversarial_mutation_and_repair_candidate_metadata(tmp_path, mutation)
    with pytest.raises(ValueError):
        validate_evidence(evidence_path, schema_path, tmp_path)
