"""Canonical evidence orchestration and source-derived full replay."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from copy import deepcopy
from fractions import Fraction
from itertools import combinations, product
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping

from jsonschema import Draft202012Validator

from .algorithm1 import audit_literal_algorithm1
from .diminishing_returns import (
    appendix_shift_witness,
    canonical_parameterized_instance_id,
    canonical_variant_parameters,
    run_diminishing_returns_audit,
)
from .equivalence import run_equivalence_audit
from .greedy import run_greedy_audit
from .proof_ledger import (
    build_symbolic_ledger,
    cardinality_b_minus_t_witness,
    declared_aggregate_ceiling,
    declared_component_ceilings,
    run_finite_ledger_control,
    run_symbolic_ledger_control,
    validate_prerequisite_graph,
)
from .provenance import (
    PAPER,
    PDF_ACQUISITION_COMMAND,
    TARGET_CLAIMS,
    TRANSCRIPTION_SET_SHA256,
    load_transcriptions,
)
from .shifts import run_shift_audit
from .types import SET_FUNCTION_VARIANTS, Instance


ATTEMPT_ID = "64bfe193-333b-4b37-9683-9ac25ca5ac27"
SCHEMA_VERSION = 1
_SOURCE_REVISION = re.compile(r"^[0-9a-f]{40}$")
_FRACTION_TEXT = re.compile(r"^-?(?:0|[1-9][0-9]*)/[1-9][0-9]*$")
_CLOCK_FIELDS = {
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
_RATIONAL_FIELDS = {
    "alpha",
    "eta",
    "coefficient",
    "threshold",
    "marginal",
    "marginal_empty",
    "marginal_y",
    "difference",
    "greedy_value",
    "optimum_value",
    "ratio",
    "value",
}
_PREMISE_FIELDS = (
    "global_nonnegativity",
    "normalization",
    "global_monotonicity",
    "global_submodularity",
)
_EXPECTED_SEARCH_IDS = (
    "algorithm1",
    "diminishing_returns",
    "greedy",
    "objective_equivalence",
    "proof_ledger",
    "shifts",
)
_AUDIT_CACHE: dict[str, dict[str, object]] = {}


def canonical_json_bytes(value: object) -> bytes:
    """Return the one canonical byte representation used by every artifact."""

    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _validate_source_revision(source_revision: str) -> None:
    if type(source_revision) is not str:
        raise TypeError("source_revision must be a string")
    if _SOURCE_REVISION.fullmatch(source_revision) is None:
        raise ValueError("source_revision must be a 40-character lowercase SHA")


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _source_root_for_output(output_dir: Path) -> Path:
    candidate = output_dir.parent
    if (candidate / "paper_transcriptions").is_dir():
        return candidate.resolve()
    return _project_root()


def _symmetric_interactions(
    edges: Iterable[tuple[str, str]],
    edge_values: Iterable[Fraction],
) -> dict[tuple[str, str], Fraction]:
    interactions: dict[tuple[str, str], Fraction] = {}
    for (left, right), value in zip(edges, edge_values, strict=True):
        interactions[(left, right)] = value
        interactions[(right, left)] = value
    return interactions


def _graph_id(graph: Instance) -> str:
    alpha, eta = canonical_variant_parameters(graph, "paper_mwcp")
    parameterized = Instance(
        graph.vertices,
        graph.vertex_weights,
        graph.interactions,
        alpha=alpha,
        eta=eta,
    )
    return canonical_parameterized_instance_id(
        parameterized,
        "paper_mwcp",
    ).split("::variant=", 1)[0]


def _parameterized_examples() -> dict[str, dict[str, str]]:
    graph = Instance(
        ("v0", "v1"),
        {"v0": Fraction(), "v1": Fraction()},
        {
            ("v0", "v1"): Fraction(-1),
            ("v1", "v0"): Fraction(-1),
        },
    )
    examples: dict[str, dict[str, str]] = {}
    for model_variant in SET_FUNCTION_VARIANTS:
        alpha, eta = canonical_variant_parameters(graph, model_variant)
        instance = Instance(
            graph.vertices,
            graph.vertex_weights,
            graph.interactions,
            alpha=alpha,
            eta=eta,
        )
        examples[model_variant] = {
            "alpha": _fraction_text(alpha),
            "eta": _fraction_text(eta),
            "instance_id": canonical_parameterized_instance_id(
                instance,
                model_variant,
            ),
        }
    return examples


def _greedy_domain() -> tuple[
    tuple[tuple[Instance, int], ...],
    dict[str, object],
]:
    vertex_domain = (Fraction(), Fraction(1), Fraction(2))
    edge_domain = (Fraction(-1), Fraction())
    finite_instances: list[tuple[Instance, int]] = []
    graph_records: list[dict[str, object]] = []

    for vertex_count in range(1, 5):
        vertices = tuple(f"v{index}" for index in range(vertex_count))
        edges = tuple(combinations(vertices, 2))
        for vertex_values in product(vertex_domain, repeat=vertex_count):
            for edge_values in product(edge_domain, repeat=len(edges)):
                graph = Instance(
                    vertices,
                    dict(zip(vertices, vertex_values, strict=True)),
                    _symmetric_interactions(edges, edge_values),
                )
                graph_records.append(
                    {
                        "id": _graph_id(graph),
                        "vertex_count": vertex_count,
                        "vertex_weights": [
                            _fraction_text(value) for value in vertex_values
                        ],
                        "edge_weights": [
                            _fraction_text(value) for value in edge_values
                        ],
                        "max_abs_interaction": _fraction_text(
                            max(
                                (abs(value) for value in edge_values),
                                default=Fraction(),
                            )
                        ),
                    }
                )
                for budget in range(1, min(3, vertex_count) + 1):
                    finite_instances.append((graph, budget))

    subset_records = [
        {
            "id": (
                f"n={vertex_count};subset="
                f"{','.join(selected) if selected else '-'}"
            ),
            "vertex_count": vertex_count,
            "selected": list(selected),
        }
        for vertex_count in range(1, 5)
        for size in range(vertex_count + 1)
        for selected in combinations(
            tuple(f"v{index}" for index in range(vertex_count)),
            size,
        )
    ]
    graph_records.sort(key=lambda record: str(record["id"]))
    subset_records.sort(key=lambda record: str(record["id"]))
    manifest_payload = {
        "graphs": graph_records,
        "subsets": subset_records,
        "parameterized_examples": _parameterized_examples(),
    }
    manifest = {
        **manifest_payload,
        "record_count": len(graph_records) + len(subset_records),
        "sha256": hashlib.sha256(
            canonical_json_bytes(manifest_payload)
        ).hexdigest(),
    }
    if len(graph_records) != 5_421 or len(finite_instances) != 16_239:
        raise AssertionError("canonical greedy domain accounting drift")
    return tuple(finite_instances), manifest


def _compute_audits(
    source_revision: str,
    source_root: Path,
    *,
    use_cache: bool,
) -> dict[str, object]:
    _validate_source_revision(source_revision)
    load_transcriptions(source_root)
    if use_cache and source_revision in _AUDIT_CACHE:
        return _AUDIT_CACHE[source_revision]

    appendix = appendix_shift_witness()
    finite_instances, domain_manifest = _greedy_domain()
    greedy = run_greedy_audit(source_revision)
    expected_examples = domain_manifest["parameterized_examples"]
    if greedy["canonical_parameter_examples"]["n=2;M=1/1"] != expected_examples:
        raise AssertionError("Task 3 canonical parameter examples drift")

    audits = {
        "equivalence": run_equivalence_audit(source_revision),
        "diminishing_returns": run_diminishing_returns_audit(source_revision),
        "shifts": run_shift_audit(source_revision),
        "greedy": greedy,
        "algorithm1": audit_literal_algorithm1(
            source_root / "paper_transcriptions" / "algorithm1.txt",
            project_root=source_root,
        ),
        "symbolic_ledger": run_symbolic_ledger_control(
            {"appendix_shift": appendix["id"]}
        ),
        "finite_ledger": run_finite_ledger_control(finite_instances),
        "appendix_witness": appendix,
        "cardinality_witness": cardinality_b_minus_t_witness(),
        "domain_manifest": domain_manifest,
    }
    if use_cache:
        _AUDIT_CACHE[source_revision] = audits
    return audits


def _component_actuals(audits: Mapping[str, object]) -> dict[str, int]:
    equivalence = audits["equivalence"]
    diminishing = audits["diminishing_returns"]
    shifts = audits["shifts"]
    greedy = audits["greedy"]
    finite = audits["finite_ledger"]
    symbolic = audits["symbolic_ledger"]
    return {
        "objective_equivalence_objective_values": (
            equivalence["search"]["cases_examined"] * 2
        ),
        "symmetric_diminishing_return_primitives": diminishing[
            "symmetric_search"
        ]["primitive_evaluations"],
        "asymmetric_literal_diagnostic_primitives": diminishing[
            "asymmetric_diagnostic"
        ]["primitive_evaluations"],
        "shift_marginal_score_values": shifts["boundary_search"][
            "values_evaluated"
        ],
        "rational_alpha_values": shifts["rational_alpha_controls"][
            "values_evaluated"
        ],
        "premise_subset_values": greedy["premise_certification"]["actual"][
            "subset_objective_values"
        ],
        "premise_marginal_values": greedy["premise_certification"]["actual"][
            "marginal_values"
        ],
        "premise_submodularity_comparisons": greedy[
            "premise_certification"
        ]["actual"]["diminishing_return_comparisons"],
        "eq7_candidate_scores": greedy["greedy_search"]["actual"][
            "eq7_candidate_scores"
        ],
        "eq7_terminal_paths": greedy["greedy_search"]["actual"][
            "eq7_terminal_paths"
        ],
        "true_marginal_candidate_lookups": greedy["greedy_search"]["actual"][
            "true_marginal_cache_lookups"
        ],
        "true_marginal_terminal_paths": greedy["greedy_search"]["actual"][
            "true_marginal_terminal_paths"
        ],
        "optimum_subset_objective_values": greedy["greedy_search"]["actual"][
            "optimum_objective_values"
        ],
        "greedy_summary_classifications": greedy["greedy_search"]["actual"][
            "classifications"
        ],
        "finite_appendix_f_conclusions": finite[
            "actual_conclusion_operations"
        ],
        "symbolic_appendix_f_conclusions": symbolic[
            "actual_conclusion_operations"
        ],
        "literal_algorithm1_audit": 1,
        "appendix_e_witness_marginals": 2,
    }


def _components(
    names: Iterable[str],
    actuals: Mapping[str, int],
    ceilings: Mapping[str, int],
) -> list[dict[str, object]]:
    return [
        {
            "id": name,
            "actual": actuals[name],
            "declared_ceiling": ceilings[name],
            "completed": True,
        }
        for name in sorted(names)
    ]


def _search_records(
    audits: Mapping[str, object],
    actuals: Mapping[str, int],
    ceilings: Mapping[str, int],
) -> list[dict[str, object]]:
    greedy_result = deepcopy(audits["greedy"])
    guarantee_summary = greedy_result["guarantee_summary"]
    guarantee_summary.pop("guarantee_violations")
    guarantee_summary.pop("out_of_premise_diagnostics")
    greedy_result["domain_manifest"] = deepcopy(audits["domain_manifest"])

    symbolic_summary = {
        key: deepcopy(value)
        for key, value in audits["symbolic_ledger"].items()
        if key != "ledgers"
    }
    groups = {
        "algorithm1": (
            "symbolic",
            ("literal_algorithm1_audit",),
            deepcopy(audits["algorithm1"]),
        ),
        "diminishing_returns": (
            "exhaustive_finite",
            (
                "symmetric_diminishing_return_primitives",
                "asymmetric_literal_diagnostic_primitives",
                "appendix_e_witness_marginals",
            ),
            deepcopy(audits["diminishing_returns"]),
        ),
        "greedy": (
            "exhaustive_finite",
            (
                "premise_subset_values",
                "premise_marginal_values",
                "premise_submodularity_comparisons",
                "eq7_candidate_scores",
                "eq7_terminal_paths",
                "true_marginal_candidate_lookups",
                "true_marginal_terminal_paths",
                "optimum_subset_objective_values",
                "greedy_summary_classifications",
            ),
            greedy_result,
        ),
        "objective_equivalence": (
            "symbolic",
            ("objective_equivalence_objective_values",),
            deepcopy(audits["equivalence"]),
        ),
        "proof_ledger": (
            "exhaustive_finite",
            (
                "finite_appendix_f_conclusions",
                "symbolic_appendix_f_conclusions",
            ),
            {
                "finite": deepcopy(audits["finite_ledger"]),
                "symbolic": symbolic_summary,
            },
        ),
        "shifts": (
            "non_exhaustive",
            ("shift_marginal_score_values", "rational_alpha_values"),
            deepcopy(audits["shifts"]),
        ),
    }
    return [
        {
            "id": search_id,
            "evidence_kind": evidence_kind,
            "completed": True,
            "status": "pass",
            "components": _components(names, actuals, ceilings),
            "result": result,
        }
        for search_id, (evidence_kind, names, result) in sorted(groups.items())
    ]


def _attach_witness_artifact(
    witness: Mapping[str, object],
) -> tuple[dict[str, object], bytes]:
    semantic = deepcopy(dict(witness))
    witness_id = semantic.get("id")
    if type(witness_id) is not str or not witness_id:
        raise ValueError("witness lacks a stable ID")
    payload = canonical_json_bytes(semantic)
    filename = f"{hashlib.sha256(witness_id.encode()).hexdigest()[:16]}.json"
    record = {
        **semantic,
        "artifact_path": f"evidence/witnesses/{filename}",
        "artifact_sha256": hashlib.sha256(payload).hexdigest(),
    }
    return record, payload


def _witness_records(
    audits: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, bytes]]:
    payloads = (
        audits["cardinality_witness"],
        audits["appendix_witness"],
        audits["equivalence"]["search"]["smallest_nonzero_edge_mismatch"],
    )
    records: list[dict[str, object]] = []
    files: dict[str, bytes] = {}
    for payload in payloads:
        record, artifact = _attach_witness_artifact(payload)
        logical_path = record["artifact_path"]
        if logical_path in files:
            raise ValueError("duplicate canonical witness artifact path")
        records.append(record)
        files[logical_path] = artifact
    records.sort(key=lambda record: str(record["id"]))
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("duplicate canonical witness ID")
    return records, files


def _claim_results(
    appendix_id: str,
    cardinality_id: str,
    objective_id: str,
) -> list[dict[str, object]]:
    records = [
        {
            "id": "appendix-f-cardinality-bound",
            "audit": "appendix_f_proof_ledger",
            "model_variant": "paper_samplewise_literal",
            "evidence_kind": "symbolic",
            "status": "contradicted",
            "witness_ids": [cardinality_id],
        },
        {
            "id": "appendix-f-literal-ledger",
            "audit": "appendix_f_proof_ledger",
            "model_variant": "appendix_inline_shift_literal",
            "evidence_kind": "symbolic",
            "status": "contradicted",
            "witness_ids": [appendix_id],
        },
        {
            "id": "appendix-inline-diminishing-returns",
            "audit": "diminishing_returns",
            "model_variant": "appendix_inline_shift_literal",
            "evidence_kind": "symbolic",
            "status": "contradicted",
            "witness_ids": [appendix_id],
        },
        {
            "id": "appendix-inline-greedy-premise",
            "audit": "greedy_guarantee_premise",
            "model_variant": "appendix_inline_shift_literal",
            "evidence_kind": "symbolic",
            "status": "contradicted",
            "witness_ids": [appendix_id],
        },
        {
            "id": "modular-shift-candidate-boundary",
            "audit": "shift_boundary",
            "model_variant": "modular_shift_candidate",
            "evidence_kind": "exhaustive_finite",
            "status": "supported",
            "witness_ids": [],
        },
        {
            "id": "paper-objective-equivalence",
            "audit": "objective_equivalence",
            "model_variant": "paper_samplewise_literal",
            "evidence_kind": "symbolic",
            "status": "contradicted",
            "witness_ids": [objective_id],
        },
    ]
    return sorted(records, key=lambda record: str(record["id"]))


def _unavailable_claims() -> list[dict[str, str]]:
    return [
        {
            "id": "cifar-imagenet-training-results",
            "status": "unavailable",
            "reason": (
                "Paper-reported CIFAR-10/100 and ImageNet-1k accuracy, "
                "training-time, and acceleration values were not recomputed."
            ),
        },
        {
            "id": "detection-segmentation-results",
            "status": "unavailable",
            "reason": (
                "Paper-reported detection and segmentation experiments "
                "were not recomputed."
            ),
        },
    ]


def _environment() -> dict[str, object]:
    return {
        "compute": "cpu",
        "network_used": False,
        "paid_api_cost_usd": "0/1",
        "python_requires": ">=3.11",
    }


def _build_evidence(
    output_dir: Path,
    source_revision: str,
    *,
    source_root: Path,
    use_cache: bool,
) -> dict[str, object]:
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a Path")
    _validate_source_revision(source_revision)
    output_dir.mkdir(parents=True, exist_ok=True)
    audits = _compute_audits(
        source_revision,
        source_root,
        use_cache=use_cache,
    )
    ceilings = declared_component_ceilings()
    actuals = _component_actuals(audits)
    if set(actuals) != set(ceilings):
        raise AssertionError("generation component set drift")
    if any(actuals[name] < 0 or actuals[name] > ceilings[name] for name in actuals):
        raise ValueError("generation component exceeds declared ceiling")
    aggregate_ceiling = declared_aggregate_ceiling()
    if aggregate_ceiling != 13_833_860:
        raise AssertionError("aggregate generation ceiling drift")
    aggregate_actual = sum(actuals.values())

    witnesses, witness_files = _witness_records(audits)
    witness_by_property = {
        witness["property"]: witness for witness in witnesses
    }
    appendix = witness_by_property[
        "appendix_inline_shift_diminishing_returns"
    ]
    cardinality = witness_by_property[
        "optimum_remainder_cardinality_exceeds_b_minus_t"
    ]
    objective = witness_by_property[
        "paper_mwcp_vs_paper_samplewise_literal"
    ]
    claims = _claim_results(
        str(appendix["id"]),
        str(cardinality["id"]),
        str(objective["id"]),
    )
    transcriptions = sorted(
        (dict(record) for record in load_transcriptions(source_root)),
        key=lambda record: str(record["record_id"]),
    )
    guarantee_violations = sorted(
        deepcopy(audits["greedy"]["guarantee_summary"]["guarantee_violations"]),
        key=lambda record: str(record["id"]),
    )
    diagnostics = sorted(
        deepcopy(
            audits["greedy"]["guarantee_summary"][
                "out_of_premise_diagnostics"
            ]
        ),
        key=lambda record: str(record["id"]),
    )
    appendix_index = next(
        index
        for index, witness in enumerate(witnesses)
        if witness["id"] == appendix["id"]
    )
    artifacts = [
        {
            "id": "render-pointers",
            "kind": "render_pointers",
            "render_pointers": [
                (
                    f"/witnesses/{appendix_index}/intermediate_values/"
                    "marginal_empty"
                ),
                (
                    f"/witnesses/{appendix_index}/intermediate_values/"
                    "marginal_y"
                ),
                "/commands/0/ceiling",
            ],
        },
        {
            "id": "witness-file-set",
            "kind": "file_set",
            "paths": sorted(witness_files),
        },
    ]
    evidence: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "attempt_id": ATTEMPT_ID,
        "source_revision": source_revision,
        "paper": {
            **PAPER,
            "pdf_acquisition_command": PDF_ACQUISITION_COMMAND,
        },
        "target_claims": list(TARGET_CLAIMS),
        "environment": _environment(),
        "transcriptions": {
            "set_sha256": TRANSCRIPTION_SET_SHA256,
            "records": transcriptions,
        },
        "searches": _search_records(audits, actuals, ceilings),
        "witnesses": witnesses,
        "guarantee_violations": guarantee_violations,
        "out_of_premise_diagnostics": diagnostics,
        "proof_ledger": {
            "symbolic": deepcopy(audits["symbolic_ledger"]),
            "finite": deepcopy(audits["finite_ledger"]),
        },
        "claim_results": claims,
        "unavailable_claims": _unavailable_claims(),
        "commands": [
            {
                "id": "recompute",
                "argv": [
                    "python",
                    "-m",
                    "graph_pruning_repro.cli",
                    "recompute",
                    "OUTPUT_DIR",
                    "--source-revision",
                    source_revision,
                ],
                "return_code": 0,
                "completed": True,
                "status": "completed",
                "actual": aggregate_actual,
                "ceiling": aggregate_ceiling,
            }
        ],
        "artifacts": artifacts,
    }

    witness_dir = output_dir / "witnesses"
    witness_dir.mkdir(parents=True, exist_ok=True)
    expected_names = {
        PurePosixPath(logical_path).name for logical_path in witness_files
    }
    for existing in witness_dir.iterdir():
        if existing.is_file() and existing.name not in expected_names:
            existing.unlink()
    for logical_path, payload in witness_files.items():
        _atomic_write(
            witness_dir / PurePosixPath(logical_path).name,
            payload,
        )
    _atomic_write(output_dir / "evidence.json", canonical_json_bytes(evidence))
    return evidence


def build_evidence(
    output_dir: Path,
    source_revision: str,
) -> dict[str, object]:
    """Run every prior audit and atomically emit canonical evidence files."""

    return _build_evidence(
        output_dir,
        source_revision,
        source_root=_source_root_for_output(output_dir),
        use_cache=True,
    )


def _load_json(path: Path, label: str) -> object:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{label} is missing or unreadable") from exc
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical UTF-8 JSON") from exc


def _source_root_for_validation(evidence_root: Path) -> Path:
    if (evidence_root / "paper_transcriptions").is_dir():
        return evidence_root.resolve()
    if (evidence_root.parent / "paper_transcriptions").is_dir():
        return evidence_root.parent.resolve()
    return _project_root()


def _flat_evidence_layout(
    evidence_path: Path,
    evidence_root: Path,
) -> bool:
    return evidence_path.absolute() == (
        evidence_root / "evidence.json"
    ).absolute()


def _artifact_path(
    logical_path: str,
    evidence_path: Path,
    evidence_root: Path,
) -> Path:
    relative = PurePosixPath(logical_path)
    if (
        relative.is_absolute()
        or relative.as_posix() != logical_path
        or len(relative.parts) != 3
        or relative.parts[:2] != ("evidence", "witnesses")
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError("witness artifact path is not canonical")
    if _flat_evidence_layout(evidence_path, evidence_root):
        return evidence_root.joinpath(*relative.parts[1:])
    return evidence_root.joinpath(*relative.parts)


def _actual_witness_dir(
    evidence_path: Path,
    evidence_root: Path,
) -> Path:
    if _flat_evidence_layout(evidence_path, evidence_root):
        return evidence_root / "witnesses"
    return evidence_root / "evidence" / "witnesses"


def _assert_no_clock_fields(value: object) -> None:
    if isinstance(value, Mapping):
        forbidden = _CLOCK_FIELDS.intersection(value)
        if forbidden:
            raise ValueError(
                f"canonical evidence contains measured clock fields: {forbidden}"
            )
        for child in value.values():
            _assert_no_clock_fields(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_clock_fields(child)


def _assert_canonical_rationals(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if (
                key in _RATIONAL_FIELDS
                and isinstance(child, str)
                and re.fullmatch(r"-?[0-9./]+", child) is not None
            ):
                if _FRACTION_TEXT.fullmatch(child) is None:
                    raise ValueError(f"{key} is not a canonical Fraction")
                parsed = Fraction(child)
                if _fraction_text(parsed) != child:
                    raise ValueError(f"{key} is not a normalized Fraction")
            _assert_canonical_rationals(child)
    elif isinstance(value, list):
        for child in value:
            _assert_canonical_rationals(child)


def _assert_ordered_unique_ids(
    records: object,
    label: str,
) -> None:
    if type(records) is not list:
        raise ValueError(f"{label} must be an array")
    identifiers = [
        record.get("id") if isinstance(record, Mapping) else None
        for record in records
    ]
    if any(type(identifier) is not str or not identifier for identifier in identifiers):
        raise ValueError(f"{label} contains an invalid ID")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{label} contains duplicate IDs")
    if identifiers != sorted(identifiers):
        raise ValueError(f"{label} is not in canonical ID order")


def _resolve_pointer(value: object, pointer: str) -> object:
    if type(pointer) is not str or not pointer.startswith("/"):
        raise ValueError("render pointer is not RFC 6901")
    current = value
    for encoded in pointer[1:].split("/"):
        if "{" in encoded or "}" in encoded:
            raise ValueError("ID-as-index pseudo-pointer is forbidden")
        token = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            if re.fullmatch(r"0|[1-9][0-9]*", token) is None:
                raise ValueError("render pointer array index is not numeric")
            index = int(token)
            if index >= len(current):
                raise ValueError("render pointer array index is out of range")
            current = current[index]
        elif isinstance(current, Mapping):
            if token not in current:
                raise ValueError("render pointer object key is missing")
            current = current[token]
        else:
            raise ValueError("render pointer traverses a scalar")
    return current


def _validate_witnesses(
    evidence: Mapping[str, object],
    evidence_path: Path,
    evidence_root: Path,
) -> None:
    witnesses = evidence["witnesses"]
    _assert_ordered_unique_ids(witnesses, "witnesses")
    expected_paths: set[str] = set()
    for witness in witnesses:
        logical_path = witness["artifact_path"]
        if logical_path in expected_paths:
            raise ValueError("duplicate witness artifact path")
        expected_paths.add(logical_path)
        actual_path = _artifact_path(logical_path, evidence_path, evidence_root)
        try:
            actual_bytes = actual_path.read_bytes()
        except OSError as exc:
            raise ValueError("canonical witness artifact is missing") from exc
        semantic = {
            key: deepcopy(value)
            for key, value in witness.items()
            if key not in {"artifact_path", "artifact_sha256"}
        }
        expected_bytes = canonical_json_bytes(semantic)
        if actual_bytes != expected_bytes:
            raise ValueError("canonical witness artifact bytes mismatch")
        if hashlib.sha256(actual_bytes).hexdigest() != witness["artifact_sha256"]:
            raise ValueError("canonical witness artifact hash mismatch")

    witness_dir = _actual_witness_dir(evidence_path, evidence_root)
    try:
        actual_names = {
            path.name
            for path in witness_dir.iterdir()
            if path.is_file()
        }
    except OSError as exc:
        raise ValueError("canonical witness directory is missing") from exc
    expected_names = {PurePosixPath(path).name for path in expected_paths}
    if actual_names != expected_names:
        raise ValueError("canonical witness file set mismatch")


def _expected_domain_manifest() -> dict[str, object]:
    _, manifest = _greedy_domain()
    return manifest


def _validate_searches(evidence: Mapping[str, object]) -> None:
    searches = evidence["searches"]
    _assert_ordered_unique_ids(searches, "searches")
    if tuple(search["id"] for search in searches) != _EXPECTED_SEARCH_IDS:
        raise ValueError("search domain set is not the source-defined set")
    declared = declared_component_ceilings()
    observed_components: dict[str, dict[str, object]] = {}
    for search in searches:
        if search["completed"] is not True or search["status"] != "pass":
            raise ValueError("canonical search is incomplete")
        components = search["components"]
        _assert_ordered_unique_ids(components, f"{search['id']} components")
        for component in components:
            component_id = component["id"]
            if component_id in observed_components:
                raise ValueError("generation component is duplicated")
            if component_id not in declared:
                raise ValueError("generation component is undeclared")
            if component["declared_ceiling"] != declared[component_id]:
                raise ValueError("generation component ceiling drift")
            if (
                type(component["actual"]) is not int
                or component["actual"] < 0
                or component["actual"] > component["declared_ceiling"]
                or component["completed"] is not True
            ):
                raise ValueError("generation component is incomplete or over ceiling")
            observed_components[component_id] = component
    if set(observed_components) != set(declared):
        raise ValueError("generation component set is incomplete")
    if sum(declared.values()) != 13_833_860:
        raise ValueError("aggregate generation ceiling drift")

    greedy = next(search for search in searches if search["id"] == "greedy")
    if greedy["result"]["model_variants"] != list(SET_FUNCTION_VARIANTS):
        raise ValueError("set-function model variants were merged or reordered")
    manifest = greedy["result"]["domain_manifest"]
    if canonical_json_bytes(manifest) != canonical_json_bytes(
        _expected_domain_manifest()
    ):
        raise ValueError("candidate domain differs from source-defined domain")

    command = evidence["commands"][0]
    actual = sum(component["actual"] for component in observed_components.values())
    if command["actual"] != actual or command["ceiling"] != 13_833_860:
        raise ValueError("canonical command accounting does not match components")


def _validate_classifications(evidence: Mapping[str, object]) -> None:
    _assert_ordered_unique_ids(evidence["claim_results"], "claim_results")
    _assert_ordered_unique_ids(
        evidence["guarantee_violations"],
        "guarantee_violations",
    )
    _assert_ordered_unique_ids(
        evidence["out_of_premise_diagnostics"],
        "out_of_premise_diagnostics",
    )
    witness_ids = {witness["id"] for witness in evidence["witnesses"]}
    for result in evidence["claim_results"]:
        if any(identifier not in witness_ids for identifier in result["witness_ids"]):
            raise ValueError("claim result links an unknown witness")

    appendix_id = next(
        witness["id"]
        for witness in evidence["witnesses"]
        if witness["property"] == "appendix_inline_shift_diminishing_returns"
    )
    cardinality_id = next(
        witness["id"]
        for witness in evidence["witnesses"]
        if witness["property"]
        == "optimum_remainder_cardinality_exceeds_b_minus_t"
    )
    objective_id = next(
        witness["id"]
        for witness in evidence["witnesses"]
        if witness["property"] == "paper_mwcp_vs_paper_samplewise_literal"
    )
    if evidence["claim_results"] != _claim_results(
        appendix_id,
        cardinality_id,
        objective_id,
    ):
        raise ValueError("claim result classifications or witness links drift")

    for violation in evidence["guarantee_violations"]:
        premise = violation.get("premise_evaluation")
        if not isinstance(premise, Mapping) or any(
            premise.get(name) is not True for name in _PREMISE_FIELDS
        ):
            raise ValueError(
                "ineligible result cannot be a guarantee violation"
            )
    diagnostic_keys = {
        "id",
        "failed_premise_ids",
        "failed_witness_ids",
        "ratio_classification",
    }
    for diagnostic in evidence["out_of_premise_diagnostics"]:
        if set(diagnostic) != diagnostic_keys:
            raise ValueError("out-of-premise diagnostic schema drift")


def _validate_proof_ledger(evidence: Mapping[str, object]) -> None:
    symbolic = evidence["proof_ledger"]["symbolic"]
    if symbolic["completed"] is not True:
        raise ValueError("symbolic proof ledger is incomplete")
    ledgers = symbolic["ledgers"]
    appendix_id = next(
        witness["id"]
        for witness in evidence["witnesses"]
        if witness["property"] == "appendix_inline_shift_diminishing_returns"
    )
    for model_variant, rows in ledgers.items():
        validate_prerequisite_graph(rows)
        expected = build_symbolic_ledger(
            model_variant,
            {"appendix_shift": appendix_id},
        )
        if canonical_json_bytes(rows) != canonical_json_bytes(expected):
            raise ValueError("symbolic proof ledger differs from source")
    finite = evidence["proof_ledger"]["finite"]
    if (
        finite["completed"] is not True
        or finite["weighted_cardinality_instances"] != 16_239
        or finite["actual_conclusion_operations"] > finite[
            "declared_conclusion_ceiling"
        ]
    ):
        raise ValueError("finite proof ledger is incomplete or over ceiling")


def _validate_artifacts(evidence: Mapping[str, object]) -> None:
    _assert_ordered_unique_ids(evidence["artifacts"], "artifacts")
    if [artifact["id"] for artifact in evidence["artifacts"]] != [
        "render-pointers",
        "witness-file-set",
    ]:
        raise ValueError("canonical artifact set drift")
    pointer_record = evidence["artifacts"][0]
    for pointer in pointer_record["render_pointers"]:
        _resolve_pointer(evidence, pointer)
    expected_paths = sorted(
        witness["artifact_path"] for witness in evidence["witnesses"]
    )
    if evidence["artifacts"][1]["paths"] != expected_paths:
        raise ValueError("witness artifact manifest drift")


def _validate_candidate_before_replay(
    evidence: Mapping[str, object],
    evidence_path: Path,
    evidence_root: Path,
    source_root: Path,
) -> None:
    if evidence["schema_version"] != SCHEMA_VERSION:
        raise ValueError("schema version drift")
    if evidence["attempt_id"] != ATTEMPT_ID:
        raise ValueError("attempt ID drift")
    _validate_source_revision(evidence["source_revision"])
    if evidence["paper"] != {
        **PAPER,
        "pdf_acquisition_command": PDF_ACQUISITION_COMMAND,
    }:
        raise ValueError("paper provenance drift")
    if evidence["target_claims"] != list(TARGET_CLAIMS):
        raise ValueError("target claims are missing, additional, or rewritten")
    if evidence["environment"] != _environment():
        raise ValueError("canonical environment record drift")
    if evidence["unavailable_claims"] != _unavailable_claims():
        raise ValueError("unavailable empirical-claim boundary drift")
    _assert_no_clock_fields(evidence)
    _assert_canonical_rationals(evidence)

    transcriptions = sorted(
        (dict(record) for record in load_transcriptions(source_root)),
        key=lambda record: str(record["record_id"]),
    )
    if evidence["transcriptions"] != {
        "set_sha256": TRANSCRIPTION_SET_SHA256,
        "records": transcriptions,
    }:
        raise ValueError("candidate transcription records differ from source")

    _assert_ordered_unique_ids(evidence["commands"], "commands")
    if evidence["commands"][0]["argv"] != [
        "python",
        "-m",
        "graph_pruning_repro.cli",
        "recompute",
        "OUTPUT_DIR",
        "--source-revision",
        evidence["source_revision"],
    ]:
        raise ValueError("canonical recompute command drift")
    _validate_searches(evidence)
    _validate_witnesses(evidence, evidence_path, evidence_root)
    _validate_classifications(evidence)
    _validate_proof_ledger(evidence)
    _validate_artifacts(evidence)


def _compare_witness_trees(
    candidate_path: Path,
    candidate_root: Path,
    replay_dir: Path,
) -> None:
    candidate_dir = _actual_witness_dir(candidate_path, candidate_root)
    replay_witness_dir = replay_dir / "witnesses"
    candidate_files = {
        path.name: path.read_bytes()
        for path in candidate_dir.iterdir()
        if path.is_file()
    }
    replay_files = {
        path.name: path.read_bytes()
        for path in replay_witness_dir.iterdir()
        if path.is_file()
    }
    if candidate_files != replay_files:
        raise ValueError("candidate witness tree differs from full replay")


def validate_evidence(
    evidence_path: Path,
    schema_path: Path,
    evidence_root: Path,
) -> None:
    """Require JSON Schema and byte-exact full deterministic replay."""

    if not all(
        isinstance(path, Path)
        for path in (evidence_path, schema_path, evidence_root)
    ):
        raise TypeError("evidence, schema, and root paths must be Path values")
    evidence_raw = _load_json(evidence_path, "evidence")
    schema_raw = _load_json(schema_path, "schema")
    if not isinstance(evidence_raw, dict) or not isinstance(schema_raw, dict):
        raise ValueError("evidence and schema must be JSON objects")
    validator = Draft202012Validator(schema_raw)
    errors = sorted(validator.iter_errors(evidence_raw), key=lambda error: list(error.path))
    if errors:
        raise ValueError(f"JSON Schema validation failed: {errors[0].message}")
    if evidence_path.read_bytes() != canonical_json_bytes(evidence_raw):
        raise ValueError("evidence.json is not canonical JSON")

    source_root = _source_root_for_validation(evidence_root)
    _validate_candidate_before_replay(
        evidence_raw,
        evidence_path,
        evidence_root,
        source_root,
    )
    with tempfile.TemporaryDirectory(prefix="graph-pruning-full-replay-") as temporary:
        replay_dir = Path(temporary)
        replay = _build_evidence(
            replay_dir,
            evidence_raw["source_revision"],
            source_root=source_root,
            use_cache=False,
        )
        if canonical_json_bytes(evidence_raw) != canonical_json_bytes(replay):
            raise ValueError("candidate evidence differs from full replay")
        if evidence_path.read_bytes() != (replay_dir / "evidence.json").read_bytes():
            raise ValueError("candidate evidence bytes differ from full replay")
        _compare_witness_trees(evidence_path, evidence_root, replay_dir)
