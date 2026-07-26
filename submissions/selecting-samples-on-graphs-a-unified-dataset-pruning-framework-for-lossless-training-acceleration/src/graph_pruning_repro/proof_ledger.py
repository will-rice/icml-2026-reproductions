"""Explicit Appendix F proof ledger and named search accounting."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations

from .diminishing_returns import (
    ASYMMETRIC_PRIMITIVE_CEILING,
    SYMMETRIC_PRIMITIVE_CEILING,
    appendix_shift_witness,
    canonical_parameterized_instance_id,
    canonical_variant_parameters,
    closed_form_marginal,
    direct_marginal,
)
from .greedy import (
    CLASSIFICATION_CEILING,
    EQ7_PATH_CEILING,
    EQ7_SCORE_CEILING,
    OPTIMUM_VALUE_CEILING,
    PREMISE_DIMINISHING_CEILING,
    PREMISE_MARGINAL_CEILING,
    PREMISE_SUBSET_CEILING,
    TRUE_MARGINAL_LOOKUP_CEILING,
    TRUE_MARGINAL_PATH_CEILING,
    greedy_domain_formulas,
    premise_domain_formulas,
)
from .objectives import evaluate_objective
from .shifts import RATIONAL_ALPHA_VALUE_CEILING, SHIFT_VALUE_CEILING
from .types import (
    MODEL_VARIANTS,
    SET_FUNCTION_VARIANTS,
    Instance,
    ModelVariant,
)

OBJECTIVE_EQUIVALENCE_VALUE_CEILING = 52
GREEDY_INSTANCE_CEILING = 16_239
CONCLUSIONS_PER_VARIANT_INSTANCE = 12
FINITE_LEDGER_CONCLUSION_CEILING = 1_169_208
SYMBOLIC_LEDGER_CONCLUSION_CEILING = 84
ALGORITHM1_AUDIT_CEILING = 1
APPENDIX_WITNESS_MARGINAL_CEILING = 2
AGGREGATE_GENERATION_CEILING = 13_833_860

_ROW_KEYS = {
    "row_id",
    "equation",
    "model_variant",
    "instance_id",
    "conclusions",
}
_CONCLUSION_KEYS = {
    "conclusion_id",
    "statement",
    "required_premise_ids",
    "prerequisite_conclusion_refs",
    "check_id",
    "evidence_kind",
    "status",
    "blocked_by",
    "witness_ids",
}
_STATUS_VALUES = {"supported", "contradicted", "not_applicable"}
_REPAIRED_TRUE_MARGINAL_VARIANTS = {
    "single_counted_pairwise",
    "half_corrected_samplewise",
    "modular_shift_candidate",
}


@dataclass(frozen=True)
class _ConclusionSpec:
    row_id: str
    equation: str
    conclusion_id: str
    statement: str
    required_premise_ids: tuple[str, ...]
    prerequisite_keys: tuple[str, ...]
    check_id: str

    @property
    def key(self) -> str:
        return f"{self.row_id}/{self.conclusion_id}"


_CONCLUSION_SPECS = (
    _ConclusionSpec(
        "eq28",
        "28",
        "eq28_union_submodular_bound",
        (
            "f(S*) is bounded by f(S_t) plus the sum of candidate "
            "marginals."
        ),
        ("global_monotonicity", "global_submodularity"),
        (),
        "union_monotonicity_and_submodular_telescoping",
    ),
    _ConclusionSpec(
        "eq28",
        "28",
        "eq28_b_minus_t_bound",
        (
            "The marginal sum is bounded by (b-t) times the maximum "
            "candidate marginal."
        ),
        (
            "global_monotonicity",
            "global_submodularity",
            "nonnegative_candidate_marginals",
            "cardinality_b_minus_t_bound",
        ),
        ("eq28/eq28_union_submodular_bound",),
        "optimum_remainder_at_most_b_not_b_minus_t",
    ),
    _ConclusionSpec(
        "eq29",
        "29",
        "eq29_true_marginal_gain",
        (
            "The selected element is a true-marginal maximizer and "
            "S_(t+1) is defined."
        ),
        ("true_marginal_argmax", "defined_s_t_plus_1"),
        (),
        "true_marginal_argmax_and_defined_next_set",
    ),
    _ConclusionSpec(
        "eq30",
        "30",
        "eq30_gap_to_next_gain",
        (
            "The optimum gap is bounded by (b-t) times the selected "
            "true marginal."
        ),
        (),
        (
            "eq28/eq28_union_submodular_bound",
            "eq28/eq28_b_minus_t_bound",
            "eq29/eq29_true_marginal_gain",
        ),
        "combine_both_eq28_conclusions_and_eq29",
    ),
    _ConclusionSpec(
        "eq31",
        "31",
        "eq31_residual_identity",
        "The next residual equals the current residual minus the gain.",
        ("defined_algebraic_quantities",),
        (),
        "defined_residual_algebra",
    ),
    _ConclusionSpec(
        "eq32",
        "32",
        "eq32_contract_residual",
        "The next residual contracts by 1-1/(b-t).",
        ("positive_b_minus_t",),
        (
            "eq30/eq30_gap_to_next_gain",
            "eq31/eq31_residual_identity",
        ),
        "divide_only_by_positive_b_minus_t",
    ),
    _ConclusionSpec(
        "eq33",
        "33",
        "eq33_recurrence",
        "The residual recurrence holds for the indexed iteration.",
        ("defined_residual_sequence",),
        ("eq32/eq32_contract_residual",),
        "residual_recurrence_definition",
    ),
    _ConclusionSpec(
        "eq34",
        "34",
        "eq34_product_recurrence",
        (
            "The recurrence product includes every t=0,...,b-1 and "
            "therefore k=1."
        ),
        ("recurrence_for_every_t", "exact_product_indices"),
        ("eq33/eq33_recurrence",),
        "product_includes_k_equals_one_zero_factor",
    ),
    _ConclusionSpec(
        "eq35",
        "35",
        "eq35_product_bound",
        "The exact product is bounded by (1-1/b)^b.",
        ("well_defined_product", "positive_integer_budget"),
        ("eq34/eq34_product_recurrence",),
        "well_defined_positive_budget_product_bound",
    ),
    _ConclusionSpec(
        "eq36",
        "36",
        "eq36_exponential_bound",
        "(1-1/b)^b is at most 1/e on its exact integer domain.",
        ("positive_integer_budget",),
        (),
        "integer_budget_exponential_bound_and_log_domain",
    ),
    _ConclusionSpec(
        "eq37",
        "37",
        "eq37_final_gap",
        "The greedy residual is at most f(S*)/e.",
        (
            "global_nonnegativity",
            "normalization",
            "global_monotonicity",
            "global_submodularity",
        ),
        (
            "eq34/eq34_product_recurrence",
            "eq35/eq35_product_bound",
            "eq36/eq36_exponential_bound",
        ),
        "complete_chain_and_all_theorem_premises",
    ),
    _ConclusionSpec(
        "eq38",
        "38",
        "eq38_greedy_ratio",
        "The stated objective has the (1-1/e) greedy guarantee.",
        ("exact_objective_relation", "nonnegative_optimum"),
        ("eq37/eq37_final_gap",),
        "ratio_transfer_from_repaired_objective",
    ),
)
_SPEC_BY_KEY = {spec.key: spec for spec in _CONCLUSION_SPECS}


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _canonical_ref(
    model_variant: str,
    instance_id: str,
    key: str,
) -> str:
    return f"{model_variant}/{instance_id}/{key}"


def _direct_status(spec: _ConclusionSpec, model_variant: str) -> str:
    if model_variant == "appendix_eq26_score":
        return "not_applicable"
    if spec.check_id == "optimum_remainder_at_most_b_not_b_minus_t":
        return "contradicted"
    if (
        spec.check_id
        == "union_monotonicity_and_submodular_telescoping"
        and model_variant == "appendix_inline_shift_literal"
    ):
        return "contradicted"
    if spec.check_id == "true_marginal_argmax_and_defined_next_set":
        return (
            "supported"
            if model_variant in _REPAIRED_TRUE_MARGINAL_VARIANTS
            else "contradicted"
        )
    if (
        spec.check_id == "ratio_transfer_from_repaired_objective"
        and model_variant == "appendix_inline_shift_literal"
    ):
        return "contradicted"
    return "supported"


def _depends_on_global_submodularity(
    key: str,
    visiting: frozenset[str] = frozenset(),
) -> bool:
    if key in visiting:
        raise ValueError("source prerequisite graph contains a cycle")
    spec = _SPEC_BY_KEY[key]
    if "global_submodularity" in spec.required_premise_ids:
        return True
    return any(
        _depends_on_global_submodularity(
            prerequisite,
            visiting.union({key}),
        )
        for prerequisite in spec.prerequisite_keys
    )


def _conclusion_witness_ids(
    spec: _ConclusionSpec,
    *,
    model_variant: str,
    appendix_witness_id: str | None,
) -> list[str]:
    identifiers: list[str] = []
    if spec.check_id == "optimum_remainder_at_most_b_not_b_minus_t":
        identifiers.append("cardinality-b-minus-t")
    if (
        model_variant == "appendix_inline_shift_literal"
        and _depends_on_global_submodularity(spec.key)
    ):
        if appendix_witness_id is None:
            raise ValueError("Appendix literal ledger requires its witness ID")
        identifiers.append(appendix_witness_id)
    return identifiers


def _build_ledger(
    model_variant: str,
    instance_id: str,
    evidence_kind: str,
    appendix_witness_id: str | None,
) -> tuple[dict[str, object], ...]:
    status_by_key: dict[str, str] = {}
    conclusions_by_row: dict[str, list[dict[str, object]]] = {
        f"eq{equation}": [] for equation in range(28, 39)
    }
    for spec in _CONCLUSION_SPECS:
        prerequisite_refs = [
            _canonical_ref(
                model_variant,
                instance_id,
                prerequisite,
            )
            for prerequisite in spec.prerequisite_keys
        ]
        direct_status = _direct_status(spec, model_variant)
        blockers = [
            reference
            for prerequisite, reference in zip(
                spec.prerequisite_keys,
                prerequisite_refs,
                strict=True,
            )
            if status_by_key[prerequisite] != "supported"
        ]
        if direct_status == "contradicted":
            status = "contradicted"
            blocked_by: list[str] = []
        elif direct_status == "not_applicable":
            status = "not_applicable"
            blocked_by = []
        elif blockers:
            status = "not_applicable"
            blocked_by = blockers
        else:
            status = "supported"
            blocked_by = []
        status_by_key[spec.key] = status
        conclusions_by_row[spec.row_id].append(
            {
                "conclusion_id": spec.conclusion_id,
                "statement": spec.statement,
                "required_premise_ids": list(spec.required_premise_ids),
                "prerequisite_conclusion_refs": prerequisite_refs,
                "check_id": spec.check_id,
                "evidence_kind": evidence_kind,
                "status": status,
                "blocked_by": blocked_by,
                "witness_ids": _conclusion_witness_ids(
                    spec,
                    model_variant=model_variant,
                    appendix_witness_id=appendix_witness_id,
                ),
            }
        )

    return tuple(
        {
            "row_id": f"eq{equation}",
            "equation": str(equation),
            "model_variant": model_variant,
            "instance_id": instance_id,
            "conclusions": conclusions_by_row[f"eq{equation}"],
        }
        for equation in range(28, 39)
    )


def build_symbolic_ledger(
    model_variant: ModelVariant,
    witness_ids: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    """Build all 11 explicit symbolic rows and their 12 conclusions."""

    if (
        type(model_variant) is not str
        or model_variant not in MODEL_VARIANTS
    ):
        raise ValueError("unknown model variant")
    if not isinstance(witness_ids, Mapping):
        raise TypeError("witness_ids must be a mapping")
    appendix_witness_id: str | None = None
    if model_variant == "appendix_inline_shift_literal":
        candidate = witness_ids.get("appendix_shift")
        if type(candidate) is not str or not candidate:
            raise ValueError("Appendix literal ledger requires its witness ID")
        appendix_witness_id = candidate
    rows = _build_ledger(
        model_variant,
        "symbolic",
        "symbolic",
        appendix_witness_id,
    )
    validate_prerequisite_graph(rows)
    return rows


def _validate_rows_shape(
    rows: Sequence[Mapping[str, object]],
) -> tuple[str, str, dict[str, dict[str, object]]]:
    if len(rows) != 11:
        raise ValueError("proof ledger must contain 11 rows")
    model_variants = {row.get("model_variant") for row in rows}
    instance_ids = {row.get("instance_id") for row in rows}
    if (
        len(model_variants) != 1
        or len(instance_ids) != 1
        or not all(type(value) is str and value for value in model_variants)
        or not all(type(value) is str and value for value in instance_ids)
    ):
        raise ValueError("proof ledger crosses variant or instance boundaries")
    model_variant = next(iter(model_variants))
    instance_id = next(iter(instance_ids))
    conclusions: dict[str, dict[str, object]] = {}
    observed_rows: set[str] = set()
    for row in rows:
        if set(row) != _ROW_KEYS:
            raise ValueError("proof ledger row schema drift")
        row_id = row["row_id"]
        equation = row["equation"]
        if (
            type(row_id) is not str
            or type(equation) is not str
            or row_id != f"eq{equation}"
            or row_id in observed_rows
        ):
            raise ValueError("invalid or duplicate proof ledger row")
        observed_rows.add(row_id)
        row_conclusions = row["conclusions"]
        if type(row_conclusions) is not list:
            raise ValueError("row conclusions must be a list")
        for conclusion in row_conclusions:
            if (
                not isinstance(conclusion, Mapping)
                or set(conclusion) != _CONCLUSION_KEYS
            ):
                raise ValueError("proof conclusion schema drift")
            conclusion_id = conclusion["conclusion_id"]
            if type(conclusion_id) is not str or not conclusion_id:
                raise ValueError("invalid conclusion ID")
            key = f"{row_id}/{conclusion_id}"
            if key in conclusions:
                raise ValueError("duplicate proof conclusion")
            if conclusion["status"] not in _STATUS_VALUES:
                raise ValueError("invalid proof conclusion status")
            conclusions[key] = dict(conclusion)
    if set(conclusions) != set(_SPEC_BY_KEY):
        raise ValueError("missing or replaced proof conclusion")
    return model_variant, instance_id, conclusions


def _reject_prerequisite_cycle(
    adjacency: Mapping[str, tuple[str, ...]],
) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(key: str) -> None:
        if key in visiting:
            raise ValueError("cyclic proof prerequisite graph")
        if key in visited:
            return
        visiting.add(key)
        for prerequisite in adjacency[key]:
            visit(prerequisite)
        visiting.remove(key)
        visited.add(key)

    for key in adjacency:
        visit(key)


def validate_prerequisite_graph(
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Reject any departure from the source-defined acyclic adjacency map."""

    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("rows must be a sequence")
    model_variant, instance_id, conclusions = _validate_rows_shape(rows)
    ref_to_key = {
        _canonical_ref(model_variant, instance_id, key): key
        for key in conclusions
    }
    adjacency: dict[str, tuple[str, ...]] = {}
    for key, conclusion in conclusions.items():
        references = conclusion["prerequisite_conclusion_refs"]
        if type(references) is not list or any(
            type(reference) is not str for reference in references
        ):
            raise ValueError("prerequisite references must be strings")
        if len(references) != len(set(references)):
            raise ValueError("duplicate proof prerequisite reference")
        unknown = [
            reference
            for reference in references
            if reference not in ref_to_key
        ]
        if unknown:
            raise ValueError("unknown or cross-instance prerequisite reference")
        adjacency[key] = tuple(ref_to_key[reference] for reference in references)
    _reject_prerequisite_cycle(adjacency)
    expected = {
        spec.key: spec.prerequisite_keys for spec in _CONCLUSION_SPECS
    }
    if adjacency != expected:
        raise ValueError("missing, replaced, or reordered prerequisite edge")


def equation_36(budget: int) -> dict[str, str]:
    """Audit the exact integer and logarithm domains of Eq. (36)."""

    if type(budget) is not int:
        raise TypeError("budget must be an integer")
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    if budget == 0:
        conclusion = "not_applicable"
        logarithm = "not_applicable"
    elif budget == 1:
        conclusion = "supported"
        logarithm = "not_applicable"
    else:
        conclusion = "supported"
        logarithm = "supported"
    return {
        "status": conclusion,
        "conclusion_status": conclusion,
        "log_derivation_status": logarithm,
    }


def run_symbolic_ledger_control(
    witness_ids: Mapping[str, object],
) -> dict[str, object]:
    """Evaluate 12 symbolic conclusions for each of seven named variants."""

    ledgers = {
        model_variant: build_symbolic_ledger(model_variant, witness_ids)
        for model_variant in MODEL_VARIANTS
    }
    actual = sum(
        len(row["conclusions"])
        for rows in ledgers.values()
        for row in rows
    )
    required = len(MODEL_VARIANTS) * CONCLUSIONS_PER_VARIANT_INSTANCE
    if required != SYMBOLIC_LEDGER_CONCLUSION_CEILING:
        raise ValueError("symbolic ledger ceiling does not match its formula")
    if actual != required:
        raise AssertionError("symbolic ledger conclusion accounting drift")
    return {
        "evidence_kind": "symbolic",
        "model_variants": list(MODEL_VARIANTS),
        "conclusions_per_variant": CONCLUSIONS_PER_VARIANT_INSTANCE,
        "actual_conclusion_operations": actual,
        "declared_conclusion_ceiling": (
            SYMBOLIC_LEDGER_CONCLUSION_CEILING
        ),
        "completed": True,
        "ledgers": ledgers,
    }


def _validate_finite_preflight() -> dict[str, int]:
    formulas = greedy_domain_formulas()
    if formulas["weighted_cardinality_instances"] != GREEDY_INSTANCE_CEILING:
        raise ValueError("finite ledger input formula drift")
    required = (
        GREEDY_INSTANCE_CEILING
        * len(SET_FUNCTION_VARIANTS)
        * CONCLUSIONS_PER_VARIANT_INSTANCE
    )
    if required != FINITE_LEDGER_CONCLUSION_CEILING:
        raise ValueError("finite ledger ceiling does not match its formula")
    build_symbolic_ledger("paper_samplewise_literal", {})
    return formulas


def _validate_greedy_domain_graph(graph: Instance) -> None:
    if not isinstance(graph, Instance):
        raise TypeError("finite ledger instances must contain Instance values")
    if graph.alpha != Fraction(1) or graph.eta != Fraction():
        raise ValueError("greedy-domain input graph must be unparameterized")
    n = len(graph.vertices)
    if n < 1 or n > 4:
        raise ValueError("finite ledger graph is outside the vertex domain")
    if graph.vertices != tuple(f"v{index}" for index in range(n)):
        raise ValueError("finite ledger graph vertices are not canonical")
    vertex_domain = (Fraction(), Fraction(1), Fraction(2))
    if any(
        graph.vertex_weights[vertex] not in vertex_domain
        for vertex in graph.vertices
    ):
        raise ValueError("finite ledger graph is outside the vertex domain")
    pairs = tuple(combinations(graph.vertices, 2))
    expected_edges = {
        directed
        for left, right in pairs
        for directed in ((left, right), (right, left))
    }
    if set(graph.interactions) != expected_edges:
        raise ValueError("finite ledger graph interactions are incomplete")
    edge_domain = (Fraction(-1), Fraction())
    for left, right in pairs:
        forward = graph.interactions[(left, right)]
        reverse = graph.interactions[(right, left)]
        if forward != reverse or forward not in edge_domain:
            raise ValueError("finite ledger graph is outside the edge domain")


def _parameterized_instance(
    graph: Instance,
    model_variant: ModelVariant,
) -> tuple[Instance, str]:
    alpha, eta = canonical_variant_parameters(graph, model_variant)
    instance = Instance(
        graph.vertices,
        graph.vertex_weights,
        graph.interactions,
        alpha=alpha,
        eta=eta,
    )
    if (instance.alpha, instance.eta) != canonical_variant_parameters(
        instance,
        model_variant,
    ):
        raise AssertionError("Task 3 canonical parameter tuple drift")
    base_id = canonical_parameterized_instance_id(instance, model_variant)
    suffix = (
        f"::variant={model_variant}"
        f"::alpha={_fraction_text(alpha)}"
        f"::eta={_fraction_text(eta)}"
    )
    if not base_id.endswith(suffix):
        raise AssertionError("Task 3 canonical parameterized ID drift")
    return instance, base_id


def _oracle_example(
    graph: Instance,
    budget: int,
    model_variant: ModelVariant,
) -> tuple[dict[str, object], str]:
    instance, base_id = _parameterized_instance(graph, model_variant)
    selected = frozenset(graph.vertices[: max(0, budget - 1)])
    candidate = next(
        vertex for vertex in graph.vertices if vertex not in selected
    )
    objective_selected = evaluate_objective(
        instance,
        selected,
        model_variant,
    )
    direct = direct_marginal(
        instance,
        selected,
        candidate,
        model_variant,
    )
    closed = closed_form_marginal(
        instance,
        selected,
        candidate,
        model_variant,
    )
    if direct != closed:
        raise AssertionError("Task 2 and Task 3 marginal oracles disagree")

    literal_base: Fraction | None = None
    modular_formula: Fraction | None = None
    if model_variant == "modular_shift_candidate":
        literal_base = evaluate_objective(
            instance,
            selected,
            "paper_samplewise_literal",
        )
        if objective_selected != literal_base + instance.eta * len(selected):
            raise AssertionError("modular objective does not use literal base")
        directed_incident = sum(
            (
                instance.interactions.get(
                    (candidate, other),
                    Fraction(),
                )
                + instance.interactions.get(
                    (other, candidate),
                    Fraction(),
                )
                for other in selected
            ),
            start=Fraction(),
        )
        modular_formula = (
            instance.vertex_weights[candidate]
            + directed_incident
            + instance.eta
        )
        if closed != modular_formula:
            raise AssertionError("modular marginal formula drift")

    return (
        {
            "base_instance_id": base_id,
            "alpha": _fraction_text(instance.alpha),
            "eta": _fraction_text(instance.eta),
            "budget": budget,
            "selected": sorted(selected),
            "candidate": candidate,
            "objective_selected": _fraction_text(objective_selected),
            "literal_base_objective_selected": (
                _fraction_text(literal_base)
                if literal_base is not None
                else None
            ),
            "direct_marginal": _fraction_text(direct),
            "closed_form_marginal": _fraction_text(closed),
            "modular_formula_marginal": (
                _fraction_text(modular_formula)
                if modular_formula is not None
                else None
            ),
        },
        base_id,
    )


def run_finite_ledger_control(
    instances: Iterable[tuple[Instance, int]],
) -> dict[str, object]:
    """Consume and audit the already-enumerated greedy instance domain."""

    formulas = _validate_finite_preflight()
    if isinstance(instances, (str, bytes)) or not isinstance(
        instances,
        Iterable,
    ):
        raise TypeError("instances must be an iterable")
    materialized = tuple(instances)
    if len(materialized) > GREEDY_INSTANCE_CEILING:
        raise ValueError("finite ledger instance count exceeds its ceiling")

    appendix_id = appendix_shift_witness()["id"]
    assert isinstance(appendix_id, str)
    witness_ids = {"appendix_shift": appendix_id}
    seen_cases: set[str] = set()
    examples: dict[str, dict[str, object]] = {}
    digest = hashlib.sha256()
    conclusion_operations = 0

    for item in materialized:
        if type(item) is not tuple or len(item) != 2:
            raise TypeError("finite ledger item must be (Instance, budget)")
        graph, budget = item
        _validate_greedy_domain_graph(graph)
        if type(budget) is not int:
            raise TypeError("finite ledger budget must be an integer")
        if budget < 1 or budget > min(3, len(graph.vertices)):
            raise ValueError("finite ledger budget is outside the domain")
        _, graph_base_id = _parameterized_instance(graph, "paper_mwcp")
        case_id = f"{graph_base_id}::budget={budget}"
        if case_id in seen_cases:
            raise ValueError("duplicate finite ledger instance")
        seen_cases.add(case_id)

        for model_variant in SET_FUNCTION_VARIANTS:
            example, base_id = _oracle_example(
                graph,
                budget,
                model_variant,
            )
            examples.setdefault(model_variant, example)
            instance_id = f"{base_id}::budget={budget}"
            rows = _build_ledger(
                model_variant,
                instance_id,
                "exhaustive_finite",
                (
                    witness_ids["appendix_shift"]
                    if model_variant == "appendix_inline_shift_literal"
                    else None
                ),
            )
            for row in rows:
                for conclusion in row["conclusions"]:
                    reference = _canonical_ref(
                        model_variant,
                        instance_id,
                        f"{row['row_id']}/{conclusion['conclusion_id']}",
                    )
                    digest.update(reference.encode("ascii"))
                    digest.update(b"\0")
                    digest.update(conclusion["status"].encode("ascii"))
                    digest.update(b"\n")
                    conclusion_operations += 1

    expected_operations = (
        len(materialized)
        * len(SET_FUNCTION_VARIANTS)
        * CONCLUSIONS_PER_VARIANT_INSTANCE
    )
    if conclusion_operations != expected_operations:
        raise AssertionError("finite ledger conclusion accounting drift")
    if conclusion_operations > FINITE_LEDGER_CONCLUSION_CEILING:
        raise AssertionError("finite ledger conclusion ceiling exceeded")
    completed = len(materialized) == formulas[
        "weighted_cardinality_instances"
    ]
    return {
        "evidence_kind": (
            "exhaustive_finite" if completed else "non_exhaustive"
        ),
        "weighted_cardinality_instances": len(materialized),
        "model_variant_instances": (
            len(materialized) * len(SET_FUNCTION_VARIANTS)
        ),
        "model_variants": list(SET_FUNCTION_VARIANTS),
        "conclusions_per_variant_instance": (
            CONCLUSIONS_PER_VARIANT_INSTANCE
        ),
        "actual_conclusion_operations": conclusion_operations,
        "declared_conclusion_ceiling": FINITE_LEDGER_CONCLUSION_CEILING,
        "record_digest_sha256": digest.hexdigest(),
        "canonical_parameter_examples": examples,
        "completed": completed,
    }


def declared_component_ceilings() -> dict[str, int]:
    """Return all 18 independently named generation components."""

    premises = premise_domain_formulas()
    greedy = greedy_domain_formulas()
    finite_formula = (
        greedy["weighted_cardinality_instances"]
        * len(SET_FUNCTION_VARIANTS)
        * CONCLUSIONS_PER_VARIANT_INSTANCE
    )
    symbolic_formula = (
        len(MODEL_VARIANTS) * CONCLUSIONS_PER_VARIANT_INSTANCE
    )
    if finite_formula != FINITE_LEDGER_CONCLUSION_CEILING:
        raise ValueError("finite ledger ceiling does not match its formula")
    if symbolic_formula != SYMBOLIC_LEDGER_CONCLUSION_CEILING:
        raise ValueError("symbolic ledger ceiling does not match its formula")
    return {
        "objective_equivalence_objective_values": (
            OBJECTIVE_EQUIVALENCE_VALUE_CEILING
        ),
        "symmetric_diminishing_return_primitives": (
            SYMMETRIC_PRIMITIVE_CEILING
        ),
        "asymmetric_literal_diagnostic_primitives": (
            ASYMMETRIC_PRIMITIVE_CEILING
        ),
        "shift_marginal_score_values": SHIFT_VALUE_CEILING,
        "rational_alpha_values": RATIONAL_ALPHA_VALUE_CEILING,
        "premise_subset_values": premises["subset_objective_values"],
        "premise_marginal_values": premises["marginal_values"],
        "premise_submodularity_comparisons": premises[
            "diminishing_return_comparisons"
        ],
        "eq7_candidate_scores": greedy["eq7_candidate_scores"],
        "eq7_terminal_paths": greedy["eq7_terminal_paths"],
        "true_marginal_candidate_lookups": greedy[
            "true_marginal_cache_lookups"
        ],
        "true_marginal_terminal_paths": greedy[
            "true_marginal_terminal_paths"
        ],
        "optimum_subset_objective_values": greedy[
            "optimum_objective_values"
        ],
        "greedy_summary_classifications": greedy["classifications"],
        "finite_appendix_f_conclusions": (
            FINITE_LEDGER_CONCLUSION_CEILING
        ),
        "symbolic_appendix_f_conclusions": (
            SYMBOLIC_LEDGER_CONCLUSION_CEILING
        ),
        "literal_algorithm1_audit": ALGORITHM1_AUDIT_CEILING,
        "appendix_e_witness_marginals": (
            APPENDIX_WITNESS_MARGINAL_CEILING
        ),
    }


def declared_aggregate_ceiling() -> int:
    """Sum the named components and reject any aggregate drift."""

    total = sum(declared_component_ceilings().values())
    if total != AGGREGATE_GENERATION_CEILING:
        raise ValueError("aggregate generation ceiling drift")
    return total
