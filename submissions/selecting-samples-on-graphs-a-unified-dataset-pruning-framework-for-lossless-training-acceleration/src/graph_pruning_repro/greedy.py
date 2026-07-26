"""Exact, separated greedy paths and premise-gated ratio audit."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from fractions import Fraction
from itertools import combinations, count, product

from .diminishing_returns import (
    canonical_parameterized_instance_id,
    canonical_variant_parameters,
    direct_marginal,
)
from .objectives import evaluate_objective
from .types import SET_FUNCTION_VARIANTS, Instance, ModelVariant, Vertex

PREMISE_SUBSET_CEILING = 508_500
PREMISE_MARGINAL_CEILING = 1_011_330
PREMISE_DIMINISHING_CEILING = 3_394_890
EQ7_SCORE_CEILING = 316_983
EQ7_PATH_CEILING = 210_675
TRUE_MARGINAL_LOOKUP_CEILING = 1_901_898
TRUE_MARGINAL_PATH_CEILING = 1_264_050
OPTIMUM_VALUE_CEILING = 444_870
CLASSIFICATION_CEILING = 584_604

_MAX_VERTICES = 4
_APPROVED_VERTEX_DOMAIN = (Fraction(), Fraction(1), Fraction(2))
_APPROVED_EDGE_DOMAIN = (Fraction(-1), Fraction())
_VERTEX_DOMAIN = _APPROVED_VERTEX_DOMAIN
_EDGE_DOMAIN = _APPROVED_EDGE_DOMAIN
_SUBSET_MEMBERSHIP_STATES = 2
_DIMINISHING_RELATION_STATES = 3
_PREMISE_NAMES = (
    "global_nonnegativity",
    "normalization",
    "global_monotonicity",
    "global_submodularity",
)
_SELECTORS = ("paper_eq7_score_greedy", "true_marginal_greedy")
_RESULT_KINDS = ("best", "worst", "canonical")


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _validate_source_revision(source_revision: str) -> None:
    if type(source_revision) is not str:
        raise TypeError("source_revision must be a string")
    if not source_revision.strip():
        raise ValueError("source_revision must be nonempty")


def _validate_budget(instance: Instance, budget: int) -> None:
    if type(budget) is not int:
        raise TypeError("budget must be an integer")
    if budget < 1 or budget > len(instance.vertices):
        raise ValueError("budget must be between one and the vertex count")


def _validate_candidate(
    instance: Instance,
    selected: frozenset[Vertex],
    candidate: Vertex,
) -> None:
    if type(selected) is not frozenset:
        raise TypeError("selected must be a frozenset")
    if not selected.issubset(instance.vertices):
        raise ValueError("selected contains an unknown vertex")
    if type(candidate) is not str or candidate not in instance.vertices:
        raise ValueError("candidate must be an instance vertex")
    if candidate in selected:
        raise ValueError("candidate must not be selected")


def eq7_score(
    instance: Instance,
    selected: frozenset[Vertex],
    candidate: Vertex,
) -> Fraction:
    """Evaluate Eq. (7) directly, without any objective or marginal call."""

    _validate_candidate(instance, selected, candidate)
    return instance.vertex_weights[candidate] + sum(
        (
            instance.interactions.get((candidate, other), Fraction())
            for other in selected
        ),
        start=Fraction(),
    )


def _enumerate_tied_paths(
    vertices: Sequence[Vertex],
    budget: int,
    score: Callable[[frozenset[Vertex], Vertex], Fraction],
) -> tuple[tuple[tuple[Vertex, ...], ...], int]:
    ordered_vertices = tuple(sorted(vertices))
    active: tuple[tuple[Vertex, ...], ...] = ((),)
    operations = 0
    for _ in range(budget):
        next_active: list[tuple[Vertex, ...]] = []
        for path in active:
            selected = frozenset(path)
            candidates = tuple(
                vertex
                for vertex in ordered_vertices
                if vertex not in selected
            )
            scored = tuple(
                (candidate, score(selected, candidate))
                for candidate in candidates
            )
            operations += len(scored)
            maximum = max(value for _, value in scored)
            next_active.extend(
                path + (candidate,)
                for candidate, value in scored
                if value == maximum
            )
        active = tuple(next_active)
    return active, operations


def enumerate_eq7_paths(
    instance: Instance,
    budget: int,
) -> tuple[tuple[Vertex, ...], ...]:
    """Return every Eq. (7) maximizing path in lexical branch order."""

    _validate_budget(instance, budget)
    paths, _ = _enumerate_tied_paths(
        instance.vertices,
        budget,
        lambda selected, candidate: eq7_score(
            instance,
            selected,
            candidate,
        ),
    )
    return paths


def enumerate_true_marginal_paths(
    instance: Instance,
    budget: int,
    model_variant: ModelVariant,
) -> tuple[tuple[Vertex, ...], ...]:
    """Return every true-objective-marginal maximizing path."""

    _validate_budget(instance, budget)

    def score(
        selected: frozenset[Vertex],
        candidate: Vertex,
    ) -> Fraction:
        base = evaluate_objective(instance, selected, model_variant)
        extended = evaluate_objective(
            instance,
            selected.union({candidate}),
            model_variant,
        )
        return extended - base

    paths, _ = _enumerate_tied_paths(instance.vertices, budget, score)
    return paths


def exhaustive_optima(
    instance: Instance,
    budget: int,
    model_variant: ModelVariant,
) -> tuple[frozenset[Vertex], ...]:
    """Enumerate exact size-budget optima and retain all ties."""

    _validate_budget(instance, budget)
    candidates = tuple(
        frozenset(items)
        for items in combinations(sorted(instance.vertices), budget)
    )
    values = {
        selected: evaluate_objective(instance, selected, model_variant)
        for selected in candidates
    }
    best = max(values.values())
    return tuple(
        selected for selected in candidates if values[selected] == best
    )


def classify_ratio(
    greedy_value: Fraction,
    optimum_value: Fraction,
) -> dict[str, object]:
    """Classify exact objective regimes without dividing by a bad optimum."""

    if type(greedy_value) is not Fraction or type(optimum_value) is not Fraction:
        raise TypeError("ratio values must be Fractions")
    if optimum_value < 0:
        return {
            "status": "negative_objective_regime",
            "ratio": None,
        }
    if optimum_value == 0:
        if greedy_value == 0:
            return {
                "status": "defined_zero_equality",
                "ratio": "1/1",
            }
        return {
            "status": "undefined_zero_optimum",
            "ratio": None,
        }
    return {
        "status": "defined_positive_optimum",
        "ratio": _fraction_text(greedy_value / optimum_value),
    }


def compare_ratio_to_one_minus_inverse_e(
    ratio: Fraction,
) -> dict[str, object]:
    """Certify a rational comparison using shrinking rational bounds on e."""

    if type(ratio) is not Fraction:
        raise TypeError("ratio must be a Fraction")
    factorial = 1
    partial = Fraction(2)
    for order in count(2):
        factorial *= order
        partial += Fraction(1, factorial)
        e_lower = partial
        e_upper = partial + Fraction(1, order * factorial)
        threshold_lower = Fraction(1) - Fraction(1, e_lower)
        threshold_upper = Fraction(1) - Fraction(1, e_upper)
        if ratio < threshold_lower:
            status = "below_bound"
        elif ratio > threshold_upper:
            status = "meets_bound"
        else:
            continue
        return {
            "status": status,
            "arithmetic": "exact_rational",
            "termination": (
                "unbounded refinement; irrational threshold separates rationals"
            ),
            "series_order": order,
            "e_lower": _fraction_text(e_lower),
            "e_upper": _fraction_text(e_upper),
            "threshold_lower": _fraction_text(threshold_lower),
            "threshold_upper": _fraction_text(threshold_upper),
        }


def _ratio_fraction(classification: Mapping[str, object]) -> Fraction | None:
    ratio = classification.get("ratio")
    if ratio is None:
        return None
    if type(ratio) is not str or "/" not in ratio:
        raise ValueError("ratio must be a normalized rational string")
    numerator, denominator = ratio.split("/", 1)
    return Fraction(int(numerator), int(denominator))


def summarize_guarantee(
    records: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    """Partition classifications under all four global theorem premises."""

    eligible_records = 0
    violations: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for record in records:
        premise = record["premise_evaluation"]
        classification = record["ratio_classification"]
        if not isinstance(premise, Mapping) or not isinstance(
            classification,
            Mapping,
        ):
            raise TypeError("guarantee records require mapping fields")
        failed = [
            name for name in _PREMISE_NAMES if premise.get(name) is not True
        ]
        record_id = record["id"]
        if type(record_id) is not str:
            raise TypeError("guarantee record ID must be a string")
        if failed:
            diagnostics.append(
                {
                    "id": record_id,
                    "failed_premise_ids": failed,
                    "failed_witness_ids": list(
                        premise.get("failed_witness_ids", [])
                    ),
                    "ratio_classification": dict(classification),
                }
            )
            continue

        eligible_records += 1
        ratio = _ratio_fraction(classification)
        below = (
            ratio is not None
            and compare_ratio_to_one_minus_inverse_e(ratio)["status"]
            == "below_bound"
        )
        if below:
            violations.append(
                {
                    "id": record_id,
                    "premise_evaluation": dict(premise),
                    "ratio_classification": dict(classification),
                    "bound_comparison": (
                        compare_ratio_to_one_minus_inverse_e(ratio)
                    ),
                }
            )
    if eligible_records == 0:
        status = "not_evaluated"
    elif violations:
        status = "contradicted"
    else:
        status = "supported"
    return {
        "status": status,
        "eligible_records": eligible_records,
        "guarantee_violations": violations,
        "out_of_premise_diagnostics": diagnostics,
    }


def premise_domain_formulas() -> dict[str, int]:
    """Compute every premise-domain total from the approved formulas."""

    _validate_exact_domains()
    vertex_values = len(_VERTEX_DOMAIN)
    edge_values = len(_EDGE_DOMAIN)
    weighted_graphs = sum(
        vertex_values**n * edge_values ** math.comb(n, 2)
        for n in range(1, _MAX_VERTICES + 1)
    )
    subsets = sum(
        vertex_values**n
        * edge_values ** math.comb(n, 2)
        * _SUBSET_MEMBERSHIP_STATES**n
        for n in range(1, _MAX_VERTICES + 1)
    )
    marginals = sum(
        vertex_values**n
        * edge_values ** math.comb(n, 2)
        * n
        * _SUBSET_MEMBERSHIP_STATES ** (n - 1)
        for n in range(1, _MAX_VERTICES + 1)
    )
    diminishing = sum(
        vertex_values**n
        * edge_values ** math.comb(n, 2)
        * n
        * _DIMINISHING_RELATION_STATES ** (n - 1)
        for n in range(1, _MAX_VERTICES + 1)
    )
    per_variant = subsets + marginals + diminishing
    variants = len(SET_FUNCTION_VARIANTS)
    return {
        "weighted_graphs": weighted_graphs,
        "subset_objective_values_per_variant": subsets,
        "marginal_values_per_variant": marginals,
        "diminishing_return_comparisons_per_variant": diminishing,
        "premise_work_per_variant": per_variant,
        "subset_objective_values": subsets * variants,
        "marginal_values": marginals * variants,
        "diminishing_return_comparisons": diminishing * variants,
        "premise_work": per_variant * variants,
    }


def greedy_domain_formulas() -> dict[str, int]:
    """Compute every greedy-domain total from the approved formulas."""

    _validate_exact_domains()
    vertex_values = len(_VERTEX_DOMAIN)
    edge_values = len(_EDGE_DOMAIN)
    weighted_cardinality_instances = 0
    optima = 0
    paths = 0
    candidates = 0
    for n in range(1, _MAX_VERTICES + 1):
        graphs = vertex_values**n * edge_values ** math.comb(n, 2)
        maximum_budget = min(3, n)
        weighted_cardinality_instances += graphs * maximum_budget
        optima += graphs * sum(
            math.comb(n, budget)
            for budget in range(1, maximum_budget + 1)
        )
        paths += graphs * sum(
            math.factorial(n) // math.factorial(n - budget)
            for budget in range(1, maximum_budget + 1)
        )
        candidates += graphs * sum(
            sum(
                math.factorial(n) // math.factorial(n - depth)
                for depth in range(1, budget + 1)
            )
            for budget in range(1, maximum_budget + 1)
        )
    variants = len(SET_FUNCTION_VARIANTS)
    classifications = (
        weighted_cardinality_instances
        * variants
        * len(_SELECTORS)
        * len(_RESULT_KINDS)
    )
    return {
        "weighted_cardinality_instances": weighted_cardinality_instances,
        "optimum_objective_values_per_variant": optima,
        "terminal_paths_per_selector": paths,
        "candidate_operations_per_selector": candidates,
        "eq7_candidate_scores": candidates,
        "eq7_terminal_paths": paths,
        "true_marginal_cache_lookups": candidates * variants,
        "true_marginal_terminal_paths": paths * variants,
        "optimum_objective_values": optima * variants,
        "classifications": classifications,
    }


def _validate_exact_domains() -> None:
    if (
        type(_VERTEX_DOMAIN) is not tuple
        or _VERTEX_DOMAIN != _APPROVED_VERTEX_DOMAIN
        or any(type(value) is not Fraction for value in _VERTEX_DOMAIN)
    ):
        raise ValueError("vertex weights do not match the approved domain")
    if (
        type(_EDGE_DOMAIN) is not tuple
        or _EDGE_DOMAIN != _APPROVED_EDGE_DOMAIN
        or any(type(value) is not Fraction for value in _EDGE_DOMAIN)
    ):
        raise ValueError("edge weights do not match the approved domain")


def _preflight_accounting() -> tuple[dict[str, int], dict[str, int]]:
    _validate_exact_domains()
    premises = premise_domain_formulas()
    greedy = greedy_domain_formulas()
    exact = (
        (
            "premise subset",
            premises["subset_objective_values"],
            PREMISE_SUBSET_CEILING,
        ),
        (
            "premise marginal",
            premises["marginal_values"],
            PREMISE_MARGINAL_CEILING,
        ),
        (
            "premise diminishing",
            premises["diminishing_return_comparisons"],
            PREMISE_DIMINISHING_CEILING,
        ),
        (
            "Eq. (7) score",
            greedy["eq7_candidate_scores"],
            EQ7_SCORE_CEILING,
        ),
        (
            "Eq. (7) path",
            greedy["eq7_terminal_paths"],
            EQ7_PATH_CEILING,
        ),
        (
            "true-marginal lookup",
            greedy["true_marginal_cache_lookups"],
            TRUE_MARGINAL_LOOKUP_CEILING,
        ),
        (
            "true-marginal path",
            greedy["true_marginal_terminal_paths"],
            TRUE_MARGINAL_PATH_CEILING,
        ),
        (
            "optimum value",
            greedy["optimum_objective_values"],
            OPTIMUM_VALUE_CEILING,
        ),
        (
            "classification",
            greedy["classifications"],
            CLASSIFICATION_CEILING,
        ),
    )
    for name, required, ceiling in exact:
        if required != ceiling:
            raise ValueError(
                f"{name} ceiling does not match the approved formula"
            )
    return premises, greedy


def _subsets(vertices: Sequence[Vertex]) -> tuple[frozenset[Vertex], ...]:
    return tuple(
        frozenset(items)
        for size in range(len(vertices) + 1)
        for items in combinations(vertices, size)
    )


def _subset_suffix(selected: frozenset[Vertex]) -> str:
    return ",".join(sorted(selected)) if selected else "-"


def _parameterize(
    graph: Instance,
    model_variant: ModelVariant,
) -> Instance:
    alpha, eta = canonical_variant_parameters(graph, model_variant)
    return Instance(
        graph.vertices,
        graph.vertex_weights,
        graph.interactions,
        alpha=alpha,
        eta=eta,
    )


def _premise_tables(
    instance: Instance,
    model_variant: ModelVariant,
    subsets: Sequence[frozenset[Vertex]],
) -> tuple[
    dict[frozenset[Vertex], Fraction],
    dict[tuple[frozenset[Vertex], Vertex], Fraction],
]:
    objective_values = {
        selected: evaluate_objective(instance, selected, model_variant)
        for selected in subsets
    }
    marginals: dict[tuple[frozenset[Vertex], Vertex], Fraction] = {}
    for selected in subsets:
        for candidate in instance.vertices:
            if candidate in selected:
                continue
            marginals[(selected, candidate)] = (
                objective_values[selected.union({candidate})]
                - objective_values[selected]
            )
    return objective_values, marginals


def _build_optimum_tables(
    instance: Instance,
    model_variant: ModelVariant,
) -> tuple[dict[int, dict[frozenset[Vertex], Fraction]], int]:
    """Evaluate each approved optimum candidate once for one variant."""

    tables: dict[int, dict[frozenset[Vertex], Fraction]] = {}
    evaluations = 0
    for budget in range(1, min(3, len(instance.vertices)) + 1):
        candidates = tuple(
            frozenset(items)
            for items in combinations(instance.vertices, budget)
        )
        table = {
            selected: evaluate_objective(
                instance,
                selected,
                model_variant,
            )
            for selected in candidates
        }
        tables[budget] = table
        evaluations += len(table)
    return tables, evaluations


def _premise_record(
    instance: Instance,
    model_variant: ModelVariant,
    objective_values: Mapping[frozenset[Vertex], Fraction],
    marginals: Mapping[tuple[frozenset[Vertex], Vertex], Fraction],
) -> tuple[dict[str, object], int]:
    base_id = canonical_parameterized_instance_id(instance, model_variant)
    nonnegative_witness = next(
        (
            f"{base_id}::premise=global_nonnegativity"
            f"::subset={_subset_suffix(selected)}"
            for selected, value in objective_values.items()
            if value < 0
        ),
        None,
    )
    empty = frozenset()
    normalization_witness = (
        f"{base_id}::premise=normalization::subset=-"
        if objective_values[empty] != 0
        else None
    )
    monotonicity_witness = next(
        (
            f"{base_id}::premise=global_monotonicity"
            f"::subset={_subset_suffix(selected)}::candidate={candidate}"
            for (selected, candidate), value in marginals.items()
            if value < 0
        ),
        None,
    )

    diminishing_witness: str | None = None
    diminishing_comparisons = 0
    for candidate in instance.vertices:
        others = tuple(
            vertex
            for vertex in instance.vertices
            if vertex != candidate
        )
        for states in product(
            range(_DIMINISHING_RELATION_STATES),
            repeat=len(others),
        ):
            a = frozenset(
                vertex
                for vertex, state in zip(others, states, strict=True)
                if state == 1
            )
            b = frozenset(
                vertex
                for vertex, state in zip(others, states, strict=True)
                if state in (1, 2)
            )
            diminishing_comparisons += 1
            if (
                diminishing_witness is None
                and marginals[(a, candidate)] < marginals[(b, candidate)]
            ):
                diminishing_witness = (
                    f"{base_id}::premise=global_submodularity"
                    f"::a={_subset_suffix(a)}::b={_subset_suffix(b)}"
                    f"::candidate={candidate}"
                )

    witness_by_premise = {
        "global_nonnegativity": nonnegative_witness,
        "normalization": normalization_witness,
        "global_monotonicity": monotonicity_witness,
        "global_submodularity": diminishing_witness,
    }
    booleans = {
        name: witness_by_premise[name] is None for name in _PREMISE_NAMES
    }
    failed_witness_ids = [
        witness
        for witness in witness_by_premise.values()
        if witness is not None
    ]
    return (
        {
            "id": f"{base_id}::premises=global",
            "base_instance_id": base_id,
            "model_variant": model_variant,
            **booleans,
            "theorem_eligible": all(booleans.values()),
            "failed_witness_ids": failed_witness_ids,
        },
        diminishing_comparisons,
    )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _summary_paths(
    paths: Sequence[tuple[Vertex, ...]],
    objective_values: Mapping[frozenset[Vertex], Fraction],
) -> dict[str, tuple[tuple[Vertex, ...], Fraction]]:
    valued = tuple(
        (path, objective_values[frozenset(path)]) for path in paths
    )
    best_value = max(value for _, value in valued)
    worst_value = min(value for _, value in valued)
    best_path = min(path for path, value in valued if value == best_value)
    worst_path = min(path for path, value in valued if value == worst_value)
    canonical_path = min(path for path, _ in valued)
    return {
        "best": (best_path, best_value),
        "worst": (worst_path, worst_value),
        "canonical": (
            canonical_path,
            objective_values[frozenset(canonical_path)],
        ),
    }


def _failed_premises(premise: Mapping[str, object]) -> list[str]:
    return [name for name in _PREMISE_NAMES if premise[name] is not True]


def _stream_guarantee_record(
    record: dict[str, object],
    violations: list[dict[str, object]],
    diagnostics: list[dict[str, object]],
) -> bool:
    premise = record["premise_evaluation"]
    classification = record["ratio_classification"]
    assert isinstance(premise, Mapping)
    assert isinstance(classification, Mapping)
    failed = _failed_premises(premise)
    ratio = _ratio_fraction(classification)
    comparison = (
        compare_ratio_to_one_minus_inverse_e(ratio)
        if ratio is not None
        else None
    )
    if not failed:
        if comparison is not None and comparison["status"] == "below_bound":
            violations.append(
                {
                    **record,
                    "bound_comparison": comparison,
                }
            )
        return True

    is_out_of_premise_failure = (
        classification["status"]
        in {
            "negative_objective_regime",
            "undefined_zero_optimum",
        }
        or (
            comparison is not None
            and comparison["status"] == "below_bound"
        )
    )
    if is_out_of_premise_failure:
        diagnostics.append(
            {
                "id": record["id"],
                "failed_premise_ids": failed,
                "failed_witness_ids": list(
                    premise["failed_witness_ids"]
                ),
                "ratio_classification": dict(classification),
            }
        )
    return False


def _classification_record(
    *,
    base_id: str,
    budget: int,
    selector: str,
    result_kind: str,
    path: tuple[Vertex, ...],
    greedy_value: Fraction,
    optimum_value: Fraction,
    premise: Mapping[str, object],
) -> dict[str, object]:
    path_text = ",".join(path)
    return {
        "id": (
            f"{base_id}::budget={budget}::selector={selector}"
            f"::result={result_kind}::path={path_text}"
        ),
        "base_instance_id": base_id,
        "budget": budget,
        "selector": selector,
        "result_kind": result_kind,
        "path": list(path),
        "greedy_value": _fraction_text(greedy_value),
        "optimum_value": _fraction_text(optimum_value),
        "ratio_classification": classify_ratio(
            greedy_value,
            optimum_value,
        ),
        "premise_evaluation": dict(premise),
    }


def _symmetric_interactions(
    edges: Sequence[tuple[Vertex, Vertex]],
    values: Sequence[Fraction],
) -> dict[tuple[Vertex, Vertex], Fraction]:
    interactions: dict[tuple[Vertex, Vertex], Fraction] = {}
    for (left, right), value in zip(edges, values, strict=True):
        interactions[(left, right)] = value
        interactions[(right, left)] = value
    return interactions


def run_greedy_audit(source_revision: str) -> dict[str, object]:
    """Run the exact premise and weighted-cardinality greedy domains."""

    _validate_source_revision(source_revision)
    premise_formula, greedy_formula = _preflight_accounting()

    premise_digest = hashlib.sha256()
    classification_digest = hashlib.sha256()
    representative_records: dict[str, dict[str, object]] = {}
    canonical_parameter_examples: dict[str, object] = {}
    guarantee_violations: list[dict[str, object]] = []
    out_of_premise_diagnostics: list[dict[str, object]] = []

    graph_count = 0
    premise_record_count = 0
    subset_objective_values = 0
    marginal_values = 0
    diminishing_comparisons = 0
    weighted_cardinality_instances = 0
    eq7_score_operations = 0
    eq7_terminal_paths = 0
    true_marginal_lookups = 0
    true_marginal_terminal_paths = 0
    optimum_objective_values = 0
    classifications = 0
    eligible_classifications = 0

    for n in range(1, _MAX_VERTICES + 1):
        vertices = tuple(f"v{index}" for index in range(n))
        edges = tuple(combinations(vertices, 2))
        graph_subsets = _subsets(vertices)
        for vertex_values in product(_VERTEX_DOMAIN, repeat=n):
            for edge_values in product(_EDGE_DOMAIN, repeat=len(edges)):
                graph = Instance(
                    vertices,
                    dict(zip(vertices, vertex_values, strict=True)),
                    _symmetric_interactions(edges, edge_values),
                )
                graph_count += 1

                instances: dict[ModelVariant, Instance] = {}
                marginal_tables: dict[
                    ModelVariant,
                    dict[
                        tuple[frozenset[Vertex], Vertex],
                        Fraction,
                    ],
                ] = {}
                optimum_tables: dict[
                    ModelVariant,
                    dict[int, dict[frozenset[Vertex], Fraction]],
                ] = {}
                premise_records: dict[
                    ModelVariant,
                    dict[str, object],
                ] = {}

                for model_variant in SET_FUNCTION_VARIANTS:
                    instance = _parameterize(graph, model_variant)
                    instances[model_variant] = instance
                    objective_table, marginal_table = _premise_tables(
                        instance,
                        model_variant,
                        graph_subsets,
                    )
                    record, comparisons = _premise_record(
                        instance,
                        model_variant,
                        objective_table,
                        marginal_table,
                    )
                    marginal_tables[model_variant] = marginal_table
                    optimum_table, optimum_evaluations = (
                        _build_optimum_tables(instance, model_variant)
                    )
                    optimum_tables[model_variant] = optimum_table
                    optimum_objective_values += optimum_evaluations
                    premise_records[model_variant] = record
                    premise_record_count += 1
                    subset_objective_values += len(objective_table)
                    marginal_values += len(marginal_table)
                    diminishing_comparisons += comparisons
                    premise_digest.update(_canonical_json_bytes(record))
                    premise_digest.update(b"\n")
                    representative_records.setdefault(
                        model_variant,
                        record,
                    )

                if (
                    n == 2
                    and vertex_values == (Fraction(), Fraction())
                    and edge_values == (Fraction(-1),)
                ):
                    canonical_parameter_examples["n=2;M=1/1"] = {
                        variant: {
                            "alpha": _fraction_text(
                                instances[variant].alpha
                            ),
                            "eta": _fraction_text(instances[variant].eta),
                            "instance_id": (
                                canonical_parameterized_instance_id(
                                    instances[variant],
                                    variant,
                                )
                            ),
                        }
                        for variant in SET_FUNCTION_VARIANTS
                    }

                for budget in range(1, min(3, n) + 1):
                    weighted_cardinality_instances += 1
                    eq7_paths, eq7_operations = _enumerate_tied_paths(
                        vertices,
                        budget,
                        lambda selected, candidate: eq7_score(
                            graph,
                            selected,
                            candidate,
                        ),
                    )
                    eq7_score_operations += eq7_operations
                    eq7_terminal_paths += len(eq7_paths)

                    true_paths: dict[
                        ModelVariant,
                        tuple[tuple[Vertex, ...], ...],
                    ] = {}
                    for model_variant in SET_FUNCTION_VARIANTS:
                        marginal_table = marginal_tables[model_variant]
                        paths, lookups = _enumerate_tied_paths(
                            vertices,
                            budget,
                            lambda selected, candidate, table=marginal_table: (
                                table[(selected, candidate)]
                            ),
                        )
                        true_paths[model_variant] = paths
                        true_marginal_lookups += lookups
                        true_marginal_terminal_paths += len(paths)

                    if (
                        true_paths["paper_samplewise_literal"]
                        != true_paths["appendix_inline_shift_literal"]
                    ):
                        raise AssertionError(
                            "Appendix cardinality shift changed selection ties"
                        )

                    for model_variant in SET_FUNCTION_VARIANTS:
                        instance = instances[model_variant]
                        base_id = canonical_parameterized_instance_id(
                            instance,
                            model_variant,
                        )
                        objective_table = optimum_tables[model_variant][budget]
                        optimum_value = max(objective_table.values())

                        selector_paths = {
                            "paper_eq7_score_greedy": eq7_paths,
                            "true_marginal_greedy": true_paths[model_variant],
                        }
                        for selector, paths in selector_paths.items():
                            summarized = _summary_paths(
                                paths,
                                objective_table,
                            )
                            for result_kind in _RESULT_KINDS:
                                path, greedy_value = summarized[result_kind]
                                record = _classification_record(
                                    base_id=base_id,
                                    budget=budget,
                                    selector=selector,
                                    result_kind=result_kind,
                                    path=path,
                                    greedy_value=greedy_value,
                                    optimum_value=optimum_value,
                                    premise=premise_records[model_variant],
                                )
                                classifications += 1
                                classification_digest.update(
                                    _canonical_json_bytes(record)
                                )
                                classification_digest.update(b"\n")
                                if _stream_guarantee_record(
                                    record,
                                    guarantee_violations,
                                    out_of_premise_diagnostics,
                                ):
                                    eligible_classifications += 1

    premise_work = (
        subset_objective_values
        + marginal_values
        + diminishing_comparisons
    )
    premise_actual = {
        "subset_objective_values": subset_objective_values,
        "marginal_values": marginal_values,
        "diminishing_return_comparisons": diminishing_comparisons,
        "premise_work": premise_work,
    }
    premise_declared = {
        "subset_objective_values": PREMISE_SUBSET_CEILING,
        "marginal_values": PREMISE_MARGINAL_CEILING,
        "diminishing_return_comparisons": (
            PREMISE_DIMINISHING_CEILING
        ),
        "premise_work": (
            PREMISE_SUBSET_CEILING
            + PREMISE_MARGINAL_CEILING
            + PREMISE_DIMINISHING_CEILING
        ),
    }
    greedy_actual = {
        "eq7_candidate_scores": eq7_score_operations,
        "eq7_terminal_paths": eq7_terminal_paths,
        "true_marginal_cache_lookups": true_marginal_lookups,
        "true_marginal_terminal_paths": true_marginal_terminal_paths,
        "optimum_objective_values": optimum_objective_values,
        "classifications": classifications,
    }
    greedy_declared = {
        "eq7_candidate_scores": EQ7_SCORE_CEILING,
        "eq7_terminal_paths": EQ7_PATH_CEILING,
        "true_marginal_cache_lookups": (
            TRUE_MARGINAL_LOOKUP_CEILING
        ),
        "true_marginal_terminal_paths": TRUE_MARGINAL_PATH_CEILING,
        "optimum_objective_values": OPTIMUM_VALUE_CEILING,
        "classifications": CLASSIFICATION_CEILING,
    }

    if graph_count != premise_formula["weighted_graphs"]:
        raise AssertionError("weighted graph accounting drift")
    if premise_record_count != graph_count * len(SET_FUNCTION_VARIANTS):
        raise AssertionError("premise record accounting drift")
    if premise_actual != premise_declared:
        raise AssertionError("premise work accounting drift")
    if (
        weighted_cardinality_instances
        != greedy_formula["weighted_cardinality_instances"]
    ):
        raise AssertionError("weighted-cardinality accounting drift")
    if any(
        greedy_actual[name] > ceiling
        for name, ceiling in greedy_declared.items()
    ):
        raise AssertionError("greedy actual exceeds declared ceiling")
    if optimum_objective_values != OPTIMUM_VALUE_CEILING:
        raise AssertionError("optimum objective accounting drift")
    if classifications != CLASSIFICATION_CEILING:
        raise AssertionError("classification accounting drift")

    guarantee_status = (
        "not_evaluated"
        if eligible_classifications == 0
        else (
            "contradicted"
            if guarantee_violations
            else "supported"
        )
    )
    return {
        "source_revision": source_revision,
        "model_variants": list(SET_FUNCTION_VARIANTS),
        "score_only_premise": {
            "model_variant": "appendix_eq26_score",
            "status": "not_applicable",
            "reason": "score_is_not_a_set_function",
            "finite_work_units": 0,
        },
        "canonical_parameter_examples": canonical_parameter_examples,
        "premise_certification": {
            "weighted_graphs": graph_count,
            "premise_records": premise_record_count,
            "per_variant_formula": {
                key: premise_formula[key]
                for key in (
                    "subset_objective_values_per_variant",
                    "marginal_values_per_variant",
                    "diminishing_return_comparisons_per_variant",
                    "premise_work_per_variant",
                )
            },
            "actual": premise_actual,
            "declared_ceiling": premise_declared,
            "record_digest_sha256": premise_digest.hexdigest(),
            "representative_records": [
                representative_records[variant]
                for variant in SET_FUNCTION_VARIANTS
            ],
        },
        "greedy_search": {
            "weighted_cardinality_instances": (
                weighted_cardinality_instances
            ),
            "actual": greedy_actual,
            "declared_ceiling": greedy_declared,
            "objective_evaluations_by_phase": {
                "premise": subset_objective_values,
                "optimum": optimum_objective_values,
            },
            "classification_digest_sha256": (
                classification_digest.hexdigest()
            ),
        },
        "appendix_selection_relation": {
            "selection_ties_match_at_fixed_iteration": True,
            "objective_ratios_recomputed": True,
            "ratio_transfer": False,
        },
        "guarantee_summary": {
            "status": guarantee_status,
            "eligible_records": eligible_classifications,
            "guarantee_violations": guarantee_violations,
            "out_of_premise_diagnostics": (
                out_of_premise_diagnostics
            ),
            "out_of_premise_storage": (
                "negative, undefined, or below-bound ratios only"
            ),
        },
    }
