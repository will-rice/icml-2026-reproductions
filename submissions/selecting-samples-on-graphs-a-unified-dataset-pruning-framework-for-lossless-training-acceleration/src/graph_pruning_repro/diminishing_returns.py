"""Exact diminishing-returns oracles and canonical graph parameters."""

from __future__ import annotations

import hashlib
from fractions import Fraction
from itertools import combinations, product

from .objectives import evaluate_objective
from .types import (
    SET_FUNCTION_VARIANTS as _SET_FUNCTION_VARIANTS,
    Instance,
    ModelVariant,
    Vertex,
)

SET_FUNCTION_VARIANTS = _SET_FUNCTION_VARIANTS
SYMMETRIC_CASE_CEILING = 79_480
ASYMMETRIC_CASE_CEILING = 19_738
SYMMETRIC_PRIMITIVE_CEILING = 2_861_280
ASYMMETRIC_PRIMITIVE_CEILING = 118_428

_EDGE_DOMAIN = (Fraction(-1), Fraction(), Fraction(1))


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _validate_source_revision(source_revision: str) -> None:
    if type(source_revision) is not str:
        raise TypeError("source_revision must be a string")
    if not source_revision.strip():
        raise ValueError("source_revision must be nonempty")


def _validate_set_function_variant(model_variant: ModelVariant) -> None:
    if (
        type(model_variant) is not str
        or model_variant not in SET_FUNCTION_VARIANTS
    ):
        if model_variant == "appendix_eq26_score":
            raise ValueError("appendix_eq26_score is not a set function")
        raise ValueError("unknown set-function variant")


def _canonical_pairs(
    instance: Instance,
) -> tuple[tuple[Vertex, Vertex], ...]:
    return tuple(combinations(instance.vertices, 2))


def _canonical_edge_values(instance: Instance) -> tuple[Fraction, ...]:
    return tuple(
        instance.interactions.get((left, right), Fraction())
        for left, right in _canonical_pairs(instance)
    )


def _require_symmetric_interactions(instance: Instance) -> None:
    for left, right in _canonical_pairs(instance):
        forward = instance.interactions.get((left, right), Fraction())
        reverse = instance.interactions.get((right, left), Fraction())
        if forward != reverse:
            raise ValueError(
                "canonical instance IDs require symmetric interactions"
            )


def canonical_variant_parameters(
    instance: Instance,
    model_variant: ModelVariant,
) -> tuple[Fraction, Fraction]:
    """Derive the sole pre-outcome alpha/eta tuple for a graph variant."""

    if not isinstance(instance, Instance):
        raise TypeError("instance must be an Instance")
    _validate_set_function_variant(model_variant)
    edge_values = _canonical_edge_values(instance)
    maximum = max((abs(value) for value in edge_values), default=Fraction())
    alpha = Fraction(1)
    if model_variant == "appendix_inline_shift_literal":
        return alpha, maximum
    if model_variant == "modular_shift_candidate":
        return alpha, 2 * (len(instance.vertices) - 1) * maximum
    return alpha, Fraction()


def _canonical_graph_id(instance: Instance) -> str:
    vertex_text = ",".join(
        _fraction_text(instance.vertex_weights[vertex])
        for vertex in instance.vertices
    )
    edge_values = _canonical_edge_values(instance)
    edge_text = (
        ",".join(_fraction_text(value) for value in edge_values)
        if edge_values
        else "-"
    )
    return f"n={len(instance.vertices)};vw={vertex_text};ew={edge_text}"


def canonical_parameterized_instance_id(
    instance: Instance,
    model_variant: ModelVariant,
) -> str:
    """Build the only canonical symmetric graph/variant parameter ID."""

    if not isinstance(instance, Instance):
        raise TypeError("instance must be an Instance")
    _validate_set_function_variant(model_variant)
    _require_symmetric_interactions(instance)
    alpha, eta = canonical_variant_parameters(instance, model_variant)
    return (
        f"graph={_canonical_graph_id(instance)}"
        f"::variant={model_variant}"
        f"::alpha={_fraction_text(alpha)}"
        f"::eta={_fraction_text(eta)}"
    )


def _validate_marginal_inputs(
    instance: Instance,
    selected: frozenset[Vertex],
    candidate: Vertex,
    model_variant: ModelVariant,
) -> None:
    if not isinstance(instance, Instance):
        raise TypeError("instance must be an Instance")
    _validate_set_function_variant(model_variant)
    if type(selected) is not frozenset:
        raise TypeError("selected must be a frozenset")
    if any(type(vertex) is not str for vertex in selected):
        raise TypeError("selected vertices must be strings")
    if not selected.issubset(instance.vertices):
        raise ValueError("selected contains vertices outside the instance")
    if type(candidate) is not str:
        raise TypeError("candidate must be a string")
    if candidate not in instance.vertices:
        raise ValueError("candidate is outside the instance")
    if candidate in selected:
        raise ValueError("candidate must not be selected")


def direct_marginal(
    instance: Instance,
    selected: frozenset[Vertex],
    candidate: Vertex,
    model_variant: ModelVariant,
) -> Fraction:
    """Compute a marginal only by two independent set-function evaluations."""

    _validate_marginal_inputs(instance, selected, candidate, model_variant)
    with_candidate = frozenset((*selected, candidate))
    return evaluate_objective(
        instance,
        with_candidate,
        model_variant,
    ) - evaluate_objective(instance, selected, model_variant)


def closed_form_marginal(
    instance: Instance,
    selected: frozenset[Vertex],
    candidate: Vertex,
    model_variant: ModelVariant,
) -> Fraction:
    """Compute one of six direct formulas without calling another oracle."""

    _validate_marginal_inputs(instance, selected, candidate, model_variant)
    directed_incident = sum(
        (
            instance.interactions.get((candidate, other), Fraction())
            + instance.interactions.get((other, candidate), Fraction())
            for other in selected
        ),
        start=Fraction(),
    )
    unordered_incident = sum(
        (
            instance.interactions.get(
                tuple(sorted((candidate, other))),
                Fraction(),
            )
            for other in selected
        ),
        start=Fraction(),
    )
    vertex = instance.vertex_weights[candidate]
    if model_variant in {"paper_mwcp", "single_counted_pairwise"}:
        return vertex + unordered_incident
    if model_variant == "paper_samplewise_literal":
        return vertex + directed_incident
    if model_variant == "half_corrected_samplewise":
        return vertex + directed_incident / 2
    if model_variant == "appendix_inline_shift_literal":
        return (
            vertex
            + directed_incident
            + instance.alpha * instance.eta * (2 * len(selected) + 1)
        )
    if model_variant == "modular_shift_candidate":
        return vertex + directed_incident + instance.eta
    raise AssertionError("validated set-function variant was not dispatched")


def _diminishing_case_id(
    base_id: str,
    a_indices: tuple[int, ...],
    b_indices: tuple[int, ...],
    candidate_index: int,
) -> str:
    a_text = ",".join(f"v{index}" for index in a_indices) or "-"
    b_text = ",".join(f"v{index}" for index in b_indices) or "-"
    return (
        f"{base_id}::a={a_text}::b={b_text}"
        f"::candidate=v{candidate_index}"
    )


def _enumerate_records(
    instance: Instance,
    model_variant: ModelVariant,
    base_id: str,
) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    identifiers: set[str] = set()
    vertices = instance.vertices
    for candidate_index, candidate in enumerate(vertices):
        other_indices = tuple(
            index for index in range(len(vertices)) if index != candidate_index
        )
        for states in product((0, 1, 2), repeat=len(other_indices)):
            b_indices = tuple(
                index
                for index, state in zip(other_indices, states, strict=True)
                if state != 0
            )
            a_indices = tuple(
                index
                for index, state in zip(other_indices, states, strict=True)
                if state == 2
            )
            selected_a = frozenset(vertices[index] for index in a_indices)
            selected_b = frozenset(vertices[index] for index in b_indices)
            direct_a = direct_marginal(
                instance,
                selected_a,
                candidate,
                model_variant,
            )
            direct_b = direct_marginal(
                instance,
                selected_b,
                candidate,
                model_variant,
            )
            closed_a = closed_form_marginal(
                instance,
                selected_a,
                candidate,
                model_variant,
            )
            closed_b = closed_form_marginal(
                instance,
                selected_b,
                candidate,
                model_variant,
            )
            if direct_a != closed_a or direct_b != closed_b:
                raise AssertionError(
                    "direct and closed-form marginals disagree"
                )
            case_id = _diminishing_case_id(
                base_id,
                a_indices,
                b_indices,
                candidate_index,
            )
            if case_id in identifiers:
                raise ValueError("duplicate diminishing-returns ID")
            identifiers.add(case_id)
            records.append(
                {
                    "id": case_id,
                    "a": [f"v{index}" for index in a_indices],
                    "b": [f"v{index}" for index in b_indices],
                    "candidate": f"v{candidate_index}",
                    "direct_a": _fraction_text(direct_a),
                    "direct_b": _fraction_text(direct_b),
                    "closed_a": _fraction_text(closed_a),
                    "closed_b": _fraction_text(closed_b),
                    "diminishing_returns_holds": direct_a >= direct_b,
                }
            )
    return tuple(records)


def enumerate_diminishing_returns(
    instance: Instance,
    model_variant: ModelVariant,
) -> tuple[dict[str, object], ...]:
    """Enumerate every A subseteq B and candidate outside B for one graph."""

    if not isinstance(instance, Instance):
        raise TypeError("instance must be an Instance")
    _validate_set_function_variant(model_variant)
    expected_parameters = canonical_variant_parameters(instance, model_variant)
    if (instance.alpha, instance.eta) != expected_parameters:
        raise ValueError("stored canonical parameters do not match graph")
    base_id = canonical_parameterized_instance_id(instance, model_variant)
    return _enumerate_records(instance, model_variant, base_id)


def _symmetric_interactions(
    edges: tuple[tuple[Vertex, Vertex], ...],
    values: tuple[Fraction, ...],
) -> dict[tuple[Vertex, Vertex], Fraction]:
    interactions: dict[tuple[Vertex, Vertex], Fraction] = {}
    for (left, right), value in zip(edges, values, strict=True):
        interactions[(left, right)] = value
        interactions[(right, left)] = value
    return interactions


def _parameterize(
    graph: Instance,
    model_variant: ModelVariant,
) -> Instance:
    alpha, eta = canonical_variant_parameters(graph, model_variant)
    return Instance(
        vertices=graph.vertices,
        vertex_weights=graph.vertex_weights,
        interactions=graph.interactions,
        alpha=alpha,
        eta=eta,
    )


def _directed_parameterized_id(instance: Instance) -> str:
    directed_pairs = tuple(
        (left, right)
        for left in instance.vertices
        for right in instance.vertices
        if left != right
    )
    directed_text = (
        ",".join(
            _fraction_text(
                instance.interactions.get((left, right), Fraction())
            )
            for left, right in directed_pairs
        )
        if directed_pairs
        else "-"
    )
    vertex_text = ",".join(
        _fraction_text(instance.vertex_weights[vertex])
        for vertex in instance.vertices
    )
    return (
        f"directed-graph=n={len(instance.vertices)}"
        f";vw={vertex_text};dw={directed_text}"
        "::variant=paper_samplewise_literal::alpha=1/1::eta=0/1"
    )


def _case_ceiling(max_vertices: int, directed: bool) -> int:
    return sum(
        n
        * 3 ** (n - 1)
        * 3 ** (n * (n - 1) if directed else n * (n - 1) // 2)
        for n in range(1, max_vertices + 1)
    )


def appendix_shift_witness() -> dict[str, object]:
    """Return the fixed symbolic two-vertex Appendix E falsification."""

    instance = Instance(
        vertices=("v0", "v1"),
        vertex_weights={"v0": Fraction(), "v1": Fraction()},
        interactions={
            ("v0", "v1"): Fraction(),
            ("v1", "v0"): Fraction(),
        },
        alpha=Fraction(1),
        eta=Fraction(1),
    )
    marginal_empty = direct_marginal(
        instance,
        frozenset(),
        "v0",
        "appendix_inline_shift_literal",
    )
    marginal_v1 = direct_marginal(
        instance,
        frozenset({"v1"}),
        "v0",
        "appendix_inline_shift_literal",
    )
    return {
        "id": (
            "graph=n=2;vw=0/1,0/1;ew=0/1"
            "::variant=appendix_inline_shift_literal"
            "::alpha=1/1::eta=1/1::diagnostic=appendix-minimal"
        ),
        "property": "appendix_inline_shift_diminishing_returns",
        "model_variant": "appendix_inline_shift_literal",
        "evidence_kind": "symbolic",
        "inputs": {
            "vertices": ["v0", "v1"],
            "a": [],
            "b": ["v1"],
            "candidate": "v0",
            "vertex_weights": ["0/1", "0/1"],
            "edge_weights": ["0/1"],
            "alpha": "1/1",
            "eta": "1/1",
        },
        "intermediate_values": {
            "marginal_empty": _fraction_text(marginal_empty),
            "marginal_y": _fraction_text(marginal_v1),
            "difference": _fraction_text(marginal_empty - marginal_v1),
        },
        "minimality_checks": {
            "one_vertex_strict_chain_exists": False,
            "two_vertices_required": True,
        },
    }


def run_diminishing_returns_audit(
    source_revision: str,
    *,
    symmetric_max_vertices: int = 4,
    asymmetric_max_vertices: int = 3,
) -> dict[str, object]:
    """Run the approved symmetric control and asymmetric diagnostic."""

    _validate_source_revision(source_revision)
    if symmetric_max_vertices != 4 or asymmetric_max_vertices != 3:
        raise ValueError("not the approved diminishing-returns domain")
    symmetric_ceiling = _case_ceiling(symmetric_max_vertices, directed=False)
    asymmetric_ceiling = _case_ceiling(
        asymmetric_max_vertices,
        directed=True,
    )
    if (
        symmetric_ceiling > SYMMETRIC_CASE_CEILING
        or asymmetric_ceiling > ASYMMETRIC_CASE_CEILING
    ):
        raise ValueError("diminishing-returns domain exceeds ceiling")
    if (
        symmetric_ceiling != SYMMETRIC_CASE_CEILING
        or asymmetric_ceiling != ASYMMETRIC_CASE_CEILING
    ):
        raise AssertionError("diminishing-returns ceiling formula drift")

    symmetric_cases = 0
    symmetric_variant_cases = 0
    symmetric_digest = hashlib.sha256()
    symmetric_violations: dict[str, dict[str, object]] = {}
    symmetric_base_ids: set[str] = set()
    zero = Fraction()

    for n in range(1, symmetric_max_vertices + 1):
        vertices = tuple(f"v{index}" for index in range(n))
        edges = tuple(combinations(vertices, 2))
        for edge_values in product(_EDGE_DOMAIN, repeat=len(edges)):
            graph = Instance(
                vertices=vertices,
                vertex_weights={vertex: zero for vertex in vertices},
                interactions=_symmetric_interactions(edges, edge_values),
            )
            graph_case_count: int | None = None
            for model_variant in SET_FUNCTION_VARIANTS:
                instance = _parameterize(graph, model_variant)
                base_id = canonical_parameterized_instance_id(
                    instance,
                    model_variant,
                )
                if base_id in symmetric_base_ids:
                    raise ValueError("duplicate canonical parameterized ID")
                symmetric_base_ids.add(base_id)
                records = enumerate_diminishing_returns(
                    instance,
                    model_variant,
                )
                if graph_case_count is None:
                    graph_case_count = len(records)
                    symmetric_cases += len(records)
                elif len(records) != graph_case_count:
                    raise AssertionError("variant triple domains disagree")
                symmetric_variant_cases += len(records)
                for record in records:
                    symmetric_digest.update(record["id"].encode("ascii"))
                    symmetric_digest.update(b"\n")
                    if (
                        not record["diminishing_returns_holds"]
                        and model_variant not in symmetric_violations
                    ):
                        symmetric_violations[model_variant] = record

    if symmetric_cases != SYMMETRIC_CASE_CEILING:
        raise AssertionError("symmetric case accounting drift")
    if symmetric_variant_cases != symmetric_cases * len(
        SET_FUNCTION_VARIANTS
    ):
        raise AssertionError("symmetric variant accounting drift")
    symmetric_subset_evaluations = symmetric_variant_cases * 4
    symmetric_closed_evaluations = symmetric_variant_cases * 2
    symmetric_primitives = (
        symmetric_subset_evaluations + symmetric_closed_evaluations
    )
    if symmetric_primitives != SYMMETRIC_PRIMITIVE_CEILING:
        raise AssertionError("symmetric primitive accounting drift")

    asymmetric_cases = 0
    asymmetric_digest = hashlib.sha256()
    asymmetric_violation: dict[str, object] | None = None
    asymmetric_base_ids: set[str] = set()
    for n in range(1, asymmetric_max_vertices + 1):
        vertices = tuple(f"v{index}" for index in range(n))
        directed_pairs = tuple(
            (left, right)
            for left in vertices
            for right in vertices
            if left != right
        )
        for directed_values in product(
            _EDGE_DOMAIN,
            repeat=len(directed_pairs),
        ):
            instance = Instance(
                vertices=vertices,
                vertex_weights={vertex: zero for vertex in vertices},
                interactions=dict(
                    zip(directed_pairs, directed_values, strict=True)
                ),
            )
            base_id = _directed_parameterized_id(instance)
            if base_id in asymmetric_base_ids:
                raise ValueError("duplicate asymmetric diagnostic ID")
            asymmetric_base_ids.add(base_id)
            records = _enumerate_records(
                instance,
                "paper_samplewise_literal",
                base_id,
            )
            asymmetric_cases += len(records)
            for record in records:
                asymmetric_digest.update(record["id"].encode("ascii"))
                asymmetric_digest.update(b"\n")
                if (
                    not record["diminishing_returns_holds"]
                    and asymmetric_violation is None
                ):
                    asymmetric_violation = record

    if asymmetric_cases != ASYMMETRIC_CASE_CEILING:
        raise AssertionError("asymmetric case accounting drift")
    asymmetric_subset_evaluations = asymmetric_cases * 4
    asymmetric_closed_evaluations = asymmetric_cases * 2
    asymmetric_primitives = (
        asymmetric_subset_evaluations + asymmetric_closed_evaluations
    )
    if asymmetric_primitives != ASYMMETRIC_PRIMITIVE_CEILING:
        raise AssertionError("asymmetric primitive accounting drift")

    return {
        "source_revision": source_revision,
        "symbolic": {
            "evidence_kind": "symbolic",
            "scope": "arbitrary symmetric edge weights",
            "differences": {
                "paper_mwcp": "-sum(B\\\\A edges)",
                "paper_samplewise_literal": "-2*sum(B\\\\A edges)",
                "single_counted_pairwise": "-sum(B\\\\A edges)",
                "half_corrected_samplewise": "-sum(B\\\\A edges)",
                "appendix_inline_shift_literal": (
                    "-2*sum(B\\\\A edges)-2*alpha*eta*(|B|-|A|)"
                ),
                "modular_shift_candidate": "-2*sum(B\\\\A edges)",
            },
        },
        "symmetric_search": {
            "evidence_kind": "exhaustive_finite",
            "domain": {
                "vertex_counts": [1, 2, 3, 4],
                "triples": "all A subseteq B and candidate outside B",
                "vertex_weights": "symbolically cancelled; stored as zero",
                "symmetric_edge_weights": ["-1/1", "0/1", "1/1"],
            },
            "model_variants": list(SET_FUNCTION_VARIANTS),
            "declared_case_ceiling": SYMMETRIC_CASE_CEILING,
            "cases_examined": symmetric_cases,
            "variant_cases_examined": symmetric_variant_cases,
            "subset_objective_evaluations": symmetric_subset_evaluations,
            "closed_form_marginal_evaluations": symmetric_closed_evaluations,
            "primitive_evaluations": symmetric_primitives,
            "declared_primitive_ceiling": SYMMETRIC_PRIMITIVE_CEILING,
            "case_digest": symmetric_digest.hexdigest(),
            "smallest_violations": symmetric_violations,
            "completed": True,
        },
        "asymmetric_diagnostic": {
            "evidence_kind": "exhaustive_finite",
            "scope": "outside the symmetric paper premise",
            "domain": {
                "vertex_counts": [1, 2, 3],
                "triples": "all A subseteq B and candidate outside B",
                "directed_edge_weights": ["-1/1", "0/1", "1/1"],
            },
            "model_variants": ["paper_samplewise_literal"],
            "declared_case_ceiling": ASYMMETRIC_CASE_CEILING,
            "cases_examined": asymmetric_cases,
            "subset_objective_evaluations": asymmetric_subset_evaluations,
            "closed_form_marginal_evaluations": asymmetric_closed_evaluations,
            "primitive_evaluations": asymmetric_primitives,
            "declared_primitive_ceiling": ASYMMETRIC_PRIMITIVE_CEILING,
            "case_digest": asymmetric_digest.hexdigest(),
            "smallest_violation": asymmetric_violation,
            "completed": True,
        },
        "appendix_minimal_witness": appendix_shift_witness(),
    }
