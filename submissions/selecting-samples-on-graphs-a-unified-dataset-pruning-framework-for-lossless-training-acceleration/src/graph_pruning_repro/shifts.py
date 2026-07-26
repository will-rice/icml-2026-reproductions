"""Exact Appendix E and fixed-modular shift boundary diagnostics."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from itertools import combinations, product

from .diminishing_returns import (
    SET_FUNCTION_VARIANTS,
    canonical_parameterized_instance_id,
    canonical_variant_parameters,
    closed_form_marginal,
)
from .types import Instance, Vertex

SHIFT_CASE_CEILING = 6_459
SHIFT_VALUE_CEILING = 45_213
RATIONAL_ALPHA_CASE_CEILING = 256
RATIONAL_ALPHA_VALUE_CEILING = 1_792
_CHANNELS = (
    *SET_FUNCTION_VARIANTS[:-1],
    "appendix_eq26_score",
    SET_FUNCTION_VARIANTS[-1],
)


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _validate_source_revision(source_revision: str) -> None:
    if type(source_revision) is not str:
        raise TypeError("source_revision must be a string")
    if not source_revision.strip():
        raise ValueError("source_revision must be nonempty")


def _validate_inputs(
    instance: Instance,
    selected: frozenset[Vertex],
    candidate: Vertex,
) -> None:
    if not isinstance(instance, Instance):
        raise TypeError("instance must be an Instance")
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


def eq26_score_marginal(
    instance: Instance,
    selected: frozenset[Vertex],
    candidate: Vertex,
) -> Fraction:
    """Evaluate displayed Eq. (26) independently as a score channel."""

    _validate_inputs(instance, selected, candidate)
    incident = sum(
        (
            instance.interactions.get((candidate, other), Fraction())
            for other in selected
        ),
        start=Fraction(),
    )
    return (
        instance.alpha * instance.vertex_weights[candidate]
        + instance.alpha * len(selected) * instance.eta
        + incident
    )


def fixed_literal_shift_marginal(
    instance: Instance,
    selected: frozenset[Vertex],
    candidate: Vertex,
    coefficient: Fraction,
) -> Fraction:
    """Evaluate a separately labeled fixed shift on the literal base."""

    _validate_inputs(instance, selected, candidate)
    if type(coefficient) is not Fraction:
        raise TypeError("coefficient must be a Fraction")
    directed_incident = sum(
        (
            instance.interactions.get((candidate, other), Fraction())
            + instance.interactions.get((other, candidate), Fraction())
            for other in selected
        ),
        start=Fraction(),
    )
    return (
        instance.vertex_weights[candidate]
        + directed_incident
        + coefficient
    )


def shift_boundary_ceiling(max_vertices: int = 4) -> int:
    """Return the approved pre-deduplication Eq. (27) case ceiling."""

    if type(max_vertices) is not int or max_vertices < 1:
        raise ValueError("max_vertices must be a positive integer")
    return 3 * sum(
        2 ** (n * (n - 1) // 2) * n * 2 ** (n - 1)
        for n in range(1, max_vertices + 1)
    )


def shift_boundary_case_count(max_vertices: int = 4) -> int:
    """Return the exact post-deduplication count for the approved domain."""

    return shift_boundary_ceiling(max_vertices) - sum(
        n * 2 ** (n - 1) for n in range(1, max_vertices + 1)
    )


def _symmetric_interactions(
    edges: tuple[tuple[Vertex, Vertex], ...],
    values: tuple[Fraction, ...],
) -> dict[tuple[Vertex, Vertex], Fraction]:
    interactions: dict[tuple[Vertex, Vertex], Fraction] = {}
    for (left, right), value in zip(edges, values, strict=True):
        interactions[(left, right)] = value
        interactions[(right, left)] = value
    return interactions


def _selected_sets(
    vertices: tuple[Vertex, ...],
    candidate_index: int,
) -> tuple[frozenset[Vertex], ...]:
    others = tuple(
        vertex
        for index, vertex in enumerate(vertices)
        if index != candidate_index
    )
    return tuple(
        frozenset(
            vertex
            for vertex, included in zip(others, bits, strict=True)
            if included
        )
        for bits in product((False, True), repeat=len(others))
    )


def _graph_base_id(instance: Instance) -> str:
    return canonical_parameterized_instance_id(
        instance,
        "paper_mwcp",
    ).split("::variant=", maxsplit=1)[0]


def _boundary_values(maximum: Fraction) -> tuple[dict[str, object], ...]:
    grouped: dict[Fraction, list[str]] = {}
    raw = (
        ("below", max(Fraction(), maximum - 1)),
        ("at", maximum),
        ("above", maximum + 1),
    )
    for position, eta in raw:
        grouped.setdefault(eta, []).append(position)
    return tuple(
        {
            "positions": positions,
            "position": "+".join(positions),
            "eta": eta,
        }
        for eta, positions in grouped.items()
    )


def _variant_instance(
    graph: Instance,
    model_variant: str,
    *,
    alpha: Fraction | None = None,
    eta: Fraction | None = None,
) -> Instance:
    canonical_alpha, canonical_eta = canonical_variant_parameters(
        graph,
        model_variant,
    )
    return Instance(
        vertices=graph.vertices,
        vertex_weights=graph.vertex_weights,
        interactions=graph.interactions,
        alpha=canonical_alpha if alpha is None else alpha,
        eta=canonical_eta if eta is None else eta,
    )


def _channel_values(
    graph: Instance,
    selected: frozenset[Vertex],
    candidate: Vertex,
    *,
    alpha: Fraction,
    boundary_eta: Fraction,
) -> dict[str, str]:
    values: dict[str, Fraction] = {}
    for model_variant in SET_FUNCTION_VARIANTS:
        if model_variant == "appendix_inline_shift_literal":
            instance = _variant_instance(
                graph,
                model_variant,
                alpha=alpha,
                eta=boundary_eta,
            )
        else:
            instance = _variant_instance(graph, model_variant)
        values[model_variant] = closed_form_marginal(
            instance,
            selected,
            candidate,
            model_variant,
        )
    eq26_instance = Instance(
        vertices=graph.vertices,
        vertex_weights=graph.vertex_weights,
        interactions=graph.interactions,
        alpha=alpha,
        eta=boundary_eta,
    )
    values["appendix_eq26_score"] = eq26_score_marginal(
        eq26_instance,
        selected,
        candidate,
    )
    return {
        channel: _fraction_text(values[channel]) for channel in _CHANNELS
    }


def _digest_record(digest: object, record: dict[str, object]) -> None:
    encoded = json.dumps(
        record,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest.update(encoded)
    digest.update(b"\n")


def _representative_boundary() -> dict[str, object]:
    graph = Instance(
        vertices=("x", "y"),
        vertex_weights={"x": Fraction(), "y": Fraction()},
        interactions={
            ("x", "y"): Fraction(-1),
            ("y", "x"): Fraction(-1),
        },
    )
    selected = frozenset({"y"})
    values = [
        {
            "position": position,
            "eta": _fraction_text(eta),
            "channels": _channel_values(
                graph,
                selected,
                "x",
                alpha=Fraction(1),
                boundary_eta=eta,
            ),
        }
        for position, eta in (
            ("below", Fraction()),
            ("at", Fraction(1)),
            ("above", Fraction(2)),
        )
    ]
    graph_base = _graph_base_id(graph)
    fixed = [
        {
            "position": position,
            "coefficient": _fraction_text(coefficient),
            "id": (
                f"{graph_base}::diagnostic=fixed-literal-shift"
                f"::coefficient={_fraction_text(coefficient)}"
                "::selected=v1::candidate=v0"
            ),
            "marginal": _fraction_text(
                fixed_literal_shift_marginal(
                    graph,
                    selected,
                    "x",
                    coefficient,
                )
            ),
        }
        for position, coefficient in (
            ("below", Fraction(1)),
            ("at", Fraction(2)),
            ("above", Fraction(3)),
        )
    ]
    modular = _variant_instance(graph, "modular_shift_candidate")
    return {
        "graph_id": graph_base,
        "selected": ["v1"],
        "candidate": "v0",
        "threshold": "1/1",
        "values": values,
        "fixed_literal_shift_diagnostics": fixed,
        "canonical_modular": {
            "instance_id": canonical_parameterized_instance_id(
                modular,
                "modular_shift_candidate",
            ),
            "coefficient": _fraction_text(modular.eta),
            "marginal": _fraction_text(
                closed_form_marginal(
                    modular,
                    selected,
                    "x",
                    "modular_shift_candidate",
                )
            ),
        },
    }


def run_shift_audit(
    source_revision: str,
    *,
    max_vertices: int = 4,
    rational_controls: int = 256,
) -> dict[str, object]:
    """Run the approved exact shift boundary and rational-alpha controls."""

    _validate_source_revision(source_revision)
    if max_vertices != 4 or rational_controls != 256:
        raise ValueError("not the approved shift domain")
    declared_cases = shift_boundary_ceiling(max_vertices)
    exact_cases = shift_boundary_case_count(max_vertices)
    if declared_cases > SHIFT_CASE_CEILING:
        raise ValueError("shift boundary domain exceeds ceiling")
    if declared_cases != SHIFT_CASE_CEILING or exact_cases != 6_410:
        raise AssertionError("shift boundary ceiling formula drift")
    required_boundary_value_ceiling = declared_cases * len(_CHANNELS)
    required_boundary_actual_values = exact_cases * len(_CHANNELS)
    required_rational_cases = rational_controls
    required_rational_values = required_rational_cases * len(_CHANNELS)
    if required_boundary_value_ceiling > SHIFT_VALUE_CEILING:
        raise ValueError("shift value ceiling is below required work")
    if required_rational_cases > RATIONAL_ALPHA_CASE_CEILING:
        raise ValueError("rational-alpha case ceiling is below required work")
    if required_rational_values > RATIONAL_ALPHA_VALUE_CEILING:
        raise ValueError("rational-alpha value ceiling is below required work")

    boundary_cases = 0
    boundary_values = 0
    boundary_digest = hashlib.sha256()
    first_negative: dict[str, dict[str, object]] = {}
    edge_domain = (Fraction(-1), Fraction())
    for n in range(1, max_vertices + 1):
        vertices = tuple(f"v{index}" for index in range(n))
        edges = tuple(combinations(vertices, 2))
        for edge_values in product(edge_domain, repeat=len(edges)):
            graph = Instance(
                vertices=vertices,
                vertex_weights={
                    vertex: Fraction() for vertex in vertices
                },
                interactions=_symmetric_interactions(edges, edge_values),
            )
            maximum = max(
                (abs(value) for value in edge_values),
                default=Fraction(),
            )
            graph_base = _graph_base_id(graph)
            for candidate_index, candidate in enumerate(vertices):
                for selected in _selected_sets(vertices, candidate_index):
                    selected_indices = [
                        f"v{index}"
                        for index, vertex in enumerate(vertices)
                        if vertex in selected
                    ]
                    for boundary in _boundary_values(maximum):
                        eta = boundary["eta"]
                        channels = _channel_values(
                            graph,
                            selected,
                            candidate,
                            alpha=Fraction(1),
                            boundary_eta=eta,
                        )
                        record = {
                            "id": (
                                f"{graph_base}::diagnostic=eq27-boundary"
                                f"::alpha=1/1"
                                f"::threshold={_fraction_text(maximum)}"
                                f"::eta={_fraction_text(eta)}"
                                f"::position={boundary['position']}"
                                f"::selected={','.join(selected_indices) or '-'}"
                                f"::candidate=v{candidate_index}"
                            ),
                            "positions": boundary["positions"],
                            "channels": channels,
                        }
                        _digest_record(boundary_digest, record)
                        boundary_cases += 1
                        boundary_values += len(channels)
                        for channel, text in channels.items():
                            if (
                                Fraction(text) < 0
                                and channel not in first_negative
                            ):
                                first_negative[channel] = {
                                    "case_id": record["id"],
                                    "value": text,
                                }

    if boundary_cases != exact_cases:
        raise AssertionError("shift boundary case accounting drift")
    if boundary_values != required_boundary_actual_values:
        raise AssertionError("shift boundary value accounting drift")
    if boundary_values > SHIFT_VALUE_CEILING:
        raise AssertionError("shift boundary value ceiling exceeded")

    rational_case_count = 0
    rational_value_count = 0
    rational_digest = hashlib.sha256()
    multipliers = (
        ("below", Fraction(1, 2)),
        ("at", Fraction(1)),
        ("above", Fraction(3, 2)),
        ("far_above", Fraction(2)),
    )
    for alpha_index in range(16):
        alpha = Fraction(2 * alpha_index + 1, 16)
        threshold = Fraction(1, 1) / alpha
        for intrinsic in map(Fraction, range(1, 5)):
            graph = Instance(
                vertices=("v0", "v1"),
                vertex_weights={"v0": intrinsic, "v1": Fraction(1)},
                interactions={
                    ("v0", "v1"): Fraction(-1),
                    ("v1", "v0"): Fraction(-1),
                },
            )
            for position, multiplier in multipliers:
                eta = threshold * multiplier
                channels = _channel_values(
                    graph,
                    frozenset({"v1"}),
                    "v0",
                    alpha=alpha,
                    boundary_eta=eta,
                )
                record = {
                    "id": (
                        "diagnostic=rational-alpha"
                        f"::alpha={_fraction_text(alpha)}"
                        f"::intrinsic={_fraction_text(intrinsic)}"
                        f"::position={position}"
                        f"::eta={_fraction_text(eta)}"
                    ),
                    "channels": channels,
                }
                _digest_record(rational_digest, record)
                rational_case_count += 1
                rational_value_count += len(channels)

    if rational_case_count != required_rational_cases:
        raise AssertionError("rational-alpha case accounting drift")
    if rational_value_count != required_rational_values:
        raise AssertionError("rational-alpha value accounting drift")

    return {
        "source_revision": source_revision,
        "boundary_search": {
            "evidence_kind": "exhaustive_finite",
            "domain": {
                "vertex_counts": [1, 2, 3, 4],
                "vertex_weights": ["0/1"],
                "symmetric_edge_weights": ["-1/1", "0/1"],
                "alpha": ["1/1"],
                "eta_positions": ["below", "at", "above"],
            },
            "declared_case_ceiling": SHIFT_CASE_CEILING,
            "cases_examined": boundary_cases,
            "channels_per_case": len(_CHANNELS),
            "values_evaluated": boundary_values,
            "declared_value_ceiling": SHIFT_VALUE_CEILING,
            "deduplication": (
                "nonnegative below/at values coincide when M=0"
            ),
            "case_digest": boundary_digest.hexdigest(),
            "first_negative_by_channel": first_negative,
            "completed": True,
        },
        "rational_alpha_controls": {
            "evidence_kind": "non_exhaustive",
            "domain": {
                "alpha": "16 fixed positive odd-sixteenths",
                "positive_intrinsic_weights": ["1/1", "2/1", "3/1", "4/1"],
                "eta_positions": [
                    "below",
                    "at",
                    "above",
                    "far_above",
                ],
            },
            "declared_case_ceiling": RATIONAL_ALPHA_CASE_CEILING,
            "cases_examined": rational_case_count,
            "channels_per_case": len(_CHANNELS),
            "values_evaluated": rational_value_count,
            "declared_value_ceiling": RATIONAL_ALPHA_VALUE_CEILING,
            "case_digest": rational_digest.hexdigest(),
            "supports_universal_conclusion": False,
            "completed": True,
        },
        "representative_eq27_boundary": _representative_boundary(),
    }
