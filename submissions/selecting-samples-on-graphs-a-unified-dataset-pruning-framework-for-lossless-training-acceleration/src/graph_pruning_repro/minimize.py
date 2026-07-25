"""Deterministic property-preserving minimization for exact witnesses."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from itertools import permutations
from typing import Callable, Mapping

from .types import Vertex

WitnessPredicate = Callable[[dict[str, object]], bool]


def _validate_witness(witness: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(witness, Mapping):
        raise TypeError("witness must be a mapping")
    required = {
        "property",
        "vertices",
        "selected",
        "candidate",
        "vertex_weights",
        "interactions",
    }
    if not required.issubset(witness):
        raise ValueError("witness is missing required fields")

    vertices = witness["vertices"]
    if type(vertices) is not tuple:
        raise TypeError("witness vertices must be a tuple")
    if not vertices:
        raise ValueError("witness vertices must be nonempty")
    if any(type(vertex) is not str or not vertex for vertex in vertices):
        raise ValueError("witness vertices must be nonempty strings")
    if len(set(vertices)) != len(vertices):
        raise ValueError("witness vertices must be unique")

    selected = witness["selected"]
    if type(selected) is not frozenset:
        raise TypeError("witness selected must be a frozenset")
    if any(type(vertex) is not str for vertex in selected):
        raise TypeError("witness selected vertices must be strings")
    if not selected.issubset(vertices):
        raise ValueError("witness selected contains an unknown vertex")

    candidate = witness["candidate"]
    if type(candidate) is not str or candidate not in vertices:
        raise ValueError("witness candidate must be a known vertex")
    if candidate in selected:
        raise ValueError("witness candidate must not be selected")

    vertex_weights = witness["vertex_weights"]
    if not isinstance(vertex_weights, Mapping):
        raise TypeError("witness vertex_weights must be a mapping")
    vertex_weights = dict(vertex_weights)
    if set(vertex_weights) != set(vertices):
        raise ValueError("witness vertex_weights must exactly cover vertices")
    if any(type(value) is not Fraction for value in vertex_weights.values()):
        raise TypeError("witness vertex weights must be Fractions")

    interactions = witness["interactions"]
    if not isinstance(interactions, Mapping):
        raise TypeError("witness interactions must be a mapping")
    interactions = dict(interactions)
    for edge, value in interactions.items():
        if type(edge) is not tuple or len(edge) != 2:
            raise TypeError("witness interaction keys must be vertex pairs")
        left, right = edge
        if left not in vertices or right not in vertices:
            raise ValueError("witness interaction endpoint is unknown")
        if left == right:
            raise ValueError("witness self interactions are not allowed")
        if type(value) is not Fraction:
            raise TypeError("witness interaction weights must be Fractions")

    normalized = {
        key: value for key, value in witness.items() if key != "id"
    }
    normalized["vertices"] = tuple(vertices)
    normalized["selected"] = frozenset(selected)
    normalized["candidate"] = candidate
    normalized["vertex_weights"] = vertex_weights
    normalized["interactions"] = interactions
    return normalized


def _property_holds(
    predicate: WitnessPredicate,
    witness: dict[str, object],
) -> bool:
    result = predicate(_copy_witness(witness))
    if type(result) is not bool:
        raise TypeError("predicate must return a bool")
    return result


def _copy_witness(witness: Mapping[str, object]) -> dict[str, object]:
    copied = {key: value for key, value in witness.items() if key != "id"}
    copied["vertices"] = tuple(witness["vertices"])
    copied["selected"] = frozenset(witness["selected"])
    copied["vertex_weights"] = dict(witness["vertex_weights"])
    copied["interactions"] = dict(witness["interactions"])
    return copied


def _delete_vertex(
    witness: Mapping[str, object],
    vertex: Vertex,
) -> dict[str, object]:
    candidate = _copy_witness(witness)
    candidate["vertices"] = tuple(
        existing
        for existing in candidate["vertices"]
        if existing != vertex
    )
    candidate["selected"] = frozenset(
        existing
        for existing in candidate["selected"]
        if existing != vertex
    )
    candidate["vertex_weights"] = {
        existing: value
        for existing, value in candidate["vertex_weights"].items()
        if existing != vertex
    }
    candidate["interactions"] = {
        edge: value
        for edge, value in candidate["interactions"].items()
        if vertex not in edge
    }
    return candidate


def _reduce_vertices(
    predicate: WitnessPredicate,
    witness: dict[str, object],
) -> dict[str, object]:
    current = witness
    changed = True
    while changed:
        changed = False
        for vertex in sorted(current["vertices"]):
            if vertex == current["candidate"]:
                continue
            proposed = _delete_vertex(current, vertex)
            if _property_holds(predicate, proposed):
                current = proposed
                changed = True
                break
    return current


def _reduce_selected(
    predicate: WitnessPredicate,
    witness: dict[str, object],
) -> dict[str, object]:
    current = witness
    for vertex in sorted(tuple(current["selected"])):
        proposed = _copy_witness(current)
        proposed["selected"] = frozenset(
            existing
            for existing in proposed["selected"]
            if existing != vertex
        )
        if _property_holds(predicate, proposed):
            current = proposed
    return current


def _zero_weights(
    predicate: WitnessPredicate,
    witness: dict[str, object],
) -> dict[str, object]:
    current = witness
    for vertex in current["vertices"]:
        if current["vertex_weights"][vertex] == 0:
            continue
        proposed = _copy_witness(current)
        proposed["vertex_weights"][vertex] = Fraction()
        if _property_holds(predicate, proposed):
            current = proposed
    for edge in sorted(current["interactions"]):
        if current["interactions"][edge] == 0:
            continue
        proposed = _copy_witness(current)
        proposed["interactions"][edge] = Fraction()
        if _property_holds(predicate, proposed):
            current = proposed
    return current


def _smaller_nonzero_magnitudes(value: Fraction) -> tuple[Fraction, ...]:
    sign = -1 if value < 0 else 1
    absolute = abs(value)
    candidates = {
        Fraction(sign * numerator, absolute.denominator)
        for numerator in range(1, absolute.numerator)
    }
    if absolute > 1:
        candidates.add(Fraction(sign))
    return tuple(sorted(candidates, key=lambda candidate: abs(candidate)))


def _reduce_magnitudes(
    predicate: WitnessPredicate,
    witness: dict[str, object],
) -> dict[str, object]:
    current = witness
    for vertex in current["vertices"]:
        value = current["vertex_weights"][vertex]
        for smaller in _smaller_nonzero_magnitudes(value):
            proposed = _copy_witness(current)
            proposed["vertex_weights"][vertex] = smaller
            if _property_holds(predicate, proposed):
                current = proposed
                break
    for edge in sorted(current["interactions"]):
        value = current["interactions"][edge]
        for smaller in _smaller_nonzero_magnitudes(value):
            proposed = _copy_witness(current)
            proposed["interactions"][edge] = smaller
            if _property_holds(predicate, proposed):
                current = proposed
                break
    return current


def _rename_vertices(
    witness: Mapping[str, object],
    order: tuple[Vertex, ...],
) -> dict[str, object]:
    renamed = _copy_witness(witness)
    mapping = {
        old: f"v{index}" for index, old in enumerate(order)
    }
    renamed["vertices"] = tuple(f"v{index}" for index in range(len(order)))
    renamed["selected"] = frozenset(
        mapping[vertex] for vertex in renamed["selected"]
    )
    renamed["candidate"] = mapping[renamed["candidate"]]
    renamed["vertex_weights"] = {
        mapping[vertex]: value
        for vertex, value in renamed["vertex_weights"].items()
    }
    renamed["interactions"] = {
        (mapping[left], mapping[right]): value
        for (left, right), value in renamed["interactions"].items()
    }
    return renamed


def _json_value(value: object) -> object:
    if type(value) is Fraction:
        return f"{value.numerator}/{value.denominator}"
    if isinstance(value, Mapping):
        if not all(type(key) is str for key in value):
            raise TypeError("canonical metadata mappings require string keys")
        return {
            key: _json_value(item)
            for key, item in sorted(value.items())
        }
    if type(value) in {tuple, list}:
        return [_json_value(item) for item in value]
    if type(value) in {frozenset, set}:
        return [_json_value(item) for item in sorted(value)]
    if value is None or type(value) in {str, int, bool}:
        return value
    raise TypeError("witness contains noncanonical metadata")


def _canonical_payload(witness: Mapping[str, object]) -> dict[str, object]:
    payload = {
        key: value
        for key, value in witness.items()
        if key not in {"id", "vertex_weights", "interactions"}
    }
    payload["vertex_weights"] = [
        [vertex, _json_value(witness["vertex_weights"][vertex])]
        for vertex in witness["vertices"]
    ]
    payload["interactions"] = [
        [left, right, _json_value(value)]
        for (left, right), value in sorted(witness["interactions"].items())
    ]
    return _json_value(payload)


def _canonical_bytes(witness: Mapping[str, object]) -> bytes:
    return json.dumps(
        _canonical_payload(witness),
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _lexicographic_canonicalize(
    predicate: WitnessPredicate,
    witness: dict[str, object],
) -> dict[str, object]:
    vertices = tuple(sorted(witness["vertices"]))
    if len(vertices) > 8:
        raise ValueError("witness canonicalization supports at most 8 vertices")
    candidates = []
    for order in permutations(vertices):
        proposed = _rename_vertices(witness, order)
        if _property_holds(predicate, proposed):
            candidates.append(proposed)
    if not candidates:
        raise AssertionError("canonical relabeling lost witness property")
    return min(candidates, key=_canonical_bytes)


def minimize_witness(
    predicate: WitnessPredicate,
    witness: Mapping[str, object],
) -> dict[str, object]:
    """Minimize one exact witness in the approved deterministic order."""

    if not callable(predicate):
        raise TypeError("predicate must be callable")
    current = _validate_witness(witness)
    if not _property_holds(predicate, current):
        raise ValueError("original witness does not satisfy predicate")
    current = _reduce_vertices(predicate, current)
    current = _reduce_selected(predicate, current)
    current = _zero_weights(predicate, current)
    current = _reduce_magnitudes(predicate, current)
    current = _lexicographic_canonicalize(predicate, current)
    if not _property_holds(predicate, current):
        raise AssertionError("minimized witness lost predicate")
    result = _copy_witness(current)
    result["id"] = hashlib.sha256(_canonical_bytes(result)).hexdigest()[:16]
    return result
