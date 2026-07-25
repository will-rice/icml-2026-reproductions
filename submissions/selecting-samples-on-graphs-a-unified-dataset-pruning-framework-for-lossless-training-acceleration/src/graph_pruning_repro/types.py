"""Exact immutable types shared by the graph-pruning evidence oracles."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Literal, Mapping, TypeAlias

Vertex = str
Edge = tuple[Vertex, Vertex]
ModelVariant = Literal[
    "paper_mwcp",
    "paper_samplewise_literal",
    "single_counted_pairwise",
    "half_corrected_samplewise",
    "appendix_inline_shift_literal",
    "appendix_eq26_score",
    "modular_shift_candidate",
]
Witness: TypeAlias = Mapping[str, object]

MODEL_VARIANTS: tuple[ModelVariant, ...] = (
    "paper_mwcp",
    "paper_samplewise_literal",
    "single_counted_pairwise",
    "half_corrected_samplewise",
    "appendix_inline_shift_literal",
    "appendix_eq26_score",
    "modular_shift_candidate",
)
SET_FUNCTION_VARIANTS: tuple[ModelVariant, ...] = (
    "paper_mwcp",
    "paper_samplewise_literal",
    "single_counted_pairwise",
    "half_corrected_samplewise",
    "appendix_inline_shift_literal",
    "modular_shift_candidate",
)


@dataclass(frozen=True)
class Instance:
    """Ordered finite graph instance with deeply frozen exact weights."""

    vertices: tuple[Vertex, ...]
    vertex_weights: Mapping[Vertex, Fraction]
    interactions: Mapping[Edge, Fraction]
    alpha: Fraction = Fraction(1)
    eta: Fraction = Fraction(0)

    def __post_init__(self) -> None:
        if type(self.vertices) is not tuple:
            raise TypeError("vertices must be a tuple")
        if not self.vertices:
            raise ValueError("vertices must be nonempty")
        if any(type(vertex) is not str or not vertex for vertex in self.vertices):
            raise ValueError("vertices must be nonempty strings")
        if len(set(self.vertices)) != len(self.vertices):
            raise ValueError("vertices must be unique")

        if not isinstance(self.vertex_weights, Mapping):
            raise TypeError("vertex_weights must be a mapping")
        vertex_weights = dict(self.vertex_weights)
        if set(vertex_weights) != set(self.vertices):
            raise ValueError("vertex_weights must exactly cover vertices")
        if any(type(weight) is not Fraction for weight in vertex_weights.values()):
            raise TypeError("vertex weights must be Fractions")

        if not isinstance(self.interactions, Mapping):
            raise TypeError("interactions must be a mapping")
        interactions = dict(self.interactions)
        for edge, weight in interactions.items():
            if type(edge) is not tuple or len(edge) != 2:
                raise TypeError("interaction keys must be vertex pairs")
            left, right = edge
            if left not in vertex_weights or right not in vertex_weights:
                raise ValueError("interaction endpoint is not a vertex")
            if left == right:
                raise ValueError("self interactions are not allowed")
            if type(weight) is not Fraction:
                raise TypeError("interaction weights must be Fractions")

        if type(self.alpha) is not Fraction:
            raise TypeError("alpha must be a Fraction")
        if type(self.eta) is not Fraction:
            raise TypeError("eta must be a Fraction")

        object.__setattr__(
            self,
            "vertex_weights",
            MappingProxyType(vertex_weights),
        )
        object.__setattr__(
            self,
            "interactions",
            MappingProxyType(interactions),
        )


def ordered_selection(
    instance: Instance,
    selected: frozenset[Vertex],
) -> tuple[Vertex, ...]:
    """Validate a selected set and return its deterministic lexical order."""

    if type(selected) is not frozenset:
        raise TypeError("selected must be a frozenset")
    if any(type(vertex) is not str for vertex in selected):
        raise TypeError("selected vertices must be strings")
    unknown = selected.difference(instance.vertices)
    if unknown:
        raise ValueError("selected contains vertices outside the instance")
    return tuple(sorted(selected))
