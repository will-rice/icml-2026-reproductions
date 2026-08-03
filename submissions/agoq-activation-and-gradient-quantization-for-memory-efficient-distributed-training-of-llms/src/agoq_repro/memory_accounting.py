"""Exact rational activation-memory arithmetic derived from the paper transcription."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction

COMPONENT_ORDER = (
    "qkv",
    "attention",
    "linear_1",
    "rmsnorm",
    "ffn_1",
    "activation",
    "ffn_2",
)


@dataclass(frozen=True)
class LayerMemoryAudit:
    method: str
    components_u: dict[str, Fraction]
    total_u: Fraction


@dataclass(frozen=True)
class ModelProjection:
    batch: int
    sequence: int
    hidden: int
    layers: int
    bytes_per_u: int
    totals_bytes: dict[str, Fraction]


def _parse_fraction(value: object) -> Fraction:
    if type(value) is not str or not re.fullmatch(r"-?\d+(?:/[1-9]\d*)?", value):
        raise ValueError(f"invalid rational value: {value!r}")
    return Fraction(value)


def audit_table_1(
    transcription: Mapping[str, object],
) -> tuple[LayerMemoryAudit, ...]:
    table = transcription.get("table_1_units")
    if type(table) is not dict or set(table) != {"bf16", "coat", "agoq"}:
        raise ValueError("table_1_units must contain bf16, coat, and agoq")
    rows: list[LayerMemoryAudit] = []
    for method in ("bf16", "coat", "agoq"):
        raw = table[method]
        if type(raw) is not dict or set(raw) != set(COMPONENT_ORDER):
            raise ValueError(f"{method} must contain exactly seven components")
        components = {name: _parse_fraction(raw[name]) for name in COMPONENT_ORDER}
        rows.append(
            LayerMemoryAudit(
                method=method,
                components_u=components,
                total_u=sum(components.values(), Fraction()),
            )
        )
    return tuple(rows)


def project_model(
    audits: Iterable[LayerMemoryAudit],
    batch: int,
    sequence: int,
    hidden: int,
    layers: int,
) -> ModelProjection:
    dimensions = {
        "batch": batch,
        "sequence": sequence,
        "hidden": hidden,
        "layers": layers,
    }
    for name, value in dimensions.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    rows = tuple(audits)
    methods = [row.method for row in rows]
    if len(methods) != len(set(methods)) or not rows:
        raise ValueError("audits must contain unique methods")
    bytes_per_u = batch * sequence * hidden * 2
    totals = {row.method: row.total_u * bytes_per_u * layers for row in rows}
    return ModelProjection(
        batch=batch,
        sequence=sequence,
        hidden=hidden,
        layers=layers,
        bytes_per_u=bytes_per_u,
        totals_bytes=totals,
    )


def fraction_text(value: Fraction) -> str:
    if not isinstance(value, Fraction):
        raise TypeError("value must be a Fraction")
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"
