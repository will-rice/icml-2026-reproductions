"""Strict loading and deterministic analysis of released DeMix artifacts."""

from collections.abc import Mapping
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
import hashlib
import json
from pathlib import Path
from typing import Any


PINNED_MIXTURE_SHA256 = (
    "2be00152f98c44a740bc2f8e2098be3740ea2f1cd31b7158ade9d54c8e852dc2"
)
_NORMALIZED_QUANTUM = Decimal("0.000000000001")


class ArtifactIntegrityError(ValueError):
    """Raised when artifact bytes do not match the pinned release."""


class ArtifactValidationError(ValueError):
    """Raised when a released artifact does not satisfy the expected schema."""


def load_pinned_manifest(
    path: Path,
    expected_sha256: str = PINNED_MIXTURE_SHA256,
) -> dict[str, dict[str, Decimal]]:
    """Load the exact pinned mixture manifest without binary float conversion."""
    payload = path.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ArtifactIntegrityError(
            "manifest SHA-256 "
            f"{actual_sha256} does not match pinned {expected_sha256}"
        )

    try:
        parsed = json.loads(
            payload.decode("utf-8"),
            parse_float=Decimal,
            parse_int=Decimal,
            parse_constant=lambda value: (_raise_non_finite_json(value)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactValidationError("manifest is not valid UTF-8 JSON") from error

    if not isinstance(parsed, dict):
        raise ArtifactValidationError("manifest root must be an object")
    return parsed


def normalize_weights(weights: Mapping[str, Decimal]) -> dict[str, str]:
    """Validate and normalize one ordered weight vector deterministically."""
    if not isinstance(weights, Mapping) or not weights:
        raise ArtifactValidationError("weights must be a non-empty mapping")

    validated: list[tuple[str, Decimal]] = []
    for domain, value in weights.items():
        if not isinstance(domain, str) or not domain:
            raise ArtifactValidationError("domain names must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, (Decimal, int)):
            raise ArtifactValidationError(
                f"weight for {domain!r} must be a Decimal or integer"
            )
        decimal_value = value if isinstance(value, Decimal) else Decimal(value)
        if not decimal_value.is_finite():
            raise ArtifactValidationError(f"weight for {domain!r} must be finite")
        if decimal_value < 0:
            raise ArtifactValidationError(
                f"weight for {domain!r} must be nonnegative"
            )
        validated.append((domain, decimal_value))

    with localcontext() as context:
        context.prec = 28
        total = sum((value for _, value in validated), start=Decimal("0"))
        if total <= 0:
            raise ArtifactValidationError("weight sum must be positive")

        normalized: dict[str, Decimal] = {}
        for domain, value in validated[:-1]:
            normalized[domain] = (value / total).quantize(
                _NORMALIZED_QUANTUM,
                rounding=ROUND_HALF_EVEN,
            )

        final_domain = validated[-1][0]
        final_value = Decimal("1") - sum(
            normalized.values(),
            start=Decimal("0"),
        )
        if final_value < 0:
            raise ArtifactValidationError("rounded normalized weights exceed one")
        normalized[final_domain] = final_value

    return {
        domain: format(value, ".12f")
        for domain, value in normalized.items()
    }


def analyze_manifest(
    manifest: Mapping[str, Mapping[str, Decimal]],
    reference_model_count: int = 16,
) -> dict[str, Any]:
    """Derive auditable facts from the released ordered mixture manifest."""
    if not isinstance(manifest, Mapping) or not manifest:
        raise ArtifactValidationError("manifest must be a non-empty mapping")
    if isinstance(reference_model_count, bool) or reference_model_count < 0:
        raise ArtifactValidationError("reference model count must be nonnegative")

    mixture_ids = list(manifest)
    first_weights = manifest[mixture_ids[0]]
    if not isinstance(first_weights, Mapping) or not first_weights:
        raise ArtifactValidationError("each mixture must be a non-empty mapping")
    domain_names = list(first_weights)

    raw_weight_sums: dict[str, str] = {}
    normalized_weights: dict[str, dict[str, str]] = {}
    normalization_required: list[str] = []
    already_unit_sum: list[str] = []

    for mixture_id, weights in manifest.items():
        if not isinstance(mixture_id, str) or not mixture_id:
            raise ArtifactValidationError(
                "mixture identifiers must be non-empty strings"
            )
        if not isinstance(weights, Mapping) or list(weights) != domain_names:
            raise ArtifactValidationError(
                f"{mixture_id!r} does not have the expected ordered domains"
            )

        normalized_weights[mixture_id] = normalize_weights(weights)
        raw_sum = sum(
            (Decimal(value) for value in weights.values()),
            start=Decimal("0"),
        )
        raw_weight_sums[mixture_id] = _plain_decimal(raw_sum)
        if raw_sum == Decimal("1"):
            already_unit_sum.append(mixture_id)
        else:
            normalization_required.append(mixture_id)

    return {
        "mixture_count": len(mixture_ids),
        "mixture_ids": mixture_ids,
        "domain_count": len(domain_names),
        "domain_names": domain_names,
        "raw_weight_sums": raw_weight_sums,
        "normalized_weights": normalized_weights,
        "normalization_required": normalization_required,
        "already_unit_sum": already_unit_sum,
        "all_weights_nonnegative": True,
        "all_sums_positive": True,
        "reference_model_count": reference_model_count,
        "manifest_reference_count_match": (
            len(mixture_ids) == reference_model_count
        ),
        "normalization_serialization": {
            "decimal_places": 12,
            "rounding": "ROUND_HALF_EVEN",
            "final_domain_is_residual": True,
        },
    }


def _plain_decimal(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _raise_non_finite_json(value: str) -> None:
    raise ArtifactValidationError(
        f"manifest contains non-finite JSON constant {value!r}"
    )
