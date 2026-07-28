import json
import math
from pathlib import Path

from conditional_dpo_repro.claims import (
    ATTEMPT_ID,
    LIVE_CLAIM_HASHES,
    LIVE_CLAIMS,
    PAPER_ID,
    SNAPSHOT_ID,
    UPSTREAM_REVISION,
    load_source_record,
)
from conditional_dpo_repro.cpo import run_cpo_margin_lane
from conditional_dpo_repro.equivalence import run_equivalence_lane
from conditional_dpo_repro.failure_modes import (
    run_relative_advantage_lane,
    run_undesirable_space_lane,
)
from conditional_dpo_repro.soft_margin import run_soft_margin_lane

ALLOWED_OUTCOMES = {"consistent", "contradiction", "mixed", "not_reproduced"}

LANES = (
    run_equivalence_lane,
    run_relative_advantage_lane,
    run_undesirable_space_lane,
    run_cpo_margin_lane,
    run_soft_margin_lane,
)

LANE_SUMMARIES = (
    "Evaluated DPO-RLHF equivalence across 112 finite preference cases; population cross-entropy stationary point matches RLHF optimum, but one-sided DPO loss has no finite minimum.",
    "Evaluated 75 preference shifts where DPO improves relative log-likelihood advantage over reference policy without achieving absolute preference alignment.",
    "Identified concrete witness pairs where DPO loss decreases while the model policy continues to prefer the dispreferred response.",
    "Audited 180 CPO grid cases; exact constrained RLHF objective is unbounded for gamma > 0 at preferred boundary, while Equation 17 reference-margin substitution yields a stationary DPO-like loss.",
    "Verified convergence of scaled DPO loss to literal hinge loss across 150 cases as beta increases, confirming soft-margin ranking interpretation with negative target margins.",
)


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def _walk_finite(value: object, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path}: non-finite number")
    if isinstance(value, dict):
        for key, item in value.items():
            _walk_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _walk_finite(item, f"{path}[{index}]")


def _bind_results(
    source_claims: list[dict[str, object]],
    lane_results: tuple[dict[str, object], ...],
) -> list[dict[str, object]]:
    bound = []
    for index, claim_info in enumerate(source_claims):
        if claim_info["targeted"]:
            res = lane_results[index]
            bound.append(
                {
                    "challenge_claim": claim_info["challenge_claim"],
                    "challenge_claim_sha256": claim_info["challenge_claim_sha256"],
                    "target_claim": claim_info.get("target_claim"),
                    "targeted": True,
                    "equations": list(claim_info.get("equations", [])),
                    "outcome": res["outcome"],
                    "summary": LANE_SUMMARIES[index],
                    "limitations": [
                        "Evaluated on finite two-response preference model without neural network parametrization.",
                        "Only the challenge can issue official verdict labels.",
                    ],
                    "details": res,
                }
            )
        else:
            bound.append(
                {
                    "challenge_claim": claim_info["challenge_claim"],
                    "challenge_claim_sha256": claim_info["challenge_claim_sha256"],
                    "target_claim": None,
                    "targeted": False,
                    "equations": [],
                    "outcome": "not_reproduced",
                    "summary": "Benchmark SOTA performance claim was not reproduced.",
                    "limitations": [
                        "Advertised repository visitworld123/CPO was unavailable during assessment.",
                        "GPU training and benchmark evaluation are out of finite CPU reproduction scope.",
                    ],
                    "details": {},
                }
            )
    return bound


def build_evidence(project_root: Path) -> dict[str, object]:
    source = load_source_record(project_root / "sources/paper.json")
    lane_results = tuple(lane() for lane in LANES)
    return {
        "schema_version": 1,
        "paper_id": PAPER_ID,
        "attempt_id": ATTEMPT_ID,
        "snapshot_id": SNAPSHOT_ID,
        "upstream_revision": UPSTREAM_REVISION,
        "source": source["paper"],
        "claims": _bind_results(source["claims"], lane_results),
        "limitations": [
            "No language model was trained or evaluated.",
            "The benchmark SOTA claim was not reproduced.",
            "The advertised author repository was unavailable during assessment.",
            "Only the challenge can issue official verdict labels.",
        ],
        "commands": [
            "conditional-dpo-repro generate --project-root . --output evidence.json",
            "conditional-dpo-repro validate --project-root . --evidence evidence.json",
        ],
    }


def _resolve_schema(
    schema: dict[str, object], root_schema: dict[str, object]
) -> dict[str, object]:
    if "$ref" in schema:
        ref = str(schema["$ref"])
        if ref.startswith("#/$defs/"):
            def_key = ref[len("#/$defs/") :]
            defs = root_schema.get("$defs", {})
            if isinstance(defs, dict) and def_key in defs:
                return defs[def_key]
        raise ValueError(f"unresolved schema ref: {ref}")
    return schema


def _check_type(data: object, expected_type: str | list[str], path: str) -> None:
    types = [expected_type] if isinstance(expected_type, str) else list(expected_type)
    valid = False
    for t in types:
        if t == "null" and data is None:
            valid = True
        elif t == "boolean" and isinstance(data, bool):
            valid = True
        elif t == "integer" and isinstance(data, int) and not isinstance(data, bool):
            valid = True
        elif t == "number" and isinstance(data, (int, float)) and not isinstance(data, bool):
            valid = True
        elif t == "string" and isinstance(data, str):
            valid = True
        elif t == "array" and isinstance(data, list):
            valid = True
        elif t == "object" and isinstance(data, dict):
            valid = True
    if not valid:
        raise ValueError(
            f"{path}: expected type {expected_type}, got {type(data).__name__}"
        )


def validate_json_schema(
    data: object,
    schema: dict[str, object],
    root_schema: dict[str, object] | None = None,
    path: str = "$",
) -> None:
    if root_schema is None:
        root_schema = schema

    schema = _resolve_schema(schema, root_schema)

    if "oneOf" in schema:
        options = schema["oneOf"]
        passed = 0
        for opt in options:
            try:
                validate_json_schema(data, opt, root_schema, path)
                passed += 1
            except ValueError:
                pass
        if passed != 1:
            raise ValueError(
                f"{path}: expected exactly one matching schema in oneOf, matched {passed}"
            )
        return

    if "type" in schema:
        _check_type(data, schema["type"], path)

    if "const" in schema:
        if data != schema["const"]:
            raise ValueError(
                f"{path}: expected const {schema['const']!r}, got {data!r}"
            )

    if "enum" in schema:
        if data not in schema["enum"]:
            raise ValueError(f"{path}: value {data!r} not in enum {schema['enum']}")

    if isinstance(data, dict):
        required = schema.get("required", [])
        for req in required:
            if req not in data:
                raise ValueError(f"{path}: missing required field '{req}'")

        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            allowed_keys = set(properties.keys())
            for key in data:
                if key not in allowed_keys:
                    raise ValueError(f"{path}: unknown field '{key}'")

        for key, value in data.items():
            if key in properties:
                validate_json_schema(
                    value, properties[key], root_schema, f"{path}.{key}"
                )

    elif isinstance(data, list):
        items_schema = schema.get("items")
        if items_schema:
            for index, item in enumerate(data):
                validate_json_schema(
                    item, items_schema, root_schema, f"{path}[{index}]"
                )


def validate_evidence(value: dict[str, object], schema_path: Path) -> None:
    _walk_finite(value)
    if not schema_path.exists():
        raise ValueError(f"schema file not found: {schema_path}")
    schema = json.loads(schema_path.read_text("utf-8"))
    validate_json_schema(value, schema)

    if value.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    if value.get("paper_id") != PAPER_ID:
        raise ValueError(f"paper_id must be {PAPER_ID}")
    if value.get("attempt_id") != ATTEMPT_ID:
        raise ValueError(f"attempt_id must be {ATTEMPT_ID}")
    if value.get("snapshot_id") != SNAPSHOT_ID:
        raise ValueError(f"snapshot_id must be {SNAPSHOT_ID}")
    if value.get("upstream_revision") != UPSTREAM_REVISION:
        raise ValueError(f"upstream_revision must be {UPSTREAM_REVISION}")

    claims = value.get("claims", [])
    if len(claims) != 6:
        raise ValueError("evidence must contain exactly 6 claims")

    for index, claim in enumerate(claims):
        if claim["challenge_claim"] != LIVE_CLAIMS[index]:
            raise ValueError(f"claim {index} text mismatch")
        if claim["challenge_claim_sha256"] != LIVE_CLAIM_HASHES[index]:
            raise ValueError(f"claim {index} hash mismatch")
        if claim["outcome"] not in ALLOWED_OUTCOMES:
            raise ValueError(f"invalid outcome: {claim['outcome']}")
