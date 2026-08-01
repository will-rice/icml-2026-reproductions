"""Static source audit for the pinned ThunderAgent release."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any


ATTEMPT_ID = "29efb851-aeef-4989-b74d-a83e2c481384"
PAPER_ID = "kR4iOTaAOJ"
SNAPSHOT_ID = "ebb6ed5d2cfc196369f149f52b51f716251ae6b3ba2cfabce1d2642aefdf3aa0"
UPSTREAM_REVISION = (
    "arxiv:2602.13692+"
    "github:ThunderAgent-org/ThunderAgent@7ddc8610270e56d3b109eed8796b3a4360fc67c9"
)

CLAIM_BINDINGS = [
    {
        "slug": "program_abstraction",
        "challenge_claim_sha256": "f3c921a44400a59b56213973efcf334f326cf8f9f3f1ede152eba85119c08faf",
        "claim": "ThunderAgent abstracts agentic workflows as LLM Programs that unify KV cache, system state, and external tool resources (Section 3).",
    },
    {
        "slug": "program_scheduler",
        "challenge_claim_sha256": "030df6bacd99e5a892294e6960768d02e9e6f7b561ec4d24cf44a5d0606ee9f1",
        "claim": "The system adds a program-aware scheduler to improve KV cache hit rates and reduce memory imbalance across agent workflows (Section 4.2).",
    },
    {
        "slug": "resource_lifecycle",
        "challenge_claim_sha256": "733855963a596a93d707e7d5b97f94849c572b0ab5bdcf24c4c215d91c961511",
        "claim": "The tool resource manager asynchronously prepares and reuses tool environments, including disk and port resources (Section 4.3).",
    },
]


def audit_source_tree(source_root: Path) -> dict[str, Any]:
    """Return deterministic observations from a ThunderAgent source tree."""
    source_root = Path(source_root)
    program_text = _read(source_root, "ThunderAgent/program/state.py")
    router_text = _read(source_root, "ThunderAgent/scheduler/router.py")
    app_text = _read(source_root, "ThunderAgent/app.py")
    backend_text = _read(source_root, "ThunderAgent/backend/state.py")

    program_fields = _dataclass_fields(program_text, "Program")
    program_enums = _enum_values(program_text)
    router_methods = _class_methods(router_text, "MultiBackendRouter")
    app_functions = _module_functions(app_text)

    has_program_abstraction = (
        {"program_id", "backend_url", "status", "state", "context_len", "total_tokens"}
        <= program_fields
        and {"REASONING", "ACTING"} <= program_enums.get("ProgramStatus", set())
        and {"ACTIVE", "PAUSED", "TERMINATED"} <= program_enums.get("ProgramState", set())
        and "get_program_id" in app_functions
        and "register_routes" in app_functions
    )
    has_scheduler = (
        {"_scheduler_loop", "_scheduled_check", "_pause_until_safe", "_greedy_resume", "_resume_program"}
        <= router_methods
        and "global_waiting_queue" in router_text
        and "remaining_capacity" in router_text
        and "shared_tokens" in backend_text
    )
    has_release = "release_program" in router_methods and "unregister_program" in router_text
    missing_resource_features = []
    router_lower = router_text.lower()
    if "disk" not in router_lower:
        missing_resource_features.append("No explicit disk resource manager was found in the pinned source.")
    if not any(term in router_lower for term in ("port resource", "port manager", "allocated_port", "release_port")):
        missing_resource_features.append("No explicit port resource manager was found in the pinned source.")

    return {
        "attempt_id": ATTEMPT_ID,
        "paper_id": PAPER_ID,
        "snapshot_id": SNAPSHOT_ID,
        "upstream_revision": UPSTREAM_REVISION,
        "source_hashes": _source_hashes(source_root),
        "claims": {
            "program_abstraction": {
                "status": "verified" if has_program_abstraction else "inconclusive",
                "observations": {
                    "program_fields": program_fields,
                    "program_enums": {key: sorted(value) for key, value in program_enums.items()},
                    "app_functions": sorted(app_functions),
                },
            },
            "program_scheduler": {
                "status": "verified" if has_scheduler else "inconclusive",
                "observations": {
                    "scheduler_methods": router_methods,
                    "has_global_waiting_queue": "global_waiting_queue" in router_text,
                    "has_capacity_accounting": "remaining_capacity" in router_text and "shared_tokens" in backend_text,
                },
            },
            "resource_lifecycle": {
                "status": "toy" if has_release and missing_resource_features else ("verified" if has_release else "inconclusive"),
                "observations": {
                    "release_paths": sorted(method for method in router_methods if "release" in method),
                    "missing_source_features": missing_resource_features,
                },
            },
        },
    }


def build_evidence_bundle(source_root: Path) -> dict[str, Any]:
    """Convert audit observations into judge-facing claim records."""
    audit = audit_source_tree(source_root)
    claims = []
    for binding in CLAIM_BINDINGS:
        result = audit["claims"][binding["slug"]]
        claims.append(
            {
                "claim": binding["claim"],
                "challenge_claim_sha256": binding["challenge_claim_sha256"],
                "status": result["status"],
                "observations": _jsonable(result["observations"]),
            }
        )
    return {
        "attempt_id": ATTEMPT_ID,
        "paper_id": PAPER_ID,
        "snapshot_id": SNAPSHOT_ID,
        "upstream_revision": UPSTREAM_REVISION,
        "source_hashes": audit["source_hashes"],
        "claims": claims,
    }


def write_evidence(source_root: Path, output_path: Path) -> dict[str, Any]:
    bundle = build_evidence_bundle(source_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bundle


def _read(source_root: Path, relative_path: str) -> str:
    return (source_root / relative_path).read_text(encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _source_hashes(source_root: Path) -> dict[str, str]:
    paths = [
        "ThunderAgent/program/state.py",
        "ThunderAgent/scheduler/router.py",
        "ThunderAgent/app.py",
        "ThunderAgent/backend/state.py",
    ]
    return {
        path: hashlib.sha256((source_root / path).read_bytes()).hexdigest()
        for path in paths
    }


def _dataclass_fields(source: str, class_name: str) -> set[str]:
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                stmt.target.id
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
            }
    return set()


def _enum_values(source: str) -> dict[str, set[str]]:
    module = ast.parse(source)
    values: dict[str, set[str]] = {}
    for node in module.body:
        if not isinstance(node, ast.ClassDef):
            continue
        members = {
            stmt.targets[0].id
            for stmt in node.body
            if isinstance(stmt, ast.Assign)
            and stmt.targets
            and isinstance(stmt.targets[0], ast.Name)
        }
        if members:
            values[node.name] = members
    return values


def _class_methods(source: str, class_name: str) -> set[str]:
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {stmt.name for stmt in node.body if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))}
    return set()


def _module_functions(source: str) -> set[str]:
    module = ast.parse(source)
    return {node.name for node in module.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
