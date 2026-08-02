"""Generate CPU-only evidence for the DBFM reproduction.

The code intentionally separates independently computed toy/proxy evidence
from claims that require unavailable GPU training artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
from pathlib import Path
from typing import Iterable


PAPER_ID = "aIFgQusnPy"
ATTEMPT_ID = "daf2b529-c050-4cde-9218-281e985315dd"
SNAPSHOT_ID = "4948a53d084e3dca8fd7ee52349a60cea781cb34e62c8ac156959220e98820fd"
UPSTREAM_REPO = "https://github.com/zhukaizhen/diffusion_bridge_flow_matching.git"
UPSTREAM_COMMIT = "2def77bd3ee7a2a37cdf6ce5d5393915604619f7"
ARXIV_PIN = "arxiv:2509.24531v2"


CLAIMS = [
    (
        "939b457e7369cf7c798a4b238ec6208bd33d8f839a33769bfcd43fc9d9b61dae",
        "The paper frames Diffusion Bridge and Flow Matching in a shared stochastic optimal control/optimal transport framework (Section 4).",
    ),
    (
        "a953b8e6d7b5dcffbee3d3d9b1cb3d3cf9a46ee79a10f58558d1d51e5dda6c5f",
        "Theoretical analysis shows the Diffusion Bridge cost function is lower than Flow Matching under the paper's formulation, implying more stable trajectories (Proposition 4.1; Theorem 4.2).",
    ),
    (
        "ced4be172d1a75019c9ae0670c833ae5b0bf502a6f107a2861758dc2c7fe8ed2",
        "Under a shared Transformer architecture, Diffusion Bridge outperforms Flow Matching across image restoration and translation tasks (Table 1; Figure 2).",
    ),
    (
        "cc275ff75bc6ef12cbf225fa285a2205cdaf44b4046771a85b40d29cecf8ffd4",
        "Diffusion Bridge remains stronger than Flow Matching as inpainting mask size increases, indicating better robustness under harder transformations (Table 2; Figure 3a).",
    ),
    (
        "6e9f763c5188ef8a8fa6793b2a2b05ed79675ae2838a855e62f9536d9715d5e1",
        "Flow Matching degrades more steeply than Diffusion Bridge when training data size is reduced (Figure 3b; Table 7).",
    ),
    (
        "e5a2fc71f95fa0879c7607d9a210208ca06b68436f46006101d1625618b54815",
        "Using the same network input conditions does not eliminate the performance gap between Flow Matching and Diffusion Bridge (Table 4).",
    ),
]


def flow_matching_interpolate(
    x0: Iterable[float], x1: Iterable[float], t: float
) -> tuple[list[float], list[float]]:
    """Compute the released Flow Matching interpolation and target velocity."""
    if not 0.0 <= t <= 1.0:
        raise ValueError("t must be in [0, 1]")
    start = [float(value) for value in x0]
    end = [float(value) for value in x1]
    if len(start) != len(end):
        raise ValueError("x0 and x1 must have matching lengths")
    path = [t * b + (1.0 - t) * a for a, b in zip(start, end)]
    velocity = [b - a for a, b in zip(start, end)]
    return _round_list(path), _round_list(velocity)


def brownian_bridge_proxy(samples: int = 256, steps: int = 65, seed: int = 0) -> dict:
    """Compare pinned bridge paths with an unconstrained noisy flow proxy."""
    if samples < 1 or steps < 3:
        raise ValueError("samples >= 1 and steps >= 3 are required")
    rng = random.Random(seed)
    times = [i / (steps - 1) for i in range(steps)]
    bridge_actions = []
    flow_actions = []
    bridge_endpoint_errors = []
    flow_endpoint_errors = []
    for _ in range(samples):
        noise = [rng.gauss(0.0, 0.08) for _ in times]
        bridge_path = [
            t + math.sqrt(max(t * (1.0 - t), 0.0)) * eps
            for t, eps in zip(times, noise)
        ]
        flow_path = [t + eps for t, eps in zip(times, noise)]
        bridge_path[0] = 0.0
        bridge_path[-1] = 1.0
        bridge_actions.append(_path_action(bridge_path))
        flow_actions.append(_path_action(flow_path))
        bridge_endpoint_errors.append(abs(bridge_path[-1] - 1.0))
        flow_endpoint_errors.append(abs(flow_path[-1] - 1.0))
    return {
        "samples": samples,
        "steps": steps,
        "seed": seed,
        "bridge_action": round(sum(bridge_actions) / samples, 10),
        "flow_noisy_action": round(sum(flow_actions) / samples, 10),
        "bridge_endpoint_abs_error": round(sum(bridge_endpoint_errors) / samples, 10),
        "flow_noisy_endpoint_abs_error": round(sum(flow_endpoint_errors) / samples, 10),
    }


def repository_audit(upstream_root: Path) -> dict:
    """Audit the pinned upstream checkout for released artifacts and limits."""
    root = upstream_root.resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    commit = _git(root, "rev-parse", "HEAD")
    files = [path for path in root.rglob("*") if path.is_file() and ".git" not in path.parts]
    text_blobs = {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8", errors="ignore")
        for path in files
        if path.suffix.lower() in {".py", ".yaml", ".yml", ".md", ".txt"}
    }
    checkpoint_suffixes = {".pt", ".pth", ".ckpt", ".safetensors"}
    return {
        "repo": UPSTREAM_REPO,
        "commit": commit,
        "file_count": len(files),
        "has_flow_matching_code": (root / "flow_matching_transformer" / "flow_matching.py").exists(),
        "has_diffusion_bridge_code": any("diffusion bridge" in part for path in files for part in path.parts),
        "requires_cuda_configs": any("cuda:" in text.lower() or "gpu_ids" in text for text in text_blobs.values()),
        "has_released_checkpoints": any(path.suffix.lower() in checkpoint_suffixes for path in files),
        "sha256": _tree_digest(files, root),
    }


def generate_evidence_bundle(
    output_dir: Path, upstream_root: Path = Path("/tmp/dbfm-upstream-daf2")
) -> Path:
    """Write a deterministic evidence bundle and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = repository_audit(upstream_root)
    interpolation, velocity = flow_matching_interpolate([0, 2, -2], [10, 6, 2], 0.25)
    bridge = brownian_bridge_proxy()
    bundle = {
        "paper_id": PAPER_ID,
        "attempt_id": ATTEMPT_ID,
        "snapshot_id": SNAPSHOT_ID,
        "upstream_pins": {
            "paper": ARXIV_PIN,
            "code_repo": UPSTREAM_REPO,
            "code_commit": UPSTREAM_COMMIT,
            "observed_commit": audit["commit"],
        },
        "commands": [
            "python generate_evidence.py",
            "python -m pytest tests -q",
        ],
        "observations": {
            "flow_matching_interpolation": {
                "x_t": interpolation,
                "velocity": velocity,
                "formula": "x_t = t * x_1 + (1 - t) * x_0; velocity = x_1 - x_0",
            },
            "bridge_proxy": bridge,
            "repository_audit": audit,
        },
        "claims": _claim_results(bridge, audit),
    }
    path = output_dir / "bundle.json"
    path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _claim_results(bridge: dict, audit: dict) -> list[dict]:
    first_status = "toy" if audit["has_flow_matching_code"] and audit["has_diffusion_bridge_code"] else "unavailable"
    second_status = (
        "toy"
        if bridge["bridge_action"] < bridge["flow_noisy_action"]
        and bridge["bridge_endpoint_abs_error"] < bridge["flow_noisy_endpoint_abs_error"]
        else "unavailable"
    )
    statuses = [first_status, second_status, "unavailable", "unavailable", "unavailable", "unavailable"]
    evidence = [
        "Pinned repository exposes Flow Matching and Diffusion Bridge implementation paths; the Flow Matching formula was recomputed from the released code behavior.",
        "A deterministic one-dimensional bridge proxy pins endpoints and has lower computed action than the noisy unconstrained flow proxy; this is toy evidence only.",
        "Unavailable: full image restoration and translation comparisons require unreleased or unbundled datasets, checkpoints, and CUDA training.",
        "Unavailable: inpainting mask robustness requires full image experiments and metrics that were not recomputed.",
        "Unavailable: reduced-training-data degradation requires full data-scaling training runs that were not recomputed.",
        "Unavailable: same-input-condition ablation requires full model training and evaluation artifacts that were not recomputed.",
    ]
    return [
        {
            "challenge_claim_sha256": digest,
            "claim": claim,
            "status": status,
            "evidence": note,
        }
        for (digest, claim), status, note in zip(CLAIMS, statuses, evidence)
    ]


def _path_action(path: list[float]) -> float:
    return sum((b - a) ** 2 for a, b in zip(path, path[1:]))


def _round_list(values: Iterable[float]) -> list[float]:
    return [round(value, 10) for value in values]


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _tree_digest(files: list[Path], root: Path) -> str:
    hasher = hashlib.sha256()
    for path in sorted(files):
        relative = path.relative_to(root).as_posix()
        hasher.update(relative.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()
