from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import torch


ATTEMPT_ID = "daee6151-3f6d-429d-a01b-c6d91b72dd1c"
PAPER_ID = "m1IRWFAMsa"
SNAPSHOT_ID = "0692f289e0163260c616a30969f5e5d5db781c4d463ff6b123403db29926e574"
CHALLENGE_REVISION = "81166abbeb76e5f79ff87e51061b5a0306507203"
UPSTREAM_COMMIT = "c6989a8354730695d9f5a9faa6c55eeb24865209"
UPSTREAM_REPO = "https://github.com/zichongli5/NorMuon.git"
UPSTREAM_REVISION = f"arxiv:2510.05491+github:zichongli5/NorMuon@{UPSTREAM_COMMIT}"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

LIVE_CLAIMS = [
    {
        "text": "NorMuon combines Muon's orthogonalized updates with neuron-wise adaptive normalization based on second-order momentum statistics (Algorithm 1).",
        "sha256": "c2b6b1756f6922c77ecc6915c14be6996d0b2111cc654462c685bebfce5d1b32",
    },
    {
        "text": "The optimizer analysis shows Muon lowers update condition numbers but leaves high variance in per-neuron update norms, while NorMuon normalizes neuron contributions (Figure 1).",
        "sha256": "ed5bd5aaea367741a42e41f1b0dc573013283967c88896699d9893b32942e08e",
    },
    {
        "text": "NorMuon achieves 21.74% better step efficiency than Adam and 11.31% improvement over Muon on the 1.1B pretraining setting (Table 1).",
        "sha256": "4a7dae74da0fb8136463ae2c090dfc7c32194619f23a57c7b340b867fc5ecf40",
    },
    {
        "text": "NorMuon also improves validation-loss trajectories over Muon and AdamW at the 5.4B scale (Figure 2).",
        "sha256": "af7240033e51f295e515e5451ec065b2792a868a9a88d0b1af74f36acb186684",
    },
    {
        "text": "On 5.4B training, NorMuon keeps optimizer-state memory close to Muon and below AdamW, with about 2.9% total training-step time overhead over AdamW (Table 2).",
        "sha256": "c6c7f62e3ee644b96ef6ac1c3127eaa7483ebe1b04569dd098c0088c36ba8dbf",
    },
    {
        "text": "NorMuon outperforms Muon in Modded-NanoGPT pretraining at 124M and 350M parameter scales (Figure 5).",
        "sha256": "f5e30e2f743849b30aea2822e34321d6e31ae28cba4912d2b0542f62af1ffcb3",
    },
]


def ensure_upstream_checkout(cache_dir: Path | None = None) -> Path:
    root = cache_dir or Path("/tmp") / f"normuon-upstream-{UPSTREAM_COMMIT[:12]}"
    if (root / ".git").is_dir():
        observed = _run(["git", "-C", str(root), "rev-parse", "HEAD"])
        if observed == UPSTREAM_COMMIT:
            return root
        raise RuntimeError(f"unexpected checkout revision at {root}: {observed}")
    _run(["git", "clone", "--depth", "1", UPSTREAM_REPO, str(root)])
    observed = _run(["git", "-C", str(root), "rev-parse", "HEAD"])
    if observed != UPSTREAM_COMMIT:
        raise RuntimeError(f"upstream revision drifted: {observed}")
    return root


def load_official_normuon(source_root: Path) -> ModuleType:
    module_path = source_root / "normuon.py"
    spec = importlib.util.spec_from_file_location("official_normuon", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load official normuon.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def zeropower_via_newtonschulz5(matrix: torch.Tensor, steps: int = 5) -> torch.Tensor:
    assert matrix.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = matrix.bfloat16()
    transposed = matrix.size(-2) > matrix.size(-1)
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = x @ x.mT
        update = b * gram + c * gram @ gram
        x = a * x + update @ x
    if transposed:
        x = x.mT
    return x


def apply_neuron_adaptive_normalization(
    update: torch.Tensor,
    second_momentum: torch.Tensor,
    *,
    beta2: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = update.clone()
    state = second_momentum.clone()
    original_norm = normalized.norm(dim=(-2, -1), keepdim=True)
    row_second_moment = torch.mean(normalized * normalized, dim=-1, keepdim=True)
    state.lerp_(row_second_moment, 1 - beta2)
    step_size = 1 / state.sqrt().add(1e-10)
    normalized.mul_(step_size)
    new_norm = normalized.norm(dim=(-2, -1), keepdim=True)
    normalized.mul_(original_norm / new_norm.add(1e-10))
    return normalized, state


def reference_normuon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    second_momentum: torch.Tensor,
    *,
    beta: float = 0.95,
    beta2: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    momentum = momentum.clone()
    second_momentum = second_momentum.clone()
    grad_work = grad.clone()
    momentum.lerp_(grad_work, 1 - beta)
    update = grad_work.lerp_(momentum, beta) if nesterov else momentum
    original_shape = None
    if update.ndim == 4:
        original_shape = update.shape
        update = update.reshape(update.size(0), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps).to(grad.dtype)
    if original_shape is not None:
        update = update.reshape(original_shape)
    update, second_momentum = apply_neuron_adaptive_normalization(
        update,
        second_momentum,
        beta2=beta2,
    )
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update, second_momentum


def row_norm_cv(matrix: torch.Tensor) -> float:
    row_norms = torch.linalg.vector_norm(matrix, dim=-1).float()
    mean = float(row_norms.mean())
    if mean == 0.0:
        return 0.0
    return float(row_norms.std(unbiased=False) / mean)


def build_evidence_bundle(source_root: Path | None = None) -> dict[str, Any]:
    source = source_root or ensure_upstream_checkout()
    official = load_official_normuon(source)
    normuon_py = source / "normuon.py"
    license_file = source / "LICENSE"

    grad = torch.tensor(
        [[3.0, -2.0, 0.5, 1.0], [0.25, 4.0, -1.0, 2.0], [1.5, 0.0, -3.5, 0.75]],
        dtype=torch.float32,
    )
    official_second = torch.zeros(grad.shape[0], 1)
    official_update = official.normuon_update(
        grad.clone(),
        torch.zeros_like(grad),
        official_second,
    )
    reference_update, reference_second = reference_normuon_update(
        grad,
        torch.zeros_like(grad),
        torch.zeros(grad.shape[0], 1),
    )

    uneven_update = torch.tensor(
        [[8.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [0.25, 0.25, 0.25, 0.25]],
        dtype=torch.float32,
    )
    normalized_update, second = apply_neuron_adaptive_normalization(
        uneven_update,
        torch.zeros(uneven_update.shape[0], 1),
    )
    before_cv = row_norm_cv(uneven_update)
    after_cv = row_norm_cv(normalized_update)

    claims = [
        {
            "claim": LIVE_CLAIMS[0]["text"],
            "challenge_claim_sha256": LIVE_CLAIMS[0]["sha256"],
            "status": "verified",
            "evidence": "The pinned official normuon.py update matches an independently implemented Algorithm 1 reference on deterministic synthetic gradients.",
            "observations": {
                "max_abs_update_delta": _round_float(
                    torch.max(torch.abs(official_update - reference_update))
                ),
                "max_abs_second_momentum_delta": _round_float(
                    torch.max(torch.abs(official_second - reference_second))
                ),
                "second_momentum_shape": list(official_second.shape),
            },
        },
        {
            "claim": LIVE_CLAIMS[1]["text"],
            "challenge_claim_sha256": LIVE_CLAIMS[1]["sha256"],
            "status": "toy",
            "evidence": "A CPU synthetic update with deliberately uneven row norms shows the NorMuon adaptive normalization step reduces row-norm dispersion while restoring total update norm.",
            "observations": {
                "row_norm_cv_before": _round_float(before_cv),
                "row_norm_cv_after": _round_float(after_cv),
                "second_momentum_shape": list(second.shape),
                "total_norm_before": _round_float(torch.linalg.vector_norm(uneven_update)),
                "total_norm_after": _round_float(torch.linalg.vector_norm(normalized_update)),
            },
        },
    ]
    for claim in LIVE_CLAIMS[2:]:
        claims.append(
            {
                "claim": claim["text"],
                "challenge_claim_sha256": claim["sha256"],
                "status": "unavailable",
                "evidence": "Not reproduced: this claim requires GPU-scale LLM pretraining or large model benchmark runs outside the CPU-only budget.",
                "observations": {},
            }
        )

    return {
        "attempt_id": ATTEMPT_ID,
        "paper_id": PAPER_ID,
        "snapshot_id": SNAPSHOT_ID,
        "challenge_revision": CHALLENGE_REVISION,
        "upstream_revision": UPSTREAM_REVISION,
        "estimated_api_cost_usd": 0.0,
        "commands": [
            "git clone --depth 1 https://github.com/zichongli5/NorMuon.git",
            "python generate_evidence.py",
            "python -m pytest tests -q",
        ],
        "inputs": {
            "normuon.py_sha256": _sha256_file(normuon_py),
            "LICENSE_sha256": _sha256_file(license_file),
            "license": "MIT",
        },
        "claims": claims,
        "limitations": [
            "GPU-scale LLM pretraining efficiency, 5.4B validation loss, memory, and Modded-NanoGPT claims were not reproduced.",
            "The mechanism tests use deterministic CPU synthetic gradients and do not claim paper-scale training performance.",
        ],
    }


def write_evidence_bundle(path: Path) -> dict[str, Any]:
    bundle = build_evidence_bundle()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bundle


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(argv: list[str]) -> str:
    completed = subprocess.run(argv, check=True, text=True, capture_output=True)
    return completed.stdout.strip()


def _round_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        value = float(value.detach().cpu())
    return round(float(value), 10)
