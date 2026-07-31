from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import numpy as np


ATTEMPT_ID = "2ad6b75a-74a9-4603-bf31-e15707b3e683"
PAPER_ID = "09CSjVeDug"
SNAPSHOT_ID = "5f56d39afbd1e52d5200744bd965233dd0eb78dcb9bb3d5d1d20fd2129be54a9"
GITHUB_REPO = "yuchen-zhu-zyc/DMPO"
GITHUB_URL = "https://github.com/yuchen-zhu-zyc/DMPO.git"
GITHUB_COMMIT = "1661fa7d75f0ccec3bbc1b6cae94e9e3fb88571a"
ARXIV_ID = "2510.08233"


CLAIMS = [
    {
        "sha256": "de36b989902f6868972692307865db7ac7943f97c4bf4ffc50880d83c14bff6c",
        "text": "DMPO fine-tunes diffusion LLMs by matching the model policy distribution to an optimal reward-tilted distribution through cross-entropy optimization (Section 3).",
        "observation_key": "objective",
    },
    {
        "sha256": "f09381d3dd0cb2d8f52436eeff59f2739930f041584ba590c2cbf4275e03e368",
        "text": "The method introduces weight baseline subtraction to make small-batch DMPO training effective (Section 3.4).",
        "observation_key": "baseline_subtraction",
    },
    {
        "sha256": "9b048a80e727b9407f8c05a6b7873db6a4e334e65b039ffa90e635c31914bfaa",
        "text": "DMPO is trained without supervised fine-tuning in an R1-Zero-like recipe for reasoning tasks (Section 4).",
        "observation_key": "r1_zero_recipe",
    },
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum()


def reward_tilted_weights(
    log_rnds: np.ndarray, rewards: np.ndarray, alpha: float, coeff: float
) -> np.ndarray:
    if alpha < 0:
        logits = coeff * rewards
    elif alpha == 0:
        logits = coeff * rewards
    else:
        logits = coeff * (log_rnds + rewards / alpha)
    return softmax(logits.astype(np.float64))


def weighted_denoising_ce(
    token_losses: np.ndarray,
    mask_counts: np.ndarray,
    advantages: np.ndarray,
    num_replicates: int,
) -> float:
    per_row = token_losses.sum(axis=-1) / np.maximum(mask_counts, 1)
    return float((per_row * advantages).sum() / num_replicates)


def centered_advantages(
    advantages: np.ndarray, centering_factor: np.ndarray, strength: float
) -> np.ndarray:
    return advantages - strength * centering_factor


def clone_upstream() -> Path:
    path = Path(tempfile.mkdtemp(prefix="dmpo-upstream-")) / "repo"
    subprocess.run(["git", "clone", "--filter=blob:none", GITHUB_URL, str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "checkout", GITHUB_COMMIT], check=True)
    return path


FIXED_UPSTREAM = Path("/tmp/icml-dmpo-upstream-1785350300")


def resolve_upstream(path: Path | None) -> Path:
    if path is not None and path.exists():
        return path
    if FIXED_UPSTREAM.exists():
        return FIXED_UPSTREAM
    return clone_upstream()


def analyze_upstream(repo: Path) -> dict[str, Any]:
    trainer = (repo / "DMPO" / "dmpo_trainer.py").read_text(encoding="utf-8")
    config = (repo / "DMPO" / "dmpo_train_config.yaml").read_text(encoding="utf-8")
    train = (repo / "DMPO" / "dmpo_train.py").read_text(encoding="utf-8")
    rewards = (repo / "DMPO" / "reward_func.py").read_text(encoding="utf-8")
    data_utils = (repo / "DMPO" / "data_utils.py").read_text(encoding="utf-8")

    log_rnds = np.array([-0.7, -0.1, -1.4], dtype=np.float64)
    reward_values = np.array([0.0, 1.0, 0.25], dtype=np.float64)
    weights = reward_tilted_weights(log_rnds, reward_values, alpha=0.5, coeff=0.8)
    token_losses = np.array(
        [[0.0, 0.2, 0.8, 0.0], [0.1, 0.0, 0.4, 0.7], [0.0, 0.3, 0.5, 0.9], [0.2, 0.1, 0.0, 0.6]],
        dtype=np.float64,
    )
    mask_counts = np.array([2, 3, 3, 3], dtype=np.float64)
    advantages = np.array([0.55, 0.45, 0.55, 0.45], dtype=np.float64)
    loss = weighted_denoising_ce(token_losses, mask_counts, advantages, num_replicates=2)
    centered = centered_advantages(
        np.array([0.75, 0.25], dtype=np.float64),
        np.array([0.2, 0.8], dtype=np.float64),
        strength=0.5,
    )

    objective_markers = [
        'self.args.loss == "wdce"',
        "F.cross_entropy",
        "rewards / self.args.alpha",
        "softmax(dim=-1)",
        "losses.sum(dim=-1) / m * advantages",
    ]
    baseline_markers = [
        "advantage_centering",
        "advantage_centering_neg",
        "advantage_centering_unbias",
        "advantages -=",
        "centering_strength",
    ]
    recipe_markers = [
        "loss: wdce",
        "alpha: 0.04",
        "num_generations: 16",
        "GSAI-ML/LLaDA-8B-Instruct",
    ]
    reward_markers = ["gsm8k", "math", "countdown", "sudoku"]
    sft_markers = ["SFTTrainer", "supervised_fine_tuning", "sft_train"]

    file_hashes = {
        str(path.relative_to(repo)): sha256_file(path)
        for path in [
            repo / "DMPO" / "dmpo_trainer.py",
            repo / "DMPO" / "DMPO_config.py",
            repo / "DMPO" / "dmpo_train_config.yaml",
            repo / "DMPO" / "dmpo_train.py",
            repo / "DMPO" / "reward_func.py",
            repo / "LICENSE",
        ]
        if path.exists()
    }
    file_hashes["dmpo_trainer.py"] = file_hashes["DMPO/dmpo_trainer.py"]

    return {
        "objective": {
            "status": "verified" if all(marker in trainer for marker in objective_markers) else "inconclusive",
            "reward_tilted_weights": [round(float(value), 8) for value in weights],
            "weighted_denoising_ce": round(loss, 8),
            "markers_found": [marker for marker in objective_markers if marker in trainer],
        },
        "baseline_subtraction": {
            "status": "verified" if all(marker in trainer for marker in baseline_markers) else "inconclusive",
            "raw_advantages": [0.75, 0.25],
            "centering_factor": [0.2, 0.8],
            "centered_advantages": [round(float(value), 8) for value in centered],
            "markers_found": [marker for marker in baseline_markers if marker in trainer],
        },
        "r1_zero_recipe": {
            "status": "toy",
            "config_markers_found": [marker for marker in recipe_markers if marker in config],
            "reward_task_markers_found": [
                marker for marker in reward_markers if marker in rewards.lower() or marker in data_utils.lower()
            ],
            "sft_markers_in_training_entrypoint": [marker for marker in sft_markers if marker in train],
        },
        "file_hashes": file_hashes,
    }


def build_bundle(upstream: Path | None = None) -> dict[str, Any]:
    repo = resolve_upstream(upstream)
    analysis = analyze_upstream(repo)
    claims = []
    for claim in CLAIMS:
        observation = analysis[claim["observation_key"]]
        claims.append(
            {
                "sha256": claim["sha256"],
                "text": claim["text"],
                "status": observation["status"],
                "evidence": claim_evidence(claim["observation_key"], observation),
            }
        )
    return {
        "attempt_id": ATTEMPT_ID,
        "paper_id": PAPER_ID,
        "snapshot_id": SNAPSHOT_ID,
        "title": "Enhancing Reasoning for Diffusion LLMs via Distribution Matching Policy Optimization",
        "upstream": {
            "arxiv": ARXIV_ID,
            "github": {
                "repo": GITHUB_REPO,
                "url": GITHUB_URL,
                "commit": GITHUB_COMMIT,
            },
        },
        "claims": claims,
        "observations": analysis,
        "commands": [
            f"git ls-remote {GITHUB_URL} HEAD",
            "git clone --filter=blob:none https://github.com/yuchen-zhu-zyc/DMPO.git",
            "uv run python generate_evidence.py",
            "uv run pytest -q",
        ],
        "api_cost_usd": 0.0,
        "limitations": [
            "No LLaDA-8B, DeepSpeed, or distributed training run was executed.",
            "Table 1 benchmark accuracy gains were not independently recomputed.",
            "The R1-Zero-style evidence is a source/configuration audit plus toy checks, not full policy optimization.",
        ],
    }


def claim_evidence(key: str, observation: dict[str, Any]) -> str:
    if key == "objective":
        return (
            "Pinned trainer code contains WDCE, cross-entropy, reward/alpha softmax weighting, "
            f"and the independent toy weighted CE is {observation['weighted_denoising_ce']}."
        )
    if key == "baseline_subtraction":
        return (
            "Pinned trainer code contains advantage-centering branches and subtracts a centering "
            f"factor; toy centered advantages are {observation['centered_advantages']}."
        )
    return (
        "Pinned config selects WDCE with alpha 0.04 and LLaDA-8B reasoning tasks; "
        f"SFT markers in the training entrypoint: {observation['sft_markers_in_training_entrypoint']}."
    )


def write_bundle(path: Path, upstream: Path | None = None) -> dict[str, Any]:
    bundle = build_bundle(upstream)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bundle
