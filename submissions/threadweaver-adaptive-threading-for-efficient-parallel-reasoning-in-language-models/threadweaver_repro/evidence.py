"""Deterministic CPU-only evidence for the ThreadWeaver reproduction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess


ATTEMPT_ID = "464da858-71f3-498e-a567-dc4fbba10d42"
PAPER_ID = "Efq2VvYk1o"
PAPER_TITLE = (
    "ThreadWeaver: Adaptive Threading for Efficient Parallel Reasoning in "
    "Language Models"
)
EXPECTED_UPSTREAM_REVISION = "b944f0139209258caa34fa7dea6a58c2502912fa"
UPSTREAM_URL = "https://github.com/facebookresearch/threadweaver"
PROJECT_PAGE = "https://threadweaver-parallel.github.io/"
ARXIV_ID = "2512.07843"

CHALLENGE_CLAIMS = [
    {
        "claim_id": "two_stage_generator",
        "text": (
            "ThreadWeaver introduces a two-stage parallel trajectory generator "
            "for producing parallel chain-of-thought data for supervised "
            "fine-tuning (Abstract)."
        ),
        "sha256": "14103fb0e4104687d642e8f62103e018dcad4f5422c176a6115999f7f9b9a936",
    },
    {
        "claim_id": "trie_rollout_design",
        "text": (
            "ThreadWeaver uses a trie-based rollout design to enable parallel "
            "reasoning on off-the-shelf autoregressive inference engines "
            "(Abstract)."
        ),
        "sha256": "e0947fee5089612e1401263eef8ae3aa49c0df19a20a714031ee1e4b3e4f1bce",
    },
    {
        "claim_id": "parallelization_aware_rl",
        "text": (
            "The framework includes parallelization-aware reinforcement "
            "learning to balance reasoning accuracy with effective "
            "parallelization (Abstract)."
        ),
        "sha256": "1a30a8a25ac3eee77f42eec9eb71e9fa4e803bf2b5bed938498f0a23fa126ef1",
    },
    {
        "claim_id": "six_benchmark_accuracy",
        "text": (
            "On six math reasoning benchmarks, ThreadWeaver trained on "
            "Qwen3-8B reaches 79.9% on AIME24 and 71.9% average performance "
            "(Abstract)."
        ),
        "sha256": "63287f3a10d2f5882c76cb40aa8453996521da594ae056e43c3158fc2348fd46",
    },
    {
        "claim_id": "reported_token_latency_speedup",
        "text": (
            "ThreadWeaver reports up to 1.53x token-latency speedup while "
            "matching comparably sized sequential reasoning models (Abstract)."
        ),
        "sha256": "12f521e3f0f73c78c88d24b6d272f553368a94b5b0357b881c378e7b05ae026a",
    },
]

REQUIRED_PATHS = [
    "README.md",
    "data_generation/src/generate_trajectories.py",
    "data_generation/README.md",
    "threadweaver_sft/src/prefix_tree_utils_v1.py",
    "threadweaver_sft/src/simple_eval.py",
    "threadweaver_rl/train_par.slurm",
    "threadweaver_rl/train_seq.slurm",
    "threadweaver_rl/deepscaler/rewards/math_rewardv2.py",
    "threadweaver_rl/deepscaler/rewards/reward_types.py",
]


@dataclass(frozen=True)
class ParsedTrajectory:
    outlines: dict[str, str]
    threads: dict[str, str]
    outline_thread_ids_match: bool
    threads_are_independent: bool


@dataclass(frozen=True)
class TrieAttention:
    tokens: list[str]
    parents: list[int]
    allowed: list[list[bool]]
    positions: dict[tuple[str, int], int]


class _TrieNode:
    def __init__(self, token: str | None = None, parent: "_TrieNode | None" = None):
        self.token = token
        self.parent = parent
        self.children: dict[str, _TrieNode] = {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def audit_upstream(upstream: Path) -> dict:
    upstream = upstream.resolve()
    readme = _read(upstream / "README.md")
    data_readme = _read(upstream / "data_generation" / "README.md")
    prefix_tree = _read(
        upstream / "threadweaver_sft" / "src" / "prefix_tree_utils_v1.py"
    )
    train_par = _read(upstream / "threadweaver_rl" / "train_par.slurm")
    rl_readme = _read(upstream / "threadweaver_rl" / "README.md")

    required = {relative: (upstream / relative).exists() for relative in REQUIRED_PATHS}
    hashes = {
        relative: sha256_file(upstream / relative)
        for relative, exists in required.items()
        if exists
    }

    return {
        "upstream_url": UPSTREAM_URL,
        "project_page": PROJECT_PAGE,
        "arxiv_id": ARXIV_ID,
        "git_revision": git_revision(upstream),
        "expected_revision": EXPECTED_UPSTREAM_REVISION,
        "revision_matches": git_revision(upstream) == EXPECTED_UPSTREAM_REVISION,
        "required_paths": required,
        "file_sha256": hashes,
        "markers": {
            "two_stage_generator": all(
                marker in readme
                for marker in [
                    "Stage 1: Lightweight Rewriting",
                    "Stage 2: Scalable Self-Training",
                ]
            )
            and "five refinement stages" in data_readme,
            "trie_training": (
                "Trie Construction" in readme
                and "ancestor-only attention mask" in readme
                and "process_input_ids" in prefix_tree
            ),
            "off_the_shelf_engines": (
                "vLLM or SGLang" in readme and "standard API calls" in readme
            ),
            "p_grpo_mean_centered": (
                "P-GRPO" in readme
                and "Mean-Centered Normalization" in readme
                and "algorithm.norm_adv_by_std_in_grpo=False" in train_par
            ),
            "parallel_branching_config": (
                "actor_rollout_ref.rollout.agent.enable_parallel_branching=True"
                in train_par
            ),
            "reward_acceleration_term": "acceleration ratio = sequential cost / parallel cost"
            in rl_readme,
            "requires_8x80g_gpu": "8x80G A100 or H100 GPUs" in rl_readme,
            "reported_accuracy_table_only": "79.9%" in readme and "71.9%" in readme,
            "reported_speedup_table_only": "1.53x" in readme,
        },
    }


def parse_parallel_trajectory(text: str) -> ParsedTrajectory:
    outlines: dict[str, str] = {}
    threads: dict[str, str] = {}

    outlines_match = re.search(r"<Outlines>(.*?)</Outlines>", text, flags=re.S)
    if outlines_match:
        for match in re.finditer(
            r"<Outline>\s*([^:<]+)\s*:\s*(.*?)\s*</Outline>",
            outlines_match.group(1),
            flags=re.S,
        ):
            outlines[match.group(1).strip()] = " ".join(match.group(2).split())

    for match in re.finditer(
        r"<Thread>\s*([^:<]+)\s*:\s*(.*?)\s*</Thread>", text, flags=re.S
    ):
        threads[match.group(1).strip()] = " ".join(match.group(2).split())

    thread_ids = set(threads)
    independent = True
    for thread_id, body in threads.items():
        for other in thread_ids - {thread_id}:
            if re.search(rf"\b(Thread|thread)\s*{re.escape(other)}\b", body):
                independent = False

    return ParsedTrajectory(
        outlines=outlines,
        threads=threads,
        outline_thread_ids_match=set(outlines) == set(threads),
        threads_are_independent=independent,
    )


def trie_attention(sequences: list[list[str]]) -> TrieAttention:
    root = _TrieNode()
    for sequence in sequences:
        node = root
        for token in sequence:
            node = node.children.setdefault(token, _TrieNode(token, node))

    tokens: list[str] = []
    parents: list[int] = []
    node_indices: dict[int, int] = {}

    def visit(node: _TrieNode, parent_index: int) -> None:
        for child in node.children.values():
            index = len(tokens)
            node_indices[id(child)] = index
            tokens.append(child.token or "")
            parents.append(parent_index)
            visit(child, index)

    visit(root, -1)

    ancestor_sets: list[set[int]] = []
    for index in range(len(tokens)):
        ancestors = {index}
        parent = parents[index]
        while parent != -1:
            ancestors.add(parent)
            parent = parents[parent]
        ancestor_sets.append(ancestors)

    allowed = [
        [source in ancestor_sets[target] for source in range(len(tokens))]
        for target in range(len(tokens))
    ]

    seen: dict[str, int] = {}
    positions: dict[tuple[str, int], int] = {}
    for index, token in enumerate(tokens):
        occurrence = seen.get(token, 0)
        positions[(token, occurrence)] = index
        seen[token] = occurrence + 1

    return TrieAttention(tokens=tokens, parents=parents, allowed=allowed, positions=positions)


def critical_path_latency(sequential_tokens: int, thread_tokens: list[int]) -> dict:
    if sequential_tokens < 0 or any(tokens < 0 for tokens in thread_tokens):
        raise ValueError("token counts must be nonnegative")
    sequential_latency = sequential_tokens + sum(thread_tokens)
    critical_path = sequential_tokens + (max(thread_tokens) if thread_tokens else 0)
    speedup = sequential_latency / critical_path if critical_path else 1.0
    return {
        "sequential_token_latency": sequential_latency,
        "critical_path_token_latency": critical_path,
        "token_latency_speedup": speedup,
    }


def p_grpo_advantages(
    correctness: list[float],
    acceleration: list[float],
    acceleration_weight: float,
) -> dict:
    if len(correctness) != len(acceleration) or not correctness:
        raise ValueError("matching nonempty reward lists required")

    rewards = [
        round(c + acceleration_weight * a, 12)
        for c, a in zip(correctness, acceleration, strict=True)
    ]
    mean = sum(rewards) / len(rewards)
    centered = [round(reward - mean, 12) for reward in rewards]
    variance = sum(value * value for value in centered) / len(centered)
    std = variance**0.5
    standard = [round(value / std, 12) if std else 0.0 for value in centered]
    return {
        "rewards": rewards,
        "mean_reward": round(mean, 12),
        "p_grpo_advantages": centered,
        "standard_grpo_advantages": standard,
    }


def _toy_trajectory_result() -> dict:
    trajectory = """
<think>
<Parallel>
<Outlines>
<Outline>1: derive identity</Outline>
<Outline>2: numeric check</Outline>
</Outlines>
<Thread>1: Use product-to-sum.</Thread>
<Thread>2: Substitute decimals.</Thread>
</Parallel>
</think>
"""
    parsed = parse_parallel_trajectory(trajectory)
    return {
        "outlines": parsed.outlines,
        "threads": parsed.threads,
        "outline_thread_ids_match": parsed.outline_thread_ids_match,
        "threads_are_independent": parsed.threads_are_independent,
    }


def _toy_trie_result() -> dict:
    trie = trie_attention(
        [
            ["prompt", "<Parallel>", "<Thread>1", "alpha", "</Thread>"],
            ["prompt", "<Parallel>", "<Thread>2", "beta", "</Thread>"],
        ]
    )
    alpha = trie.positions[("alpha", 0)]
    beta = trie.positions[("beta", 0)]
    parallel = trie.positions[("<Parallel>", 0)]
    return {
        "tokens": trie.tokens,
        "parents": trie.parents,
        "alpha_attends_parallel": trie.allowed[alpha][parallel],
        "beta_attends_parallel": trie.allowed[beta][parallel],
        "alpha_attends_beta": trie.allowed[alpha][beta],
        "beta_attends_alpha": trie.allowed[beta][alpha],
    }


def build_evidence(upstream: Path) -> dict:
    audit = audit_upstream(upstream)
    trajectory = _toy_trajectory_result()
    trie = _toy_trie_result()
    latency = critical_path_latency(sequential_tokens=8, thread_tokens=[5, 13, 7])
    p_grpo = p_grpo_advantages(
        correctness=[1.0, 1.0, 1.0],
        acceleration=[0.0, 0.1, 0.2],
        acceleration_weight=0.1,
    )

    claims = [
        {
            **CHALLENGE_CLAIMS[0],
            "status": "verified",
            "evidence": [
                "Pinned upstream contains data_generation/ and README-described Stage 1/Stage 2 pipeline.",
                "No paper-reported benchmark values were used as reproduced measurements.",
            ],
        },
        {
            **CHALLENGE_CLAIMS[1],
            "status": "toy",
            "evidence": [
                "Pinned upstream implements prefix-tree training utilities.",
                "Toy trie check confirms sibling branches cannot attend to each other while descendants attend to shared ancestors.",
            ],
        },
        {
            **CHALLENGE_CLAIMS[2],
            "status": "toy",
            "evidence": [
                "Pinned RL training command disables GRPO standard-deviation normalization.",
                "Toy reward check shows mean-centered P-GRPO preserves small acceleration scale where standard normalization inflates it.",
            ],
        },
        {
            **CHALLENGE_CLAIMS[3],
            "status": "unreplicated",
            "evidence": [
                "The public artifact documents the reported table, but recomputing Qwen3-8B six-benchmark accuracy requires 8x80G GPUs.",
                "This CPU-only attempt did not train or evaluate Qwen3-8B and does not claim the paper value as reproduced.",
            ],
        },
        {
            **CHALLENGE_CLAIMS[4],
            "status": "unreplicated",
            "evidence": [
                "The CPU toy check recomputes critical-path token accounting only.",
                "The paper's reported token-latency speedup and wall-clock measurements require the released multi-GPU evaluation path.",
            ],
        },
    ]

    return {
        "attempt_id": ATTEMPT_ID,
        "paper_id": PAPER_ID,
        "title": PAPER_TITLE,
        "generated_at": "2026-08-01T00:00:00+00:00",
        "upstream": audit,
        "toy_checks": {
            "parallel_trajectory_parser": trajectory,
            "trie_attention": trie,
            "critical_path_latency": latency,
            "p_grpo_advantages": p_grpo,
        },
        "claims": claims,
        "costs": {
            "metered_api_usd": 0.0,
            "gpu_hours": 0.0,
            "notes": "CPU-only static and toy checks; no paid API calls.",
        },
        "limitations": [
            "Qwen3-8B training and six-benchmark evaluation were not run.",
            "The 1.53x reported token-latency claim was not recomputed on released model outputs.",
            "The public README states SFT and RL require a single node with 8x80G A100 or H100 GPUs.",
        ],
    }


def write_evidence(upstream: Path, output: Path) -> dict:
    evidence = build_evidence(upstream)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence
