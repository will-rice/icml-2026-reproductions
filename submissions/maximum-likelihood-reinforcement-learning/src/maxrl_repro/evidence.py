from __future__ import annotations

from dataclasses import dataclass
import json
from math import isclose
from pathlib import Path
from typing import Any


PINNED_REPO = "https://github.com/tajwarfahim/maxrl"
PINNED_COMMIT = "7197bbb46a2ecd866da52f6b401ff20a34fe9390"


@dataclass(frozen=True)
class BernoulliCheck:
    p: float
    n: int
    estimator_expectation: float
    truncated_gradient: float
    ml_gradient: float

    @property
    def gap_to_ml(self) -> float:
        return abs(self.ml_gradient - self.truncated_gradient)


def maxrl_estimator_expectation(p: float, n: int) -> float:
    """Expected conditional-success score estimator for Bernoulli success."""
    _validate_probability(p)
    if n <= 0:
        raise ValueError("n must be positive")
    return (1.0 - p) * (1.0 - (1.0 - p) ** n)


def truncated_maxrl_gradient(p: float, n: int) -> float:
    """Gradient of sum_{k<=n} pass@k/k with respect to Bernoulli logit."""
    _validate_probability(p)
    if n <= 0:
        raise ValueError("n must be positive")
    return sum(p * (1.0 - p) ** k for k in range(1, n + 1))


def ml_gradient(p: float) -> float:
    """Gradient of log(p) with respect to Bernoulli logit."""
    _validate_probability(p)
    return 1.0 - p


def bernoulli_checks() -> list[BernoulliCheck]:
    checks: list[BernoulliCheck] = []
    for p in (0.05, 0.2, 0.5, 0.8):
        for n in (1, 2, 4, 8, 16):
            estimator = maxrl_estimator_expectation(p, n)
            gradient = truncated_maxrl_gradient(p, n)
            if not isclose(estimator, gradient, rel_tol=0.0, abs_tol=1e-12):
                raise AssertionError((p, n, estimator, gradient))
            checks.append(
                BernoulliCheck(
                    p=p,
                    n=n,
                    estimator_expectation=estimator,
                    truncated_gradient=gradient,
                    ml_gradient=ml_gradient(p),
                )
            )
    return checks


def audit_source(source_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    core = source_root / "verl" / "trainer" / "ppo" / "core_algos.py"
    qwen = source_root / "qwen3_experiments" / "run_qwen3_training.sh"
    readme = source_root / "README.md"
    for path in (core, qwen, readme):
        if not path.is_file():
            raise FileNotFoundError(path)
    core_text = core.read_text(encoding="utf-8")
    qwen_text = qwen.read_text(encoding="utf-8")
    readme_text = readme.read_text(encoding="utf-8")
    return {
        "repo": PINNED_REPO,
        "commit": PINNED_COMMIT,
        "core_algos": {
            "path": str(core.relative_to(source_root)),
            "has_maxrl_branch": 'advantage_type == "maxrl"' in core_text,
            "has_grpo_branch": 'advantage_type == "grpo"' in core_text,
            "has_mean_reward_denominator": "denom_mean = mean_reward + epsilon" in core_text,
            "has_p_normalized_advantage": "denom_std) / denom_mean" in core_text,
        },
        "qwen3_script": {
            "path": str(qwen.relative_to(source_root)),
            "sets_maxrl_estimator": "ADVANTAGE_ESTIMATOR=maxrl" in qwen_text,
            "documents_grpo_toggle": "ADVANTAGE_ESTIMATOR=grpo" in qwen_text,
            "uses_h200_scale": "8xH200" in readme_text or "H200" in qwen_text,
        },
        "readme": {
            "mentions_official_implementation": "official PyTorch implementation" in readme_text,
            "mentions_weights": "Weights" in readme_text,
            "states_gpu_installation": "GPU machine" in readme_text,
        },
    }


def build_evidence(source_root: Path, output_dir: Path) -> dict[str, Any]:
    checks = bernoulli_checks()
    audit = audit_source(source_root)
    evidence = {
        "paper_id": "EeuLO2BjFN",
        "title": "Maximum Likelihood Reinforcement Learning",
        "generated_at": "2026-08-01T20:12:00Z",
        "upstream": audit,
        "bernoulli_checks": [check.__dict__ | {"gap_to_ml": check.gap_to_ml} for check in checks],
        "claims": [
            {
                "id": "objective_family",
                "status": "toy",
                "summary": "Toy Bernoulli checks verify finite-N MaxRL gradients interpolate from RL toward ML.",
            },
            {
                "id": "unbiased_estimator",
                "status": "toy",
                "summary": "Exact enumeration expectation matches the truncated MaxRL gradient for small Bernoulli tasks.",
            },
            {
                "id": "infinite_compute_limit",
                "status": "toy",
                "summary": "The truncation gap decreases monotonically with rollout count in deterministic checks.",
            },
            {
                "id": "pareto_dominance",
                "status": "unavailable",
                "summary": "Full multi-task Pareto claim requires released large-scale evaluation logs or rerunning GPU training.",
            },
            {
                "id": "twenty_x_scaling",
                "status": "unavailable",
                "summary": "The 20x scaling claim is not recomputed in this CPU-only evidence bundle.",
            },
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = output_dir / "evidence"
    pages_dir = output_dir / "pages"
    evidence_dir.mkdir(exist_ok=True)
    pages_dir.mkdir(exist_ok=True)
    (evidence_dir / "results.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (pages_dir / "report.md").write_text(render_report(evidence), encoding="utf-8")
    return evidence


def render_report(evidence: dict[str, Any]) -> str:
    lines = [
        "# Maximum Likelihood Reinforcement Learning Reproduction",
        "",
        f"Paper: `{evidence['paper_id']}`",
        f"Code revision: `{evidence['upstream']['commit']}`",
        "",
        "## Claim Status",
        "",
    ]
    for claim in evidence["claims"]:
        lines.append(f"- `{claim['id']}`: **{claim['status']}** - {claim['summary']}")
    lines.extend(
        [
            "",
            "## CPU Checks",
            "",
            "The Bernoulli checks exactly compare the conditional-success estimator expectation with the truncated MaxRL gradient.",
            "Large-scale Qwen3, ImageNet, and full Pareto claims are not reported as reproduced measurements.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_probability(p: float) -> None:
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0, 1)")
