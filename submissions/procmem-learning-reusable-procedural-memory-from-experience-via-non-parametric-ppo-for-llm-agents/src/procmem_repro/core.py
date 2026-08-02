from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ClaimStatus:
    TOY = "toy"
    INCONCLUSIVE = "inconclusive"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class Skill:
    name: str
    activation: str
    execution: list[str]
    termination: str
    model_parameter_updates: int = 0

    def apply(self, state: dict[str, Any]) -> dict[str, Any]:
        activated = bool(state.get("closed_container") or state.get("activate"))
        actions = list(self.execution) if activated else []
        return {
            "activated": activated,
            "actions": actions,
            "terminated": activated and bool(actions),
        }

    def token_count(self) -> int:
        fields = [self.activation, self.termination, *self.execution]
        return sum(len(field.split()) for field in fields)


@dataclass(frozen=True)
class SkillEntry:
    skill: Skill
    gain: float
    uses: int

    @property
    def online_score(self) -> float:
        return self.gain / max(1, self.uses)


class SkillPool:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity")
        self.capacity = capacity
        self.entries: list[SkillEntry] = []

    def add(self, skill: Skill, *, gain: float, uses: int) -> None:
        self.entries.append(SkillEntry(skill=skill, gain=gain, uses=uses))
        self.entries = sorted(
            self.entries,
            key=lambda entry: (-entry.online_score, entry.skill.name),
        )[: self.capacity]
        self.entries.sort(key=lambda entry: entry.skill.name)

    def total_memory_tokens(self) -> int:
        return sum(entry.skill.token_count() for entry in self.entries)


def propose_skill(skill: Skill, semantic_gradient: dict[str, Any]) -> Skill:
    return Skill(
        name=f"{skill.name}-candidate",
        activation=str(semantic_gradient.get("activation", skill.activation)),
        execution=list(semantic_gradient.get("execution", skill.execution)),
        termination=str(semantic_gradient.get("termination", skill.termination)),
        model_parameter_updates=0,
    )


def ppo_gate_accepts(
    *, ratio: float, advantage: float, clip_epsilon: float
) -> bool:
    if ratio <= 0 or clip_epsilon < 0:
        return False
    clipped_ratio = min(max(ratio, 1.0 - clip_epsilon), 1.0 + clip_epsilon)
    surrogate = min(ratio * advantage, clipped_ratio * advantage)
    return surrogate > 0


CLAIMS = [
    {
        "target_claim": "ProcMEM/Skill-Pro learns reusable procedural skills from interaction experience without updating model parameters (Section 3).",
        "challenge_claim": "ProcMEM/Skill-Pro learns reusable procedural skills from interaction experience without updating model parameters (Section 3).",
        "challenge_claim_sha256": "28eea780f130523e50495149accae17c64cf5e4759450530b8362166ca67b8eb",
        "status": ClaimStatus.TOY,
        "observation": "A synthetic Skill-MDP update path proposes executable procedural skills and records zero model parameter updates.",
        "paper_reported_context": "The paper describes non-parametric procedural-memory learning in Section 3; this bundle does not run the paper's agent benchmark.",
    },
    {
        "target_claim": "The Skill-MDP formalism converts passive episodic narratives into executable skills with activation, execution, and termination conditions (Section 3).",
        "challenge_claim": "The Skill-MDP formalism converts passive episodic narratives into executable skills with activation, execution, and termination conditions (Section 3).",
        "challenge_claim_sha256": "ac04b54896fef8a79d03b25254f0520f6e42f41064e8ae3a7818c550f5af0157",
        "status": ClaimStatus.TOY,
        "observation": "The local Skill object enforces activation, execution, and termination fields and executes them on a controlled synthetic state.",
        "paper_reported_context": "The paper defines this structure in Section 3; this is a mechanism check, not a full environment reproduction.",
    },
    {
        "target_claim": "Non-Parametric PPO uses semantic gradients for candidate skill generation and a PPO Gate for robust skill verification (Section 4).",
        "challenge_claim": "Non-Parametric PPO uses semantic gradients for candidate skill generation and a PPO Gate for robust skill verification (Section 4).",
        "challenge_claim_sha256": "9f1c59f31d308e29ade8bc2006442c0de5c2e35d79136a6a55410984a87528b6",
        "status": ClaimStatus.TOY,
        "observation": "The local semantic-gradient proposal and clipped-surrogate gate reproduce the decision logic on deterministic numeric fixtures.",
        "paper_reported_context": "The paper presents the NP-PPO gate in Section 4; no LLM-generated semantic gradients are sampled here.",
    },
    {
        "target_claim": "Skill-Pro achieves higher reuse rates than baselines in in-domain, cross-task, and cross-agent evaluations (Table 1).",
        "challenge_claim": "Skill-Pro achieves higher reuse rates than baselines in in-domain, cross-task, and cross-agent evaluations (Table 1).",
        "challenge_claim_sha256": "80ca84d9709de610ffda4e7b701cace2e858c289369ecb206b1b3216f4fa0d7b",
        "status": ClaimStatus.INCONCLUSIVE,
        "observation": "No primary raw run outputs or executable benchmark harness were available in this reproduction.",
        "paper_reported_context": "Table 1 reports reuse-rate comparisons; those values are not emitted as reproduced measurements.",
    },
    {
        "target_claim": "Skill-Pro maintains only 816 memory tokens while achieving the highest reported ALFWorld success rate of 0.90 under extreme compression (Table 2).",
        "challenge_claim": "Skill-Pro maintains only 816 memory tokens while achieving the highest reported ALFWorld success rate of 0.90 under extreme compression (Table 2).",
        "challenge_claim_sha256": "b4d70952d1bd41fb322bcd2a426ca41e2a1462811db71e2edae967f6cdefd599",
        "status": ClaimStatus.INCONCLUSIVE,
        "observation": "The local SkillPool accounts deterministic memory tokens, but it does not reproduce the ALFWorld evaluation or the reported 816-token/0.90 result.",
        "paper_reported_context": "Table 2 values remain paper-reported context only.",
    },
    {
        "target_claim": "Ablations evaluate the contribution of skill use, online score, and PPO Gate pass rate to Skill-Pro performance (Table 3).",
        "challenge_claim": "Ablations evaluate the contribution of skill use, online score, and PPO Gate pass rate to Skill-Pro performance (Table 3).",
        "challenge_claim_sha256": "f9714af3f3e8a942bd55ce3d364f074b63d501d02047991857e28ddaa9d1b9f4",
        "status": ClaimStatus.INCONCLUSIVE,
        "observation": "The local code checks online-score pruning and PPO Gate behavior independently, but no benchmark ablation runs were executed.",
        "paper_reported_context": "Table 3 ablation values are not reproduced measurements.",
    },
]


def generate_evidence_bundle() -> dict[str, Any]:
    return {
        "paper_id": "9kJQjx2B80",
        "attempt_id": "69599dee-e0f4-4f62-b6cf-2f4c6d35493d",
        "snapshot_id": "c797d3cfc3dccc0d6e34854ee2969147dce439e23c7dac0f6a8a57e3baeb54e9",
        "challenge_revision": "81166abbeb76e5f79ff87e51061b5a0306507203",
        "upstream": {
            "paper": "arxiv:2602.01869v1",
            "openreview": "https://openreview.net/forum?id=9kJQjx2B80",
            "code": None,
        },
        "claims": [dict(claim) for claim in CLAIMS],
        "reproduced_table_measurements": [],
        "cost_usd": 0.0,
        "cpu_only": True,
    }
