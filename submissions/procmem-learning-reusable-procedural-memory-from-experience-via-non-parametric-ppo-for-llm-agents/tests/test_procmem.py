import json
from pathlib import Path

from procmem_repro.core import (
    ClaimStatus,
    Skill,
    SkillPool,
    generate_evidence_bundle,
    ppo_gate_accepts,
    propose_skill,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_skill_mdp_executes_activation_policy_and_termination():
    skill = Skill(
        name="open-container",
        activation="state contains closed container",
        execution=["inspect container", "open container"],
        termination="container is open",
    )

    result = skill.apply({"closed_container": True})

    assert result == {
        "activated": True,
        "actions": ["inspect container", "open container"],
        "terminated": True,
    }


def test_semantic_gradient_proposes_skill_without_model_parameter_updates():
    original = Skill(
        name="find-object",
        activation="object may be nearby",
        execution=["look around"],
        termination="object found",
    )
    proposed = propose_skill(
        original,
        {
            "activation": "object is visible or mentioned",
            "execution": ["scan room", "pick up object"],
            "termination": "object is in inventory",
        },
    )

    assert proposed.name == "find-object-candidate"
    assert proposed.activation == "object is visible or mentioned"
    assert proposed.execution == ["scan room", "pick up object"]
    assert proposed.termination == "object is in inventory"
    assert original.execution == ["look around"]
    assert proposed.model_parameter_updates == 0


def test_ppo_gate_requires_positive_clipped_surrogate_improvement():
    assert ppo_gate_accepts(ratio=1.4, advantage=2.0, clip_epsilon=0.2)
    assert not ppo_gate_accepts(ratio=1.1, advantage=-0.5, clip_epsilon=0.2)
    assert not ppo_gate_accepts(ratio=0.0, advantage=2.0, clip_epsilon=0.2)


def test_skill_pool_prunes_by_online_score_with_stable_tie_break():
    pool = SkillPool(capacity=2)
    pool.add(Skill("beta", "b", ["b"], "done"), gain=1.0, uses=2)
    pool.add(Skill("alpha", "a", ["a"], "done"), gain=1.0, uses=2)
    pool.add(Skill("gamma", "g", ["g"], "done"), gain=3.0, uses=3)

    assert [entry.skill.name for entry in pool.entries] == ["alpha", "gamma"]
    assert pool.total_memory_tokens() == 6


def test_evidence_bundle_records_claim_bindings_and_does_not_promote_tables():
    bundle = generate_evidence_bundle()

    statuses = {claim["challenge_claim_sha256"]: claim["status"] for claim in bundle["claims"]}
    assert statuses["28eea780f130523e50495149accae17c64cf5e4759450530b8362166ca67b8eb"] == ClaimStatus.TOY
    assert statuses["ac04b54896fef8a79d03b25254f0520f6e42f41064e8ae3a7818c550f5af0157"] == ClaimStatus.TOY
    assert statuses["9f1c59f31d308e29ade8bc2006442c0de5c2e35d79136a6a55410984a87528b6"] == ClaimStatus.TOY
    assert statuses["80ca84d9709de610ffda4e7b701cace2e858c289369ecb206b1b3216f4fa0d7b"] == ClaimStatus.INCONCLUSIVE
    assert statuses["b4d70952d1bd41fb322bcd2a426ca41e2a1462811db71e2edae967f6cdefd599"] == ClaimStatus.INCONCLUSIVE
    assert statuses["f9714af3f3e8a942bd55ce3d364f074b63d501d02047991857e28ddaa9d1b9f4"] == ClaimStatus.INCONCLUSIVE

    assert bundle["paper_id"] == "9kJQjx2B80"
    assert bundle["snapshot_id"] == "c797d3cfc3dccc0d6e34854ee2969147dce439e23c7dac0f6a8a57e3baeb54e9"
    assert bundle["upstream"]["paper"] == "arxiv:2602.01869v1"
    assert bundle["reproduced_table_measurements"] == []
    assert all("paper_reported_context" in claim for claim in bundle["claims"])
    json.dumps(bundle)


def test_space_metadata_and_scoring_page_bind_proc_mem_attempt():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    report = (PROJECT_ROOT / "pages" / "report.md").read_text(encoding="utf-8")

    assert "icml2026-repro" in readme
    assert "paper-9kJQjx2B80" in readme
    assert "ProcMEM" in report
    assert "28eea780f130523e50495149accae17c64cf5e4759450530b8362166ca67b8eb" in report
    assert "toy mechanism checks" in report
    assert "not reproduced measurements" in report
