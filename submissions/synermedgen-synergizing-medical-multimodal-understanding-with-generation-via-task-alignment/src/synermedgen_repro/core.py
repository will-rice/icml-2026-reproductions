from __future__ import annotations

from copy import deepcopy


ATTEMPT_ID = "0c7a06a4-b5f9-4559-aa36-9c5ab50d65de"
PAPER_ID = "Tyv61ZKb9s"
SNAPSHOT_ID = "9d95e572c4edfbddd28e7e1a4135afc244b3829e63ce123726c610b35f0ab698"
CHALLENGE_REVISION = "81166abbeb76e5f79ff87e51061b5a0306507203"
AUDIT_TIMESTAMP = "2026-08-01T13:58:21Z"

ARTIFACTS = {
    "paper": {
        "kind": "paper",
        "pin": "arxiv:2605.08724v1",
        "url": "https://arxiv.org/abs/2605.08724",
        "accessible": True,
        "observation": "arXiv v1 is accessible and states the SynerMedGen method, dataset-size claims, benchmark claims, and project link.",
    },
    "github_project": {
        "kind": "code_project",
        "pin": "github:Mhilab/SynerMedGen",
        "url": "https://github.com/Mhilab/SynerMedGen",
        "accessible": False,
        "observation": "The project URL stated by the paper/arXiv returned 404 during the controller audit.",
    },
    "released_dataset": {
        "kind": "dataset",
        "pin": None,
        "url": None,
        "accessible": False,
        "observation": "No primary SynerMed dataset or checkpoint artifact was found from the stated project URL or web/Hugging Face search.",
    },
}

CLAIMS = [
    {
        "sha256": "de5b83cda10b4e7f29dc8cd10df6f0172106ca34269883562fb6cce262b0c4c1",
        "text": "SynerMedGen derives three generation-aligned understanding tasks from paired synthesis data: Conditional Target Selection, Modality Identification, and Transformation Instruction Alignment (Figure 2)",
        "paper_support": "The paper text defines CTS, MI, and TIA as the three generation-aligned understanding task families.",
        "requires_released_artifact": False,
    },
    {
        "sha256": "d40b38c6a084b9c6a917f0905d7e57f71f89c1ba588fa45281375562f5f36a03",
        "text": "The released SynerMed dataset contains 1M paired medical image synthesis samples and 2M generation-aligned understanding instances (Figure 2)",
        "paper_support": "The paper/arXiv states the 1M paired-sample and 2M understanding-instance release claim.",
        "requires_released_artifact": True,
    },
    {
        "sha256": "31b94e893509101e96e08423e6f869a2d9547d0d7061025e0127d22b8ccceb30",
        "text": "Generation-aligned understanding improves synthesis performance over traditional understanding supervision across 22 image synthesis tasks (Figure 4)",
        "paper_support": "The paper reports a 22-task comparison, but no independent synthesis run is possible without released code/data/checkpoints.",
        "requires_released_artifact": True,
    },
    {
        "sha256": "2d08be03fa0be6353d754fd497dd1826c664110f17336abc5fcef7dbdbff59e3",
        "text": "SynerMedGen outperforms specialized synthesis methods and unified medical multimodal baselines on SynthRAD2023, AutoPET, and BraTS cross-modality synthesis tasks (Tables 1 and 2)",
        "paper_support": "The paper reports Table 1 and Table 2 benchmark superiority, but the audit did not locate runnable primary artifacts.",
        "requires_released_artifact": True,
    },
    {
        "sha256": "255670d6ded46f0c6a90d179e720db0ae6a7b6fed2936cc53ade26a325d256c6",
        "text": "Ablations show that adding CTS, MI, and TIA progressively improves synthesis performance (Figure 9)",
        "paper_support": "The paper reports ablations, but no ablation code, data, or checkpoints were available for recomputation.",
        "requires_released_artifact": True,
    },
    {
        "sha256": "74f4ac704fae1eec2391070c1732e9566357c91310d9b6c9394944c8b675cfab",
        "text": "SynerMedGen shows generalization on unseen MyoPS cardiac MRI and SynthRAD2025 datasets (Figures 6 and 8)",
        "paper_support": "The paper reports generalization figures, but no independent generalization run was possible from accessible primary artifacts.",
        "requires_released_artifact": True,
    },
]


def summarize_artifacts(artifacts: dict[str, dict] | None = None) -> dict[str, object]:
    artifacts = deepcopy(artifacts or ARTIFACTS)
    return {
        "paper": artifacts["paper"],
        "github_project": artifacts["github_project"],
        "released_dataset_found": bool(artifacts["released_dataset"]["accessible"]),
        "released_code_found": bool(artifacts["github_project"]["accessible"]),
        "artifact_release_complete": bool(
            artifacts["github_project"]["accessible"] and artifacts["released_dataset"]["accessible"]
        ),
    }


def classify_claims(claims: list[dict] | None = None, artifacts: dict[str, dict] | None = None) -> list[dict[str, str]]:
    claims = deepcopy(claims or CLAIMS)
    artifact_summary = summarize_artifacts(artifacts)
    release_complete = bool(artifact_summary["artifact_release_complete"])
    classified = []
    for claim in claims:
        if claim["requires_released_artifact"]:
            status = "verified" if release_complete else "inconclusive"
            observation = (
                "Accessible primary code and dataset artifacts would be required to recompute this claim; "
                "the stated project URL was inaccessible and no replacement primary release was found."
            )
        else:
            status = "toy"
            observation = (
                "The task names and alignment mechanism are confirmed from the accessible paper text, "
                "but no released implementation was available for executable mechanism verification."
            )
        classified.append(
            {
                "sha256": claim["sha256"],
                "challenge_claim_sha256": claim["sha256"],
                "target_claim": claim["text"],
                "status": status,
                "observation": observation,
                "paper_support": claim["paper_support"],
            }
        )
    return classified


def build_evidence_bundle() -> dict[str, object]:
    return {
        "attempt_id": ATTEMPT_ID,
        "paper_id": PAPER_ID,
        "snapshot_id": SNAPSHOT_ID,
        "challenge_revision": CHALLENGE_REVISION,
        "generated_at": AUDIT_TIMESTAMP,
        "cpu_only": True,
        "cost_usd": 0.0,
        "upstream": {
            "paper": "arxiv:2605.08724v1",
            "project": "github:Mhilab/SynerMedGen inaccessible during audit",
            "dataset": "no accessible primary SynerMed release located",
        },
        "artifact_audit": summarize_artifacts(),
        "claims": classify_claims(),
        "reproduced_synthesis_measurements": [],
        "reproduced_ablation_measurements": [],
        "limitations": [
            "No released code, dataset manifest, checkpoint, or evaluation script was available from the stated project URL.",
            "Paper-reported table and figure values are not counted as reproduced measurements.",
            "All metric-heavy claims remain inconclusive unless a primary artifact becomes available.",
        ],
    }
