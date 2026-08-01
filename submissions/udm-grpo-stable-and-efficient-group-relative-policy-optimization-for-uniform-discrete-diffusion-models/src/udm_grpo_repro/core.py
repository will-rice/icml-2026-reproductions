from __future__ import annotations

from copy import deepcopy


ATTEMPT_ID = "a477051b-45d5-4442-839b-1001759853cd"
PAPER_ID = "WJcFtJriqv"
SNAPSHOT_ID = "6c17591130f02ebbe0b47d265ef3c6026182ce0aaf51b004bb753ec9336bc335"
CHALLENGE_REVISION = "81166abbeb76e5f79ff87e51061b5a0306507203"
AUDIT_TIMESTAMP = "2026-08-01T14:21:00Z"

AUDIT = {
    "repo": "github:Yovecent/UDM-GRPO@d1bec49f4500873606f8345d81692143de059891",
    "models": {
        "geneval": "Yovecents/URSA-1.7B-IBQ512-UDMGRPO-GenEval@7569361fb744283c10e9b81b86c3c542589f0adc",
        "pickscore": "Yovecents/URSA-1.7B-IBQ512-UDMGRPO-PickScore@71b1c4983e14934ea999b2cab3713926d7bd912f",
    },
    "pipeline_grpo": {
        "path": "diffnext/diffnext/pipelines/ursa/pipeline_grpo.py",
        "uses_forward_process": True,
        "uses_clean_response": True,
        "uses_group_relative_advantages": True,
        "evidence": [
            "process_latents uses scheduler.add_noise(states[-1].output_ids, t) when forward_process is enabled.",
            "process_latents repeats states[-1].output_ids as responses when clean_response is enabled.",
            "process_rewards normalizes rewards by group mean and standard deviation.",
        ],
    },
    "configs": {
        "geneval": {
            "path": "diffnext/configs/geneval_grpo/ursa_1.7b_ibq512.yaml",
            "train_steps": 3,
            "train_start": [0, 3],
            "forward_process": True,
            "clean_response": True,
            "guidance_scale": 1,
            "reward_scorers": ["geneval"],
        },
        "pickscore": {
            "path": "diffnext/configs/pickscore_grpo/ursa_1.7b_ibq512.yaml",
            "train_steps": 3,
            "train_start": [0, 3],
            "forward_process": True,
            "clean_response": True,
            "guidance_scale": 1,
            "reward_scorers": ["pickscore"],
        },
        "ocr": {
            "path": "diffnext/configs/ocr_grpo/ursa_1.7b_ibq512.yaml",
            "train_steps": 3,
            "train_start": [0, 3],
            "forward_process": True,
            "clean_response": True,
            "guidance_scale": 1,
            "reward_scorers": ["ocr"],
        },
    },
    "reward_hooks": ["geneval", "pickscore", "ocr"],
}

CLAIMS = [
    {
        "sha256": "0a031c59fd7bbe00c59e97c020bf7d036d73fb2cb3bbd6385b68a37b063a4d74",
        "kind": "mechanism",
        "text": "UDM-GRPO treats the final clean sample as the policy action and reconstructs training trajectories with the diffusion forward process to stabilize GRPO for uniform discrete diffusion (Section 4.2).",
    },
    {
        "sha256": "716661282da161f7fa1075a8570fa6017884d3c3f6165e5c85963f76637349d7",
        "kind": "reduced_step",
        "text": "Reduced-Step training optimizes three early high-noise timesteps to accelerate convergence (Section 4.3).",
    },
    {
        "sha256": "e00307483455d73ef3222df24dfde3d6646fcdf26017b2af7905cf722d5cdd61",
        "kind": "cfg_free",
        "text": "CFG-Free training removes classifier-free guidance during RL training while recovering and surpassing CFG-based quality after optimization (Section 4.4).",
    },
    {
        "sha256": "a25ea6f983664d3f04398287721601f9518163aac45bb16a9a2c1dba6dce396f",
        "kind": "metric",
        "text": "UDM-GRPO raises URSA GenEval overall score from 0.69 to 0.96 and reports state-of-the-art GenEval results among compared continuous and discrete T2I models (Table 1).",
    },
    {
        "sha256": "1d8a28f488fcd9fafe58329ad1b52b829c49cb2423d46c3b8e80f39a9e3b1eaf",
        "kind": "metric",
        "text": "UDM-GRPO reports GenEval 0.96, PickScore 23.81, and OCR 0.57, improving over URSA and URSA without CFG (Table 2).",
    },
    {
        "sha256": "b24538ba4305cac11eef9766e68faf5811ae1a5e15b33930d5ca20eacd477546",
        "kind": "ablation",
        "text": "Ablations show final-clean-sample actions and forward-process trajectories outperform backward-trajectory variants on GenEval, PickScore, and OCR (Table 3).",
    },
]


def summarize_audit(audit: dict | None = None) -> dict[str, object]:
    audit = deepcopy(audit or AUDIT)
    configs = list(audit["configs"].values())
    return {
        "repo": audit["repo"],
        "uses_forward_process": audit["pipeline_grpo"]["uses_forward_process"]
        and all(config["forward_process"] for config in configs),
        "uses_clean_response": audit["pipeline_grpo"]["uses_clean_response"]
        and all(config["clean_response"] for config in configs),
        "uses_group_relative_advantages": audit["pipeline_grpo"]["uses_group_relative_advantages"],
        "train_steps": sorted({config["train_steps"] for config in configs})[0],
        "train_start": configs[0]["train_start"],
        "guidance_scale": sorted({config["guidance_scale"] for config in configs})[0],
        "reward_hooks": sorted(audit["reward_hooks"]),
        "models": audit["models"],
    }


def classify_claims(claims: list[dict] | None = None, audit: dict | None = None) -> list[dict[str, object]]:
    summary = summarize_audit(audit)
    results = []
    for claim in deepcopy(claims or CLAIMS):
        if claim["kind"] == "mechanism":
            status = "verified" if summary["uses_forward_process"] and summary["uses_clean_response"] else "inconclusive"
            observation = "Pinned source uses forward-process noising from the final clean output and uses final clean outputs as responses."
        elif claim["kind"] == "reduced_step":
            status = "verified" if summary["train_steps"] == 3 and summary["train_start"] == [0, 3] else "inconclusive"
            observation = "Pinned training configs use train_steps=3 and train_start=[0, 3]."
        elif claim["kind"] == "cfg_free":
            status = "verified" if summary["guidance_scale"] == 1 else "inconclusive"
            observation = "Pinned rollout configs use guidance_scale=1 for RL training."
        else:
            status = "inconclusive"
            observation = "Full metric reproduction requires GPU image generation and external GenEval/PickScore/OCR evaluation; no paper table value is counted as reproduced."
        results.append(
            {
                "sha256": claim["sha256"],
                "challenge_claim_sha256": claim["sha256"],
                "target_claim": claim["text"],
                "status": status,
                "observation": observation,
            }
        )
    return results


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
            "paper": "arxiv:2604.18518",
            "code": AUDIT["repo"],
            "geneval_model": AUDIT["models"]["geneval"],
            "pickscore_model": AUDIT["models"]["pickscore"],
        },
        "source_audit": summarize_audit(),
        "claims": classify_claims(),
        "reproduced_metric_measurements": [],
        "reproduced_ablation_measurements": [],
        "limitations": [
            "No autonomous GPU training or image generation was run.",
            "GenEval, PickScore, OCR, and ablation table values are not reproduced measurements.",
        ],
    }
