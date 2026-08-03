from fac_evidence.core import compute_fac, missing_features
from fac_evidence.manifest import build_manifest
from fac_evidence.tables import main_result_context


CLAIMS = [
    {
        "claim_sha256": "010f1831b580319c66761587c1e8cb2ed533475cee7ee199b46c68d4eaaa4f77",
        "claim": "Feature Activation Coverage measures post-training data diversity in an interpretable LLM feature space rather than using only text-level diversity metrics (Section 3).",
    },
    {
        "claim_sha256": "2eb2a85344edfdb5f2e017b8ede642fd6098dc48fe0bfb3506bec207aa5c11c8",
        "claim": "FAC Synthesis uses sparse autoencoders to identify missing features in a seed dataset and generate synthetic samples that activate those features (Section 4).",
    },
    {
        "claim_sha256": "aa863c883e3570fb5243d7d552f94d618ab83d157571a3ce3dec6a6886424789",
        "claim": "The paper reports consistent diversity and downstream-performance improvements on instruction following, toxicity detection, reward modeling, and behavior steering (Section 5).",
    },
    {
        "claim_sha256": "d51e98ace1136e4e71581061b941585a4f682d2aecdc13b3acdc78e5708f9baf",
        "claim": "Using 100% of selected missing features outperforms lower selected-feature ratios across the four downstream tasks in the reported ablation (Table 7).",
    },
    {
        "claim_sha256": "6a15477d006c35f360169101410ea481348874a4374e64e3ea15f91a90a490f5",
        "claim": "The paper identifies a shared interpretable feature space across LLaMA, Mistral, and Qwen that enables cross-model knowledge transfer (Section 6).",
    },
]


def build_evidence_bundle() -> dict:
    fac_value = compute_fac(anchor_task_features={1, 2, 3, 4}, generated_features={2, 4})
    missing = sorted(missing_features(anchor_relevant={1, 2, 3}, seed_relevant={2}))
    manifest = build_manifest()
    return {
        "manifest": manifest,
        "claims": [
            {
                **CLAIMS[0],
                "status": "toy",
                "reproduced_measurements": [{"name": "toy_fac", "value": fac_value, "anchor_features": 4, "covered_features": 2}],
                "evidence": "Independent toy FAC computation uses the anchor-task feature set as denominator and generated feature intersection as numerator. Official artifacts pin SAE feature-space code paths, but no full SAE activation run was performed.",
            },
            {
                **CLAIMS[1],
                "status": "toy",
                "reproduced_measurements": [{"name": "toy_missing_features", "value": missing}],
                "evidence": "Toy set arithmetic reproduces the missing-feature operation. Official scripts expose SAE missing-feature selection and two-stage targeted generation, but large-model synthesis was not run.",
            },
            {
                **CLAIMS[2],
                "status": "inconclusive",
                "reproduced_measurements": [],
                "paper_reported_context": {"main_result": main_result_context()},
                "evidence": "The TeX source and official training scripts cover the four named tasks, but GPU fine-tuning and external evaluators were not rerun. Paper-reported values are context only.",
            },
            {
                **CLAIMS[3],
                "status": "toy",
                "reproduced_measurements": [{"name": "table_7_monotonicity_check", "value": "100_percent_row_dominates_30_and_60_percent_rows"}],
                "evidence": "The TeX feature-ratio table was parsed and checked for monotonic ordering. This verifies table integrity, not independent performance.",
            },
            {
                **CLAIMS[4],
                "status": "toy",
                "reproduced_measurements": [{"name": "public_sae_checkpoint_families", "value": ["llama", "mistral", "qwen"]}],
                "evidence": "Three public SAE checkpoint repositories and model-specific layer settings were pinned. Cross-model transfer performance was not recomputed.",
            },
        ],
    }
