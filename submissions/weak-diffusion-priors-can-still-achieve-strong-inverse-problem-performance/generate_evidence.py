#!/usr/bin/env python3
import json
import os
from weak_diffusion_priors.theory import simulate_theorem_3_1_posterior_concentration
from weak_diffusion_priors.inverse_problem import evaluate_table_1_inverse_problem_baselines


def generate_evidence():
    print("Generating reproduction evidence for weak diffusion priors...")
    
    # Claim 2: Theorem 3.1 Posterior Concentration
    theory_results = simulate_theorem_3_1_posterior_concentration()
    
    # Claim 1: Table 1 High-Informative Inverse Problem Baselines
    table_1_results = evaluate_table_1_inverse_problem_baselines()

    evidence = {
        "paper_id": "fdkSA4F0lN",
        "title": "Weak Diffusion Priors Can Still Achieve Strong Inverse-Problem Performance",
        "claims": [
            {
                "challenge_claim_sha256": "f92ef3142d3eb9876b4885e506e2318923f6277bd326f60f7e741fa6259e7ba9",
                "claim_text": "Weak diffusion priors can match strong-prior inverse-problem baselines when measurements are highly informative, such as many observed pixels (Table 1).",
                "verified": table_1_results["claim_1_verified"],
                "evidence_details": table_1_results["table_1_metrics"],
            },
            {
                "challenge_claim_sha256": "4d2832c903b2d7d6e55947d20468d734b233f664c678deed688b9c37ae5b8aac",
                "claim_text": "The theory gives conditions under which high-dimensional measurements make the Bayesian posterior concentrate near the true signal despite weak priors (Theorem 3.1).",
                "verified": theory_results["theorem_3_1_verified"],
                "evidence_details": {
                    "n_dim": theory_results["n_dim"],
                    "noise_std": theory_results["noise_std"],
                    "sweep_results": theory_results["sweep_results"],
                },
            }
        ],
        "summary": {
            "all_claims_verified": bool(table_1_results["claim_1_verified"] and theory_results["theorem_3_1_verified"]),
            "table_1_verified": table_1_results["claim_1_verified"],
            "theorem_3_1_verified": theory_results["theorem_3_1_verified"],
        }
    }

    evidence_dir = os.path.join(os.path.dirname(__file__), "evidence")
    os.makedirs(evidence_dir, exist_ok=True)
    evidence_file = os.path.join(evidence_dir, "evidence.json")
    
    with open(evidence_file, "w") as f:
        json.dump(evidence, f, indent=2)

    print(f"Evidence written to {evidence_file}")
    print(f"All claims verified: {evidence['summary']['all_claims_verified']}")
    return evidence


if __name__ == "__main__":
    generate_evidence()
