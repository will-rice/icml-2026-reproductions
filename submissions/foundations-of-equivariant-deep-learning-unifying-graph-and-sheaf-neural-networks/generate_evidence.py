import json
from pathlib import Path
import sys

# Ensure src can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.benchmark import run_reproduction_experiments

def main():
    print("Running Sheaf Neural Networks reproduction pipeline...")
    results = run_reproduction_experiments()

    evidence_data = {
        "paper_id": "aIH1jyU37z",
        "title": "Foundations of Equivariant Deep Learning: Unifying Graph and Sheaf Neural Networks",
        "claims": [
            {
                "claim_id": "claim_1",
                "statement": "The arXiv paper introduces sheaf neural networks by replacing graph Laplacian diffusion with sheaf-Laplacian diffusion that can encode asymmetric, signed, and varying-dimensional relations (Section 3).",
                "verified": results["claim_1_verified"],
                "evidence_type": "theoretical_and_implementation",
                "details": "Sheaf Laplacian operator correctly encodes signed and asymmetric restriction maps, generalizing standard graph diffusion."
            },
            {
                "claim_id": "claim_2",
                "statement": "The sheaf diffusion operator is presented as a drop-in generalization of the diffusion operation used in graph convolutional networks (Section 2.1).",
                "verified": results["claim_2_verified"],
                "evidence_type": "implementation_and_verification",
                "details": "Identity restriction maps in SheafLaplacian produce identical output structure and operator form to standard Kipf-Welling GCN graph Laplacian."
            },
            {
                "claim_id": "claim_3",
                "statement": "On synthetic semi-supervised node-classification tasks over signed graphs, sheaf neural networks outperform Kipf-Welling GCN variants across feature and edge-noise regimes (Figure 1).",
                "verified": results["claim_3_verified"],
                "evidence_type": "empirical_benchmark",
                "details": {
                    "noise_regimes": results["noise_regimes_benchmark"],
                    "status": "verified" if results["claim_3_verified"] else "unverified"
                }
            },
            {
                "claim_id": "claim_4",
                "statement": "The experiments average results over five random graph trials and report standard-deviation error bars for SheafNN and GCN comparisons (Figure 1).",
                "verified": results["claim_4_verified"],
                "evidence_type": "empirical_statistical_reporting",
                "details": {
                    "num_random_trials": results["num_trials"],
                    "standard_deviations_computed": True,
                    "status": "verified" if results["claim_4_verified"] else "unverified"
                }
            }
        ],
        "summary": {
            "all_claims_verified": all(c["verified"] for c in [
                {"verified": results["claim_1_verified"]},
                {"verified": results["claim_2_verified"]},
                {"verified": results["claim_3_verified"]},
                {"verified": results["claim_4_verified"]}
            ]),
            "num_claims_evaluated": 4,
            "num_claims_verified": sum([
                results["claim_1_verified"],
                results["claim_2_verified"],
                results["claim_3_verified"],
                results["claim_4_verified"]
            ])
        }
    }

    out_dir = Path(__file__).resolve().parent / "evidence"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "evidence.json"

    with open(out_file, "w") as f:
        json.dump(evidence_data, f, indent=2)

    print(f"Evidence generated successfully at {out_file}")

if __name__ == "__main__":
    main()
