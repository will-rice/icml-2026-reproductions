#!/usr/bin/env python3
"""Generate machine-readable evidence bundle for Optimizing Rank for High-Fidelity INRs."""

import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from optimizing_rank_inr_repro.benchmarks import run_all_benchmarks


def main():
    results = run_all_benchmarks()

    c1_ver = results["claim1_stable_rank"]["status"] == "verified"
    c2_ver = results["claim2_image_overfitting"]["status"] == "verified"
    c3_ver = results["claim3_sparse_ct"]["status"] == "verified"
    c4_ver = results["claim4_multidomain"]["status"] == "verified"

    evidence = {
        "paper_id": "2azIa9tfl3",
        "title": "Optimizing Rank for High-Fidelity Implicit Neural Representations",
        "claims": [
            {
                "challenge_claim_sha256": "0670e0c7422f892ec13151b030fba3772815a6b8afa2911bc65bf4ebda3a2fb9",
                "claim_text": "The paper argues that vanilla MLP INR low-frequency bias is a symptom of stable-rank degradation during training rather than an intrinsic architectural limitation (Section 3).",
                "verified": c1_ver,
                "evidence_details": results["claim1_stable_rank"]
            },
            {
                "challenge_claim_sha256": "a61547a408e0908bb8456fb9e932ef0b9b6dfcf95250b0b7ad03dd637c40ef10",
                "claim_text": "Rank-regulating, near-orthogonal Muon updates improve image overfitting quality across multiple INR architectures compared with Adam (Table 1).",
                "verified": c2_ver,
                "evidence_details": results["claim2_image_overfitting"]
            },
            {
                "challenge_claim_sha256": "fcc893efa61fac59f23688ac2cddf3eddfa930ddda3ee6db84c46d414c7e5933",
                "claim_text": "Muon improves sparse-view CT reconstruction quality across multiple INR architectures compared with Adam (Table 4).",
                "verified": c3_ver,
                "evidence_details": results["claim3_sparse_ct"]
            },
            {
                "challenge_claim_sha256": "088d6fac2fdfdbef61cefc94e0c7c7ada940c86b14ddd19114fa022aed352d6d",
                "claim_text": "The reported improvements extend to natural images, medical images, audio, super-resolution, and novel-view synthesis, with up to about +9 dB PSNR over the same architecture (Tables 1-6).",
                "verified": c4_ver,
                "evidence_details": results["claim4_multidomain"]
            }
        ],
        "summary": {
            "all_claims_verified": c1_ver and c2_ver and c3_ver and c4_ver
        }
    }

    out_dir = Path(__file__).parent / "evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "evidence.json"
    with open(out_file, "w") as f:
        json.dump(evidence, f, indent=2)

    print(f"Evidence successfully generated and written to {out_file}")


if __name__ == "__main__":
    main()
