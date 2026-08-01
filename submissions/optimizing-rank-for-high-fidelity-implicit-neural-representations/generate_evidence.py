import json
from pathlib import Path
from optimizing_rank_inr_repro.benchmarks import run_all_benchmarks

def main():
    bench_res = run_all_benchmarks()
    c1_ver = bench_res["claim1_stable_rank"]["status"] == "verified"
    c2_ver = bench_res["claim2_image_overfitting"]["status"] == "verified"
    c3_ver = bench_res["claim3_sparse_ct"]["status"] == "verified"
    c4_ver = bench_res["claim4_multidomain"]["status"] == "verified"

    evidence = {
        "paper_id": "2azIa9tfl3",
        "claims": [
            {
                "claim_id": "claim_1",
                "statement": "The paper argues that vanilla MLP INR low-frequency bias is a symptom of stable-rank degradation during training rather than an intrinsic architectural limitation (Section 3).",
                "verified": c1_ver,
                "evidence_type": "empirical_benchmark",
                "details": bench_res["claim1_stable_rank"]
            },
            {
                "claim_id": "claim_2",
                "statement": "Rank-regulating, near-orthogonal Muon updates improve image overfitting quality across multiple INR architectures compared with Adam (Table 1).",
                "verified": c2_ver,
                "evidence_type": "empirical_benchmark",
                "details": bench_res["claim2_image_overfitting"]
            },
            {
                "claim_id": "claim_3",
                "statement": "Muon improves sparse-view CT reconstruction quality across multiple INR architectures compared with Adam (Table 4).",
                "verified": c3_ver,
                "evidence_type": "empirical_benchmark",
                "details": bench_res["claim3_sparse_ct"]
            },
            {
                "claim_id": "claim_4",
                "statement": "The reported improvements extend to natural images, medical images, audio, super-resolution, and novel-view synthesis, with up to about +9 dB PSNR over the same architecture (Tables 1-6).",
                "verified": c4_ver,
                "evidence_type": "empirical_ablation",
                "details": bench_res["claim4_multidomain"]
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
        f.write("\n")

    print(f"Evidence successfully generated and written to {out_file}")


if __name__ == "__main__":
    main()

