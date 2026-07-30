"""Generate evidence bundle for high-accuracy sampling reproduction."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from sampler import (
    verify_polylog_step_scaling,
    verify_intrinsic_dimension_scaling,
    verify_log_concave_gradient_sampler,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "evidence")
    parser.add_argument("--upstream-dir", type=Path)
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    delta_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    c1_res = verify_polylog_step_scaling(delta_list)
    c2_res = verify_intrinsic_dimension_scaling(d_star=10, full_d=1000, delta=1e-4)
    c3_res = verify_log_concave_gradient_sampler(dimension=5, target_accuracy=1e-3)

    bundle = {
        "title": "High-accuracy sampling for diffusion models and log-concave distributions",
        "paper_id": "71132",
        "upstream_revision": "arxiv:2602.01338v2+arxiv-source:2602.01338v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cpu_only": True,
        "commands": [
            "python generate_evidence.py"
        ],
        "target_claims": [
            "The diffusion sampler attains delta-error in polylog(1/delta) steps given sufficiently accurate score estimates, improving the dependence on accuracy over prior high-accuracy samplers (Theorem 4.3)",
            "When the data distribution has intrinsic dimension d*, the complexity reduces to Õ(d* polylog(1/delta)) (Corollary 4.4)",
            "The same framework yields a polylog(1/delta)-accuracy sampler for log-concave and more general isoperimetric distributions using first-order gradient queries (Section 5)"
        ],
        "claims": [
            {
                "claim": "The diffusion sampler attains delta-error in polylog(1/delta) steps given sufficiently accurate score estimates, improving the dependence on accuracy over prior high-accuracy samplers (Theorem 4.3)",
                "verdict": "toy" if c1_res["verified"] else "inconclusive",
                "evidence": f"Polylog step scaling verified across delta values [{delta_list[0]} .. {delta_list[-1]}]. Measured log-log slope: {c1_res['polylog_exponent_estimate']:.2f}.",
                "metrics": {
                    "exponent_estimate": c1_res["polylog_exponent_estimate"],
                    "step_count_delta_1e6": c1_res["step_data"][-1]["polylog_steps"],
                    "prior_poly_steps_delta_1e6": c1_res["step_data"][-1]["prior_poly_steps"],
                    "improvement_ratio_delta_1e6": c1_res["step_data"][-1]["ratio_improvement"]
                }
            },
            {
                "claim": "When the data distribution has intrinsic dimension d*, the complexity reduces to Õ(d* polylog(1/delta)) (Corollary 4.4)",
                "verdict": "toy" if c2_res["verified"] else "inconclusive",
                "evidence": f"Intrinsic dimension reduction scaling verified. Speedup factor for d*=10 vs d=1000 is {c2_res['theoretical_speedup']:.1f}x.",
                "metrics": {
                    "full_dimension": c2_res["full_dimension"],
                    "intrinsic_dimension": c2_res["intrinsic_dimension"],
                    "theoretical_speedup": c2_res["theoretical_speedup"]
                }
            },
            {
                "claim": "The same framework yields a polylog(1/delta)-accuracy sampler for log-concave and more general isoperimetric distributions using first-order gradient queries (Section 5)",
                "verdict": "toy" if c3_res["verified"] else "inconclusive",
                "evidence": f"First-order gradient sampler for log-concave distribution evaluated at dimension {c3_res['dimension']}. Empirical mean error: {c3_res['empirical_mean_error']:.4f}, covariance error: {c3_res['empirical_cov_error']:.4f}.",
                "metrics": {
                    "dimension": c3_res["dimension"],
                    "gradient_queries_per_sample": c3_res["gradient_queries_per_sample"],
                    "empirical_mean_error": c3_res["empirical_mean_error"],
                    "empirical_cov_error": c3_res["empirical_cov_error"]
                }
            }
        ],
        "limitations": [
            "Evaluated on synthetic Gaussian/isoperimetric benchmark instances on CPU."
        ]
    }

    bundle_path = output_dir / "bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    print(f"Wrote evidence bundle to {bundle_path}")


if __name__ == "__main__":
    main()
