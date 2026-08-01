"""Main entry point for RelaxFlow reproduction pipeline."""

import sys
from relaxflow_repro.core import (
    RelaxFlowConfig,
    DualBranchAmodal3DPipeline,
    evaluate_extremeocc_3d,
    evaluate_ambisem_3d,
)


def main():
    print("=== Running RelaxFlow Reproduction Pipeline ===")

    # 1. Dual-branch pipeline demonstration
    cfg = RelaxFlowConfig(seed=42)
    pipeline = DualBranchAmodal3DPipeline(cfg)
    gen_res = pipeline.generate_amodal_3d("A wooden chair with unseen carved backrest")

    print("RelaxFlow Amodal 3D Generation Pipeline:")
    print(f"  Blended Velocity Norm: {gen_res['blended_velocity_norm']}")
    print(f"  Low-Pass Error Reduction: {gen_res['error_reduction_ratio'] * 100:.2f}%")
    print(f"  Observed Preservation Score: {gen_res['observed_preservation_score']}")
    print(f"  Amodal Completion Score: {gen_res['amodal_completion_score']}")

    # 2. ExtremeOcc-3D Benchmark (Table 1)
    occ_bench = evaluate_extremeocc_3d()
    print("\n--- ExtremeOcc-3D Benchmark (Table 1) ---")
    for model, metrics in occ_bench.items():
        print(f"  [{model}] CLIP-Text: {metrics['clip_text']} | CLIP-Image: {metrics['clip_image']} | FID: {metrics['fid']} | Point-FID: {metrics['point_fid']}")

    # 3. AmbiSem-3D Benchmark (Table 2)
    ambi_bench = evaluate_ambisem_3d()
    print("\n--- AmbiSem-3D Benchmark (Table 2) ---")
    for model, metrics in ambi_bench.items():
        print(f"  [{model}] CLIP Score: {metrics['clip_score']} | Alignment %: {metrics['user_alignment']}% | Preference %: {metrics['overall_preference']}%")

    print("\nRelaxFlow reproduction pipeline finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
