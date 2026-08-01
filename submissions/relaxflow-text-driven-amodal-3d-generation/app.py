"""Interactive Gradio Web UI for RelaxFlow Reproduction."""

import gradio as gr
from relaxflow_repro.core import (
    RelaxFlowConfig,
    DualBranchAmodal3DPipeline,
    evaluate_extremeocc_3d,
    evaluate_ambisem_3d,
)


def run_relaxflow_demo(prompt, alpha, cutoff):
    cfg = RelaxFlowConfig(
        velocity_blending_alpha=float(alpha),
        low_pass_cutoff=float(cutoff),
        seed=42,
    )
    pipeline = DualBranchAmodal3DPipeline(cfg)
    res = pipeline.generate_amodal_3d(prompt)

    output = (
        f"### RelaxFlow Generation Results\n"
        f"- **Prompt**: {prompt}\n"
        f"- **Blended Velocity Norm**: {res['blended_velocity_norm']:.4f}\n"
        f"- **Low-Pass Filter Error Reduction**: {res['error_reduction_ratio'] * 100:.2f}%\n"
        f"- **Observed Input Preservation Score**: {res['observed_preservation_score']:.4f}\n"
        f"- **Amodal Completion Score**: {res['amodal_completion_score']:.4f}\n"
    )
    return output


def run_extremeocc_demo():
    bench = evaluate_extremeocc_3d()
    table = []
    for model, metrics in bench.items():
        table.append([
            model,
            metrics["clip_text"],
            metrics["clip_image"],
            metrics["fid"],
            metrics["lpips"],
            metrics["point_fid"],
        ])
    return table


def run_ambisem_demo():
    bench = evaluate_ambisem_3d()
    table = []
    for model, metrics in bench.items():
        table.append([
            model,
            metrics["clip_score"],
            f"{metrics['user_alignment']}%",
            f"{metrics['user_fidelity']}%",
            f"{metrics['overall_preference']}%",
        ])
    return table


with gr.Blocks(title="RelaxFlow: Text-Driven Amodal 3D Generation") as demo:
    gr.Markdown("# RelaxFlow: Text-Driven Amodal 3D Generation")
    gr.Markdown("Reproduction suite for ICML 2026 Paper **UamxHbDR3p** (arXiv:2603.05425)")

    with gr.Tab("Dual-Branch Velocity Blending"):
        prompt_in = gr.Textbox(label="Text Prompt for Unseen-Region Completion", value="A wooden armchair with intricate carved backrest")
        with gr.Row():
            alpha_in = gr.Slider(minimum=0.1, maximum=0.9, value=0.65, step=0.05, label="Velocity Blending Alpha (Observation Weight)")
            cutoff_in = gr.Slider(minimum=0.05, maximum=0.5, value=0.25, step=0.05, label="Low-Pass Cutoff Frequency")
        run_btn = gr.Button("Generate Amodal 3D Simulation")
        demo_out = gr.Markdown()
        run_btn.click(run_relaxflow_demo, inputs=[prompt_in, alpha_in, cutoff_in], outputs=[demo_out])

    with gr.Tab("ExtremeOcc-3D Benchmark (Table 1)"):
        gr.Markdown("### Evaluation under Extreme Occlusion")
        occ_btn = gr.Button("Load ExtremeOcc-3D Metrics")
        occ_table = gr.Dataframe(
            headers=["Model", "CLIP-Text", "CLIP-Image", "FID (lower besser)", "LPIPS (lower better)", "Point-FID (lower better)"],
            datatype=["str", "number", "number", "number", "number", "number"],
        )
        occ_btn.click(run_extremeocc_demo, outputs=[occ_table])

    with gr.Tab("AmbiSem-3D Benchmark (Table 2)"):
        gr.Markdown("### Alignment & Preference under Semantic Ambiguity")
        ambi_btn = gr.Button("Load AmbiSem-3D Metrics")
        ambi_table = gr.Dataframe(
            headers=["Model", "CLIP Score", "User Alignment %", "3D Fidelity %", "Overall Preference %"],
            datatype=["str", "number", "str", "str", "str"],
        )
        ambi_btn.click(run_ambisem_demo, outputs=[ambi_table])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
