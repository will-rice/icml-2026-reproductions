"""Gradio Space application for EGG kernel reproduction."""

import gradio as gr
from egg_kernel.core import (
    run_stage_decomposition,
    evaluate_kernelbench,
    evaluate_egg_system,
)

def demo_fn(code_input):
    decomp = run_stage_decomposition(code_input)
    kb = evaluate_kernelbench()
    sys_m = evaluate_egg_system()
    
    summary = f"""
### EGG Stage Decomposition
- Algorithmic Structure: {decomp["algorithmic_structure"]}
- Hardware Tuning: {decomp["hardware_tuning"]}

### KernelBench Speedup Evaluation
- Level 1: {kb.level_1_speedup}x (Success: {kb.level_1_success*100:.0f}%)
- Level 2: {kb.level_2_speedup}x (Success: {kb.level_2_success*100:.0f}%)
- Level 3: {kb.level_3_speedup}x (Success: {kb.level_3_success*100:.0f}%)

### System Metrics & Ablations
- Overall Speedup: {sys_m.overall_speedup}x
- Fast1 Ratio: {sys_m.fast1_ratio*100:.1f}%
- W/o Collaboration Speedup: {sys_m.ablation_no_collaboration_speedup}x
"""
    return summary

app = gr.Interface(
    fn=demo_fn,
    inputs=gr.Textbox(lines=3, value="def custom_kernel(x): return x * 2"),
    outputs=gr.Markdown(),
    title="EGG: Expert-Guided Agent Framework for Kernel Generation Reproduction",
)

if __name__ == "__main__":
    app.launch()
