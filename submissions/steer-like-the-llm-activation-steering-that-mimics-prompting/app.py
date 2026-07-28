"""Hugging Face Space Gradio Application for Steer Like the LLM Reproduction."""

import gradio as gr
import json
import os
import torch
from steer_like_llm.evidence_bundle import run_evidence_pipeline

# Generate results bundle
bundle = run_evidence_pipeline(output_dir="results")

def get_summary_text():
    statuses = bundle["claim_statuses"]
    verified_count = sum(1 for c in statuses.values() if c["status"] == "verified")
    total_count = len(statuses)
    
    text = f"### Reproduction Summary: {verified_count}/{total_count} Claims Verified\n\n"
    text += f"**Paper ID:** `{bundle['paper_id']}`  \n"
    text += f"**Title:** {bundle['paper_title']}  \n"
    text += f"**Upstream Revision:** `{bundle['upstream_revision']}`  \n\n"
    
    text += "#### Verified Claims Status:\n"
    for claim_key, claim_info in statuses.items():
        icon = "✅" if claim_info["status"] == "verified" else "❌"
        text += f"- {icon} **{claim_key}**: {claim_info['evidence']}\n"
        
    return text

def get_table_1_json():
    return json.dumps(bundle["persona_vectors"]["table_1_coherence"], indent=2)

def get_table_3_json():
    return json.dumps(bundle["axbench"]["table_3_axbench"], indent=2)

def get_full_results_json():
    return json.dumps(bundle, indent=2)

with gr.Blocks(title="Steer Like the LLM Reproduction") as demo:
    gr.Markdown("# Steer Like the LLM: Activation Steering that Mimics Prompting")
    gr.Markdown("Independent Reproduction Evidence Space for ICML 2026 Agent Repro Challenge.")
    
    with gr.Tab("Summary & Claims"):
        gr.Markdown(get_summary_text())
        
    with gr.Tab("Table 1: Persona Vectors Coherence"):
        gr.Markdown("### Persona Vectors Benchmark Coherence Evaluation Across LLMs")
        gr.JSON(bundle["persona_vectors"]["table_1_coherence"])
        
    with gr.Tab("Table 3: AxBench Gemma Layer Subsets"):
        gr.Markdown("### AxBench Gemma Layer Subsets Baseline Comparison")
        gr.JSON(bundle["axbench"]["table_3_axbench"])
        
    with gr.Tab("Full Results JSON"):
        gr.JSON(bundle)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
