"""Streamlit web application for RelayCaching reproduction demonstration."""

import json
from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="RelayCaching ICML 2026 Reproduction",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ RelayCaching: Accelerating LLM Collaboration via Decoding KV Cache Reuse")
st.caption("ICML 2026 Reproduction | Paper ID: 1tbhBSXcyX | arXiv: 2603.13289")

st.markdown("""
### Overview
Multi-agent LLM collaboration often suffers from high Time-To-First-Token (TTFT) due to redundant prefill computation.
**RelayCaching** reuses decoding-phase KV caches during downstream prefilling and rectifies only localized deviations at critical layer and token positions.
""")

evidence_path = Path(__file__).parent / "evidence.json"
if evidence_path.exists():
    with open(evidence_path, "r", encoding="utf-8") as f:
        evidence = json.load(f)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Macro Alignment", f"{evidence['macro_alignment_similarity']:.4f}")
    col2.metric("Avg KV Cache Reuse", f"{evidence['average_kv_cache_reuse_rate']*100:.1f}%")
    col3.metric("Max Per-Agent Speedup", f"{evidence['max_per_agent_ttft_speedup']:.1f}x")
    col4.metric("VS KVCOMM Speedup", f"{evidence['cumulative_context_benchmark']['avg_speedup_vs_kvcomm']:.1f}x")

    st.subheader("Verified Reproduction Claims")
    for claim_key, claim_info in evidence["claim_verifications"].items():
        st.success(f"**{claim_key.replace('_', ' ').title()}**: {claim_info['details']}")

    st.subheader("Multi-Agent Workflows")
    st.json(evidence["workflows"])

    st.subheader("Ablation Study")
    st.json(evidence["ablation_study"])
else:
    st.warning("Evidence file not found. Run generate_evidence.py to compute metrics.")
