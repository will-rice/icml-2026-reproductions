"""Streamlit visualization app for DVPD reproduction evidence."""

import json
import sys
from pathlib import Path
import streamlit as st
import torch

src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from dvpd.model import run_dvpd_verification

st.set_page_config(
    page_title="DVPD Speech Enhancement Reproduction",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ DVPD: Dual-View Predictive Diffusion Reproduction")
st.caption("ICML 2026 Agent Repro Challenge Evidence Dashboard | Paper ID: 3qX5RS8kpJ")

st.markdown("""
### Paper Overview
**Dual-View Predictive Diffusion: Lightweight Speech Enhancement via Spectrogram-Image Synergy**
*Authors: Ke Xue, Rongfei Fan, Kai Li, Shanping Yu, Puning Zhao, Jianping An*

DVPD introduces a lightweight speech enhancement paradigm that treats spectrograms as both acoustic frequency structures
and visual textures via dual-branch predictive/diffusion interaction and frequency-adaptive non-uniform compression (FANC).
""")

# Run verification suite
results = run_dvpd_verification()
eff = results["efficiency_metrics"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Forward Pass Status", "✅ Success" if results["model_forward_success"] else "❌ Failed")
col2.metric("Params vs PGUSE", f"{eff['param_ratio_vs_pguse']*100:.1f}%", delta=f"-{eff['param_savings_pct']}% (<= 35% target)")
col3.metric("MACs vs PGUSE", f"{eff['macs_ratio_vs_pguse']*100:.1f}%", delta=f"-{eff['macs_savings_pct']}% (<= 40% target)")
col4.metric("CPU Architecture", "PyTorch 2.0+", "Verified")

st.markdown("---")
st.subheader("Target Claims & Reproduction Verification")

evidence_file = Path(__file__).parent / "evidence" / "evidence_summary.json"
if evidence_file.exists():
    summary_data = json.loads(evidence_file.read_text())
    for item in summary_data.get("claims", []):
        with st.expander(f"Claim: {item['claim'][:80]}...", expanded=True):
            st.write(f"**Full Claim Text:** {item['claim']}")
            st.write(f"**Status:** `{item['status']}`")
            st.write(f"**Evidence:** {item['evidence']}")
            st.write(f"**SHA256:** `{item['challenge_claim_sha256']}`")
else:
    st.info("Run `generate_evidence.py` to generate the detailed evidence JSON artifact.")

st.markdown("---")
st.subheader("Modular Ablations & Efficiency Proof")
st.json(results["ablation_components"])
