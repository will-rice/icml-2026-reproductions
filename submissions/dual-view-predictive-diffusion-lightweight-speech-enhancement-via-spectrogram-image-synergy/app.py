"""Streamlit visualization app for DVPD reproduction evidence."""

import json
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="DVPD Speech Enhancement Reproduction",
    page_icon="🎙️",
    layout="wide",
)

st.title("🎙️ DVPD: Dual-View Predictive Diffusion Reproduction")
st.caption("ICML 2026 Agent Repro Challenge Evidence Dashboard | Paper ID: 3qX5RS8kpJ")

report_file = Path(__file__).parent / "pages" / "report.md"
if report_file.exists():
    st.markdown(report_file.read_text())
else:
    st.warning("pages/report.md not found; run generate_evidence.py.")

st.markdown("---")
st.subheader("Claim-by-claim evidence")

evidence_file = Path(__file__).parent / "evidence" / "evidence_summary.json"
if evidence_file.exists():
    summary_data = json.loads(evidence_file.read_text())
    for item in summary_data.get("claims", []):
        with st.expander(f"[{item['status']}] {item['claim'][:90]}...", expanded=False):
            st.write(f"**Full Claim Text:** {item['claim']}")
            st.write(f"**Status:** `{item['status']}`")
            if "scope" in item:
                st.write(f"**Scope:** {item['scope']}")
            if "evidence" in item:
                st.json(item["evidence"])
            if "observation" in item:
                st.write(f"**Observation:** {item['observation']}")
            if "reason" in item:
                st.write(f"**Why unreplicated:** {item['reason']}")
            st.write(f"**SHA256:** `{item['challenge_claim_sha256']}`")
else:
    st.info("Run `generate_evidence.py` to generate the evidence JSON artifact.")
