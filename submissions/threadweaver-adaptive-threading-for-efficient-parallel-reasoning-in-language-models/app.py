"""Streamlit report for the ThreadWeaver evidence bundle."""

import json
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
EVIDENCE_PATH = ROOT / "evidence" / "bundle.json"

st.set_page_config(page_title="ThreadWeaver Reproduction", layout="wide")
st.title("ThreadWeaver Reproduction Evidence")
st.caption("ICML 2026 Agent Repro Challenge | Paper ID: Efq2VvYk1o")

if not EVIDENCE_PATH.exists():
    st.error("Evidence bundle is missing. Run generate_evidence.py first.")
    st.stop()

evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
upstream = evidence["upstream"]

cols = st.columns(4)
cols[0].metric("Upstream pin", upstream["git_revision"][:8])
cols[1].metric("API cost", f"${evidence['costs']['metered_api_usd']:.2f}")
cols[2].metric("GPU hours", f"{evidence['costs']['gpu_hours']:.1f}")
cols[3].metric("Claims", len(evidence["claims"]))

st.subheader("Claim Outcomes")
for claim in evidence["claims"]:
    st.markdown(f"**{claim['claim_id']}**: `{claim['status']}`")
    st.write(claim["text"])
    for item in claim["evidence"]:
        st.write(f"- {item}")

st.subheader("Toy Checks")
st.json(evidence["toy_checks"])

st.subheader("Artifact Audit")
st.json(
    {
        "revision_matches": upstream["revision_matches"],
        "required_paths": upstream["required_paths"],
        "markers": upstream["markers"],
    }
)
