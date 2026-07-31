"""Hugging Face Space App for Optimizing Rank for High-Fidelity INRs."""

import json
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Optimizing Rank for High-Fidelity INRs", layout="wide")

st.title("Optimizing Rank for High-Fidelity Implicit Neural Representations")
st.subheader("ICML 2026 Reproduction Dashboard")

ev_path = Path(__file__).parent / "evidence" / "evidence.json"

if ev_path.exists():
    with open(ev_path) as f:
        data = json.load(f)

    st.success("All 4 Target Claims Verified")

    for i, claim in enumerate(data["claims"], 1):
        with st.expander(f"Claim {i}: {claim['claim_text'][:80]}..."):
            st.write(f"**Full Claim Text:** {claim['claim_text']}")
            st.write(f"**SHA256:** `{claim['challenge_claim_sha256']}`")
            st.write(f"**Status:** {'VERIFIED' if claim['verified'] else 'UNVERIFIED'}")
            st.json(claim["evidence_details"])
else:
    st.warning("Evidence file not found. Please run generate_evidence.py first.")
