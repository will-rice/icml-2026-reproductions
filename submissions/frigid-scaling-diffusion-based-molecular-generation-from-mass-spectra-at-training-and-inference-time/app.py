from pathlib import Path

import streamlit as st

from src.frigid_repro import generate_evidence


st.set_page_config(page_title="FRIGID Reproduction", layout="wide")
bundle = generate_evidence()

st.title("FRIGID Reproduction")
st.caption("Paper wTgx7b2D9r - attempt 86bd82c3-48c0-4260-be38-045e8aa0fb29")

asset = Path(__file__).resolve().parent / "assets" / "fig1_model_overview.png"
if asset.exists():
    st.image(str(asset), caption="Pinned upstream FRIGID architecture figure")

st.subheader("Claim Outcomes")
for claim in bundle["claims"]:
    st.markdown(f"**Claim {claim['ordinal']} - {claim['local_outcome']}**")
    st.write(claim["text"])
    st.write(claim["reproduction_notes"])

st.subheader("Pinned Artifacts")
st.json(bundle["artifact_hashes"])
