"""Hugging Face Space Streamlit application for Chain-of-Thought Reasoning In The Wild Is Not Always Faithful."""

import json
from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="CoT Reasoning Unfaithfulness Reproduction",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Chain-of-Thought Reasoning In The Wild Is Not Always Faithful")
st.subheader("ICML 2026 Reproduction Challenge — Paper ID: NUyt4uxzx0")

# Load evidence
ROOT_DIR = Path(__file__).parent
EVIDENCE_PATH = ROOT_DIR / "evidence.json"

if EVIDENCE_PATH.exists():
    with open(EVIDENCE_PATH, "r", encoding="utf-8") as f:
        evidence = json.load(f)
else:
    evidence = {}

# Sidebar navigation
st.sidebar.header("Navigation")
page_selection = st.sidebar.radio(
    "Select Logbook Page:",
    [
        "01: Overview & Summary",
        "02: IPHR Unfaithfulness Rates",
        "03: Qualitative Patterns",
        "04: Hard Math Shortcuts",
        "05: Restoration Errors",
        "Raw Evidence Data",
    ],
)

page_file_map = {
    "01: Overview & Summary": "pages/01_overview.md",
    "02: IPHR Unfaithfulness Rates": "pages/02_iphr_rates.md",
    "03: Qualitative Patterns": "pages/03_unfaithfulness_patterns.md",
    "04: Hard Math Shortcuts": "pages/04_hard_math_shortcuts.md",
    "05: Restoration Errors": "pages/05_restoration_errors.md",
}

if page_selection in page_file_map:
    md_path = ROOT_DIR / page_file_map[page_selection]
    if md_path.exists():
        with open(md_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.warning(f"Page file {md_path.name} not found.")
elif page_selection == "Raw Evidence Data":
    st.header("Verified Evidence JSON")
    st.json(evidence)

# Claim verification status cards
st.sidebar.markdown("---")
st.sidebar.header("Verified Claims Summary")
if "claims" in evidence:
    for i, claim in enumerate(evidence["claims"], 1):
        status_color = "🟢" if claim["status"] == "verified" else "🔴"
        st.sidebar.markdown(f"**Claim {i}**: {status_color} `{claim['status']}`")
