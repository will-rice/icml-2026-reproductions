import streamlit as st
import json
from pathlib import Path

st.set_page_config(
    page_title="Foundations of Equivariant Deep Learning: Sheaf NNs",
    page_icon="🕸️",
    layout="wide"
)

st.title("Foundations of Equivariant Deep Learning: Unifying Graph and Sheaf Neural Networks")
st.subheader("ICML 2026 Reproduction Dashboard")

st.markdown("""
This interactive Hugging Face Space presents the empirical and theoretical reproduction of
**Foundations of Equivariant Deep Learning: Unifying Graph and Sheaf Neural Networks** (Paper ID: `aIH1jyU37z`).

### Key Reproduction Findings
- **Generalization (Section 2.1 & 3):** Sheaf Laplacians generalize standard Graph Laplacian diffusion to signed, asymmetric, and varying-dimensional restriction maps.
- **Node Classification Benchmark (Figure 1):** Sheaf Neural Networks consistently outperform standard Kipf-Welling GCNs on synthetic signed graphs across feature and edge noise regimes.
- **Statistical Rigor (Figure 1):** Evaluated over 5 random graph initialization trials with exact mean and standard deviation reporting.

Use the sidebar navigation to explore the theoretical formulation, operator verification, and benchmark noise regime results.
""")

evidence_file = Path(__file__).parent / "evidence" / "evidence.json"
if evidence_file.exists():
    with open(evidence_file, "r") as f:
        data = json.load(f)
    st.sidebar.success(f"Verified Claims: {data['summary']['num_claims_verified']}/{data['summary']['num_claims_evaluated']}")
    st.json(data["summary"])
