import streamlit as st
import json
from pathlib import Path

st.set_page_config(page_title="Overview - Sheaf NNs", page_icon="📄", layout="wide")
st.title("Paper Overview & Target Claims")

st.markdown("""
### Paper Details
- **Title:** Foundations of Equivariant Deep Learning: Unifying Graph and Sheaf Neural Networks
- **Authors:** Yoshihiro Maruyama
- **Paper ID:** `aIH1jyU37z`
- **Upstream Pin:** `arxiv:2012.06333v3+github:twitter-research/graph-neural-sheaves@57002ef2c2c0199d7990be10f0dfc8c83a54d658`

### Evaluated Reproduction Claims
1. **Sheaf-Laplacian Diffusion (Section 3):** Encodes asymmetric, signed, and varying-dimensional relations via cellular sheaf restriction maps.
2. **Drop-in Generalization (Section 2.1):** When restriction maps are identity matrices, Sheaf Laplacian diffusion matches standard Kipf-Welling GCN diffusion.
3. **Signed Graph Node Classification (Figure 1):** SheafNN outperforms GCN variants across feature/edge noise regimes.
4. **Statistical Error Bars (Figure 1):** Results are averaged over 5 random graph trials with standard deviations.
""")

evidence_file = Path(__file__).parent.parent / "evidence" / "evidence.json"
if evidence_file.exists():
    with open(evidence_file, "r") as f:
        evidence = json.load(f)
    st.subheader("Claims Verification Table")
    st.table(evidence["claims"])
