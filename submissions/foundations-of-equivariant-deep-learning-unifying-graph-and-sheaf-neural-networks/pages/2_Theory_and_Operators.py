import streamlit as st
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.sheaf import SheafLaplacian

st.set_page_config(page_title="Theory & Operators", page_icon="🧮", layout="wide")
st.title("Sheaf Laplacian Operator Verification")

st.latex(r"""
\Delta_{\mathcal{F}, \text{norm}} = D_{\mathcal{F}}^{-1/2} \Delta_{\mathcal{F}} D_{\mathcal{F}}^{-1/2}, \quad P_{\mathcal{F}} = I - \eta \Delta_{\mathcal{F}, \text{norm}}
""")

st.markdown("""
### Drop-in GCN Generalization
When the restriction maps $\mathcal{F}_{v \to e} = I_d$ are identity maps on every edge, 
the Sheaf Laplacian $\Delta_{\mathcal{F}}$ reduces exactly to the Kronecker product $L_G \otimes I_d$ 
of the standard graph Laplacian $L_G$ and feature identity matrix $I_d$.
""")

num_nodes = st.slider("Number of Nodes", 3, 10, 4)
feature_dim = st.slider("Feature Dimension (d)", 2, 8, 3)

X = torch.ones(num_nodes, feature_dim)
edge_index = torch.tensor([[i for i in range(num_nodes-1)], [i+1 for i in range(num_nodes-1)]], dtype=torch.long)

sheaf_op = SheafLaplacian(num_nodes, feature_dim, is_identity=True)
out = sheaf_op(X, edge_index)

st.write("Diffused Output Tensor shape:", out.shape)
st.dataframe(out.numpy())
