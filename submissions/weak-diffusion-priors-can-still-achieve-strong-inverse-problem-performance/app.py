import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from weak_diffusion_priors.theory import simulate_theorem_3_1_posterior_concentration
from weak_diffusion_priors.inverse_problem import evaluate_table_1_inverse_problem_baselines

st.set_page_config(page_title="Weak Diffusion Priors Reproduction", layout="wide")

st.title("Weak Diffusion Priors Can Still Achieve Strong Inverse-Problem Performance")
st.caption("ICML 2026 Paper Reproduction (fdkSA4F0lN / arXiv:2601.22443)")

st.markdown("""
This Space presents the empirical verification of two core claims from the paper:
1. **Claim 1 (Table 1)**: Weak diffusion priors match strong-prior baselines when measurements are highly informative ($m/n \\ge 0.75$).
2. **Claim 2 (Theorem 3.1)**: High-dimensional measurements make the Bayesian posterior concentrate near the true signal despite weak/mismatched priors.
""")

tab1, tab2 = st.tabs(["Theorem 3.1 Posterior Concentration", "Table 1 Inverse Problem Baselines"])

with tab1:
    st.header("Theorem 3.1 Posterior Concentration Simulation")
    col1, col2 = st.columns(2)
    with col1:
        n_dim = st.slider("Signal Dimension (n)", 64, 256, 128, step=32)
        noise_std = st.slider("Noise Standard Deviation", 0.01, 0.20, 0.05, step=0.01)
    with col2:
        seed = st.number_input("Random Seed", 1, 9999, 42)
    
    res = simulate_theorem_3_1_posterior_concentration(n_dim=n_dim, noise_std=noise_std, seed=seed)
    
    st.success(f"Theorem 3.1 Verified: **{res['theorem_3_1_verified']}**")

    ratios = [r["measurement_ratio"] for r in res["sweep_results"]]
    errors_weak = [r["reconstruction_error_weak"] for r in res["sweep_results"]]
    errors_true = [r["reconstruction_error_true"] for r in res["sweep_results"]]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ratios, errors_weak, "o-", label="Weak Prior Reconstruction Error", color="#e74c3c", linewidth=2)
    ax.plot(ratios, errors_true, "s--", label="True (Strong) Prior Reconstruction Error", color="#2ecc71", linewidth=2)
    ax.set_xlabel("Measurement Ratio (m / n)")
    ax.set_ylabel("Reconstruction Error (||x_hat - x*||)")
    ax.set_title("Posterior Concentration Under Measurement Scaling")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

with tab2:
    st.header("Table 1 Baseline Comparison")
    eval_res = evaluate_table_1_inverse_problem_baselines(signal_length=128, num_samples=30, seed=42)
    st.success(f"Claim 1 Verified: **{eval_res['claim_1_verified']}**")

    st.json(eval_res["table_1_metrics"])
