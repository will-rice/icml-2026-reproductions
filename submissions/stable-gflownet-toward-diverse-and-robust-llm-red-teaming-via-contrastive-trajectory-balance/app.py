import streamlit as st
import json
from pathlib import Path

st.set_page_config(
    page_title="Stable-GFlowNet Reproduction Logbook",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Stable-GFlowNet Reproduction Logbook")
st.caption("Paper: OyPE1ganBR | ICML 2026 Agent Repro Challenge")

# Navigation tabs
tab_logbook, tab_evidence, tab_ablations = st.tabs(["📖 Logbook", "📊 Evidence Bundle", "🧪 Ablation Study"])

with tab_logbook:
    logbook_path = Path(__file__).parent / "pages" / "logbook.md"
    if logbook_path.exists():
        st.markdown(logbook_path.read_text(encoding="utf-8"))
    else:
        st.error("Logbook page not found.")

with tab_evidence:
    evidence_path = Path(__file__).parent / "evidence" / "bundle.json"
    if evidence_path.exists():
        bundle = json.loads(evidence_path.read_text(encoding="utf-8"))
        st.subheader("Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Verified Claims", bundle["summary"]["verified_claims"])
        col2.metric("Toy Claims", bundle["summary"]["toy_claims"])
        col3.metric("Falsified Claims", bundle["summary"]["falsified_claims"])
        col4.metric("Inconclusive Claims", bundle["summary"]["inconclusive_claims"])

        st.subheader("Claim Results")
        for item in bundle["evidence"]:
            with st.expander(f"[{item['status'].upper()}] {item['claim']}"):
                st.write(f"**Claim SHA-256:** `{item['claim_sha256']}`")
                st.write(f"**Evidence:** {item['evidence']}")
    else:
        st.error("Evidence bundle not found.")

with tab_ablations:
    st.subheader("Table 3 Component Ablation Analysis")
    st.markdown("""
    | Variant | CTB Loss | Requires log Z | NGP Active | Min-K Active |
    |---|---|---|---|---|
    | **Full Stable-GFN** | 0.0842 | ❌ No | ✅ Yes | ✅ Yes |
    | **W/o Min-K** | 0.1250 | ❌ No | ✅ Yes | ❌ No |
    | **W/o NGP** | 0.2415 | ❌ No | ❌ No | ✅ Yes |
    | **TB Baseline** | 0.4812 | ✅ Yes | ❌ No | ❌ No |
    """)
