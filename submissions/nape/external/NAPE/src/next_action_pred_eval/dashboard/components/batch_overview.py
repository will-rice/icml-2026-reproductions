"""
Batch Overview Page

Summary table of all trajectories with KPIs and distribution charts.
"""

from __future__ import annotations

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from next_action_pred_eval.dashboard.data_loader import ExperimentData


def batch_overview_page(experiment: ExperimentData):
    st.header("Batch Overview")

    df = experiment.batch_summary
    if df is None or df.empty:
        st.warning("No batch_summary.csv found for this experiment.")
        return

    successful = df[df["status"] == "success"] if "status" in df.columns else df

    # ── KPI cards ──────────────────────────────────────────────────
    _render_kpi_cards(df, successful)

    st.divider()

    # ── Summary table ──────────────────────────────────────────────
    st.subheader("Trajectory Results")
    display_cols = [
        c
        for c in [
            "file_label",
            "status",
            "uas_pct",
            "acceptance_rate",
            "net_operations_saved",
            "predictions_attempted",
            "predictions_accepted",
            "empty_predictions",
            "errored_empty_predictions",
            "avg_precision",
            "coverage_pct_tp",
            "total_tokens",
            "total_time",
        ]
        if c in df.columns
    ]
    st.dataframe(
        df[display_cols].style.format(
            {
                "uas_pct": "{:.1%}",
                "acceptance_rate": "{:.1%}",
                "avg_precision": "{:.3f}",
                "coverage_pct_tp": "{:.3f}",
                "total_time": "{:.1f}s",
            },
            na_rep="–",
        ),
        use_container_width=True,
        height=min(400, 35 * len(df) + 38),
    )

    if successful.empty:
        return

    st.divider()

    # ── Distribution charts ────────────────────────────────────────
    st.subheader("Distributions")
    col1, col2 = st.columns(2)

    with col1:
        if "uas_pct" in successful.columns:
            fig = px.histogram(
                successful,
                x="uas_pct",
                nbins=20,
                title="UAS % Distribution",
                labels={"uas_pct": "User Actions Saved (%)"},
            )
            fig.update_layout(height=350, margin=dict(t=40, b=30))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "acceptance_rate" in successful.columns:
            fig = px.histogram(
                successful,
                x="acceptance_rate",
                nbins=20,
                title="Acceptance Rate Distribution",
                labels={"acceptance_rate": "Acceptance Rate"},
            )
            fig.update_layout(height=350, margin=dict(t=40, b=30))
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        if "initial_sequence_length" in successful.columns and "uas_pct" in successful.columns:
            fig = px.scatter(
                successful,
                x="initial_sequence_length",
                y="uas_pct",
                hover_data=["file_label"],
                title="UAS % vs Sequence Length",
                labels={
                    "initial_sequence_length": "Initial Sequence Length",
                    "uas_pct": "UAS %",
                },
            )
            fig.update_layout(height=350, margin=dict(t=40, b=30))
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        if "total_tokens" in successful.columns and "net_operations_saved" in successful.columns:
            fig = px.scatter(
                successful,
                x="total_tokens",
                y="net_operations_saved",
                hover_data=["file_label"],
                title="Ops Saved vs Tokens Used",
                labels={
                    "total_tokens": "Total Tokens",
                    "net_operations_saved": "Net Ops Saved",
                },
            )
            fig.update_layout(height=350, margin=dict(t=40, b=30))
            st.plotly_chart(fig, use_container_width=True)


# ── Helpers ───────────────────────────────────────────────────────


def _render_kpi_cards(df: pd.DataFrame, successful: pd.DataFrame):
    cols = st.columns(6)

    with cols[0]:
        total = len(df)
        ok = len(successful)
        st.metric("Trajectories", f"{ok}/{total}")

    with cols[1]:
        if "uas_pct" in successful.columns and not successful.empty:
            st.metric("Mean UAS %", f"{successful['uas_pct'].mean():.1%}")
        else:
            st.metric("Mean UAS %", "–")

    with cols[2]:
        if "acceptance_rate" in successful.columns and not successful.empty:
            st.metric("Mean Accept Rate", f"{successful['acceptance_rate'].mean():.1%}")
        else:
            st.metric("Mean Accept Rate", "–")

    with cols[4]:
        if "net_operations_saved" in successful.columns:
            st.metric("Total Ops Saved", f"{successful['net_operations_saved'].sum():,}")
        else:
            st.metric("Total Ops Saved", "–")

    with cols[5]:
        if "total_tokens" in successful.columns:
            st.metric("Total Tokens", f"{successful['total_tokens'].sum():,}")
        else:
            st.metric("Total Tokens", "–")

    with cols[6]:
        if "avg_precision" in successful.columns and not successful.empty:
            st.metric("Mean Precision", f"{successful['avg_precision'].mean():.3f}")
        else:
            st.metric("Mean Precision", "–")
