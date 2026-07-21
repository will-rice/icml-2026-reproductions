"""
Cost Analysis Page

Token usage, cost per operation saved, wasted cost, generation time.
"""

from __future__ import annotations

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from next_action_pred_eval.dashboard.data_loader import ExperimentData


def cost_analysis_page(experiment: ExperimentData):
    st.header("Cost Analysis")

    batch_df = experiment.batch_summary
    if batch_df is None or batch_df.empty:
        st.warning("No batch summary data.")
        return

    successful = batch_df[batch_df["status"] == "success"] if "status" in batch_df.columns else batch_df

    # ── Token KPIs ─────────────────────────────────────────────────
    _render_token_kpis(successful)

    st.divider()

    # ── Token usage per trajectory ─────────────────────────────────
    st.subheader("Token Usage per Trajectory")
    _render_token_bar(successful)

    st.divider()

    # ── Cost efficiency scatter ────────────────────────────────────
    st.subheader("Cost Efficiency")
    _render_efficiency_scatter(successful)

    st.divider()

    # ── Per-prediction token and timing analysis ───────────────────
    st.subheader("Per-Prediction Token & Timing Analysis")
    _render_per_prediction_analysis(experiment)


def _render_token_kpis(df: pd.DataFrame):
    cols = st.columns(6)

    total_input = df["input_tokens"].sum() if "input_tokens" in df.columns else 0
    total_output = df["output_tokens"].sum() if "output_tokens" in df.columns else 0
    total_tokens = df["total_tokens"].sum() if "total_tokens" in df.columns else 0
    total_ops_saved = df["net_operations_saved"].sum() if "net_operations_saved" in df.columns else 0
    total_preds = df["predictions_attempted"].sum() if "predictions_attempted" in df.columns else 0
    total_accepted = df["predictions_accepted"].sum() if "predictions_accepted" in df.columns else 0

    with cols[0]:
        st.metric("Total Tokens", f"{total_tokens:,}")
    with cols[1]:
        st.metric("Input / Output", f"{total_input:,} / {total_output:,}")
    with cols[2]:
        per_pred = total_tokens / total_preds if total_preds > 0 else 0
        st.metric("Tokens / Prediction", f"{per_pred:,.0f}")
    with cols[3]:
        per_accepted = total_tokens / total_accepted if total_accepted > 0 else 0
        st.metric("Tokens / Accepted", f"{per_accepted:,.0f}")
    with cols[4]:
        per_op = total_tokens / total_ops_saved if total_ops_saved > 0 else 0
        st.metric("Tokens / Op Saved", f"{per_op:,.0f}")
    with cols[5]:
        rejected_preds = total_preds - total_accepted
        wasted_pct = rejected_preds / total_preds if total_preds > 0 else 0
        st.metric("Wasted Predictions %", f"{wasted_pct:.1%}")


def _render_token_bar(df: pd.DataFrame):
    if "file_label" not in df.columns or "total_tokens" not in df.columns:
        st.info("Missing token data.")
        return

    sorted_df = df.sort_values("total_tokens", ascending=True)

    fig = go.Figure()

    if "input_tokens" in df.columns and "output_tokens" in df.columns:
        fig.add_trace(
            go.Bar(
                y=sorted_df["file_label"],
                x=sorted_df["input_tokens"],
                name="Input",
                orientation="h",
                marker_color="#3b82f6",
            )
        )
        fig.add_trace(
            go.Bar(
                y=sorted_df["file_label"],
                x=sorted_df["output_tokens"],
                name="Output",
                orientation="h",
                marker_color="#f59e0b",
            )
        )
        fig.update_layout(barmode="stack")
    else:
        fig.add_trace(
            go.Bar(
                y=sorted_df["file_label"],
                x=sorted_df["total_tokens"],
                name="Total",
                orientation="h",
            )
        )

    fig.update_layout(
        height=max(300, 25 * len(df)),
        margin=dict(t=20, b=30, l=120),
        xaxis_title="Tokens",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_efficiency_scatter(df: pd.DataFrame):
    if "total_tokens" not in df.columns or "net_operations_saved" not in df.columns:
        st.info("Missing data for efficiency analysis.")
        return

    fig = px.scatter(
        df,
        x="total_tokens",
        y="net_operations_saved",
        hover_data=["file_label"] if "file_label" in df.columns else None,
        labels={
            "total_tokens": "Total Tokens",
            "net_operations_saved": "Net Ops Saved",
        },
    )

    # Add efficiency reference lines
    if not df.empty:
        max_tokens = df["total_tokens"].max()
        if max_tokens > 0:
            for ratio_label, ratio in [("1 op / 1K tokens", 0.001), ("1 op / 5K tokens", 0.0002)]:
                fig.add_trace(
                    go.Scatter(
                        x=[0, max_tokens],
                        y=[0, max_tokens * ratio],
                        mode="lines",
                        line=dict(dash="dash", width=1),
                        name=ratio_label,
                        showlegend=True,
                    )
                )

    fig.update_layout(height=400, margin=dict(t=20, b=30))
    st.plotly_chart(fig, use_container_width=True)


def _render_per_prediction_analysis(experiment: ExperimentData):
    selected = st.selectbox(
        "Trajectory", experiment.trajectory_labels, key="cost_traj"
    )
    if not selected:
        return

    traj = experiment.load_trajectory(selected)
    pred_df = traj.load_predictions()
    if pred_df.empty:
        st.info("No predictions.")
        return

    col1, col2 = st.columns(2)

    with col1:
        if "tokens_total" in pred_df.columns:
            fig = px.line(
                pred_df,
                x="prediction_index" if "prediction_index" in pred_df.columns else pred_df.index,
                y="tokens_total",
                title="Tokens per Prediction",
                labels={"tokens_total": "Total Tokens", "prediction_index": "Prediction #"},
            )
            if "tokens_input" in pred_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=pred_df.get("prediction_index", pred_df.index),
                        y=pred_df["tokens_input"],
                        name="Input",
                        line=dict(dash="dot"),
                    )
                )
            fig.update_layout(height=350, margin=dict(t=40, b=30))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "generation_time_s" in pred_df.columns:
            fig = px.histogram(
                pred_df,
                x="generation_time_s",
                nbins=20,
                title="Generation Time Distribution",
                labels={"generation_time_s": "Time (seconds)"},
            )
            fig.update_layout(height=350, margin=dict(t=40, b=30))
            st.plotly_chart(fig, use_container_width=True)
