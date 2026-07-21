"""
Experiment Comparison Page

Cross-experiment comparison with side-by-side metrics, box plots, and Pareto frontier.
"""

from __future__ import annotations

from typing import List

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from next_action_pred_eval.dashboard.data_loader import ExperimentData


def experiment_comparison_page(experiments: List[ExperimentData]):
    st.header("Experiment Comparison")

    if len(experiments) < 2:
        st.info(
            "Select at least 2 experiments to compare. "
            "Use the sidebar to enable comparison mode and add experiments."
        )
        # Still show single experiment summary
        if experiments:
            _render_single_summary(experiments[0])
        return

    # ── Aggregate metrics table ────────────────────────────────────
    st.subheader("Aggregate Metrics")
    _render_comparison_table(experiments)

    st.divider()

    # ── Box plots ──────────────────────────────────────────────────
    st.subheader("Distribution Comparison")
    _render_box_plots(experiments)

    st.divider()

    # ── Per-trajectory comparison ──────────────────────────────────
    st.subheader("Per-Trajectory Comparison")
    _render_per_trajectory_comparison(experiments)

    st.divider()

    # ── Pareto frontier ────────────────────────────────────────────
    st.subheader("Quality vs Cost Pareto")
    _render_pareto(experiments)


def _render_single_summary(experiment: ExperimentData):
    """Fallback: show summary for a single experiment."""
    st.subheader(f"Summary: {experiment.experiment_dir.name}")
    agg = experiment.experiment_summary.get("aggregate_metrics", {})
    if agg:
        cols = st.columns(4)
        with cols[0]:
            st.metric("Mean UAS %", f"{agg.get('mean_uas_pct', 0):.1%}")
        with cols[1]:
            st.metric("Accept Rate", f"{agg.get('overall_acceptance_rate', 0):.1%}")
        with cols[2]:
            st.metric("Total Ops Saved", f"{agg.get('total_net_ops_saved', 0):,}")
        with cols[3]:
            st.metric("Total Tokens", f"{agg.get('total_tokens', 0):,}")


def _render_comparison_table(experiments: List[ExperimentData]):
    rows = []
    for exp in experiments:
        agg = exp.experiment_summary.get("aggregate_metrics", {})
        counts = exp.experiment_summary.get("counts", {})
        rows.append({
            "Experiment": exp.experiment_dir.name,
            "Trajectories": counts.get("successful", 0),
            "Mean UAS %": agg.get("mean_uas_pct", 0),
            "Accept Rate": agg.get("overall_acceptance_rate", 0),
            "Mean Precision": agg.get("mean_avg_precision", 0),
            "Total Ops Saved": agg.get("total_net_ops_saved", 0),
            "Empty Preds": agg.get("total_empty_predictions", 0),
            "Errored Preds": agg.get("total_errored_empty_predictions", 0),
            "Total Tokens": agg.get("total_tokens", 0),
            "Efficiency": (
                agg.get("total_net_ops_saved", 0) / agg.get("total_tokens", 1)
                if agg.get("total_tokens", 0) > 0
                else 0
            ),
        })

    comp_df = pd.DataFrame(rows)
    st.dataframe(
        comp_df.style.format({
            "Mean UAS %": "{:.1%}",
            "Accept Rate": "{:.1%}",
            "Mean Precision": "{:.3f}",
            "Total Ops Saved": "{:,.0f}",
            "Empty Preds": "{:,.0f}",
            "Errored Preds": "{:,.0f}",
            "Total Tokens": "{:,.0f}",
            "Efficiency": "{:.6f}",
        }),
        use_container_width=True,
    )


def _render_box_plots(experiments: List[ExperimentData]):
    all_rows = []
    for exp in experiments:
        df = exp.batch_summary
        if df is None or df.empty:
            continue
        successful = df[df["status"] == "success"] if "status" in df.columns else df
        for _, row in successful.iterrows():
            all_rows.append({
                "experiment": exp.experiment_dir.name,
                "uas_pct": row.get("uas_pct", 0),
                "acceptance_rate": row.get("acceptance_rate", 0),
                "avg_precision": row.get("avg_precision", 0),
            })

    if not all_rows:
        st.info("No data for box plots.")
        return

    box_df = pd.DataFrame(all_rows)

    col1, col2, col3 = st.columns(3)

    with col1:
        fig = px.box(box_df, x="experiment", y="uas_pct", title="UAS %", color="experiment")
        fig.update_layout(height=350, margin=dict(t=40, b=80), showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(box_df, x="experiment", y="acceptance_rate", title="Acceptance Rate", color="experiment")
        fig.update_layout(height=350, margin=dict(t=40, b=80), showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        fig = px.box(box_df, x="experiment", y="avg_precision", title="Avg Precision", color="experiment")
        fig.update_layout(height=350, margin=dict(t=40, b=80), showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)


def _render_per_trajectory_comparison(experiments: List[ExperimentData]):
    """Grouped bar chart comparing UAS% per trajectory across experiments."""
    # Find common trajectories
    all_labels: dict = {}
    for exp in experiments:
        if exp.batch_summary is not None and "file_label" in exp.batch_summary.columns:
            for label in exp.batch_summary["file_label"]:
                all_labels[label] = all_labels.get(label, 0) + 1

    common = [l for l, count in all_labels.items() if count >= 2]
    if not common:
        st.info("No trajectories shared across experiments.")
        return

    # Show top N
    max_show = st.slider("Max trajectories to show", 5, min(50, len(common)), min(20, len(common)), key="comp_max")
    common = common[:max_show]

    rows = []
    for exp in experiments:
        df = exp.batch_summary
        if df is None:
            continue
        for label in common:
            match = df[df["file_label"] == label]
            if not match.empty:
                rows.append({
                    "trajectory": label,
                    "experiment": exp.experiment_dir.name,
                    "uas_pct": match.iloc[0].get("uas_pct", 0),
                })

    if not rows:
        return

    comp_df = pd.DataFrame(rows)
    fig = px.bar(
        comp_df,
        x="trajectory",
        y="uas_pct",
        color="experiment",
        barmode="group",
        labels={"uas_pct": "UAS %", "trajectory": "Trajectory"},
    )
    fig.update_layout(
        height=max(350, 20 * len(common)),
        margin=dict(t=20, b=80),
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_pareto(experiments: List[ExperimentData]):
    """Quality vs cost Pareto frontier."""
    points = []
    for exp in experiments:
        agg = exp.experiment_summary.get("aggregate_metrics", {})
        uas = agg.get("mean_uas_pct", 0)
        tokens = agg.get("total_tokens", 0)
        if tokens > 0:
            points.append({
                "experiment": exp.experiment_dir.name,
                "Mean UAS %": uas,
                "Total Tokens": tokens,
            })

    if len(points) < 2:
        st.info("Need at least 2 experiments with token data for Pareto analysis.")
        return

    pareto_df = pd.DataFrame(points)
    fig = px.scatter(
        pareto_df,
        x="Total Tokens",
        y="Mean UAS %",
        text="experiment",
        labels={"Total Tokens": "Total Tokens (cost)", "Mean UAS %": "Mean UAS % (quality)"},
    )
    fig.update_traces(textposition="top center", marker=dict(size=12))
    fig.update_layout(height=450, margin=dict(t=20, b=30))
    st.plotly_chart(fig, use_container_width=True)
