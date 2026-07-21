"""
Temporal Analysis Page

How prediction quality changes over trajectory progression.
"""

from __future__ import annotations

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from next_action_pred_eval.dashboard.data_loader import (
    ExperimentData,
    categorize_op,
    op_type_from_symbolic,
)


def temporal_analysis_page(experiment: ExperimentData):
    st.header("Temporal Analysis")

    if not experiment.trajectory_labels:
        st.warning("No trajectory data found.")
        return

    # ── Rolling quality metrics ────────────────────────────────────
    st.subheader("Rolling Quality Metrics")
    _render_rolling_metrics(experiment)

    st.divider()

    # ── Quality by trajectory phase ────────────────────────────────
    st.subheader("Quality by Trajectory Phase")
    _render_phase_analysis(experiment)

    st.divider()

    # ── Prediction density ─────────────────────────────────────────
    st.subheader("Prediction Density")
    _render_prediction_density(experiment)

    st.divider()

    # ── Last context operation analysis ────────────────────────────
    st.subheader("Last Context Operation vs Acceptance")
    _render_last_context_op(experiment)


def _render_rolling_metrics(experiment: ExperimentData):
    selected = st.selectbox(
        "Trajectory", experiment.trajectory_labels, key="temp_traj"
    )
    if not selected:
        return

    traj = experiment.load_trajectory(selected)
    pred_df = traj.load_predictions()
    if pred_df.empty or len(pred_df) < 2:
        st.info("Not enough predictions for rolling analysis.")
        return

    window = st.slider("Rolling window size", 2, max(3, len(pred_df) // 2), min(5, len(pred_df)), key="temp_window")

    fig = go.Figure()

    if "eval_final_state_precision" in pred_df.columns:
        rolling_prec = pred_df["eval_final_state_precision"].rolling(window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=pred_df.get("prediction_index", pred_df.index),
            y=rolling_prec,
            name="Precision (rolling)",
            line=dict(color="#3b82f6"),
        ))

    if "eval_final_state_recall" in pred_df.columns:
        rolling_rec = pred_df["eval_final_state_recall"].rolling(window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=pred_df.get("prediction_index", pred_df.index),
            y=rolling_rec,
            name="Recall (rolling)",
            line=dict(color="#22c55e"),
        ))

    if "accepted" in pred_df.columns:
        rolling_acc = pred_df["accepted"].astype(float).rolling(window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=pred_df.get("prediction_index", pred_df.index),
            y=rolling_acc,
            name="Acceptance Rate (rolling)",
            line=dict(color="#f59e0b", dash="dash"),
        ))

    if "eval_final_state_ops_saved" in pred_df.columns:
        rolling_ops = pred_df["eval_final_state_ops_saved"].rolling(window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=pred_df.get("prediction_index", pred_df.index),
            y=rolling_ops,
            name="Ops Saved (rolling)",
            line=dict(color="#8b5cf6", dash="dot"),
            yaxis="y2",
        ))

    fig.update_layout(
        height=450,
        margin=dict(t=20, b=30),
        xaxis_title="Prediction Index",
        yaxis_title="Rate / Precision",
        yaxis2=dict(title="Ops Saved", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_phase_analysis(experiment: ExperimentData):
    """Split trajectories into early/mid/late phases and compare quality."""
    all_rows = []

    for label in experiment.trajectory_labels:
        traj = experiment.load_trajectory(label)
        pred_df = traj.load_predictions()
        if pred_df.empty:
            continue

        total_steps = traj.summary.get("initial_sequence_length", 1)
        if total_steps <= 0:
            continue

        for _, row in pred_df.iterrows():
            step_t = row.get("step_t", 0)
            progress = step_t / total_steps
            if progress < 0.33:
                phase = "Early (0-33%)"
            elif progress < 0.66:
                phase = "Mid (33-66%)"
            else:
                phase = "Late (66-100%)"

            all_rows.append({
                "phase": phase,
                "precision": row.get("eval_final_state_precision", 0) or 0,
                "recall": row.get("eval_final_state_recall", 0) or 0,
                "accepted": 1 if row.get("accepted") else 0,
                "ops_saved": row.get("eval_final_state_ops_saved", 0) or 0,
            })

    if not all_rows:
        st.info("No per-prediction data available.")
        return

    phase_df = pd.DataFrame(all_rows)
    agg = phase_df.groupby("phase").agg(
        mean_precision=("precision", "mean"),
        mean_recall=("recall", "mean"),
        acceptance_rate=("accepted", "mean"),
        mean_ops_saved=("ops_saved", "mean"),
        count=("precision", "count"),
    ).reindex(["Early (0-33%)", "Mid (33-66%)", "Late (66-100%)"])

    # Table
    st.dataframe(
        agg.style.format({
            "mean_precision": "{:.3f}",
            "mean_recall": "{:.3f}",
            "acceptance_rate": "{:.1%}",
            "mean_ops_saved": "{:.1f}",
        }),
        use_container_width=True,
    )

    # Grouped bar chart
    fig = go.Figure()
    metrics_to_plot = [
        ("mean_precision", "Precision", "#3b82f6"),
        ("mean_recall", "Recall", "#22c55e"),
        ("acceptance_rate", "Accept Rate", "#f59e0b"),
    ]
    for col_name, display_name, color in metrics_to_plot:
        if col_name in agg.columns:
            fig.add_trace(go.Bar(
                name=display_name,
                x=agg.index,
                y=agg[col_name],
                marker_color=color,
            ))
    fig.update_layout(
        barmode="group",
        height=400,
        margin=dict(t=20, b=30),
        yaxis_title="Value",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_prediction_density(experiment: ExperimentData):
    """Histogram of where predictions occur in trajectories."""
    all_progress = []
    all_accepted = []

    for label in experiment.trajectory_labels:
        traj = experiment.load_trajectory(label)
        pred_df = traj.load_predictions()
        if pred_df.empty:
            continue

        total_steps = traj.summary.get("initial_sequence_length", 1)
        if total_steps <= 0:
            continue

        for _, row in pred_df.iterrows():
            step_t = row.get("step_t", 0)
            all_progress.append(step_t / total_steps)
            all_accepted.append("Accepted" if row.get("accepted") else "Rejected")

    if not all_progress:
        st.info("No prediction data available.")
        return

    density_df = pd.DataFrame({"progress": all_progress, "status": all_accepted})
    fig = px.histogram(
        density_df,
        x="progress",
        color="status",
        nbins=20,
        barmode="stack",
        color_discrete_map={"Accepted": "#22c55e", "Rejected": "#ef4444"},
        labels={"progress": "Trajectory Progress (0=start, 1=end)"},
        title="Prediction Density by Trajectory Progress",
    )
    fig.update_layout(height=400, margin=dict(t=40, b=30))
    st.plotly_chart(fig, use_container_width=True)


def _render_last_context_op(experiment: ExperimentData):
    """Acceptance rate by last operation category in the context before prediction.

    For each prediction, the last operation in history_tail is categorized.
    Shows which operation types as context lead to higher/lower acceptance.
    """
    cat_accepted: dict = {}  # category -> count of accepted
    cat_total: dict = {}     # category -> total count

    for label in experiment.trajectory_labels:
        traj = experiment.load_trajectory(label)
        timeline_df = traj.load_timeline()
        if timeline_df.empty:
            continue

        pred_events = timeline_df[timeline_df["event"] == "prediction"]
        for _, row in pred_events.iterrows():
            history_tail = row.get("history_tail")
            if not isinstance(history_tail, list) or not history_tail:
                continue

            last_op = history_tail[-1]
            op_type = op_type_from_symbolic(last_op)
            cat = categorize_op(op_type)
            if cat == "Other":
                continue

            cat_total[cat] = cat_total.get(cat, 0) + 1
            if row.get("accepted", False):
                cat_accepted[cat] = cat_accepted.get(cat, 0) + 1

    if not cat_total:
        st.info("No prediction context data available.")
        return

    rows = []
    for cat, total in cat_total.items():
        acc = cat_accepted.get(cat, 0)
        rows.append({
            "Category": cat,
            "Total": total,
            "Accepted": acc,
            "Acceptance Rate": acc / total * 100 if total > 0 else 0,
        })

    df = pd.DataFrame(rows).sort_values("Acceptance Rate", ascending=True)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=df["Category"],
            x=df["Acceptance Rate"],
            orientation="h",
            marker_color="#60a5fa",
            text=[f"{r:.1f}%" for r in df["Acceptance Rate"]],
            textposition="inside",
            textfont=dict(color="white", size=14),
        )
    )

    # Add n= labels outside bars
    for i, row in df.iterrows():
        fig.add_annotation(
            x=row["Acceptance Rate"] + 1,
            y=row["Category"],
            text=f"n={row['Total']:,}",
            showarrow=False,
            font=dict(size=12, color="#666"),
            xanchor="left",
        )

    fig.update_layout(
        height=max(300, 40 * len(df)),
        margin=dict(t=10, b=30, l=120, r=60),
        xaxis_title="Acceptance Rate (%)",
        xaxis=dict(range=[0, df["Acceptance Rate"].max() * 1.2]),
    )
    st.plotly_chart(fig, use_container_width=True)
