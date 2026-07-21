"""
Coverage Analysis Page

TP/FP/FN breakdown by property type, precision/recall scatter plots.
"""

from __future__ import annotations

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from next_action_pred_eval.dashboard.data_loader import (
    ExperimentData,
    categorize_op,
    op_type_from_symbolic,
    cell_count_from_symbolic,
)


def coverage_analysis_page(experiment: ExperimentData):
    st.header("Coverage Analysis")

    if not experiment.trajectory_labels:
        st.warning("No trajectory data found.")
        return

    # ── Aggregate coverage ─────────────────────────────────────────
    st.subheader("Aggregate Coverage")
    agg = experiment.experiment_summary.get("aggregate_metrics", {})
    batch_df = experiment.batch_summary

    _render_coverage_pie(experiment)

    st.divider()

    # ── Property breakdown ─────────────────────────────────────────
    st.subheader("Property-Level Breakdown")
    _render_property_breakdown(experiment)

    st.divider()

    # ── Precision/recall scatter ───────────────────────────────────
    st.subheader("Per-Prediction Precision vs Recall")
    _render_precision_recall_scatter(experiment)

    st.divider()

    # ── Prediction volume by size ──────────────────────────────────
    st.subheader("Acceptance Rate by Prediction Size")
    ps_mode = st.radio(
        "Counting mode",
        ["Range-wise (# operations)", "Cell-wise (# cells affected)"],
        horizontal=True,
        key="ps_count_mode",
    )
    _render_prediction_size_analysis(experiment, cell_wise=ps_mode.startswith("Cell"))

    st.divider()

    # ── Correctional operations ───────────────────────────────────
    st.subheader("Correctional Operations")
    _render_correctional_ops(experiment)

    st.divider()

    # ── Cumulative metrics ─────────────────────────────────────────
    st.subheader("Cumulative Metrics Over Trajectory")
    _render_cumulative_metrics(experiment)


def _render_coverage_pie(experiment: ExperimentData):
    """Aggregate TP/FP/FN pie chart across all trajectories."""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_mm = 0

    for label in experiment.trajectory_labels:
        traj = experiment.load_trajectory(label)
        cov = traj.summary.get("coverage", {})
        total_tp += cov.get("tp", 0)
        total_fp += cov.get("fp", 0)
        total_fn += cov.get("fn", 0)
        total_mm += cov.get("mm", 0)

    if total_tp + total_fp + total_fn + total_mm == 0:
        st.info("No coverage data available.")
        return

    labels = ["True Positive", "False Positive", "False Negative", "Mismatch"]
    values = [total_tp, total_fp, total_fn, total_mm]
    colors = ["#22c55e", "#ef4444", "#f59e0b", "#8b5cf6"]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker=dict(colors=colors),
            textinfo="label+value+percent",
        )
    )
    fig.update_layout(height=350, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)


def _render_property_breakdown(experiment: ExperimentData):
    """Aggregate property breakdown across trajectories."""
    agg_props: dict = {}

    for label in experiment.trajectory_labels:
        traj = experiment.load_trajectory(label)
        prop_bd = traj.summary.get("property_breakdown", {})
        for prop, stats in prop_bd.items():
            if prop not in agg_props:
                agg_props[prop] = {"tp": 0, "fp": 0, "fn": 0, "mm": 0}
            if isinstance(stats, dict):
                for k in ("tp", "fp", "fn", "mm"):
                    agg_props[prop][k] += stats.get(k, 0)

    if not agg_props:
        st.info("No property breakdown data available.")
        return

    rows = []
    for prop, stats in sorted(agg_props.items()):
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        mm = stats["mm"]
        total_pred = tp + fp + mm
        precision = tp / total_pred if total_pred > 0 else 0
        total_true = tp + fn + mm
        recall = tp / total_true if total_true > 0 else 0
        rows.append({
            "Property": prop,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Mismatch": mm,
            "Precision": precision,
            "Recall": recall,
        })

    prop_df = pd.DataFrame(rows)

    # Table
    st.dataframe(
        prop_df.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}"}, na_rep="–"),
        use_container_width=True,
    )

    # Stacked horizontal bar
    fig = go.Figure()
    fig.add_trace(go.Bar(name="TP", y=prop_df["Property"], x=prop_df["TP"], orientation="h", marker_color="#22c55e"))
    fig.add_trace(go.Bar(name="FP", y=prop_df["Property"], x=prop_df["FP"], orientation="h", marker_color="#ef4444"))
    fig.add_trace(go.Bar(name="FN", y=prop_df["Property"], x=prop_df["FN"], orientation="h", marker_color="#f59e0b"))
    fig.add_trace(go.Bar(name="MM", y=prop_df["Property"], x=prop_df["Mismatch"], orientation="h", marker_color="#8b5cf6"))
    fig.update_layout(barmode="stack", height=max(300, 30 * len(prop_df)), margin=dict(t=20, b=30, l=150))
    st.plotly_chart(fig, use_container_width=True)


def _render_precision_recall_scatter(experiment: ExperimentData):
    """Scatter of precision vs recall per prediction across all trajectories."""
    all_rows = []
    for label in experiment.trajectory_labels:
        traj = experiment.load_trajectory(label)
        pred_df = traj.load_predictions()
        if pred_df.empty:
            continue
        for _, row in pred_df.iterrows():
            prec = row.get("eval_final_state_precision")
            rec = row.get("eval_final_state_recall")
            if prec is not None and rec is not None:
                all_rows.append({
                    "precision": prec,
                    "recall": rec,
                    "accepted": row.get("accepted", False),
                    "predicted_count": row.get("predicted_count", 1),
                    "trajectory": label,
                })

    if not all_rows:
        st.info("No per-prediction data available.")
        return

    scatter_df = pd.DataFrame(all_rows)
    scatter_df["status"] = scatter_df["accepted"].map({True: "Accepted", False: "Rejected"})

    fig = px.scatter(
        scatter_df,
        x="precision",
        y="recall",
        color="status",
        size="predicted_count",
        hover_data=["trajectory"],
        color_discrete_map={"Accepted": "#22c55e", "Rejected": "#ef4444"},
        labels={"precision": "Precision", "recall": "Recall"},
    )
    fig.update_layout(height=450, margin=dict(t=20, b=30))
    st.plotly_chart(fig, use_container_width=True)


def _render_cumulative_metrics(experiment: ExperimentData):
    """Cumulative TP/FP/FN over predictions for a selected trajectory."""
    selected = st.selectbox(
        "Trajectory for cumulative view",
        experiment.trajectory_labels,
        key="cov_cum_traj",
    )
    if not selected:
        return

    traj = experiment.load_trajectory(selected)
    pred_df = traj.load_predictions()
    if pred_df.empty:
        st.info("No predictions.")
        return

    cum_tp, cum_fp, cum_fn, cum_mm = 0, 0, 0, 0
    rows = []
    for _, row in pred_df.iterrows():
        cum_tp += row.get("eval_final_state_tp", 0) or 0
        cum_fp += row.get("eval_final_state_fp", 0) or 0
        cum_fn += row.get("eval_final_state_fn", 0) or 0
        cum_mm += row.get("eval_final_state_mm", 0) or 0
        total_pred = cum_tp + cum_fp
        rows.append({
            "prediction_index": row.get("prediction_index", 0),
            "Cum TP": cum_tp,
            "Cum FP": cum_fp,
            "Cum FN": cum_fn,
            "Cum MM": cum_mm,
            "Cum Precision": cum_tp / total_pred if total_pred > 0 else 0,
        })

    cum_df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum_df["prediction_index"], y=cum_df["Cum TP"], name="TP", line=dict(color="#22c55e")))
    fig.add_trace(go.Scatter(x=cum_df["prediction_index"], y=cum_df["Cum FP"], name="FP", line=dict(color="#ef4444")))
    fig.add_trace(go.Scatter(x=cum_df["prediction_index"], y=cum_df["Cum FN"], name="FN", line=dict(color="#f59e0b")))
    fig.add_trace(go.Scatter(x=cum_df["prediction_index"], y=cum_df["Cum MM"], name="MM", line=dict(color="#8b5cf6")))
    fig.update_layout(
        height=400,
        margin=dict(t=20, b=30),
        xaxis_title="Prediction Index",
        yaxis_title="Cumulative Count",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_prediction_size_analysis(experiment: ExperimentData, cell_wise: bool = False):
    """Stacked bar: accepted vs rejected by prediction size, with acceptance rate labels.

    Args:
        cell_wise: If True, prediction size = total cells across ops (not op count).
    """
    import numpy as np

    if cell_wise:
        bins = [0, 1, 5, 10, 20, 50, 100, 10000]
        bin_labels = ["1", "2-5", "6-10", "11-20", "21-50", "51-100", "100+"]
    else:
        bins = [0, 1, 2, 3, 5, 10, 15, 100]
        bin_labels = ["1", "2", "3", "4-5", "6-10", "11-15", "15+"]

    accepted_counts = [0] * len(bin_labels)
    rejected_counts = [0] * len(bin_labels)

    for label in experiment.trajectory_labels:
        traj = experiment.load_trajectory(label)
        pred_df = traj.load_predictions()
        if pred_df.empty:
            continue

        for _, row in pred_df.iterrows():
            if cell_wise:
                # Sum cell counts across all predicted ops
                ops = row.get("predicted_ops", [])
                if not isinstance(ops, list):
                    continue
                count = sum(
                    cell_count_from_symbolic(op) for op in ops
                    if isinstance(op, str) and "|" in op
                )
            else:
                count = row.get("predicted_count", 0) or 0

            accepted = row.get("accepted", False)
            if count <= 0:
                continue

            for i in range(len(bins) - 1):
                if bins[i] < count <= bins[i + 1]:
                    if accepted:
                        accepted_counts[i] += 1
                    else:
                        rejected_counts[i] += 1
                    break

    if sum(accepted_counts) + sum(rejected_counts) == 0:
        st.info("No prediction size data available.")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Bar(name="Accepted", x=bin_labels, y=accepted_counts, marker_color="#22c55e")
    )
    fig.add_trace(
        go.Bar(name="Rejected", x=bin_labels, y=rejected_counts, marker_color="#ef4444")
    )

    # Add acceptance rate annotations
    for i in range(len(bin_labels)):
        total = accepted_counts[i] + rejected_counts[i]
        if total > 0:
            rate = accepted_counts[i] / total * 100
            fig.add_annotation(
                x=bin_labels[i],
                y=total,
                text=f"{rate:.0f}%",
                showarrow=False,
                yshift=12,
                font=dict(size=13, color="#333"),
            )

    fig.update_layout(
        barmode="stack",
        height=400,
        margin=dict(t=20, b=40),
        xaxis_title="Prediction Size (# cells)" if cell_wise else "Prediction Size (# operations)",
        yaxis_title="Number of Predictions",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_correctional_ops(experiment: ExperimentData):
    """Correctional (inverse) operations analysis.

    Left: scatter of inverse ops count vs acceptance rate per file.
    Right: bar chart of inverse ops by operation category.
    """
    from collections import defaultdict

    # Collect per-file data
    file_rows = []
    all_inverse_ops: list = []

    for label in experiment.trajectory_labels:
        traj = experiment.load_trajectory(label)
        summary = traj.summary
        inv_count = summary.get("inverse_ops_added", 0)
        acc_rate = summary.get("acceptance_rate", 0)
        if isinstance(acc_rate, (int, float)):
            file_rows.append({
                "file": label,
                "inverse_ops": inv_count,
                "acceptance_rate": acc_rate * 100,
            })

        # Collect individual inverse ops from predictions.jsonl future_if_accepted
        pred_df = traj.load_predictions()
        if not pred_df.empty:
            for _, row in pred_df.iterrows():
                undo = row.get("future_if_accepted") if "future_if_accepted" in pred_df.columns else None
                # future_if_accepted might be stored as a nested dict in the flattened df
                # Check the raw predictions.jsonl via the non-flattened path
            # Fall back to loading raw predictions for inverse ops
            import json
            preds_path = traj.trajectory_dir / "predictions.jsonl"
            if preds_path.exists():
                for line in open(preds_path, "r", encoding="utf-8"):
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    undo = rec.get("future_if_accepted", {})
                    if undo and isinstance(undo, dict):
                        inv_ops = undo.get("inverse_ops_added", [])
                        if isinstance(inv_ops, list):
                            all_inverse_ops.extend(inv_ops)

    col1, col2 = st.columns(2)

    # Left: scatter plot
    with col1:
        if file_rows:
            file_df = pd.DataFrame(file_rows)
            fig = px.scatter(
                file_df,
                x="acceptance_rate",
                y="inverse_ops",
                hover_data=["file"],
                labels={
                    "acceptance_rate": "Acceptance Rate (%)",
                    "inverse_ops": "Total Correctional Ops",
                },
                title="Correctional Ops vs Acceptance Rate",
            )
            fig.update_traces(marker=dict(size=10, color="#4A90A4", line=dict(color="#2C5F6E", width=1.5)))
            fig.update_layout(height=400, margin=dict(t=40, b=30))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No file-level data.")

    # Right: bar chart by category
    with col2:
        if all_inverse_ops:
            cat_counts = defaultdict(int)
            for op in all_inverse_ops:
                if isinstance(op, str) and "|" in op:
                    ot = op_type_from_symbolic(op)
                    cat = categorize_op(ot)
                    cat_counts[cat] += 1

            if cat_counts:
                cat_df = pd.DataFrame(
                    sorted(cat_counts.items(), key=lambda x: x[1], reverse=True),
                    columns=["Category", "Count"],
                )
                fig = px.bar(
                    cat_df,
                    y="Category",
                    x="Count",
                    orientation="h",
                    title="Correctional Ops by Category",
                    color_discrete_sequence=["#4A90A4"],
                )
                fig.update_layout(height=400, margin=dict(t=40, b=30, l=120))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No categorizable inverse ops.")
        else:
            st.info("No inverse operations found (may not be in online mode).")
