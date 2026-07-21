"""
Operation Breakdown Page

Operation type distribution charts: initial vs final, predicted vs accepted vs rejected.
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

import numpy as np


def operation_breakdown_page(experiment: ExperimentData):
    st.header("Operation Breakdown")

    if not experiment.trajectory_labels:
        st.warning("No trajectory data found.")
        return

    # Allow single trajectory or aggregate view
    view_mode = st.radio(
        "View", ["Aggregate (all trajectories)", "Single trajectory"],
        horizontal=True,
    )

    if view_mode.startswith("Single"):
        selected = st.selectbox("Trajectory", experiment.trajectory_labels)
        if not selected:
            return
        traj = experiment.load_trajectory(selected)
        summary = traj.summary
        _render_single_breakdown(summary)
    else:
        _render_aggregate_breakdown(experiment)

    st.divider()

    # ── Category frequency vs acceptance heatmap ───────────────────
    st.subheader("Category Frequency vs Acceptance Rate")
    count_mode = st.radio(
        "Counting mode",
        ["Range-wise (1 op = 1)", "Cell-wise (1 op = N cells)"],
        horizontal=True,
        key="ob_count_mode",
        help="Range-wise: each operation counts as 1. Cell-wise: FONT_BOLD on A1:A10 counts as 10.",
    )
    cell_wise = count_mode.startswith("Cell")
    _render_category_frequency_heatmap(experiment, cell_wise=cell_wise)


def _render_single_breakdown(summary: dict):
    """Show operation breakdown for a single trajectory."""
    op_breakdown = summary.get("operation_breakdown", {})

    # Initial vs final
    initial = op_breakdown.get("initial", {})
    final = op_breakdown.get("final", {})
    predicted = op_breakdown.get("predicted", {})
    accepted = op_breakdown.get("accepted", {})
    rejected = op_breakdown.get("rejected", {})

    if not initial and not predicted:
        st.info("No operation breakdown data in experiment summary.")
        return

    # ── Grouped bar chart ──────────────────────────────────────────
    st.subheader("Operation Distribution by Sequence")
    all_op_types = sorted(
        set(list(initial) + list(final) + list(predicted) + list(accepted) + list(rejected))
    )

    if all_op_types:
        fig = go.Figure()
        for name, data, color in [
            ("Initial", initial, "#3b82f6"),
            ("Final", final, "#8b5cf6"),
            ("Predicted", predicted, "#f59e0b"),
            ("Accepted", accepted, "#22c55e"),
            ("Rejected", rejected, "#ef4444"),
        ]:
            if data:
                fig.add_trace(
                    go.Bar(
                        name=name,
                        x=all_op_types,
                        y=[data.get(op, 0) for op in all_op_types],
                        marker_color=color,
                    )
                )
        fig.update_layout(
            barmode="group",
            height=450,
            margin=dict(t=30, b=80),
            xaxis_tickangle=-45,
            xaxis_title="Operation Type",
            yaxis_title="Count",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Pie charts ─────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        if initial:
            st.subheader("Initial Composition")
            fig = px.pie(
                names=list(initial.keys()),
                values=list(initial.values()),
                hole=0.3,
            )
            fig.update_layout(height=350, margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if accepted:
            st.subheader("Accepted Composition")
            fig = px.pie(
                names=list(accepted.keys()),
                values=list(accepted.values()),
                hole=0.3,
            )
            fig.update_layout(height=350, margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

    # ── Accepted vs Rejected stacked bar ───────────────────────────
    if accepted or rejected:
        st.subheader("Accepted vs Rejected by Type")
        all_types = sorted(set(list(accepted) + list(rejected)))
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="Accepted",
                x=all_types,
                y=[accepted.get(t, 0) for t in all_types],
                marker_color="#22c55e",
            )
        )
        fig.add_trace(
            go.Bar(
                name="Rejected",
                x=all_types,
                y=[rejected.get(t, 0) for t in all_types],
                marker_color="#ef4444",
            )
        )
        fig.update_layout(
            barmode="stack",
            height=400,
            margin=dict(t=30, b=80),
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_aggregate_breakdown(experiment: ExperimentData):
    """Aggregate operation breakdown across all trajectories."""
    agg_initial: dict = {}
    agg_predicted: dict = {}
    agg_accepted: dict = {}
    agg_rejected: dict = {}

    for label in experiment.trajectory_labels:
        traj = experiment.load_trajectory(label)
        op_bd = traj.summary.get("operation_breakdown", {})
        for src, target in [
            ("initial", agg_initial),
            ("predicted", agg_predicted),
            ("accepted", agg_accepted),
            ("rejected", agg_rejected),
        ]:
            for op_type, count in op_bd.get(src, {}).items():
                target[op_type] = target.get(op_type, 0) + count

    if not agg_initial and not agg_predicted:
        st.info("No operation breakdown data available across trajectories.")
        return

    all_types = sorted(
        set(list(agg_initial) + list(agg_predicted) + list(agg_accepted) + list(agg_rejected))
    )

    st.subheader("Aggregate Operation Distribution")
    fig = go.Figure()
    for name, data, color in [
        ("Initial", agg_initial, "#3b82f6"),
        ("Predicted", agg_predicted, "#f59e0b"),
        ("Accepted", agg_accepted, "#22c55e"),
        ("Rejected", agg_rejected, "#ef4444"),
    ]:
        if data:
            fig.add_trace(
                go.Bar(
                    name=name,
                    x=all_types,
                    y=[data.get(op, 0) for op in all_types],
                    marker_color=color,
                )
            )
    fig.update_layout(
        barmode="group",
        height=500,
        margin=dict(t=30, b=80),
        xaxis_tickangle=-45,
        xaxis_title="Operation Type",
        yaxis_title="Count (aggregate)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Acceptance rate per op type
    if agg_predicted:
        st.subheader("Acceptance Rate by Operation Type")
        rates = []
        for op in all_types:
            total = agg_predicted.get(op, 0)
            acc = agg_accepted.get(op, 0)
            rates.append({
                "Operation": op,
                "Predicted": total,
                "Accepted": acc,
                "Rate": acc / total if total > 0 else 0,
            })
        rate_df = pd.DataFrame(rates).sort_values("Rate", ascending=False)
        fig = px.bar(
            rate_df,
            x="Operation",
            y="Rate",
            title="Acceptance Rate per Operation Type",
            labels={"Rate": "Acceptance Rate"},
        )
        fig.update_layout(height=400, margin=dict(t=40, b=80), xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)


def _render_category_frequency_heatmap(experiment: ExperimentData, cell_wise: bool = False):
    """Heatmap: rows = op categories, cols = frequency bins, values = acceptance rate.

    For each prediction, count how many operations belong to each category.
    Then compute acceptance rate for each (category, frequency_bin) cell.
    Answers: "If a prediction has N border ops, how likely is it to be accepted?"

    Args:
        cell_wise: If True, count cells (A1:A10=10) instead of ops (A1:A10=1).
    """
    from collections import defaultdict

    categories_order = [
        "Input", "Border", "Font", "Fill", "Alignment",
        "Number Format", "Merge", "Paste",
    ]
    freq_bins = [0, 1, 2, 3, 5, 10, float("inf")]
    freq_labels = ["0", "1", "2", "3-5", "6-10", "10+"]

    # (category, bin_idx) -> {"accepted": int, "total": int}
    cells = defaultdict(lambda: {"accepted": 0, "total": 0})

    for label in experiment.trajectory_labels:
        traj = experiment.load_trajectory(label)
        pred_df = traj.load_predictions()
        if pred_df.empty or "predicted_ops" not in pred_df.columns:
            continue

        for _, row in pred_df.iterrows():
            ops = row.get("predicted_ops", [])
            if not isinstance(ops, list) or not ops:
                continue

            accepted = bool(row.get("accepted", False))

            # Count ops per category (range-wise: 1 per op; cell-wise: N per op)
            cat_counts = defaultdict(int)
            for op in ops:
                if isinstance(op, str) and "|" in op:
                    ot = op_type_from_symbolic(op)
                    cat = categorize_op(ot)
                    if cat != "Other":
                        weight = cell_count_from_symbolic(op) if cell_wise else 1
                        cat_counts[cat] += weight

            # Assign to bins for each category
            for cat in categories_order:
                count = cat_counts.get(cat, 0)
                for bi in range(len(freq_bins) - 1):
                    if freq_bins[bi] <= count < freq_bins[bi + 1]:
                        key = (cat, bi)
                        cells[key]["total"] += 1
                        if accepted:
                            cells[key]["accepted"] += 1
                        break

    if not cells:
        st.info("No per-prediction operation data available.")
        return

    # Build matrix
    matrix = []
    for cat in categories_order:
        row_vals = []
        for bi in range(len(freq_labels)):
            cell = cells.get((cat, bi), {"accepted": 0, "total": 0})
            if cell["total"] > 0:
                row_vals.append(cell["accepted"] / cell["total"] * 100)
            else:
                row_vals.append(None)
        matrix.append(row_vals)

    # Annotation text: show rate + count
    annot_text = []
    for ci, cat in enumerate(categories_order):
        row_texts = []
        for bi in range(len(freq_labels)):
            cell = cells.get((cat, bi), {"accepted": 0, "total": 0})
            if cell["total"] > 0:
                rate = cell["accepted"] / cell["total"] * 100
                row_texts.append(f"{rate:.0f}%")
            else:
                row_texts.append("")
        annot_text.append(row_texts)

    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=freq_labels,
            y=categories_order,
            colorscale="RdYlGn",
            zmin=0,
            zmax=70,
            text=annot_text,
            texttemplate="%{text}",
            textfont=dict(size=14),
            hovertemplate=(
                "Category: %{y}<br>"
                "# ops in prediction: %{x}<br>"
                "Acceptance rate: %{z:.1f}%"
                "<extra></extra>"
            ),
            colorbar=dict(title="Accept %"),
        )
    )

    fig.update_layout(
        height=max(350, 45 * len(categories_order)),
        margin=dict(t=10, b=40, l=120, r=20),
        xaxis_title="# Ops of Category in Prediction" + (" (cells)" if cell_wise else ""),
        yaxis_title="Operation Category",
        xaxis=dict(type="category"),
        yaxis=dict(type="category", autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)
