"""
Trajectory Detail Page

Per-trajectory timeline visualization and prediction table.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from next_action_pred_eval.dashboard.data_loader import ExperimentData


def trajectory_detail_page(experiment: ExperimentData):
    st.header("Trajectory Detail")

    if not experiment.trajectory_labels:
        st.warning("No trajectory data found.")
        return

    selected_label = st.selectbox(
        "Select trajectory", experiment.trajectory_labels
    )
    if not selected_label:
        return

    traj = experiment.load_trajectory(selected_label)
    summary = traj.summary

    # ── KPI cards ──────────────────────────────────────────────────
    _render_trajectory_kpis(summary)

    st.divider()

    # ── Timeline visualization ─────────────────────────────────────
    st.subheader("Timeline")
    timeline_df = traj.load_timeline()
    if not timeline_df.empty:
        _render_timeline_chart(timeline_df)
    else:
        st.info("No timeline data available.")

    # ── Progress curve ────────────────────────────────────────────
    if not timeline_df.empty:
        _render_progress_curve(timeline_df, summary)

    st.divider()

    # ── Prediction table ───────────────────────────────────────────
    st.subheader("Predictions")
    pred_df = traj.load_predictions()
    if not pred_df.empty:
        _render_prediction_table(pred_df)
    else:
        st.info("No prediction data available.")

    # ── Operation type progression ─────────────────────────────────
    if not pred_df.empty and "predicted_ops" in pred_df.columns:
        st.divider()
        st.subheader("Operation Type per Prediction")
        _render_op_type_progression(pred_df)


# ── Helpers ───────────────────────────────────────────────────────


def _render_trajectory_kpis(summary: dict):
    cols = st.columns(6)

    with cols[0]:
        st.metric(
            "Sequence Length",
            f"{summary.get('initial_sequence_length', '–')} -> {summary.get('final_sequence_length', '–')}",
        )
    with cols[1]:
        st.metric("User Steps", summary.get("user_steps_taken", "–"))
    with cols[2]:
        uas = summary.get("uas_pct", 0)
        st.metric("UAS %", f"{uas:.1%}" if isinstance(uas, (int, float)) else "–")
    with cols[3]:
        st.metric(
            "Predictions",
            f"{summary.get('predictions_accepted', 0)}/{summary.get('predictions_attempted', 0)}",
        )
    with cols[5]:
        empty = summary.get('empty_predictions', 0)
        errored = summary.get('errored_empty_predictions', 0)
        if errored > 0:
            st.metric("Empty / Errored", f"{empty} / {errored}")
        else:
            st.metric("Empty", str(empty))
    with cols[6]:
        cov = summary.get("coverage", {})
        tp = cov.get("tp", 0)
        fp = cov.get("fp", 0)
        fn = cov.get("fn", 0)
        mm = cov.get("mm", 0)
        st.metric("TP / FP / FN / MM", f"{tp} / {fp} / {fn} / {mm}")


def _render_timeline_chart(timeline_df: pd.DataFrame):
    """User-step timeline: one position per user step on a horizontal track.

    X-axis = user step (1, 2, 3, ...).
    Each step is a small blue tick by default.  If a prediction was triggered
    at that step, the tick is replaced by a green circle (accepted),
    red X (rejected), or grey square (empty).  Hover shows details.
    """
    if "event" not in timeline_df.columns or "user_step" not in timeline_df.columns:
        st.info("No timeline data.")
        return

    # Build per-user-step data: default is "user", overlay prediction outcomes
    user_events = timeline_df[timeline_df["event"] == "user_step"]
    if user_events.empty:
        st.info("No user step data.")
        return

    max_us = int(user_events["user_step"].max())

    # Initialise every step as plain user step
    step_type = ["user"] * (max_us + 1)       # index 0 unused
    step_hover = [""] * (max_us + 1)
    step_prec = [0.0] * (max_us + 1)
    step_ops = [0] * (max_us + 1)

    for _, row in user_events.iterrows():
        us = int(row["user_step"])
        op = row.get("op", "")
        op_short = (op[:80] + "...") if isinstance(op, str) and len(op) > 80 else op
        step_hover[us] = f"<b>Step {us}</b> | User action<br>{op_short}"

    # Overlay predictions
    for _, row in timeline_df.iterrows():
        event = row.get("event", "")
        us_raw = row.get("user_step")
        if us_raw is None or (isinstance(us_raw, float) and pd.isna(us_raw)):
            continue
        us = int(us_raw)
        if us < 1 or us > max_us:
            continue

        if event == "prediction":
            accepted = row.get("accepted", False)
            prec = row.get("precision", 0) or 0
            ops_saved = row.get("ops_saved", 0) or 0
            tp = row.get("tp", 0) or 0
            fp = row.get("fp", 0) or 0
            step_prec[us] = prec
            step_ops[us] = ops_saved

            # Show predicted ops in hover (first 3 + count)
            pred_ops = row.get("predicted_ops", [])
            ops_lines = ""
            if isinstance(pred_ops, list) and pred_ops:
                shown = pred_ops[:3]
                ops_lines = "<br>".join(
                    (o[:70] + "...") if isinstance(o, str) and len(o) > 70 else str(o)
                    for o in shown
                )
                if len(pred_ops) > 3:
                    ops_lines += f"<br>... +{len(pred_ops) - 3} more"

            status = "ACCEPTED" if accepted else "REJECTED"
            hover = (
                f"<b>Step {us}</b> | <b>{status}</b><br>"
                f"Precision: {prec:.3f} | Ops saved: {ops_saved} | TP: {tp} FP: {fp}"
            )
            if ops_lines:
                hover += f"<br><br><i>Predicted:</i><br>{ops_lines}"

            step_type[us] = "accepted" if accepted else "rejected"
            step_hover[us] = hover

        elif event == "empty_prediction":
            if step_type[us] == "user":
                step_type[us] = "empty"
                step_hover[us] = f"<b>Step {us}</b> | Empty prediction (model returned nothing)"

        elif event == "errored_prediction":
            if step_type[us] == "user":
                step_type[us] = "errored"
                reason = row.get("error_reason", "unknown error")
                step_hover[us] = f"<b>Step {us}</b> | Errored prediction<br>{reason}"

    # Split into groups for separate traces (different markers)
    groups = {
        "user": {"x": [], "hover": [], "color": "#60a5fa", "symbol": "circle",
                 "size": 6, "name": "User Step"},
        "accepted": {"x": [], "hover": [], "color": "#2E8B57", "symbol": "circle",
                     "size": 13, "name": "Accepted"},
        "rejected": {"x": [], "hover": [], "color": "#DC3545", "symbol": "x",
                     "size": 12, "name": "Rejected"},
        "empty": {"x": [], "hover": [], "color": "#9ca3af", "symbol": "diamond",
                  "size": 9, "name": "Empty"},
        "errored": {"x": [], "hover": [], "color": "#f39c12", "symbol": "triangle-up",
                    "size": 10, "name": "Errored"},
    }

    for us in range(1, max_us + 1):
        g = groups[step_type[us]]
        g["x"].append(us)
        g["hover"].append(step_hover[us])

    fig = go.Figure()

    # Baseline track line
    fig.add_trace(
        go.Scatter(
            x=[1, max_us], y=[0, 0],
            mode="lines", line=dict(color="#e5e7eb", width=2),
            showlegend=False, hoverinfo="skip",
        )
    )

    # Add each group as a trace
    for key in ["user", "accepted", "rejected", "empty"]:
        g = groups[key]
        if not g["x"]:
            continue
        fig.add_trace(
            go.Scatter(
                x=g["x"],
                y=[0] * len(g["x"]),
                mode="markers",
                marker=dict(
                    color=g["color"], size=g["size"], symbol=g["symbol"],
                    line=dict(color="white", width=1) if key in ("accepted",) else dict(width=0),
                ),
                name=g["name"],
                hovertext=g["hover"],
                hoverinfo="text",
            )
        )

    fig.update_layout(
        height=150,
        margin=dict(t=10, b=30, l=20, r=20),
        xaxis_title="User Step",
        yaxis=dict(visible=False, range=[-0.5, 0.5]),
        showlegend=True,
        legend=dict(orientation="h", y=1.3),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_progress_curve(timeline_df: pd.DataFrame, summary: dict):
    """Progress curve: operations remaining vs user steps, with manual baseline.

    Uses only ``future_len`` from user_step events for a clean monotonic line.
    Accepted predictions show as green markers at the user step where the
    drop occurred (the drop is visible between consecutive user steps).
    """
    st.subheader("Progress Curve")

    initial_len = summary.get("initial_sequence_length", 0)
    if initial_len == 0:
        st.info("No sequence length data.")
        return

    def _safe_int(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        return int(val)

    # Build the progress line from user_step events only
    x_points = []
    y_points = []

    user_events = timeline_df[timeline_df["event"] == "user_step"]
    for _, row in user_events.iterrows():
        us = _safe_int(row.get("user_step"))
        future_len = _safe_int(row.get("future_len"))
        if us is not None and future_len is not None:
            x_points.append(us)
            y_points.append(future_len)

    # Track which user steps had an accepted prediction right after them
    # (for green markers)
    accept_user_steps = set()
    pred_events = timeline_df[
        (timeline_df["event"] == "prediction") & (timeline_df["accepted"] == True)
    ]
    for _, row in pred_events.iterrows():
        us = _safe_int(row.get("user_step"))
        if us is not None:
            accept_user_steps.add(us)

    # Build accepted marker positions: the future_len at the user step
    # RIGHT AFTER the acceptance (i.e. the point the line drops to)
    accept_x = []
    accept_y = []
    for i, us in enumerate(x_points):
        if us in accept_user_steps:
            # Find the next user step's future_len (the post-drop value)
            if i + 1 < len(x_points):
                accept_x.append(x_points[i + 1])
                accept_y.append(y_points[i + 1])
            else:
                # Last step — use current value
                accept_x.append(us)
                accept_y.append(y_points[i])

    if len(x_points) < 2:
        st.info("Not enough progress data to plot.")
        return

    fig = go.Figure()

    # Actual progress line
    fig.add_trace(
        go.Scatter(
            x=x_points,
            y=y_points,
            mode="lines",
            line=dict(color="#1F77B4", width=3),
            name="With model",
            hovertemplate="User step %{x}<br>Ops remaining: %{y}<extra></extra>",
        )
    )

    # Manual baseline (straight line from initial_len to 0 over initial_len steps)
    fig.add_trace(
        go.Scatter(
            x=[0, initial_len],
            y=[initial_len, 0],
            mode="lines",
            line=dict(color="#DC3545", width=2.5, dash="dash"),
            name="Manual baseline",
        )
    )

    # Accepted prediction markers (at the drop-to point)
    if accept_x:
        fig.add_trace(
            go.Scatter(
                x=accept_x,
                y=accept_y,
                mode="markers",
                marker=dict(color="#2E8B57", size=8, line=dict(color="white", width=1)),
                name="Accepted prediction",
                hovertemplate="User step %{x}<br>Ops remaining: %{y}<extra>Accepted</extra>",
            )
        )

    ops_saved_pct = summary.get("uas_pct", 0)
    fig.update_layout(
        height=350,
        margin=dict(t=30, b=40, l=60, r=20),
        xaxis_title="User Steps",
        yaxis_title="Operations Remaining",
        title=f"UAS: {ops_saved_pct:.1%}" if isinstance(ops_saved_pct, (int, float)) else None,
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_prediction_table(pred_df: pd.DataFrame):
    display_cols = [
        c
        for c in [
            "prediction_index",
            "step_t",
            "accepted",
            "predicted_count",
            "eval_final_state_precision",
            "eval_final_state_recall",
            "eval_final_state_ops_saved",
            "eval_final_state_tp",
            "eval_final_state_fp",
            "eval_final_state_fn",
            "eval_final_state_mm",
            "tokens_total",
            "generation_time_s",
        ]
        if c in pred_df.columns
    ]
    if not display_cols:
        st.dataframe(pred_df, use_container_width=True)
        return

    st.dataframe(
        pred_df[display_cols].style.format(
            {
                "eval_final_state_precision": "{:.3f}",
                "eval_final_state_recall": "{:.3f}",
                "generation_time_s": "{:.2f}s",
            },
            na_rep="–",
        ),
        use_container_width=True,
        height=min(400, 35 * len(pred_df) + 38),
    )


def _render_op_type_progression(pred_df: pd.DataFrame):
    """Stacked bar chart of operation types per prediction."""
    rows = []
    for _, row in pred_df.iterrows():
        ops = row.get("predicted_ops", [])
        if not isinstance(ops, list):
            continue
        for op in ops:
            if isinstance(op, str) and "|" in op:
                op_type = op.split("|")[0].strip()
                rows.append(
                    {
                        "prediction_index": row.get("prediction_index", 0),
                        "op_type": op_type,
                    }
                )
    if not rows:
        st.info("No operation data to display.")
        return

    op_df = pd.DataFrame(rows)
    counts = (
        op_df.groupby(["prediction_index", "op_type"])
        .size()
        .reset_index(name="count")
    )
    import plotly.express as px

    fig = px.bar(
        counts,
        x="prediction_index",
        y="count",
        color="op_type",
        title="Operation Types per Prediction",
        labels={
            "prediction_index": "Prediction Index",
            "count": "Count",
            "op_type": "Operation Type",
        },
    )
    fig.update_layout(height=400, margin=dict(t=40, b=30))
    st.plotly_chart(fig, use_container_width=True)
