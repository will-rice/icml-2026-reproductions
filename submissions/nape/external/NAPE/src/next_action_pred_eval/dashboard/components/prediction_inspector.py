"""
Prediction Inspector Page

Deep dive into individual predictions: prompt, response, metrics, GT comparison.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from next_action_pred_eval.dashboard.data_loader import (
    ExperimentData,
    PredictionFolderData,
)


def prediction_inspector_page(experiment: ExperimentData):
    st.header("Prediction Inspector")

    if not experiment.trajectory_labels:
        st.warning("No trajectory data found.")
        return

    # ── Trajectory + prediction selectors ──────────────────────────
    col_traj, col_pred = st.columns([1, 1])

    with col_traj:
        selected_label = st.selectbox(
            "Trajectory", experiment.trajectory_labels, key="pi_traj"
        )
    if not selected_label:
        return

    traj = experiment.load_trajectory(selected_label)
    pred_df = traj.load_predictions()

    if pred_df.empty:
        st.info("No predictions for this trajectory.")
        return

    indices = pred_df["prediction_index"].tolist() if "prediction_index" in pred_df.columns else []
    if not indices:
        st.info("No prediction indices found.")
        return

    with col_pred:
        selected_idx = st.selectbox("Prediction #", indices, key="pi_pred")

    # Load detailed folder data if available
    folder_data = traj.load_prediction_folder(selected_idx)
    pred_row = pred_df[pred_df["prediction_index"] == selected_idx].iloc[0] if not pred_df.empty else None

    st.divider()

    # ── Layout: two columns ────────────────────────────────────────
    left, right = st.columns(2)

    with left:
        _render_prompt_response(folder_data, pred_row)

    with right:
        _render_gt_comparison(folder_data, pred_row)
        _render_metrics_panel(folder_data, pred_row)
        _render_acceptance(folder_data, pred_row)

    # ── Bottom section ─────────────────────────────────────────────
    st.divider()
    _render_history_context(folder_data, pred_row)
    _render_future_edits(folder_data, pred_row)


# ── Section renderers ─────────────────────────────────────────────


def _render_prompt_response(
    folder: PredictionFolderData | None, pred_row: pd.Series | None
):
    st.subheader("Prompt")
    if folder and folder.prompt_text:
        with st.expander("View full prompt", expanded=False):
            st.code(folder.prompt_text, language="text")
    else:
        st.caption("Prompt not available (enable `save_prediction_folders` in config)")

    st.subheader("Response")
    if folder and folder.response_text:
        st.code(folder.response_text, language="text")

        # Response metadata
        if folder.response_meta:
            meta = folder.response_meta
            cols = st.columns(4)
            with cols[0]:
                st.metric("Model", meta.get("model", "–"))
            with cols[1]:
                tokens = meta.get("tokens", {})
                st.metric("Tokens", tokens.get("total", "–"))
            with cols[2]:
                st.metric("Time", f"{meta.get('generation_time_s', 0):.2f}s")
            with cols[3]:
                st.metric("Temperature", meta.get("temperature", "–"))
    elif pred_row is not None:
        # Fallback to predictions.jsonl data
        ops = pred_row.get("predicted_ops", [])
        if isinstance(ops, list) and ops:
            st.code("\n".join(ops), language="text")
        else:
            st.caption("No response data.")
    else:
        st.caption("No response data.")

    st.subheader("Parsed Operations")
    if folder and folder.predicted_ops:
        for i, op in enumerate(folder.predicted_ops, 1):
            st.text(f"{i}. {op}")
    elif pred_row is not None:
        ops = pred_row.get("predicted_ops", [])
        if isinstance(ops, list):
            for i, op in enumerate(ops, 1):
                st.text(f"{i}. {op}")


def _render_gt_comparison(
    folder: PredictionFolderData | None, pred_row: pd.Series | None
):
    st.subheader("Ground Truth Segment")

    gt_ops = []
    pred_ops = []
    if folder:
        gt_ops = folder.gt_segment
        pred_ops = folder.predicted_ops
    elif pred_row is not None:
        gt_ops = pred_row.get("gt_segment", []) or []
        pred_ops = pred_row.get("predicted_ops", []) or []

    if not gt_ops and not pred_ops:
        st.caption("No comparison data.")
        return

    # Side-by-side comparison
    max_len = max(len(gt_ops), len(pred_ops))
    rows = []
    for i in range(max_len):
        gt = gt_ops[i] if i < len(gt_ops) else ""
        pred = pred_ops[i] if i < len(pred_ops) else ""
        match = "exact" if gt == pred else ("extra" if not gt else ("missing" if not pred else "diff"))
        rows.append({"#": i + 1, "Ground Truth": gt, "Predicted": pred, "Match": match})

    comp_df = pd.DataFrame(rows)
    st.dataframe(comp_df, use_container_width=True, height=min(300, 35 * len(comp_df) + 38))

    # Matched pairs (from evaluation.json)
    if folder and folder.evaluation and "matched_pairs" in folder.evaluation:
        with st.expander("Detailed matched pairs"):
            pairs = folder.evaluation["matched_pairs"]
            st.dataframe(pd.DataFrame(pairs), use_container_width=True)


def _render_metrics_panel(
    folder: PredictionFolderData | None, pred_row: pd.Series | None
):
    st.subheader("Metrics")

    metrics = {}
    if folder and folder.evaluation and "metrics" in folder.evaluation:
        metrics = folder.evaluation["metrics"]
    elif pred_row is not None:
        # Extract from flattened prediction row
        for col in pred_row.index:
            if col.startswith("eval_"):
                metrics[col.replace("eval_", "")] = pred_row[col]

    if not metrics:
        st.caption("No metrics available.")
        return

    cols = st.columns(4)
    with cols[0]:
        st.metric("Precision", f"{metrics.get('final_state_precision', 0):.3f}")
    with cols[1]:
        st.metric("Recall", f"{metrics.get('final_state_recall', 0):.3f}")
    with cols[2]:
        st.metric("Ops Saved", metrics.get("final_state_ops_saved", "–"))
    with cols[3]:
        tp = metrics.get("final_state_tp", 0)
        fp = metrics.get("final_state_fp", 0)
        fn = metrics.get("final_state_fn", 0)
        mm = metrics.get("final_state_mm", 0)
        st.metric("TP / FP / FN / MM", f"{tp} / {fp} / {fn} / {mm}")

    # Property breakdown
    if folder and folder.evaluation and "property_breakdown" in folder.evaluation:
        with st.expander("Property-level breakdown"):
            breakdown = folder.evaluation["property_breakdown"]
            rows = []
            for prop, stats in breakdown.items():
                if isinstance(stats, dict):
                    rows.append({"Property": prop, **stats})
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)


def _render_acceptance(
    folder: PredictionFolderData | None, pred_row: pd.Series | None
):
    st.subheader("Acceptance Decision")

    accepted = None
    heuristic = {}
    if folder and folder.acceptance:
        accepted = folder.acceptance.get("accepted")
        heuristic = folder.acceptance.get("heuristic", {})
    elif pred_row is not None:
        accepted = pred_row.get("accepted")
        heuristic = {"name": pred_row.get("heuristic_name", "")}

    if accepted is None:
        st.caption("No acceptance data.")
        return

    if accepted:
        st.success(f"ACCEPTED ({heuristic.get('name', '–')})")
    else:
        st.error(f"REJECTED ({heuristic.get('name', '–')})")

    # Heuristic check details
    checks = heuristic.get("checks", [])
    if checks:
        for check in checks:
            icon = "passed" if check.get("passed") else "failed"
            metric_name = check.get("metric", "?")
            value = check.get("value", "?")
            minimum = check.get("min")
            maximum = check.get("max")
            bounds = ""
            if minimum is not None:
                bounds += f" min={minimum}"
            if maximum is not None:
                bounds += f" max={maximum}"
            st.text(f"  {'PASS' if check.get('passed') else 'FAIL'}: {metric_name} = {value}{bounds}")


def _render_history_context(
    folder: PredictionFolderData | None, pred_row: pd.Series | None
):
    history = []
    if folder and folder.history_context:
        history = folder.history_context

    if not history:
        return

    with st.expander(f"History Context ({len(history)} operations)", expanded=False):
        st.code("\n".join(f"{i}. {op}" for i, op in enumerate(history, 1)), language="text")


def _render_future_edits(
    folder: PredictionFolderData | None, pred_row: pd.Series | None
):
    future = None
    if folder and folder.future_edits:
        future = folder.future_edits
    elif pred_row is not None:
        net_gain = pred_row.get("undo_net_gain")
        if net_gain is not None:
            future = {
                "future_if_accepted": {
                    "net_gain": net_gain,
                    "dedup_gain": pred_row.get("undo_dedup_gain", 0),
                    "inverse_cost": pred_row.get("undo_inverse_cost", 0),
                }
            }

    if not future:
        return

    with st.expander("Future Edits (Online Mode)", expanded=False):
        cols = st.columns(4)
        with cols[0]:
            st.metric("GT Before", future.get("gt_len_before", "–"))
        with cols[1]:
            st.metric("GT After", future.get("gt_len_after", "–"))

        preview = future.get("future_if_accepted", {})
        with cols[2]:
            st.metric("Net Gain", preview.get("net_gain", "–"))
        with cols[3]:
            st.metric("Inverse Cost", preview.get("inverse_cost", "–"))
