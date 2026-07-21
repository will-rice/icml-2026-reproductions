"""
Error Analysis Page

Cross-experiment comparison of prediction outcomes: accepted, rejected, empty,
errors (by type), and rejection-reason breakdown.
"""

from __future__ import annotations

from typing import List, Dict, Any

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from next_action_pred_eval.dashboard.data_loader import ExperimentData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gather_trajectory_stats(exp: ExperimentData) -> pd.DataFrame:
    """Build a per-trajectory stats DataFrame from experiment_summary.json files.

    Each row has: experiment, trajectory, predictions_attempted, accepted,
    rejected, empty, error_total, error_parse, error_prediction_failure, etc.
    """
    rows: list[dict] = []
    exp_name = exp.experiment_dir.name
    for label in exp.trajectory_labels:
        traj = exp.load_trajectory(label)
        s = traj.summary
        if not s or s.get("status") != "success":
            continue

        attempted = s.get("predictions_attempted", 0)
        accepted = s.get("predictions_accepted", 0)
        rejected = s.get("predictions_rejected", 0)
        empty = s.get("empty_predictions", 0)
        errored_empty = s.get("errored_empty_predictions", 0)
        errors_info = s.get("errors", {})
        error_total = errors_info.get("total", 0)
        error_counts = errors_info.get("counts", {})

        # Total "prediction opportunities" = attempted + empty + errored_empty + error
        # (attempted already excludes empty & error — it's the ones that
        #  produced parseable ops and went through acceptance)
        total_opportunities = attempted + empty + errored_empty + error_total

        row: Dict[str, Any] = {
            "experiment": exp_name,
            "trajectory": label,
            "total_opportunities": total_opportunities,
            "predictions_attempted": attempted,
            "predictions_accepted": accepted,
            "predictions_rejected": rejected,
            "empty_predictions": empty,
            "errored_empty_predictions": errored_empty,
            "error_total": error_total,
        }

        # Per-error-type columns
        for err_type, cnt in error_counts.items():
            row[f"error_{err_type}"] = cnt

        # Rates (against total opportunities)
        if total_opportunities > 0:
            row["accept_rate"] = accepted / total_opportunities
            row["reject_rate"] = rejected / total_opportunities
            row["empty_rate"] = empty / total_opportunities
            row["errored_empty_rate"] = errored_empty / total_opportunities
            row["error_rate"] = error_total / total_opportunities
        else:
            row["accept_rate"] = 0
            row["reject_rate"] = 0
            row["empty_rate"] = 0
            row["errored_empty_rate"] = 0
            row["error_rate"] = 0

        # Extra context
        row["avg_precision"] = s.get("avg_precision", 0)
        row["uas_pct"] = s.get("uas_pct", 0)
        row["tokens_total"] = s.get("tokens", {}).get("total", 0)
        row["user_steps_taken"] = s.get("user_steps_taken", 0)

        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _aggregate_experiments(
    all_stats: List[pd.DataFrame],
) -> pd.DataFrame:
    """Aggregate per-trajectory stats into one row per experiment."""
    rows = []
    for df in all_stats:
        if df.empty:
            continue
        exp_name = df["experiment"].iloc[0]
        row = {
            "Experiment": exp_name,
            "Trajectories": len(df),
            "Total Opportunities": int(df["total_opportunities"].sum()),
            "Attempted (valid)": int(df["predictions_attempted"].sum()),
            "Accepted": int(df["predictions_accepted"].sum()),
            "Rejected": int(df["predictions_rejected"].sum()),
            "Empty": int(df["empty_predictions"].sum()),
            "Errored": int(df["errored_empty_predictions"].sum()),
            "Errors": int(df["error_total"].sum()),
        }

        total = row["Total Opportunities"]
        if total > 0:
            row["Accept %"] = row["Accepted"] / total
            row["Reject %"] = row["Rejected"] / total
            row["Empty %"] = row["Empty"] / total
            row["Errored %"] = row["Errored"] / total
            row["Error %"] = row["Errors"] / total
        else:
            row["Accept %"] = row["Reject %"] = row["Empty %"] = row["Errored %"] = row["Error %"] = 0

        # Per-error-type totals
        err_cols = [c for c in df.columns if c.startswith("error_") and c != "error_total" and c != "error_rate"]
        for c in err_cols:
            nice_name = c.replace("error_", "").replace("_", " ").title()
            row[nice_name] = int(df[c].fillna(0).sum())

        rows.append(row)

    return pd.DataFrame(rows)


def _collect_error_samples(exp: ExperimentData) -> pd.DataFrame:
    """Collect up-to-50 error sample_details per trajectory into a table."""
    rows = []
    for label in exp.trajectory_labels:
        traj = exp.load_trajectory(label)
        s = traj.summary
        if not s:
            continue
        samples = s.get("errors", {}).get("sample_details", [])
        for sample in samples:
            rows.append({
                "experiment": exp.experiment_dir.name,
                "trajectory": label,
                "step_t": sample.get("t", "?"),
                "error_type": sample.get("type", "unknown"),
                "description": sample.get("description", ""),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Page renderer
# ---------------------------------------------------------------------------


def error_analysis_page(experiments: List[ExperimentData]):
    """Render the Error Analysis comparison page."""
    st.header("Error Analysis")

    if not experiments:
        st.info("Select at least one experiment.")
        return

    # ── Gather data ────────────────────────────────────────────────
    with st.spinner("Aggregating prediction outcomes across trajectories…"):
        all_stats = [_gather_trajectory_stats(exp) for exp in experiments]

    agg_df = _aggregate_experiments(all_stats)
    if agg_df.empty:
        st.warning("No trajectory data found.")
        return

    # ── 1. Summary table ──────────────────────────────────────────
    st.subheader("Prediction Outcome Summary")
    st.caption(
        "**Total Opportunities** = Attempted (valid predictions that reached "
        "acceptance) + Empty (model returned nothing) + Errored (LLM call / "
        "token errors) + Errors (parse or call failures). "
        "Accepted + Rejected = Attempted."
    )

    fmt = {
        "Accept %": "{:.1%}",
        "Reject %": "{:.1%}",
        "Empty %": "{:.1%}",
        "Errored %": "{:.1%}",
        "Error %": "{:.1%}",
    }
    st.dataframe(agg_df.style.format(fmt), use_container_width=True)

    st.divider()

    # ── 2. Stacked bar: outcome distribution ──────────────────────
    st.subheader("Outcome Distribution")

    outcome_rows = []
    for _, r in agg_df.iterrows():
        base = {
            "Experiment": r["Experiment"],
        }
        for cat, val in [
            ("Accepted", r["Accepted"]),
            ("Rejected", r["Rejected"]),
            ("Empty", r["Empty"]),
            ("Errored", r["Errored"]),
            ("Error", r["Errors"]),
        ]:
            outcome_rows.append({**base, "Outcome": cat, "Count": val})

    outcome_df = pd.DataFrame(outcome_rows)
    color_map = {
        "Accepted": "#2ecc71",
        "Rejected": "#e74c3c",
        "Empty": "#95a5a6",
        "Errored": "#f39c12",
        "Error": "#e67e22",
    }

    view_mode = st.radio(
        "View", ["Absolute counts", "Percentage"], horizontal=True, key="_ea_view"
    )

    if view_mode == "Percentage":
        # Compute percentages
        totals = outcome_df.groupby("Experiment")["Count"].transform("sum")
        outcome_df["Pct"] = outcome_df["Count"] / totals.replace(0, 1)
        fig = px.bar(
            outcome_df,
            x="Experiment",
            y="Pct",
            color="Outcome",
            color_discrete_map=color_map,
            labels={"Pct": "Proportion"},
            category_orders={"Outcome": ["Accepted", "Rejected", "Empty", "Errored", "Error"]},
        )
        fig.update_layout(yaxis_tickformat=".0%")
    else:
        fig = px.bar(
            outcome_df,
            x="Experiment",
            y="Count",
            color="Outcome",
            color_discrete_map=color_map,
            labels={"Count": "Predictions"},
            category_orders={"Outcome": ["Accepted", "Rejected", "Empty", "Errored", "Error"]},
        )

    fig.update_layout(
        barmode="stack",
        height=400,
        margin=dict(t=20, b=80),
        xaxis_tickangle=-30,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── 3. Error type breakdown ───────────────────────────────────
    st.subheader("Error Type Breakdown")

    err_type_rows = []
    for stats_df in all_stats:
        if stats_df.empty:
            continue
        exp_name = stats_df["experiment"].iloc[0]
        err_cols = [c for c in stats_df.columns if c.startswith("error_") and c not in ("error_total", "error_rate")]
        for c in err_cols:
            nice = c.replace("error_", "").replace("_", " ").title()
            err_type_rows.append({
                "Experiment": exp_name,
                "Error Type": nice,
                "Count": int(stats_df[c].fillna(0).sum()),
            })

    if err_type_rows:
        err_type_df = pd.DataFrame(err_type_rows)
        fig2 = px.bar(
            err_type_df,
            x="Experiment",
            y="Count",
            color="Error Type",
            barmode="group",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig2.update_layout(height=350, margin=dict(t=20, b=80), xaxis_tickangle=-30)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No errors recorded across experiments.")

    st.divider()

    # ── 4. Per-trajectory heatmap ─────────────────────────────────
    st.subheader("Per-Trajectory Breakdown")
    st.caption("Rates are computed against total prediction opportunities per trajectory.")

    combined = pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()
    if combined.empty:
        st.info("No trajectory-level data available.")
    else:
        metric_choice = st.selectbox(
            "Metric to compare",
            ["error_rate", "empty_rate", "errored_empty_rate", "accept_rate", "reject_rate",
             "error_total", "empty_predictions", "errored_empty_predictions",
             "predictions_accepted", "predictions_rejected"],
            format_func=lambda x: x.replace("_", " ").title(),
            key="_ea_metric",
        )

        # Pivot: rows = trajectory, columns = experiment
        pivot = combined.pivot_table(
            index="trajectory",
            columns="experiment",
            values=metric_choice,
            aggfunc="first",
        )

        if pivot.shape[0] > 50:
            st.caption(f"Showing top 50 of {pivot.shape[0]} trajectories (sorted by max value).")
            pivot["_max"] = pivot.max(axis=1)
            pivot = pivot.nlargest(50, "_max").drop(columns=["_max"])

        is_rate = metric_choice.endswith("_rate")

        fig3 = px.imshow(
            pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            color_continuous_scale="RdYlGn_r" if metric_choice in ("error_rate", "empty_rate", "errored_empty_rate", "reject_rate", "error_total", "empty_predictions", "errored_empty_predictions", "predictions_rejected") else "RdYlGn",
            labels={"color": metric_choice.replace("_", " ").title()},
            aspect="auto",
        )
        fig3.update_layout(
            height=max(400, 18 * len(pivot)),
            margin=dict(t=20, b=30, l=200),
        )
        if is_rate:
            fig3.update_coloraxes(colorbar_tickformat=".0%")
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # ── 5. Box plots: rates across trajectories ──────────────────
    st.subheader("Rate Distributions Across Trajectories")

    if not combined.empty:
        rate_cols = ["accept_rate", "reject_rate", "empty_rate", "errored_empty_rate", "error_rate"]
        melted = combined.melt(
            id_vars=["experiment"],
            value_vars=rate_cols,
            var_name="rate_type",
            value_name="value",
        )
        melted["rate_type"] = melted["rate_type"].str.replace("_", " ").str.title()

        fig4 = px.box(
            melted,
            x="rate_type",
            y="value",
            color="experiment",
            labels={"value": "Rate", "rate_type": ""},
        )
        fig4.update_layout(
            height=400,
            margin=dict(t=20, b=80),
            yaxis_tickformat=".0%",
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()

    # ── 6. Error sample details ───────────────────────────────────
    st.subheader("Error Sample Details")
    st.caption("Up to 50 error samples per trajectory are saved during evaluation.")

    all_samples = pd.concat(
        [_collect_error_samples(exp) for exp in experiments], ignore_index=True
    )

    if all_samples.empty:
        st.info("No error samples recorded.")
    else:
        # Filters
        fcols = st.columns(3)
        with fcols[0]:
            exp_filter = st.multiselect(
                "Experiment",
                all_samples["experiment"].unique(),
                default=list(all_samples["experiment"].unique()),
                key="_ea_exp_filt",
            )
        with fcols[1]:
            type_filter = st.multiselect(
                "Error type",
                all_samples["error_type"].unique(),
                default=list(all_samples["error_type"].unique()),
                key="_ea_type_filt",
            )
        with fcols[2]:
            max_rows = st.number_input("Max rows", 10, 500, 100, key="_ea_max")

        filtered = all_samples[
            all_samples["experiment"].isin(exp_filter)
            & all_samples["error_type"].isin(type_filter)
        ].head(int(max_rows))

        st.dataframe(filtered, use_container_width=True)

        # Quick stats on displayed samples
        st.caption(
            f"Showing {len(filtered)} of {len(all_samples)} total error samples "
            f"across {all_samples['experiment'].nunique()} experiment(s)."
        )
