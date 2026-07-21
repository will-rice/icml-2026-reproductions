"""
Evaluation Dashboard - Main Entry Point

Multi-page Streamlit dashboard for visualizing and analyzing
next-action prediction evaluation results.

Usage:
    streamlit run src/next_action_pred_eval/dashboard/app.py -- --dir results
    python scripts/run_dashboard.py --dir results/evaluation/main_results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is on the path when launched via `streamlit run`
# app.py is at src/next_action_pred_eval/dashboard/app.py → parents[2] = src/
_src_dir = str(Path(__file__).resolve().parents[2])
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import streamlit as st

from next_action_pred_eval.dashboard.data_loader import (
    discover_experiments,
    cached_discover_experiments,
    load_experiment,
    build_experiment_tree,
    ExperimentData,
)
from next_action_pred_eval.dashboard.components.batch_overview import batch_overview_page
from next_action_pred_eval.dashboard.components.trajectory_detail import trajectory_detail_page
from next_action_pred_eval.dashboard.components.prediction_inspector import prediction_inspector_page
from next_action_pred_eval.dashboard.components.operation_breakdown import operation_breakdown_page
from next_action_pred_eval.dashboard.components.coverage_analysis import coverage_analysis_page
from next_action_pred_eval.dashboard.components.cost_analysis import cost_analysis_page
from next_action_pred_eval.dashboard.components.temporal_analysis import temporal_analysis_page
from next_action_pred_eval.dashboard.components.experiment_comparison import experiment_comparison_page
from next_action_pred_eval.dashboard.components.error_analysis import error_analysis_page


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments passed after ``--``."""
    parser = argparse.ArgumentParser(description="Evaluation Dashboard")
    parser.add_argument(
        "--dir",
        type=str,
        default="results",
        help="Experiment directory or parent directory containing experiments",
    )
    # Streamlit strips the '--' separator and sets sys.argv to
    # [script, ...user_args], so parse_known_args safely ignores any
    # leftover Streamlit-internal flags.
    args, _ = parser.parse_known_args()
    return args


# ── Callback helpers (run *before* the next render, so state is ready) ──────

def _cb_set_all(n: int, value: bool) -> None:
    for i in range(n):
        st.session_state[f"_esd_{i}"] = value


def _cb_invert(n: int) -> None:
    for i in range(n):
        st.session_state[f"_esd_{i}"] = not st.session_state.get(f"_esd_{i}", False)


def _cb_set_folder(indices: list, value: bool) -> None:
    for i in indices:
        st.session_state[f"_esd_{i}"] = value





# ── Experiment selector dialog (requires Streamlit ≥ 1.35) ─────────────────

@st.dialog("Select Experiments", width="large")
def _show_experiment_selector(experiments, tree):
    """Hierarchical experiment selector with folder-level controls."""
    n_exp = len(experiments)

    # Initialise per-checkbox state from current selection on first open.
    if not st.session_state.get("_esd_init"):
        current = st.session_state.get("selected_experiment_indices", {0})
        for i in range(n_exp):
            st.session_state[f"_esd_{i}"] = i in current
        st.session_state["_esd_init"] = True

    # Quick-action buttons — use on_click callbacks so state updates
    # happen *before* the next render (no manual st.rerun needed).
    qcols = st.columns([1, 1, 1, 3])
    with qcols[0]:
        st.button("Select all", use_container_width=True,
                   on_click=_cb_set_all, args=(n_exp, True))
    with qcols[1]:
        st.button("Select none", use_container_width=True,
                   on_click=_cb_set_all, args=(n_exp, False))
    with qcols[2]:
        st.button("Invert", use_container_width=True,
                   on_click=_cb_invert, args=(n_exp,))

    # Text filter
    filter_text = st.text_input(
        "filter",
        placeholder="🔍 Type to filter experiments…",
        label_visibility="collapsed",
    )

    # Hierarchical tree — auto-collapse when many folders
    auto_expand = len(tree) <= 5

    for folder_name, items in tree:
        if filter_text:
            items = [
                (name, idx)
                for name, idx in items
                if filter_text.lower() in name.lower()
                or filter_text.lower() in folder_name.lower()
            ]
            if not items:
                continue

        folder_indices = [idx for _, idx in items]
        n_checked = sum(
            1 for i in folder_indices if st.session_state.get(f"_esd_{i}", False)
        )
        all_checked = n_checked == len(items)

        with st.expander(
            f"📁 **{folder_name}** — {n_checked}/{len(items)} selected",
            expanded=auto_expand or bool(filter_text),
        ):
            # Folder-level toggle
            lbl = "Deselect folder" if all_checked else "Select folder"
            st.button(
                lbl,
                key=f"_esd_f_{folder_name}",
                use_container_width=True,
                on_click=_cb_set_folder,
                args=(folder_indices, not all_checked),
            )

            # Individual experiment checkboxes
            for leaf_name, idx in items:
                st.checkbox(leaf_name, key=f"_esd_{idx}")

    # Footer
    st.divider()
    total_sel = sum(
        1 for i in range(n_exp) if st.session_state.get(f"_esd_{i}", False)
    )
    st.markdown(f"**{total_sel}** experiment(s) selected")

    bcols = st.columns(2)
    with bcols[0]:
        if st.button("Cancel", use_container_width=True):
            st.session_state["_esd_init"] = False
            st.rerun(scope="app")
    with bcols[1]:
        if st.button(
            "✅ Confirm",
            type="primary",
            use_container_width=True,
            disabled=(total_sel == 0),
        ):
            new_sel = {
                i for i in range(n_exp)
                if st.session_state.get(f"_esd_{i}", False)
            }
            st.session_state["selected_experiment_indices"] = new_sel
            st.session_state["_selection_confirmed"] = True
            st.session_state["_esd_init"] = False
            st.rerun(scope="app")


def main():
    st.set_page_config(
        page_title="Prediction Eval Dashboard",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    args = parse_args()

    # ----- Sidebar -----
    with st.sidebar:
        st.title("Experiment Explorer")

        # ── Directory picker ───────────────────────────────────────
        st.subheader("Results Directory")
        results_dir = st.text_input(
            "Paste path or edit below",
            value=args.dir,
            key="results_dir",
            help="Path to a directory containing experiment results (batch_summary.csv)",
        )

        # Discover experiments from the directory (cached to avoid repeated rglob)
        experiments = cached_discover_experiments(results_dir)

        if not experiments:
            st.warning(
                f"No experiments found in `{results_dir}`.\n\n"
                "Enter the path to a directory that contains "
                "`batch_summary.csv` (directly or in subdirectories)."
            )
            st.stop()

        st.caption(f"Found {len(experiments)} experiment(s)")

        st.divider()

        # ── Experiment selection (hierarchical dialog) ─────────────
        st.subheader("Select Experiments")
        experiment_names = [name for name, _ in experiments]
        tree = build_experiment_tree(experiments)

        # Reset selection when the results directory changes
        if st.session_state.get("_prev_results_dir") != results_dir:
            st.session_state["_prev_results_dir"] = results_dir
            st.session_state.pop("selected_experiment_indices", None)
            st.session_state["_selection_confirmed"] = False
            st.session_state["_esd_init"] = False

        # Initialise default selection
        if "selected_experiment_indices" not in st.session_state:
            st.session_state["selected_experiment_indices"] = set()
        if "_selection_confirmed" not in st.session_state:
            st.session_state["_selection_confirmed"] = False

        sel = st.session_state["selected_experiment_indices"]
        sel = {i for i in sel if 0 <= i < len(experiments)}
        st.session_state["selected_experiment_indices"] = sel
        confirmed = st.session_state["_selection_confirmed"]

        if sel:
            st.caption(f"**{len(sel)}** of {len(experiments)} experiment(s) selected")
        else:
            st.caption("No experiments selected yet")

        if st.button("📂 Choose experiments…", use_container_width=True):
            st.session_state["_esd_init"] = False
            _show_experiment_selector(experiments, tree)

        selected_indices = sorted(sel)

        if not selected_indices or not confirmed:
            if selected_indices and not confirmed:
                st.info(
                    "Selection made but not yet confirmed. "
                    "Open the chooser and click **✅ Confirm** to load data."
                )
            else:
                st.info("Click **📂 Choose experiments** to get started.")
            st.stop()

        # Primary experiment selector
        if len(selected_indices) > 1:
            primary_idx = st.selectbox(
                "Primary experiment",
                selected_indices,
                format_func=lambda i: experiment_names[i],
                key="_primary_exp",
            )
        else:
            primary_idx = selected_indices[0]

        primary_name, primary_path = experiments[primary_idx]

        with st.expander(f"Selected ({len(selected_indices)})", expanded=False):
            for i in selected_indices:
                marker = "🔹" if i == primary_idx else "▫️"
                st.caption(f"{marker} {experiment_names[i]}")

        st.divider()

        # ── Page navigation ────────────────────────────────────────
        pages = [
            "Batch Overview",
            "Trajectory Detail",
            "Prediction Inspector",
            "Operation Breakdown",
            "Coverage Analysis",
            "Cost Analysis",
            "Temporal Analysis",
            "Experiment Comparison",
            "Error Analysis",
        ]
        selected_page = st.radio("Pages", pages)

    # ----- Load experiment data -----
    primary_data = load_experiment(str(primary_path))

    all_experiment_data = [primary_data]
    for idx in selected_indices:
        if idx == primary_idx:
            continue
        _, path = experiments[idx]
        all_experiment_data.append(load_experiment(str(path)))

    # ----- Render selected page -----
    if selected_page == "Batch Overview":
        batch_overview_page(primary_data)
    elif selected_page == "Trajectory Detail":
        trajectory_detail_page(primary_data)
    elif selected_page == "Prediction Inspector":
        prediction_inspector_page(primary_data)
    elif selected_page == "Operation Breakdown":
        operation_breakdown_page(primary_data)
    elif selected_page == "Coverage Analysis":
        coverage_analysis_page(primary_data)
    elif selected_page == "Cost Analysis":
        cost_analysis_page(primary_data)
    elif selected_page == "Temporal Analysis":
        temporal_analysis_page(primary_data)
    elif selected_page == "Experiment Comparison":
        experiment_comparison_page(all_experiment_data)
    elif selected_page == "Error Analysis":
        error_analysis_page(all_experiment_data)


if __name__ == "__main__":
    main()
