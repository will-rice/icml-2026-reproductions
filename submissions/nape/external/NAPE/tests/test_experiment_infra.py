"""
Tests for the experiment infrastructure:
- ExperimentConfig loading and sweep expansion
- OutputLayout directory management and resume detection
- CSV writing and reading
- ExperimentRecorder timeline/prediction/attribution format
"""

import csv
import json
import tempfile
from pathlib import Path

import pytest
import yaml

from next_action_pred_eval.evaluation.experiment_config import (
    ExperimentConfig,
    SolverConfig,
    StrideSpec,
    load_experiment_config,
    expand_sweep,
)
from next_action_pred_eval.evaluation.output_layout import (
    OutputLayout,
    TrajectoryResult,
    CSV_COLUMNS,
)
from next_action_pred_eval.evaluation.experiment_recorder import (
    ExperimentRecorder,
)


# ============================================================================
# ExperimentConfig
# ============================================================================

class TestExperimentConfig:

    def test_load_config_from_yaml(self, tmp_path):
        """Load a YAML config and verify all fields."""
        config_data = {
            "name": "test_exp",
            "trajectory_paths": ["data/traj1.json", "data/traj2.json"],
            "max_runs": 5,
            "workers": 2,
            "max_context_ops": 25,
            "online_mode": True,
            "output_dir": "outputs/test",
            "heuristics": ["ideal_user", "precision_90"],
            "solver": {
                "type": "llm",
                "adapter": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.1,
            },
            "stride": {"mode": "fixed_interval", "interval": 3},
        }
        config_path = tmp_path / "test.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        cfg = load_experiment_config(config_path)
        assert cfg.name == "test_exp"
        assert cfg.variant_name == "test_exp"
        assert cfg.max_runs == 5
        assert cfg.workers == 2
        assert cfg.max_context_ops == 25
        assert cfg.online_mode is True
        assert cfg.solver.type == "llm"
        assert cfg.solver.adapter == "openai"
        assert cfg.solver.model == "gpt-4o-mini"
        assert cfg.solver.temperature == 0.1
        assert cfg.stride.mode == "fixed_interval"
        assert cfg.stride.interval == 3
        assert cfg.heuristics == ["ideal_user", "precision_90"]

    def test_expand_sweep_no_sweep(self):
        """No sweep section returns [base]."""
        cfg = ExperimentConfig(name="base")
        variants = expand_sweep(cfg)
        assert len(variants) == 1
        assert variants[0].name == "base"

    def test_expand_sweep_single_param(self):
        """Single param with 3 values produces 3 variants."""
        cfg = ExperimentConfig(
            name="test",
            sweep={"max_context_ops": [10, 25, 50]},
        )
        variants = expand_sweep(cfg)
        assert len(variants) == 3
        assert variants[0].max_context_ops == 10
        assert variants[1].max_context_ops == 25
        assert variants[2].max_context_ops == 50
        # Variant names contain the value
        assert "max_context_ops=10" in variants[0].variant_name
        assert "max_context_ops=50" in variants[2].variant_name

    def test_expand_sweep_cross_product(self):
        """Two params produce cross-product of variants."""
        cfg = ExperimentConfig(
            name="cross",
            sweep={
                "max_context_ops": [10, 50],
                "solver.model": ["model-a", "model-b"],
            },
        )
        variants = expand_sweep(cfg)
        assert len(variants) == 4  # 2 x 2

        # All combinations present
        models = {v.solver.model for v in variants}
        ctx_ops = {v.max_context_ops for v in variants}
        assert models == {"model-a", "model-b"}
        assert ctx_ops == {10, 50}

    def test_nested_override(self):
        """Dot-notation like solver.model works."""
        cfg = ExperimentConfig(
            name="nested",
            solver=SolverConfig(model="base_model"),
            sweep={"solver.model": ["ModelA", "ModelB"]},
        )
        variants = expand_sweep(cfg)
        assert len(variants) == 2
        assert variants[0].solver.model == "ModelA"
        assert variants[1].solver.model == "ModelB"
        # Sweep cleared from variants
        assert variants[0].sweep is None

    def test_resolve_trajectories_with_max_runs(self, tmp_path):
        """resolve_trajectories respects max_runs."""
        for i in range(5):
            (tmp_path / f"traj_{i}.json").touch()

        cfg = ExperimentConfig(
            trajectory_paths=[str(tmp_path / "*.json")],
            max_runs=3,
        )
        paths = cfg.resolve_trajectories()
        assert len(paths) == 3


# ============================================================================
# OutputLayout
# ============================================================================

class TestOutputLayout:

    def test_get_run_dir_creates_path(self, tmp_path):
        """get_run_dir creates the 2-level directory."""
        layout = OutputLayout(tmp_path / "experiment")
        run_dir = layout.get_run_dir("0000afae")
        assert run_dir.exists()
        assert run_dir == tmp_path / "experiment" / "0000afae"

    def test_is_completed_detects_summary(self, tmp_path):
        """is_completed returns True when experiment_summary.json exists."""
        layout = OutputLayout(tmp_path / "exp")
        assert not layout.is_completed("0000afae")

        # Create the summary file
        run_dir = layout.get_run_dir("0000afae")
        (run_dir / "experiment_summary.json").write_text("{}")
        assert layout.is_completed("0000afae")

    def test_csv_write_and_read(self, tmp_path):
        """Write CSV and verify column names and values."""
        layout = OutputLayout(tmp_path / "exp")
        results = [
            TrajectoryResult(
                file_label="traj1", config_variant="test",
                status="success", net_operations_saved=10,
                predictions_attempted=5, predictions_accepted=3,
                initial_sequence_length=100, final_sequence_length=90,
                user_steps_taken=87, uas_pct=0.13,
                acceptance_rate=0.6, avg_precision=0.8,
                coverage_pct_tp=0.5, ops_saved_per_prediction=2.0,
                total_tokens=1000, input_tokens=800, output_tokens=200,
                total_time=10.5, inverse_ops_added=2,
            ),
            TrajectoryResult(
                file_label="traj2", config_variant="test",
                status="error", error_message="Test error",
            ),
        ]

        csv_path = layout.write_csv(results)
        assert csv_path.exists()

        # Read back
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert set(reader.fieldnames) == set(CSV_COLUMNS)
        assert rows[0]["file_label"] == "traj1"
        assert rows[0]["status"] == "success"
        assert rows[0]["net_operations_saved"] == "10"
        assert rows[1]["status"] == "error"
        assert rows[1]["error_message"] == "Test error"

    def test_trajectory_result_csv_row_has_all_columns(self):
        """to_csv_row() includes all CSV_COLUMNS."""
        result = TrajectoryResult(
            file_label="test", config_variant="v1", status="success",
        )
        row = result.to_csv_row()
        for col in CSV_COLUMNS:
            assert col in row, f"Missing column: {col}"


# ============================================================================
# ExperimentRecorder
# ============================================================================

class TestExperimentRecorder:

    def test_timeline_user_step_format(self, tmp_path):
        """record_user_step writes correct JSONL format."""
        recorder = ExperimentRecorder(tmp_path, "test_traj")
        recorder.record_user_step(
            t=1, user_step=1,
            op_symbolic="INPUT | Sheet1!A1 | hello",
            history_len=1, future_len=83,
        )

        timeline_path = tmp_path / "timeline.jsonl"
        assert timeline_path.exists()

        with open(timeline_path, "r") as f:
            event = json.loads(f.readline())

        assert event["event"] == "user_step"
        assert event["t"] == 1
        assert event["user_step"] == 1
        assert event["op"] == "INPUT | Sheet1!A1 | hello"
        assert event["history_len"] == 1
        assert event["future_len"] == 83

    def test_prediction_writes_to_both_files(self, tmp_path):
        """record_prediction writes to predictions.jsonl AND timeline.jsonl."""
        recorder = ExperimentRecorder(tmp_path, "test_traj")
        recorder.record_prediction(
            prediction_index=1, t=5, user_step=5,
            pred_ops_symbolic=["FONT_BOLD | Sheet1!A1 | True"],
            gt_segment_symbolic=["FONT_BOLD | Sheet1!A1 | True"],
            eval_metrics={
                "final_state_tp": 1, "final_state_fp": 0, "final_state_fn": 0,
                "final_state_mm": 0,
                "final_state_precision": 1.0, "final_state_ops_saved": 1,
            },
            accepted=True,
            heuristic_details={"name": "ideal_user", "accepted": True, "checks": []},
            tokens={"input": 100, "output": 20, "total": 120},
            generation_time_s=0.5,
            history_tail=["op1", "op2"],
            future_head=["op3", "op4"],
        )

        # Check predictions.jsonl
        pred_path = tmp_path / "predictions.jsonl"
        assert pred_path.exists()
        with open(pred_path, "r") as f:
            pred = json.loads(f.readline())
        assert pred["prediction_index"] == 1
        assert pred["accepted"] is True
        assert pred["predicted_count"] == 1

        # Check timeline.jsonl
        timeline_path = tmp_path / "timeline.jsonl"
        with open(timeline_path, "r") as f:
            event = json.loads(f.readline())
        assert event["event"] == "prediction"
        assert event["prediction_index"] == 1
        assert event["accepted"] is True

    def test_final_trajectory_attribution(self, tmp_path):
        """record_final_trajectory writes attribution data."""
        recorder = ExperimentRecorder(tmp_path, "test_traj")
        attribution = [
            {"index": 0, "op": "INPUT | Sheet1!A1 | hello", "source": "user", "user_step": 1},
            {"index": 1, "op": "FONT_BOLD | Sheet1!A1 | True", "source": "predicted", "prediction_index": 1},
            {"index": 2, "op": "FONT_ITALIC | Sheet1!A1 | False", "source": "inverse", "prediction_index": 1},
            {"index": 3, "op": "FILL_COLOR | Sheet1!B1 | #FF0000", "source": "original"},
        ]
        recorder.record_final_trajectory(attribution)

        traj_path = tmp_path / "final_trajectory.jsonl"
        assert traj_path.exists()

        with open(traj_path, "r") as f:
            lines = [json.loads(line) for line in f]

        assert len(lines) == 4
        assert lines[0]["source"] == "user"
        assert lines[1]["source"] == "predicted"
        assert lines[1]["prediction_index"] == 1
        assert lines[2]["source"] == "inverse"
        assert lines[3]["source"] == "original"

    def test_finalize_enriches_summary(self, tmp_path):
        """finalize() adds avg_precision, ops_saved_per_prediction, etc."""
        recorder = ExperimentRecorder(tmp_path, "test_traj")

        # Simulate some predictions
        for i in range(3):
            recorder.record_prediction(
                prediction_index=i + 1, t=i * 2 + 2, user_step=i + 1,
                pred_ops_symbolic=["FONT_BOLD | Sheet1!A1 | True"],
                gt_segment_symbolic=["FONT_BOLD | Sheet1!A1 | True"],
                eval_metrics={
                    "final_state_tp": 1, "final_state_fp": 0, "final_state_fn": 0,
                    "final_state_mm": 0,
                    "final_state_precision": 0.5 + i * 0.2,  # 0.5, 0.7, 0.9
                    "final_state_ops_saved": 1,
                },
                accepted=i > 0,  # first rejected, next two accepted
                heuristic_details={"name": "test", "accepted": i > 0, "checks": []},
                tokens={"input": 100, "output": 20, "total": 120},
                generation_time_s=0.5,
                history_tail=[], future_head=[],
            )

        summary = {
            "predictions_attempted": 3,
            "net_operations_saved": 2,
        }
        recorder.finalize(summary)

        # Check enriched fields
        assert abs(summary["avg_precision"] - 0.7) < 0.01  # mean(0.5, 0.7, 0.9)
        assert abs(summary["ops_saved_per_prediction"] - 2 / 3) < 0.01
        assert summary["empty_predictions"] == 0
        assert summary["divergences"] == 0
        assert "predicted" in summary["operation_breakdown"]

    def test_divergence_recording(self, tmp_path):
        """record_divergence writes to both divergence_log.jsonl and timeline.jsonl."""
        recorder = ExperimentRecorder(tmp_path, "test_traj")
        recorder.record_divergence(
            prediction_index=3, t=10,
            description="State mismatch on cell A2",
            action="rejected",
        )

        div_path = tmp_path / "divergence_log.jsonl"
        assert div_path.exists()
        with open(div_path, "r") as f:
            entry = json.loads(f.readline())
        assert entry["prediction_index"] == 3
        assert entry["action"] == "rejected"

        timeline_path = tmp_path / "timeline.jsonl"
        with open(timeline_path, "r") as f:
            event = json.loads(f.readline())
        assert event["event"] == "divergence"
