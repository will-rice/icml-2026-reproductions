"""
Evaluation Module - Online evaluation framework for action prediction.

This module provides the core evaluation framework for the paper's methodology:
- ISolver: Abstract interface for prediction solvers
- StrideConfig: Configuration for prediction frequency
- AcceptanceHeuristic: Criteria for accepting/rejecting predictions
- StateComparator: State-based comparison for TP/FP/FN metrics
- StepEvaluator: Evaluates single predictions
- Orchestrator: Runs end-to-end evaluation experiments

Example usage::

    from next_action_pred_eval.evaluation import (
        ISolver, PredictionResult,
        Orchestrator, every_step,
        HEURISTIC_IDEAL_USER,
    )

    # Implement your solver
    class MySolver(ISolver):
        def predict(self, previous_actions, **kwargs):
            # Your prediction logic here
            return PredictionResult(...)

    # Run evaluation
    orchestrator = Orchestrator(
        solver=MySolver(),
        stride_config=every_step(),
        acceptance_heuristics=[HEURISTIC_IDEAL_USER],
    )

    summary = orchestrator.run_experiment(operations)
    print(summary.summary_str())
"""

# Solver interface
from next_action_pred_eval.evaluation.solver import (
    DecodingConfig,
    ISolver,
    PredictionResult,
    ConstantSolver,
)

# Stride configuration
from next_action_pred_eval.evaluation.stride import (
    StrideMode,
    StrideConfig,
    every_step,
    every_n_steps,
    adaptive,
    stride_after_format_change,
    stride_after_data_entry,
    stride_after_n_same_type,
)

# Metrics
from next_action_pred_eval.evaluation.metrics import (
    EvaluationMetrics,
    DocumentMetrics,
    create_empty_metrics,
)

# Acceptance heuristics
from next_action_pred_eval.evaluation.acceptance import (
    AcceptanceHeuristic,
    ConstraintCheck,
    HEURISTIC_PERFECT,
    HEURISTIC_PRECISION_100,
    HEURISTIC_PRECISION_90,
    HEURISTIC_IDEAL_USER,
    HEURISTIC_STEPS_SAVED,
    HEURISTIC_ACCEPT_ALL,
    HEURISTIC_REJECT_ALL,
    COMMON_HEURISTICS,
    ALL_PREDEFINED_HEURISTICS,
    get_heuristic_by_name,
    create_custom_heuristic,
)

# State comparator
from next_action_pred_eval.evaluation.state_comparator import (
    StateComparator,
    ComparisonResult,
    PropertyDifference,
)

# Evaluator
from next_action_pred_eval.evaluation.evaluator import (
    StepEvaluator,
    EvaluationResult,
)

# Future edits
from next_action_pred_eval.evaluation.future_edits import (
    FutureEditsManager,
    FutureEditsConfig,
    FutureEditChanges,
)

# Orchestrator
from next_action_pred_eval.evaluation.orchestrator import (
    Orchestrator,
    ExperimentSummary,
    WorkbookStateTracker,
)

# Experiment config
from next_action_pred_eval.evaluation.experiment_config import (
    ExperimentConfig,
    SolverConfig,
    StrideSpec,
    load_experiment_config,
    expand_sweep,
)

# Output layout
from next_action_pred_eval.evaluation.output_layout import (
    OutputLayout,
    TrajectoryResult,
    CSV_COLUMNS,
)

# Experiment recorder
from next_action_pred_eval.evaluation.experiment_recorder import (
    ExperimentRecorder,
)

# Transformed solver
from next_action_pred_eval.evaluation.transformed_solver import (
    TransformedSolver,
)

__all__ = [
    # Solver interface
    "DecodingConfig",
    "ISolver",
    "PredictionResult",
    "ConstantSolver",
    # Stride configuration
    "StrideMode",
    "StrideConfig",
    "every_step",
    "every_n_steps",
    "adaptive",
    "stride_after_format_change",
    "stride_after_data_entry",
    "stride_after_n_same_type",
    # Metrics
    "EvaluationMetrics",
    "DocumentMetrics",
    "create_empty_metrics",
    # Acceptance heuristics
    "AcceptanceHeuristic",
    "ConstraintCheck",
    "HEURISTIC_PERFECT",
    "HEURISTIC_PRECISION_100",
    "HEURISTIC_PRECISION_90",
    "HEURISTIC_IDEAL_USER",
    "HEURISTIC_STEPS_SAVED",
    "HEURISTIC_ACCEPT_ALL",
    "HEURISTIC_REJECT_ALL",
    "COMMON_HEURISTICS",
    "ALL_PREDEFINED_HEURISTICS",
    "get_heuristic_by_name",
    "create_custom_heuristic",
    # State comparator
    "StateComparator",
    "ComparisonResult",
    "PropertyDifference",
    # Evaluator
    "StepEvaluator",
    "EvaluationResult",
    # Future edits
    "FutureEditsManager",
    "FutureEditsConfig",
    "FutureEditChanges",
    # Orchestrator
    "Orchestrator",
    "ExperimentSummary",
    "WorkbookStateTracker",
    # Experiment config
    "ExperimentConfig",
    "SolverConfig",
    "StrideSpec",
    "load_experiment_config",
    "expand_sweep",
    # Output layout
    "OutputLayout",
    "TrajectoryResult",
    "CSV_COLUMNS",
    # Experiment recorder
    "ExperimentRecorder",
    # Transformed solver
    "TransformedSolver",
]
