#!/usr/bin/env python
"""
Run Evaluation Script
Runs a single evaluation experiment from a configuration file.

Usage:
    python scripts/run_evaluation.py --config configs/evaluation/default.yaml
    python scripts/run_evaluation.py --trajectory data/trajectories/example.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from next_action_pred_eval.evaluation import (
    Orchestrator,
    every_step,
    every_n_steps,
    HEURISTIC_IDEAL_USER,
    HEURISTIC_PRECISION_90,
    get_heuristic_by_name,
    create_custom_heuristic,
)
from next_action_pred_eval.evaluation.solver import ConstantSolver
from next_action_pred_eval.core.symbolic import symbolic_to_operations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_trajectory(trajectory_path: Path) -> dict:
    """Load trajectory from JSON file."""
    with open(trajectory_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_solver(config: dict):
    """Create solver from configuration."""
    solver_config = config.get("solver", {})
    solver_type = solver_config.get("type", "constant")

    if solver_type == "constant":
        solver = ConstantSolver()
    elif solver_type in ("llm", "chat", "completion"):
        # Import LLM solvers and adapter
        from next_action_pred_eval.evaluation.baselines import ChatSolver, CompletionSolver
        from next_action_pred_eval.utils.llm import OpenAIAdapter

        # Create adapter
        adapter_config = solver_config.get("adapter", {})
        api_key = adapter_config.get("api_key") or None
        model = adapter_config.get("model", "gpt-4")

        if api_key is None:
            import os
            api_key = os.environ.get(
                adapter_config.get("api_key_env", "OPENAI_API_KEY")
            )

        adapter = OpenAIAdapter(api_key=api_key, model=model)

        # Select solver class based on type
        SolverClass = CompletionSolver if solver_type == "completion" else ChatSolver

        solver = SolverClass(
            llm_adapter=adapter,
            max_context_ops=solver_config.get("max_context_ops", 50),
            temperature=solver_config.get("temperature", 0.0),
            system_prompt=solver_config.get("system_prompt"),
            system_prompt_file=solver_config.get("system_prompt_file"),
            user_prompt=solver_config.get("user_prompt"),
            user_prompt_file=solver_config.get("user_prompt_file"),
            completion_prompt=solver_config.get("completion_prompt"),
            completion_prompt_file=solver_config.get("completion_prompt_file"),
        )
    elif solver_type == "ngram":
        from next_action_pred_eval.evaluation.baselines import NGramSolver
        solver = NGramSolver(
            training_data_path=solver_config.get("training_data_path"),
            max_n=solver_config.get("max_n", 5),
            num_predictions=solver_config.get("num_predictions", 1),
            use_cell_patterns=solver_config.get("use_cell_patterns", True),
        )
    elif solver_type == "xgboost":
        from next_action_pred_eval.evaluation.baselines import XGBoostSolver
        solver = XGBoostSolver(
            model_dir=solver_config.get("model_dir", "models/xgboost"),
            window_size=solver_config.get("window_size", 10),
            max_predictions=solver_config.get("max_predictions", 5),
            stop_on_type_change=solver_config.get("stop_on_type_change", True),
        )
    elif solver_type == "lstm":
        from next_action_pred_eval.evaluation.baselines import LSTMSolver
        solver = LSTMSolver(
            model_dir=solver_config.get("model_dir", "models/lstm"),
            device=solver_config.get("device", "cpu"),
        )
    elif solver_type == "online_ngram":
        from next_action_pred_eval.evaluation.baselines import OnlineNGramSolver
        solver = OnlineNGramSolver(
            max_ngram_n=solver_config.get("max_ngram_n", 5),
            min_match_length=solver_config.get("min_match_length", 2),
            max_predictions=solver_config.get("max_predictions", 5),
        )
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")

    # Wrap with transforms if configured
    transforms_cfg = solver_config.get("transforms")
    if transforms_cfg:
        from next_action_pred_eval.core.transforms import build_transforms
        from next_action_pred_eval.evaluation.transformed_solver import TransformedSolver

        transforms = build_transforms(transforms_cfg)
        solver = TransformedSolver(inner=solver, transforms=transforms)

    return solver


def create_stride_config(config: dict):
    """Create stride configuration."""
    stride_config = config.get("stride", {})
    mode = stride_config.get("mode", "every_step")

    if mode == "every_step":
        return every_step()
    elif mode == "fixed_interval":
        interval = stride_config.get("interval", 5)
        return every_n_steps(interval)
    else:
        # Default to every step
        return every_step()


def create_heuristics(config: dict):
    """Create acceptance heuristics from configuration."""
    heuristic_configs = config.get("heuristics", [])
    heuristics = []

    for h_config in heuristic_configs:
        if isinstance(h_config, str):
            # Predefined heuristic by name
            heuristic = get_heuristic_by_name(h_config)
            if heuristic:
                heuristics.append(heuristic)
            else:
                logger.warning(f"Unknown heuristic: {h_config}")
        elif isinstance(h_config, dict):
            # Custom heuristic
            heuristic = create_custom_heuristic(
                name=h_config.get("name", "custom"),
                constraints=h_config.get("constraints", {}),
                description=h_config.get("description"),
            )
            heuristics.append(heuristic)

    if not heuristics:
        # Default to ideal user
        heuristics = [HEURISTIC_IDEAL_USER]

    return heuristics


def run_evaluation(config_path: Path = None, trajectory_path: Path = None):
    """Run evaluation experiment."""
    # Load configuration
    if config_path:
        config = load_config(config_path)
        trajectory_path = Path(config.get("trajectory_path", trajectory_path))
    else:
        config = {}

    # Load trajectory
    if not trajectory_path:
        logger.error("No trajectory path provided")
        return None

    trajectory = load_trajectory(trajectory_path)
    operations_symbolic = trajectory.get("operations", [])

    if not operations_symbolic:
        logger.error("No operations in trajectory")
        return None

    logger.info(f"Loaded trajectory with {len(operations_symbolic)} operations")

    # Convert to operations
    operations = symbolic_to_operations(operations_symbolic)

    # Create components
    solver = create_solver(config)
    stride_config = create_stride_config(config)
    heuristics = create_heuristics(config)

    output_dir = Path(config.get("output_dir", "results/evaluation"))
    experiment_name = config.get("experiment_name", trajectory_path.stem)

    # Create orchestrator
    orchestrator = Orchestrator(
        solver=solver,
        stride_config=stride_config,
        acceptance_heuristics=heuristics,
        output_dir=output_dir,
    )

    # Run experiment
    logger.info(f"Running evaluation: {experiment_name}")
    summary = orchestrator.run_experiment(
        action_stream=operations,
        experiment_name=experiment_name,
        max_context_ops=config.get("max_context_ops"),
        online_mode=config.get("online_mode", False),
    )

    # Print summary
    print("\n" + "=" * 60)
    print(summary.summary_str())
    print("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run evaluation experiment")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        help="Path to trajectory JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/evaluation"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.config and not args.trajectory:
        parser.error("Either --config or --trajectory must be provided")

    summary = run_evaluation(
        config_path=args.config,
        trajectory_path=args.trajectory,
    )

    if summary:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
