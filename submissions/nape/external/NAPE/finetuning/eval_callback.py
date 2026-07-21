"""
Prediction Accuracy Callback for Finetuning
============================================

Evaluates the model on a fixed set of test cases (context → expected answer)
at the end of each epoch (and optionally before training starts, to capture
the base model's performance).

Each test case has an operation history (``context``) and a single expected
next operation (``answer``).  The callback constructs the same completion
prompt used during training, generates one line of output via greedy
decoding, and checks whether it matches the expected answer exactly.

Metrics logged to TensorBoard:
    - ``eval/prediction_accuracy`` — fraction of test cases where the
      generated first line matches the expected answer exactly.
    - ``eval/prediction_accuracy_op_type`` — fraction where the
      operation type (first pipe-delimited field) matches.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import TrainerCallback

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from next_action_pred_eval.core.symbolic import compress_symbolic
from next_action_pred_eval.evaluation.baselines.prompts import (
    shorten_symbolic_values,
)

logger = logging.getLogger(__name__)


def _preprocess_ops(
    ops: List[str],
    *,
    enable_context_shortening: bool = True,
    context_shortening_max_chars: int = 32,
    context_shortening_corner_cells_dim: int = 2,
    remove_sheet_name: bool = True,
) -> List[str]:
    """Mirror the on-the-fly preprocessing from dataset.py."""
    result = list(ops)
    if enable_context_shortening:
        result = shorten_symbolic_values(
            result,
            max_value_length=context_shortening_max_chars,
            corner_cells_dim=context_shortening_corner_cells_dim,
        )
    if remove_sheet_name:
        result = compress_symbolic(result, remove_sheet_name=True)
    return result


def _build_prompt(
    ops: List[str],
    prompt_template: str,
) -> str:
    """Build the completion prompt from preprocessed ops."""
    actions_str = "\n".join(ops)
    return prompt_template.format(actions=actions_str)


def load_test_cases(path: Path) -> List[Dict[str, Any]]:
    """Load precision-1 test cases from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class PredictionAccuracyCallback(TrainerCallback):
    """Evaluates exact-match prediction accuracy on fixed test cases.

    Parameters
    ----------
    test_cases : list[dict]
        Each dict has ``"context"`` (list of symbolic op strings) and
        ``"answer"`` (single symbolic op string).
    tokenizer
        The tokenizer used for training.
    prompt_template : str
        The same prompt template used for training data.
    preprocessing : dict
        Preprocessing config (remove_sheet_name, context_shortening, etc.).
    max_new_tokens : int
        Max tokens to generate per test case (one line is typically < 50).
    eval_before_training : bool
        If True, run evaluation before the first training step to capture
        base model performance.
    """

    def __init__(
        self,
        test_cases: List[Dict[str, Any]],
        tokenizer,
        prompt_template: str,
        preprocessing: Optional[Dict[str, Any]] = None,
        max_new_tokens: int = 64,
        eval_before_training: bool = True,
    ):
        self.test_cases = test_cases
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.max_new_tokens = max_new_tokens
        self.eval_before_training = eval_before_training

        # Preprocessing config
        prep = preprocessing or {}
        shortening = prep.get("context_shortening", {})
        self.remove_sheet_name = prep.get("remove_sheet_name", True)
        self.enable_context_shortening = shortening.get("enabled", True)
        self.context_shortening_max_chars = shortening.get("max_chars", 32)
        self.context_shortening_corner_cells_dim = shortening.get(
            "corner_cells_dim", 2
        )

        # Pre-build prompts (context doesn't change during training)
        self.prompts: List[str] = []
        self.answers: List[str] = []
        for tc in self.test_cases:
            processed = _preprocess_ops(
                tc["context"],
                enable_context_shortening=self.enable_context_shortening,
                context_shortening_max_chars=self.context_shortening_max_chars,
                context_shortening_corner_cells_dim=self.context_shortening_corner_cells_dim,
                remove_sheet_name=self.remove_sheet_name,
            )
            prompt = _build_prompt(processed, self.prompt_template)
            self.prompts.append(prompt)

            # Preprocess the answer the same way for comparison
            answer_processed = _preprocess_ops(
                [tc["answer"]],
                enable_context_shortening=self.enable_context_shortening,
                context_shortening_max_chars=self.context_shortening_max_chars,
                context_shortening_corner_cells_dim=self.context_shortening_corner_cells_dim,
                remove_sheet_name=self.remove_sheet_name,
            )
            self.answers.append(answer_processed[0])

        logger.info(
            "PredictionAccuracyCallback: %d test cases loaded", len(self.test_cases)
        )

    def _evaluate(self, model, state, control) -> Dict[str, float]:
        """Run generation on all test cases and compute accuracy."""
        model.eval()
        device = next(model.parameters()).device

        exact_match = 0
        op_type_match = 0
        total = len(self.prompts)

        t0 = time.time()
        for prompt_text, expected in zip(self.prompts, self.answers):
            # Tokenize the prompt
            inputs = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048 - self.max_new_tokens,
            )
            input_ids = inputs["input_ids"].to(device)

            # Generate (greedy, one line)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode only the generated part
            generated_ids = outputs[0, input_ids.shape[1]:]
            generated_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

            # Extract the first non-empty line (the predicted operation)
            predicted_line = ""
            for line in generated_text.split("\n"):
                line = line.strip()
                if line and " | " in line:
                    predicted_line = line
                    break

            # Exact match
            if predicted_line == expected:
                exact_match += 1

            # Operation type match (first field before |)
            pred_type = predicted_line.split(" | ")[0].strip() if " | " in predicted_line else ""
            expected_type = expected.split(" | ")[0].strip() if " | " in expected else ""
            if pred_type and pred_type == expected_type:
                op_type_match += 1

        elapsed = time.time() - t0
        exact_acc = exact_match / total if total > 0 else 0.0
        op_type_acc = op_type_match / total if total > 0 else 0.0

        logger.info(
            "Prediction accuracy: exact=%.2f%% (%d/%d), op_type=%.2f%% (%d/%d) [%.1fs]",
            exact_acc * 100, exact_match, total,
            op_type_acc * 100, op_type_match, total,
            elapsed,
        )

        # Log to TensorBoard via trainer's log method
        metrics = {
            "eval/prediction_accuracy": exact_acc,
            "eval/prediction_accuracy_op_type": op_type_acc,
        }

        if state is not None and hasattr(state, "log_history"):
            # Use the trainer's logging infrastructure
            for key, value in metrics.items():
                if hasattr(control, "_trainer") and control._trainer is not None:
                    control._trainer.log({key: value})

        return metrics

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Evaluate the base model before any training."""
        if self.eval_before_training and model is not None:
            logger.info("Evaluating base model (before training)...")
            metrics = self._evaluate(model, state, control)
            # Log at step 0
            if hasattr(state, "log_history"):
                entry = {"step": 0, **metrics}
                state.log_history.append(entry)
            # Write to TensorBoard via the callback's log method
            if hasattr(args, "logging_dir"):
                self._log_to_tensorboard(args.logging_dir, metrics, global_step=0)

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """Evaluate at the end of each epoch (piggybacks on eval_strategy)."""
        if model is not None:
            metrics = self._evaluate(model, state, control)
            # Log at current step
            global_step = state.global_step if state else 0
            if hasattr(state, "log_history"):
                entry = {"step": global_step, **metrics}
                state.log_history.append(entry)
            if hasattr(args, "logging_dir"):
                self._log_to_tensorboard(args.logging_dir, metrics, global_step)

    @staticmethod
    def _log_to_tensorboard(
        logging_dir: str, metrics: Dict[str, float], global_step: int
    ) -> None:
        """Write metrics to TensorBoard summary."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=logging_dir)
            for key, value in metrics.items():
                writer.add_scalar(key, value, global_step)
            writer.flush()
            writer.close()
        except ImportError:
            logger.debug("tensorboard not available for direct logging")
