"""
Baselines Module
Provides baseline solver implementations for evaluation.
"""

from next_action_pred_eval.evaluation.baselines.llm_solver import (
    BaseLLMSolver,
    ChatSolver,
    CompletionSolver,
    LLMSolver,  # backward-compatible alias for ChatSolver
)
from next_action_pred_eval.evaluation.baselines.prompts import (
    DEFAULT_SYSTEM_TEMPLATE,
    DEFAULT_USER_TEMPLATE,
    DEFAULT_COMPLETION_TEMPLATE,
    # Backward-compatible aliases
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT_TEMPLATE,
    create_prediction_prompt,
    load_prompt_template,
    render_prompt_template,
)
from next_action_pred_eval.evaluation.baselines.feature_solver import FeatureSolver
from next_action_pred_eval.evaluation.baselines.ngram_solver import NGramSolver
from next_action_pred_eval.evaluation.baselines.online_ngram_solver import OnlineNGramSolver

# Optional imports — these solvers have heavier dependencies (xgboost, torch)
try:
    from next_action_pred_eval.evaluation.baselines.xgboost_solver import XGBoostSolver
except ImportError:
    pass

try:
    from next_action_pred_eval.evaluation.baselines.lstm_solver import LSTMSolver
except ImportError:
    pass

__all__ = [
    "BaseLLMSolver",
    "ChatSolver",
    "CompletionSolver",
    "LLMSolver",
    "NGramSolver",
    "XGBoostSolver",
    "LSTMSolver",
    "OnlineNGramSolver",
    "DEFAULT_SYSTEM_TEMPLATE",
    "DEFAULT_USER_TEMPLATE",
    "DEFAULT_COMPLETION_TEMPLATE",
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_USER_PROMPT_TEMPLATE",
    "create_prediction_prompt",
    "load_prompt_template",
    "render_prompt_template",
]
