"""
LLM-based Solver Implementation
A baseline solver that uses LLMs for prediction.

Three classes are provided:

* :class:`BaseLLMSolver` — shared logic (prompt building, parsing, context
  shortening, config, …).
* :class:`ChatSolver` — sends the prompt as a **chat** conversation (system +
  user messages) via ``adapter.chat_with_response()``.
* :class:`CompletionSolver` — concatenates the prompt into a single string
  and calls ``adapter.complete_text_with_response()``.

For backward compatibility ``LLMSolver`` is an alias for ``ChatSolver``.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union

from next_action_pred_eval.core.operation import Operation
from next_action_pred_eval.core.symbolic import (
    compress_symbolic,
    operations_to_symbolic,
    symbolic_to_operations_detailed,
)
from next_action_pred_eval.evaluation.solver import ISolver, PredictionResult
from next_action_pred_eval.evaluation.baselines.prompts import (
    create_prediction_prompt,
    shorten_symbolic_values,
)
from next_action_pred_eval.utils.llm import LLMAdapter

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# BaseLLMSolver — shared logic for Chat and Completion solvers
# ═══════════════════════════════════════════════════════════════════════════

class BaseLLMSolver(ISolver):
    """
    Base class with all shared logic for LLM-based solvers.

    Subclasses only need to implement :meth:`_call_llm` which performs the
    actual LLM request and returns ``(response_text, input_tokens,
    output_tokens, total_tokens)``.
    """

    def __init__(
        self,
        llm_adapter: LLMAdapter,
        max_context_ops: int = 50,
        max_tokens: int = 2000,
        temperature: float = 0.0,
        max_retries: int = 0,
        num_op_to_pred: Optional[int] = None,
        enable_context_shortening: bool = True,
        context_shortening_max_chars: int = 128,
        context_shortening_corner_cells_dim: int = 3,
        context_shortening_max_cells_2d: Optional[int] = None,
        emit_intent: bool = True,
        remove_sheet_name: bool = True,
        emit_stop_instruction: bool = False,
        custom_system_prompt: Optional[str] = None,
        custom_user_template: Optional[str] = None,
        # Prompt template configuration
        system_prompt: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
        user_prompt: Optional[str] = None,
        user_prompt_file: Optional[str] = None,
        completion_prompt: Optional[str] = None,
        completion_prompt_file: Optional[str] = None,
        # Generation control
        stop_sequences: Optional[List[str]] = None,
        repetition_penalty: Optional[float] = None,
        confidence_threshold: Optional[float] = None,
        # Repetition detection
        detect_repetition: bool = False,
        max_cycle_len: int = 8,
        max_repeats: int = 3,
    ):
        """
        Initialize BaseLLMSolver.

        Args:
            llm_adapter: LLM adapter for making API calls. The adapter handles
                its own retry logic for LLM call errors (rate limits, etc.).
            max_context_ops: Maximum previous operations to include.
            max_tokens: Maximum tokens for generation.
            temperature: LLM temperature for generation.
            max_retries: Solver-level retries for validation failures (e.g., when
                LLM output cannot be parsed into operations). Default 0 means no
                solver-level retries. LLM call retries are handled by the adapter.
            num_op_to_pred: Optional upper bound on the number of operations to
                predict. Injected into the prompt to limit LLM output.
            enable_context_shortening: Shorten long values in context.
            context_shortening_max_chars: Max chars for individual values.
            context_shortening_corner_cells_dim: Rows/cols to keep from each
                corner of large 2D arrays.
            context_shortening_max_cells_2d: Max cells before 2D truncation.
                None = auto (corner_cells_dim^2 * 4).
            emit_intent: Whether LLM should emit an intent line before ops.
            remove_sheet_name: Strip sheet names from context, re-add after.
            custom_system_prompt: *Deprecated* — use *system_prompt* instead.
            custom_user_template: *Deprecated* — use *user_prompt* instead.
            system_prompt: Inline system prompt Jinja2 template string.
            system_prompt_file: Path to system prompt Jinja2 template file.
            user_prompt: Inline user prompt Jinja2 template string.
            user_prompt_file: Path to user prompt Jinja2 template file.
            completion_prompt: Inline completion prompt Jinja2 template string
                (used by CompletionSolver).
            completion_prompt_file: Path to completion prompt Jinja2 template
                file (used by CompletionSolver).
            stop_sequences: Optional list of stop strings passed to the adapter.
                Generation halts when any of these sequences is produced.
            repetition_penalty: Optional repetition penalty for local models.
                Requires an adapter that advertises ``"repetition_penalty"``
                in :attr:`~LLMAdapter.supported_generation_params`.
            confidence_threshold: Optional minimum mean log-probability per
                predicted operation.  When set, the solver requests per-token
                log probabilities from the adapter and drops trailing
                operations whose mean token logprob falls below this value.
                Typical values are in the range ``[-2.0, -0.5]``.
            detect_repetition: Enable post-hoc repetition detection on parsed
                operations.  When ``True``, the solver scans for repeating
                cycles of operations (abstracting away cell references) and
                truncates when a cycle repeats more than *max_repeats* times.
            max_cycle_len: Maximum cycle length (in operations) to scan for.
                Default 8.
            max_repeats: Maximum allowed consecutive repetitions of a cycle
                before truncation.  Output retains up to *max_repeats* copies
                of the cycle plus any non-repeating prefix.  Default 3.
        """
        self.llm_adapter = llm_adapter
        self.max_context_ops = max_context_ops
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.num_op_to_pred = num_op_to_pred
        self.enable_context_shortening = enable_context_shortening
        self.context_shortening_max_chars = context_shortening_max_chars
        self.context_shortening_corner_cells_dim = context_shortening_corner_cells_dim
        self.context_shortening_max_cells_2d = context_shortening_max_cells_2d
        self.emit_intent = emit_intent
        self.remove_sheet_name = remove_sheet_name
        self.emit_stop_instruction = emit_stop_instruction
        self.custom_system_prompt = custom_system_prompt
        self.custom_user_template = custom_user_template
        self.system_prompt = system_prompt
        self.system_prompt_file = system_prompt_file
        self.user_prompt = user_prompt
        self.user_prompt_file = user_prompt_file
        self.completion_prompt = completion_prompt
        self.completion_prompt_file = completion_prompt_file
        self.stop_sequences = stop_sequences
        self.repetition_penalty = repetition_penalty
        self.confidence_threshold = confidence_threshold
        self.detect_repetition = detect_repetition
        self.max_cycle_len = max_cycle_len
        self.max_repeats = max_repeats

        # Validate that the adapter supports the requested parameters
        self._validate_adapter_config()

        logger.info(
            f"{type(self).__name__} initialized: max_context={max_context_ops}, "
            f"max_tokens={max_tokens}, temperature={temperature}"
        )

    # ------------------------------------------------------------------
    # Prompt mode flag — overridden by CompletionSolver
    # ------------------------------------------------------------------

    _is_completion: bool = False
    """Whether this solver uses raw-completion mode."""

    # ------------------------------------------------------------------
    # Configuration validation
    # ------------------------------------------------------------------

    def _validate_adapter_config(self) -> None:
        """Validate that the adapter supports all requested generation parameters.

        Called at ``__init__`` time so that invalid configurations fail fast
        rather than on the first :meth:`predict` call.

        Raises:
            ValueError: If the adapter does not advertise support for a
                requested parameter.
        """
        supported = getattr(self.llm_adapter, "supported_generation_params", frozenset())
        unsupported = []

        if self.repetition_penalty is not None and "repetition_penalty" not in supported:
            unsupported.append("repetition_penalty")

        if unsupported:
            adapter_name = type(self.llm_adapter).__name__
            raise ValueError(
                f"{adapter_name} does not support: {', '.join(unsupported)}. "
                f"Supported extra params: {sorted(supported) if supported else 'none'}. "
                f"Use a compatible adapter (e.g., LocalModelAdapter for repetition_penalty)."
            )

    # ------------------------------------------------------------------
    # Confidence-based truncation
    # ------------------------------------------------------------------

    def _truncate_by_confidence(
        self,
        predicted_symbolic: List[str],
        token_logprobs: Optional[List[float]],
        tokens: Optional[List[str]],
    ) -> List[str]:
        """Drop trailing operations whose tokens have low mean log-probability.

        For each parsed operation string the method locates the corresponding
        token span in the full generated text, computes the mean log-prob of
        those tokens, and keeps the operation only if it meets the threshold.
        The first operation that falls below the threshold causes all
        subsequent operations to be dropped as well (greedy left-to-right
        truncation).

        Args:
            predicted_symbolic: Parsed operation strings from
                :meth:`_parse_response`.
            token_logprobs: Per-token log probabilities from the adapter.
            tokens: Token strings (parallel to *token_logprobs*).

        Returns:
            The (possibly truncated) list of operations.
        """
        if (
            self.confidence_threshold is None
            or not predicted_symbolic
            or not token_logprobs
            or not tokens
        ):
            return predicted_symbolic

        # Rebuild the full generated text and build a char→token-index map.
        char_to_tok: List[int] = []
        for tok_idx, tok in enumerate(tokens):
            char_to_tok.extend([tok_idx] * len(tok))
        full_text = "".join(tokens)

        confident: List[str] = []
        search_start = 0

        for op in predicted_symbolic:
            pos = full_text.find(op, search_start)
            if pos == -1:
                # Cannot locate the operation in the token stream — keep it
                # since we have no evidence to reject it.
                confident.append(op)
                continue

            end = pos + len(op)
            tok_start = char_to_tok[pos]
            tok_end = char_to_tok[min(end, len(char_to_tok)) - 1] + 1
            span_lps = token_logprobs[tok_start:tok_end]

            if not span_lps:
                confident.append(op)
                continue

            mean_lp = sum(span_lps) / len(span_lps)
            if mean_lp < self.confidence_threshold:
                logger.debug(
                    f"Confidence cutoff at operation {len(confident)}: "
                    f"mean_logprob={mean_lp:.4f} < {self.confidence_threshold}"
                )
                break

            confident.append(op)
            search_start = end

        if len(confident) < len(predicted_symbolic):
            logger.info(
                f"Confidence truncation: kept {len(confident)}/{len(predicted_symbolic)} operations"
            )

        return confident

    # ------------------------------------------------------------------
    # Repetition detection
    # ------------------------------------------------------------------

    @staticmethod
    def _make_op_signature(op: str) -> str:
        """Create a cell-reference-agnostic signature for an operation.

        Splits on ``' | '``, drops ``parts[1]`` (the cell reference),
        and rejoins the operation type with the value parts.

        Examples::

            "FILL_COLOR | D2:K2 | #FFE69A"  ->  "FILL_COLOR | #FFE69A"
            "FONT_BOLD | D2:K2 | True"       ->  "FONT_BOLD | True"
            "MERGE | N34:N35 | true"         ->  "MERGE | true"
        """
        parts = op.split(" | ")
        if len(parts) < 2:
            return op
        op_type = parts[0]
        value_parts = parts[2:]
        if value_parts:
            return op_type + " | " + " | ".join(value_parts)
        return op_type

    def _detect_and_truncate_repetition(
        self,
        predicted_symbolic: List[str],
    ) -> tuple:
        """Detect repeating operation cycles and truncate excess repetitions.

        Builds a *signature* for each operation by stripping cell references,
        then scans for cycles of length ``1..max_cycle_len`` that repeat more
        than ``max_repeats`` consecutive times.  When found, the output is
        truncated to keep only ``max_repeats`` copies of the cycle plus any
        non-repeating prefix.

        Args:
            predicted_symbolic: Parsed operation strings.

        Returns:
            ``(ops, info)`` where *ops* is the (possibly truncated) list and
            *info* is a dict with detection metadata.
        """
        if not predicted_symbolic:
            return predicted_symbolic, {"repetition_detected": False}

        n = len(predicted_symbolic)
        signatures = [self._make_op_signature(op) for op in predicted_symbolic]

        for k in range(1, self.max_cycle_len + 1):
            if k > n:
                break
            for start in range(n - k):
                cycle = signatures[start : start + k]
                count = 1
                pos = start + k
                while pos + k <= n and signatures[pos : pos + k] == cycle:
                    count += 1
                    pos += k
                if count > self.max_repeats:
                    keep = start + k * self.max_repeats
                    truncated = predicted_symbolic[:keep]
                    info = {
                        "repetition_detected": True,
                        "cycle_length": k,
                        "cycle_count": count,
                        "cycle_start_index": start,
                        "ops_before_cycle": start,
                        "ops_kept": keep,
                        "ops_removed": n - keep,
                        "cycle_signatures": cycle,
                    }
                    logger.info(
                        "Repetition detected: %d-op cycle repeated %d times "
                        "starting at op %d. Truncated to %d/%d ops.",
                        k, count, start, keep, n,
                    )
                    return truncated, info

        return predicted_symbolic, {"repetition_detected": False}

    # ------------------------------------------------------------------
    # Abstract LLM call — implemented by ChatSolver / CompletionSolver
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
        messages: List[Dict[str, str]],
    ) -> tuple:
        """Perform the LLM request.

        Returns:
            ``(response_text, input_tokens, output_tokens, total_tokens,
            token_logprobs, tokens, top_logprobs)``

            *token_logprobs*, *tokens*, and *top_logprobs* are ``None``
            when log probabilities were not requested.

        Subclasses **must** implement this.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Predict (shared)
    # ------------------------------------------------------------------

    def predict(
        self,
        previous_actions: List[Union[Operation, str]],
        workbook_state: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> PredictionResult:
        """
        Predict next operations using LLM.

        Args:
            previous_actions: Previous operations.
            workbook_state: Optional current workbook state (unused).
            context: Optional additional context (unused).

        Returns:
            PredictionResult with predictions and metadata.
        """
        start_time = time.time()

        # Convert to symbolic if needed
        if previous_actions and isinstance(previous_actions[0], Operation):
            symbolic_actions = operations_to_symbolic(previous_actions)
        else:
            symbolic_actions = list(previous_actions)

        # Apply context window limit
        if len(symbolic_actions) > self.max_context_ops:
            symbolic_actions = symbolic_actions[-self.max_context_ops:]

        # Shorten values if enabled
        if self.enable_context_shortening:
            symbolic_actions = shorten_symbolic_values(
                symbolic_actions,
                max_value_length=self.context_shortening_max_chars,
                max_cells_2d=self.context_shortening_max_cells_2d,
                corner_cells_dim=self.context_shortening_corner_cells_dim,
            )

        # Extract sheet_name from context if provided
        sheet_name = None
        if context and isinstance(context, dict):
            sheet_name = context.get("sheet_name")

        # Strip sheet names to save tokens — we re-add them after LLM response
        if self.remove_sheet_name:
            symbolic_actions = compress_symbolic(symbolic_actions, remove_sheet_name=True)

        # Create prompt
        system_prompt, user_prompt = create_prediction_prompt(
            symbolic_actions,
            max_context=self.max_context_ops,
            sheet_name=sheet_name if self.remove_sheet_name else None,
            num_op_to_pred=self.num_op_to_pred,
            emit_intent=self.emit_intent,
            is_completion=self._is_completion,
            system_prompt_file=self.system_prompt_file,
            system_prompt=self.system_prompt,
            user_prompt_file=self.user_prompt_file,
            user_prompt=self.user_prompt,
            completion_prompt_file=self.completion_prompt_file,
            completion_prompt=self.completion_prompt,
            custom_system_prompt=self.custom_system_prompt,
            custom_user_template=self.custom_user_template,
            emit_stop_instruction=self.emit_stop_instruction,
        )

        # Build messages — chat mode gives (system_str, user_str),
        # completion mode gives (None, completion_str).
        if system_prompt is not None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        else:
            messages = [
                {"role": "user", "content": user_prompt},
            ]

        # Make LLM call
        response_text = ""
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        token_logprobs = None
        tokens = None
        top_logprobs = None

        try:
            (
                response_text, input_tokens, output_tokens, total_tokens,
                token_logprobs, tokens, top_logprobs,
            ) = self._call_llm(
                system_prompt, user_prompt, messages,
            )
        except RuntimeError:
            raise
        except Exception as e:
            import traceback
            logger.error(f"LLM call failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return PredictionResult(
                predicted_operations=[],
                predicted_symbolic=[],
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                generation_time=time.time() - start_time,
                metadata={
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "prompt_messages": messages,
                },
            )

        # Extract intent line from response (if emit_intent is enabled)
        intent = None
        if self.emit_intent:
            for line in response_text.strip().split("\n"):
                line = line.strip()
                if line.lower().startswith("intent:"):
                    intent = line[len("intent:"):].strip()
                    break

        # Parse response
        predicted_symbolic = self._parse_response(response_text)

        # Apply repetition detection (if enabled)
        repetition_info = {"repetition_detected": False}
        if self.detect_repetition:
            predicted_symbolic, repetition_info = self._detect_and_truncate_repetition(
                predicted_symbolic,
            )

        # Apply confidence-based truncation before sheet-name restoration
        predicted_symbolic = self._truncate_by_confidence(
            predicted_symbolic, token_logprobs, tokens,
        )

        predicted_operations = []

        # Re-add sheet name to LLM output (stripped before sending)
        if self.remove_sheet_name and sheet_name and predicted_symbolic:
            restored = []
            for op in predicted_symbolic:
                parts = op.split(" | ")
                if len(parts) >= 2 and "!" not in parts[1]:
                    parts[1] = f"{sheet_name}!{parts[1]}"
                # PASTE_FROM source range (parts[2]) — add sheet if missing
                if len(parts) >= 3 and parts[0] == "PASTE_FROM" and "!" not in parts[2]:
                    parts[2] = f"{sheet_name}!{parts[2]}"
                restored.append(" | ".join(parts))
            predicted_symbolic = restored

        parse_failures: list = []
        if predicted_symbolic:
            parse_result = symbolic_to_operations_detailed(predicted_symbolic)
            predicted_operations = parse_result.valid_operations
            predicted_symbolic = parse_result.valid_symbolic  # sync with valid only
            if parse_result.failed_entries:
                for entry, reason in zip(
                    parse_result.failed_entries, parse_result.failed_reasons
                ):
                    logger.warning(
                        "Dropped unparseable prediction: '%s' (%s)", entry, reason
                    )
                parse_failures = [
                    {"symbolic": s, "reason": r}
                    for s, r in zip(
                        parse_result.failed_entries, parse_result.failed_reasons
                    )
                ]

        generation_time = time.time() - start_time

        logger.debug(
            f"LLM prediction: {len(predicted_operations)} ops in {generation_time:.2f}s"
        )

        return PredictionResult(
            predicted_operations=predicted_operations,
            predicted_symbolic=predicted_symbolic,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            generation_time=generation_time,
            metadata={
                "context_size": len(symbolic_actions),
                "response_length": len(response_text),
                "intent": intent,
                "prompt_messages": messages,
                "raw_response": response_text,
                "token_logprobs": token_logprobs,
                "tokens": tokens,
                "top_logprobs": top_logprobs,
                "response_metadata": {
                    "model": getattr(self.llm_adapter, "model_name", "unknown"),
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
                "parse_failures": parse_failures,
                "repetition_info": repetition_info,
            },
        )

    # ------------------------------------------------------------------
    # Parse / reset / config (shared)
    # ------------------------------------------------------------------

    def _parse_response(self, response_text: str) -> List[str]:
        """
        Parse LLM response to extract symbolic operations.

        Args:
            response_text: Raw LLM response.

        Returns:
            List of symbolic operation strings.
        """
        lines = response_text.strip().split("\n")
        operations = []

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('//'):
                continue

            # Skip markdown code blocks
            if line.startswith("```"):
                continue

            # Remove numbering if present (e.g., "1. VALUE | ...")
            if line and line[0].isdigit():
                line = line.split('.', 1)[-1].strip()

            # Check if it looks like an operation (has | separator)
            if ' | ' not in line:
                continue

            operations.append(line)

        return operations

    def reset(self) -> None:
        """Reset solver state (no persistent state in this implementation)."""
        pass

    def get_config(self) -> Dict[str, Any]:
        """Return solver configuration."""
        config = {
            "solver_class": type(self).__name__,
            "max_context_ops": self.max_context_ops,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "max_retries": self.max_retries,
            "enable_context_shortening": self.enable_context_shortening,
            "llm_adapter": type(self.llm_adapter).__name__,
        }
        if self.stop_sequences is not None:
            config["stop_sequences"] = self.stop_sequences
        if self.repetition_penalty is not None:
            config["repetition_penalty"] = self.repetition_penalty
        if self.confidence_threshold is not None:
            config["confidence_threshold"] = self.confidence_threshold
        if self.detect_repetition:
            config["detect_repetition"] = True
            config["max_cycle_len"] = self.max_cycle_len
            config["max_repeats"] = self.max_repeats
        return config


# ═══════════════════════════════════════════════════════════════════════════
# ChatSolver — sends prompt as chat messages
# ═══════════════════════════════════════════════════════════════════════════

class ChatSolver(BaseLLMSolver):
    """
    LLM solver that sends the prompt as a chat conversation.

    Uses ``adapter.chat_with_response()`` to get predictions.

    Example::

        from next_action_pred_eval.utils.llm import OpenAIAdapter

        adapter = OpenAIAdapter(api_key="your-key", model="gpt-4")
        solver = ChatSolver(llm_adapter=adapter, max_context_ops=50)
        result = solver.predict(previous_actions)
    """

    def _call_llm(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
        messages: List[Dict[str, str]],
    ) -> tuple:
        """Send messages via chat mode."""
        # Build extra kwargs for the adapter call
        extra_kwargs: Dict[str, Any] = {}
        if self.stop_sequences is not None:
            extra_kwargs["stop"] = self.stop_sequences
        if self.repetition_penalty is not None:
            extra_kwargs["repetition_penalty"] = self.repetition_penalty
        if self.confidence_threshold is not None:
            extra_kwargs["logprobs"] = True

        token_logprobs = None
        tokens = None
        top_logprobs = None

        # Prefer retry-wrapped response method (handles token expiry, rate
        # limits, etc.), then plain response, then string-only retry.
        call_kwargs = dict(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **extra_kwargs,
        )
        if hasattr(self.llm_adapter, "chat_with_response_and_retry"):
            llm_response = self.llm_adapter.chat_with_response_and_retry(**call_kwargs)
        elif hasattr(self.llm_adapter, "chat_with_response"):
            llm_response = self.llm_adapter.chat_with_response(**call_kwargs)
        else:
            llm_response = None

        if llm_response is not None:
            response_text = llm_response.content
            input_tokens = llm_response.prompt_tokens or sum(len(m["content"]) for m in messages) // 4
            output_tokens = llm_response.completion_tokens or len(response_text) // 4
            total_tokens = llm_response.total_tokens or (input_tokens + output_tokens)
            token_logprobs = llm_response.token_logprobs
            tokens = llm_response.tokens
            top_logprobs = llm_response.top_logprobs
        else:
            response_text = self.llm_adapter.chat_with_retry(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            input_tokens = sum(len(m["content"]) for m in messages) // 4
            output_tokens = len(response_text) // 4
            total_tokens = input_tokens + output_tokens

        return response_text, input_tokens, output_tokens, total_tokens, token_logprobs, tokens, top_logprobs


# ═══════════════════════════════════════════════════════════════════════════
# CompletionSolver — sends prompt as a single string
# ═══════════════════════════════════════════════════════════════════════════

class CompletionSolver(BaseLLMSolver):
    """
    LLM solver that uses a single, minimalistic completion prompt and
    the raw ``/completions`` endpoint.

    Uses ``adapter.complete_text_with_response()`` to get predictions.
    The default prompt (``DEFAULT_COMPLETION_TEMPLATE``) is intentionally
    short — override via *completion_prompt* or *completion_prompt_file*.

    Example::

        from next_action_pred_eval.utils.llm import OpenAIAdapter

        adapter = OpenAIAdapter(api_key="your-key", model="gpt-4")
        solver = CompletionSolver(llm_adapter=adapter, max_context_ops=50)
        result = solver.predict(previous_actions)
    """

    _is_completion: bool = True

    def _call_llm(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
        messages: List[Dict[str, str]],
    ) -> tuple:
        """Send the single completion prompt via the completion endpoint."""
        # In completion mode, create_prediction_prompt returns (None, prompt_str).
        # user_prompt already contains the full prompt.
        prompt = user_prompt

        # Build extra kwargs for the adapter call
        extra_kwargs: Dict[str, Any] = {}
        # Pass stop sequences to the adapter.  When self.stop_sequences is
        # None (default) we add "STOP" since the default completion prompt
        # tells the model to write it.  When explicitly set (even to []),
        # honour the caller's choice without force-adding "STOP".
        if self.stop_sequences is None:
            extra_kwargs["stop"] = ["STOP"]
        else:
            extra_kwargs["stop"] = list(self.stop_sequences)
        if self.repetition_penalty is not None:
            extra_kwargs["repetition_penalty"] = self.repetition_penalty
        # Always request logprobs for completion mode (needed for confidence
        # truncation and useful for diagnostics)
        extra_kwargs["logprobs"] = True

        token_logprobs = None
        tokens = None
        top_logprobs = None

        # Prefer retry-wrapped response method, then plain response, then
        # string-only retry.
        call_kwargs = dict(
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **extra_kwargs,
        )
        if hasattr(self.llm_adapter, "complete_text_with_response_and_retry"):
            llm_response = self.llm_adapter.complete_text_with_response_and_retry(**call_kwargs)
        elif hasattr(self.llm_adapter, "complete_text_with_response"):
            llm_response = self.llm_adapter.complete_text_with_response(**call_kwargs)
        else:
            llm_response = None

        if llm_response is not None:
            response_text = llm_response.content
            input_tokens = llm_response.prompt_tokens or len(prompt) // 4
            output_tokens = llm_response.completion_tokens or len(response_text) // 4
            total_tokens = llm_response.total_tokens or (input_tokens + output_tokens)
            token_logprobs = llm_response.token_logprobs
            tokens = llm_response.tokens
            top_logprobs = llm_response.top_logprobs
        else:
            response_text = self.llm_adapter.complete_text_with_retry(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            input_tokens = len(prompt) // 4
            output_tokens = len(response_text) // 4
            total_tokens = input_tokens + output_tokens

        return response_text, input_tokens, output_tokens, total_tokens, token_logprobs, tokens, top_logprobs


# ═══════════════════════════════════════════════════════════════════════════
# Backward-compatible alias
# ═══════════════════════════════════════════════════════════════════════════

LLMSolver = ChatSolver
"""Backward-compatible alias — ``LLMSolver`` is now :class:`ChatSolver`."""


__all__ = ["BaseLLMSolver", "ChatSolver", "CompletionSolver", "LLMSolver"]
