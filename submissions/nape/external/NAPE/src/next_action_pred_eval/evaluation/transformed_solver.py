"""Solver wrapper that applies composable symbolic transforms.

The ``TransformedSolver`` wraps any ``ISolver`` with a pipeline of
``SymbolicTransform`` instances. It encodes the operation history
before passing it to the inner solver, and decodes predictions back
to standard DSL.

Encoding is incremental: each call to ``predict()`` only processes new
operations since the last call, so no work is repeated across steps.

The solver is robust to truncated context windows: the orchestrator may
pass only the last *N* operations (``max_context_ops``), but the solver
keeps the full encoded history internally and correctly identifies new
operations by matching the tail of the incoming list against its own
raw history.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from next_action_pred_eval.core.operation import Operation
from next_action_pred_eval.core.symbolic import symbolic_to_operations_detailed
from next_action_pred_eval.core.transforms.base import SymbolicTransform
from next_action_pred_eval.evaluation.solver import ISolver, PredictionResult

logger = logging.getLogger(__name__)


class TransformedSolver(ISolver):
    """Wraps any ISolver with a pipeline of SymbolicTransforms.

    The transform pipeline:

    1. **Encode** — Converts standard DSL history through each transform
       in order (e.g. RelativeFormula → RelativeRange → ValueLookup).
    2. **Predict** — Passes encoded history to the inner solver.
    3. **Decode** — Converts predictions back through the transforms in
       *reverse* order.

    Handles truncated context windows correctly: the orchestrator may
    truncate history via ``max_context_ops`` before calling ``predict()``.
    This solver keeps the full encoded history internally and identifies
    truly new operations by matching the incoming list's tail against its
    stored raw history.
    """

    def __init__(
        self,
        inner: ISolver,
        transforms: List[SymbolicTransform],
    ):
        self.inner = inner
        self.transforms = transforms
        self._encoded_history: List[str] = []
        self._raw_history: List[str] = []

    def predict(
        self,
        previous_actions: List[Union[Operation, str]],
        workbook_state: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> PredictionResult:
        # 1. Convert to symbolic strings
        sym_ops: list = []
        for a in previous_actions:
            if isinstance(a, str):
                sym_ops.append(a)
            else:
                sym_ops.append(a.to_symbolic())

        # 2. Detect new operations — robust to context truncation.
        #
        # The incoming list may be a truncated suffix of the full history
        # (e.g. last 32 ops when max_context_ops=32). We find the overlap
        # between the incoming list and our stored raw history to identify
        # which operations are truly new.
        n_new = self._find_new_ops(sym_ops)

        if n_new < 0:
            # No overlap found — new trajectory or incompatible rewind.
            self._reset_state()
            n_new = len(sym_ops)

        # 3. Encode only the truly new operations incrementally
        new_ops = sym_ops[len(sym_ops) - n_new:] if n_new > 0 else []
        for raw_op in new_ops:
            encoded = raw_op
            for t in self.transforms:
                encoded = t.encode_one(encoded)
            self._encoded_history.append(encoded)
            self._raw_history.append(raw_op)

        # 4. Pass the encoded window (matching input length) to inner solver
        window = len(sym_ops)
        encoded_context = self._encoded_history[-window:]

        result = self.inner.predict(
            encoded_context, workbook_state, context
        )

        # 5. Recover ALL raw prediction strings before decoding.
        #
        # The inner solver (e.g. CompletionSolver) may parse predictions
        # internally via symbolic_to_operations_detailed(), discarding any
        # that fail CellRange validation. When transforms are active, delta
        # ranges like (0,0,9,-3) fail validation and get dropped into
        # metadata["parse_failures"]. We recover them here.
        raw_predictions = self._recover_raw_predictions(result)

        # 6. Decode predictions (reverse transform order)
        decoded_sym = list(raw_predictions)
        for t in reversed(self.transforms):
            decoded_sym = t.decode_predictions(decoded_sym)

        # 7. Parse decoded strings to Operation objects
        if decoded_sym:
            parsed = symbolic_to_operations_detailed(decoded_sym)
            pred_ops = parsed.valid_operations
            pred_sym = parsed.valid_symbolic
        else:
            pred_ops, pred_sym = [], []

        return PredictionResult(
            predicted_operations=pred_ops,
            predicted_symbolic=pred_sym,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            total_tokens=result.total_tokens,
            generation_time=result.generation_time,
            metadata={
                **result.metadata,
                "transforms": [t.get_config() for t in self.transforms],
                "raw_predictions": list(raw_predictions),
            },
        )

    @staticmethod
    def _recover_raw_predictions(result: PredictionResult) -> List[str]:
        """Reconstruct the full ordered list of raw prediction strings.

        The inner solver may split predictions into valid (predicted_symbolic)
        and failed (metadata["parse_failures"]). We need them all for
        transform decoding.

        If the inner solver stored parse_failures, we reconstruct the
        original order: valid entries keep their positions, failed entries
        fill the gaps. When all predictions are either valid or all failed
        (the common cases), this is trivial.
        """
        valid = list(result.predicted_symbolic)
        failures = result.metadata.get("parse_failures", [])

        if not failures:
            # No parse failures: inner solver's valid list is complete
            return valid

        if not valid:
            # All failed (typical for transform-encoded predictions):
            # parse_failures is already in order
            return [f["symbolic"] for f in failures]

        # Mixed case: both valid and failed predictions exist.
        # We don't have perfect ordering info, so return valid first,
        # then failed. This is a best-effort approximation — in practice
        # transforms will cause either all-valid or all-failed.
        logger.debug(
            "Mixed valid/failed predictions: %d valid, %d failed",
            len(valid), len(failures),
        )
        return valid + [f["symbolic"] for f in failures]

    def _find_new_ops(self, sym_ops: List[str]) -> int:
        """Return the number of truly new operations in *sym_ops*.

        Compares the incoming list (which may be a truncated window)
        against the stored ``_raw_history``. Returns ≥0 when a valid
        suffix match is found, or -1 when no overlap is detected
        (indicating a new trajectory / incompatible rewind).

        Strategy: the incoming list should be a suffix of
        ``_raw_history + new_ops``.  We check how much of the incoming
        list overlaps with the tail of ``_raw_history``.
        """
        if not self._raw_history:
            return len(sym_ops)

        stored_len = len(self._raw_history)

        # Fast path: incoming list extends the full history (no truncation).
        if len(sym_ops) >= stored_len:
            # Spot-check: first op and last stored op must still match.
            if (sym_ops[0] == self._raw_history[0]
                    and sym_ops[stored_len - 1] == self._raw_history[-1]):
                return len(sym_ops) - stored_len

        # Truncated window: incoming list is shorter than stored history.
        # The incoming list should be _raw_history[-k:] + new_ops for some k.
        # Find the largest k such that sym_ops[:k] == _raw_history[-k:].
        incoming_len = len(sym_ops)
        max_overlap = min(incoming_len, stored_len)
        for k in range(max_overlap, 0, -1):
            if sym_ops[:k] == self._raw_history[-k:]:
                return incoming_len - k

        # No overlap found — new trajectory or incompatible history.
        return -1

    def _reset_state(self) -> None:
        for t in self.transforms:
            t.reset()
        self._encoded_history = []
        self._raw_history = []

    def reset(self) -> None:
        self._reset_state()
        self.inner.reset()

    def get_config(self) -> Dict[str, Any]:
        return {
            "solver_class": "TransformedSolver",
            "transforms": [t.get_config() for t in self.transforms],
            "inner_solver": self.inner.get_config(),
        }
