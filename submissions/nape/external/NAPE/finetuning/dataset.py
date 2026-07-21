"""
PyTorch Dataset for Operation Sequence Training
================================================

Provides :class:`OperationSequenceDataset` — a map-style Dataset that
creates training examples from **raw** operation sequences.

Each example is a window of operations preprocessed **on-the-fly**
(value shortening, sheet-name stripping), formatted with a configurable
prompt template, and tokenised.

**Context-target mode** (``max_target_ops > 0``):
    For each sequence the *first* window (``start=0``) uses **full-loss**
    causal-LM loss on all tokens — this implicitly trains the model on
    every short-context length (1-op, 5-op, …, ``max_context_ops``),
    matching the eval setting where the orchestrator calls predict at
    every step.  All subsequent windows use the **context / target
    split**: the first ``max_context_ops`` ops are masked (``-100``)
    and loss is computed only on the next ``max_target_ops`` ops.  This
    teaches the model to generate continuations given a full context
    prompt — the primary eval task.

**Full-loss mode** (``max_target_ops`` is ``None`` or 0):
    Every window uses full causal-LM loss on all tokens.  This is the
    default.

Preprocessing happens in ``__getitem__``, not at data-export time.
Changing a preprocessing param in the config takes effect immediately
without re-exporting the JSONL file.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Ensure the parent package is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from next_action_pred_eval.core.symbolic import compress_symbolic          # noqa: E402
from next_action_pred_eval.evaluation.baselines.prompts import (           # noqa: E402
    shorten_symbolic_values,
)

logger = logging.getLogger(__name__)

# Default prompt template — matches CompletionSolver's DEFAULT_COMPLETION_TEMPLATE.
# Must contain ``{actions}`` placeholder where the numbered action list goes.
DEFAULT_PROMPT_TEMPLATE = (
    "Complete the sequence of actions to build the following "
    "spreadsheet by identifying and extending key patterns.\n\n{actions}"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_sheet_name(raw_ops: List[str]) -> Optional[str]:
    """Extract the sheet name from the first operation that has one."""
    for op in raw_ops:
        parts = op.split(" | ")
        if len(parts) >= 2 and "!" in parts[1]:
            return parts[1].rsplit("!", 1)[0]
    return None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OperationSequenceDataset(Dataset):
    """Map-style dataset with on-the-fly preprocessing and tokenisation.

    Parameters
    ----------
    sequences : list[dict]
        Raw sequences from JSONL.  Each dict must have ``"id"`` (str)
        and ``"ops"`` (list[str] — raw operation strings).
    tokenizer
        A HuggingFace tokenizer instance.
    max_context_ops : int
        Maximum context operations (prompt portion).  When
        ``max_target_ops`` is set, context labels are masked (-100).
    max_target_ops : int or None
        Maximum target operations (continuation portion).  When set,
        each window = context + target; only target tokens receive
        loss.  When ``None`` or 0, full-loss mode: every window up to
        ``max_context_ops`` with loss on all tokens.
    example_stride : int
        Step (in operations) between consecutive windows within a
        sequence.  Smaller → more examples per sequence, more overlap.
    max_seq_len : int
        Hard token limit per example.  Truncated after tokenisation.
    enable_context_shortening : bool
        Apply ``shorten_symbolic_values`` to truncate long cell values.
    context_shortening_max_chars : int
        Max characters per cell value  (default 32 — matches ft_models/base_run.yaml).
    context_shortening_corner_cells_dim : int
        Corner rows/cols for large 2-D array values  (default 2).
    remove_sheet_name : bool
        Strip sheet-name prefixes via ``compress_symbolic``.
    include_sheet_name : bool
        Prepend ``Active sheet: …`` to the prompt (experimental, off by
        default — not in the standard CompletionSolver prompt).
    prompt_template : str
        Format string for the completion prompt.  Must contain an
        ``{actions}`` placeholder where the numbered action list goes.
        Default matches ``DEFAULT_COMPLETION_TEMPLATE`` from the solver.
    """

    def __init__(
        self,
        sequences: List[Dict],
        tokenizer,
        max_context_ops: int = 128,
        max_target_ops: Optional[int] = None,
        example_stride: int = 64,
        max_seq_len: int = 2048,
        # Preprocessing
        enable_context_shortening: bool = True,
        context_shortening_max_chars: int = 32,
        context_shortening_corner_cells_dim: int = 2,
        remove_sheet_name: bool = True,
        # Prompt
        include_sheet_name: bool = False,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    ):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_context_ops = max_context_ops
        self.max_target_ops = max_target_ops or 0
        self.max_seq_len = max_seq_len

        # Preprocessing config
        self.enable_context_shortening = enable_context_shortening
        self.context_shortening_max_chars = context_shortening_max_chars
        self.context_shortening_corner_cells_dim = context_shortening_corner_cells_dim
        self.remove_sheet_name = remove_sheet_name

        # Prompt config
        self.include_sheet_name = include_sheet_name
        self.prompt_template = prompt_template

        # Validate template
        if "{actions}" not in prompt_template:
            raise ValueError(
                "prompt_template must contain an {actions} placeholder.  "
                f"Got: {prompt_template!r}"
            )

        # Total ops per window (context + optional target)
        self._total_window = max_context_ops + self.max_target_ops

        # Build flat index: (sequence_index, start_op_position, use_full_loss)
        #
        # In context-target mode (max_target_ops > 0):
        #   - start=0 → full-loss window (causal LM loss on all
        #     tokens, up to max_context_ops ops).  Trains short-context
        #     prediction.
        #   - start>0 → context/target window (context masked, target
        #     has loss).  Trains continuation from full context.
        #
        # In full-loss mode (max_target_ops == 0): all windows use
        # full-loss (the default).
        self.index: List[Tuple[int, int, bool]] = []
        for seq_idx, seq_data in enumerate(sequences):
            n = len(seq_data["ops"])
            if n < 2:
                continue
            for start in range(0, n, example_stride):
                if self.max_target_ops and start > 0:
                    # Context/target window
                    end = min(start + self._total_window, n)
                    if end - start < 2:
                        continue
                    self.index.append((seq_idx, start, False))
                else:
                    # Full-loss window (start==0 in ctx-tgt mode, or all in full-loss mode)
                    end = min(start + self.max_context_ops, n)
                    if end - start < 2:
                        continue
                    self.index.append((seq_idx, start, True))

        n_full_loss = sum(1 for _, _, fl in self.index if fl)
        n_ctx_tgt = len(self.index) - n_full_loss
        logger.info(
            "Dataset: %d sequences -> %d examples (%d full-loss, %d ctx/tgt)  "
            "(max_context_ops=%d, max_target_ops=%d, stride=%d, max_seq_len=%d)",
            len(sequences),
            len(self.index),
            n_full_loss,
            n_ctx_tgt,
            max_context_ops,
            self.max_target_ops,
            example_stride,
            max_seq_len,
        )

    # ------------------------------------------------------------------
    # Preprocessing  (mirrors BaseLLMSolver.predict transform order)
    # ------------------------------------------------------------------

    def _preprocess_ops(self, ops: List[str]) -> List[str]:
        """Apply the same transforms as CompletionSolver at inference.

        Order:  shorten_symbolic_values  →  compress_symbolic
        """
        result = list(ops)

        if self.enable_context_shortening:
            result = shorten_symbolic_values(
                result,
                max_value_length=self.context_shortening_max_chars,
                corner_cells_dim=self.context_shortening_corner_cells_dim,
            )

        if self.remove_sheet_name:
            result = compress_symbolic(result, remove_sheet_name=True)

        return result

    def _format_text(
        self, ops: List[str], sheet_name: Optional[str] = None
    ) -> str:
        """Format preprocessed ops into a completion-prompt string."""
        actions_str = "\n".join(ops)
        text = self.prompt_template.format(actions=actions_str)

        if self.include_sheet_name and sheet_name:
            # Insert "Active sheet: ..." after the first line (experimental)
            parts = text.split("\n", 1)
            text = (
                parts[0]
                + f"\nActive sheet: {sheet_name}\n"
                + (parts[1] if len(parts) > 1 else "")
            )

        return text

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        seq_idx, start, use_full_loss = self.index[idx]
        seq_data = self.sequences[seq_idx]
        raw_ops = seq_data["ops"]

        if use_full_loss:
            return self._getitem_full_loss(raw_ops, start)
        else:
            return self._getitem_context_target(raw_ops, start)

    def _getitem_full_loss(self, raw_ops: List[str], start: int) -> Dict[str, List[int]]:
        """Full-loss window: loss on all tokens."""
        window = raw_ops[start : start + self.max_context_ops]
        processed = self._preprocess_ops(window)

        sheet_name = None
        if self.include_sheet_name:
            sheet_name = _extract_sheet_name(window)

        text = self._format_text(processed, sheet_name)
        encoding = self.tokenizer(
            text, truncation=True, max_length=self.max_seq_len,
            add_special_tokens=True,
        )
        input_ids = encoding["input_ids"]
        return {"input_ids": input_ids, "labels": list(input_ids)}

    def _getitem_context_target(
        self, raw_ops: List[str], start: int,
    ) -> Dict[str, List[int]]:
        """Context/target split: mask context tokens, train on target.

        Layout:  [prompt + context_ops | target_ops]
                  ^^^^ labels = -100 ^^^  ^^ loss ^^^
        """
        full_window = raw_ops[start : start + self._total_window]
        n_window = len(full_window)

        # Split into context and target ops
        n_ctx = min(self.max_context_ops, n_window - 1)  # at least 1 target
        context_ops = full_window[:n_ctx]
        target_ops = full_window[n_ctx:]

        # Preprocess both parts together (shortening sees full context)
        all_processed = self._preprocess_ops(full_window)
        ctx_processed = all_processed[:n_ctx]
        tgt_processed = all_processed[n_ctx:]

        sheet_name = None
        if self.include_sheet_name:
            sheet_name = _extract_sheet_name(full_window)

        # Build context-only text to find the token boundary
        context_text = self._format_text(ctx_processed, sheet_name)
        # Build full text (context + target)
        full_text = self._format_text(all_processed, sheet_name)

        # Tokenize context to find the boundary
        ctx_encoding = self.tokenizer(
            context_text, truncation=True, max_length=self.max_seq_len,
            add_special_tokens=True,
        )
        n_ctx_tokens = len(ctx_encoding["input_ids"])

        # Tokenize full sequence
        full_encoding = self.tokenizer(
            full_text, truncation=True, max_length=self.max_seq_len,
            add_special_tokens=True,
        )
        input_ids = full_encoding["input_ids"]

        # Build labels: -100 for context tokens, real ids for target
        labels = [-100] * min(n_ctx_tokens, len(input_ids))
        labels += input_ids[len(labels):]
        # Ensure same length
        labels = labels[:len(input_ids)]

        # If context alone fills the entire max_seq_len, all labels are
        # -100 → CrossEntropyLoss returns NaN.  Fall back to full-loss
        # on the truncated content so the example still contributes.
        if all(l == -100 for l in labels):
            return {"input_ids": input_ids, "labels": list(input_ids)}

        return {"input_ids": input_ids, "labels": labels}


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

class CausalLMCollator:
    """Pads variable-length examples to the longest in the batch.

    ``labels`` are padded with ``-100`` so that ``CrossEntropyLoss``
    ignores padding positions.
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(
        self, features: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)

        batch_ids: List[List[int]] = []
        batch_labels: List[List[int]] = []
        batch_mask: List[List[int]] = []

        for f in features:
            ids = f["input_ids"]
            lab = f["labels"]
            pad_n = max_len - len(ids)

            batch_ids.append(ids + [self.pad_token_id] * pad_n)
            batch_labels.append(lab + [-100] * pad_n)
            batch_mask.append([1] * len(ids) + [0] * pad_n)

        return {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
        }
