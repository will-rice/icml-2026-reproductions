"""Sequence refinement package.

Provides an LLM-based pipeline for refining operation sequences to read
more naturally (as if a human were building the spreadsheet step by step).

Key components:
- RefinementConfig: Runtime configuration for the pipeline
- SequenceRefinementPipeline: Main orchestrator (judge-editor loop)
- RefinementOutcome: Result dataclass
- RefinementLLMAdapter: Adapter bridging next_action_pred_eval's LLMAdapter
  to the refinement pipeline's expected interface
- Input normalization utilities for compressing/decompressing large INPUT
  operations to reduce token usage
"""

from .config import RefinementConfig
from .pipeline import SequenceRefinementPipeline, RefinementOutcome
from .llm_adapter import (
    LLMServiceUnavailableError,
    Message,
    RefinementLLMAdapter,
    Role,
)
from .input_normalizer import (
    compress_operations,
    decompress_operations,
    compress_input_operation,
    decompress_input_operation,
    is_compressed_input,
    get_compression_docs,
    CompressionStats,
    normalize_for_prompt,
    denormalize_from_response,
)

__all__ = [
    "RefinementConfig",
    "SequenceRefinementPipeline",
    "RefinementOutcome",
    # LLM adapter
    "RefinementLLMAdapter",
    "LLMServiceUnavailableError",
    "Message",
    "Role",
    # Input normalization utilities
    "compress_operations",
    "decompress_operations",
    "compress_input_operation",
    "decompress_input_operation",
    "is_compressed_input",
    "get_compression_docs",
    "CompressionStats",
    "normalize_for_prompt",
    "denormalize_from_response",
]
