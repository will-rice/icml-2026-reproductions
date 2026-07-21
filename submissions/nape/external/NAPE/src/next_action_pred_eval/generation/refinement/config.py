from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class RefinementConfig:
    """Runtime configuration for the sequence refinement pipeline."""

    step_file: Path
    final_workbook: Path
    sheet_name: str

    output_dir: Path = Path("outputs/sequence_refinement")
    sheet_image_path: Optional[Path] = None
    allow_image_capture: bool = True
    capture_dir: Optional[Path] = None

    max_dimension: Optional[int] = 120
    reference_operation_limit: Optional[int] = None

    max_iterations: int = 4
    max_retries: int = 2
    temperature: float = 0.15
    max_completion_tokens: Optional[int] = 30000
    reasoning_effort: str = "low"

    provider: str = "substrate"
    model: str = "<your-model-id>"
    cache_path: Optional[Path] = Path("caches/sequence_refinement_cache.json")
    use_cache: bool = True

    feedback_cell_limit: Optional[int] = None
    feedback_diff_limit: Optional[int] = None

    humaneness_keyword: str = ""
    compare_formatting: bool = True

    history_summary_limit: int = 2
    repair_operation_limit: int = 8
    api_down_retry_attempts: int = 2
    api_down_retry_delay: float = 3.0
    log_progress: bool = True

    judge_provider: Optional[str] = None
    judge_model: Optional[str] = None
    judge_temperature: float = 0.0
    judge_max_completion_tokens: Optional[int] = 16000  # Higher limit for reasoning models
    judge_reasoning_effort: str = "high"
    judge_cache_path: Optional[Path] = None
    judge_use_cache: bool = True
    judge_keyword: str = "HUMAN_SEQUENCE"

    # Input compression settings (to reduce token usage for large INPUT operations)
    compress_inputs: bool = True  # Enable compression for large INPUT operations
    input_compression_threshold: int = 20  # Compress INPUTs with ranges larger than this many cells
    input_string_truncate_threshold: int = 500  # Truncate individual strings longer than this

    metadata: Dict[str, Any] = field(default_factory=dict)

    def resolve(self) -> "RefinementConfig":
        """Return a copy with absolute paths and defaults applied."""
        resolved = RefinementConfig(
            step_file=self.step_file.resolve(),
            final_workbook=self.final_workbook.resolve(),
            sheet_name=self.sheet_name,
            output_dir=self.output_dir.resolve(),
            sheet_image_path=self.sheet_image_path.resolve() if self.sheet_image_path else None,
            allow_image_capture=self.allow_image_capture,
            capture_dir=(self.capture_dir.resolve() if self.capture_dir else self.output_dir.resolve() / "images"),
            max_dimension=self.max_dimension,
            reference_operation_limit=self.reference_operation_limit,
            max_iterations=self.max_iterations,
            max_retries=self.max_retries,
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
            reasoning_effort=self.reasoning_effort,
            provider=self.provider,
            model=self.model,
            cache_path=self.cache_path.resolve() if self.cache_path else None,
            use_cache=self.use_cache,
            feedback_cell_limit=self.feedback_cell_limit,
            feedback_diff_limit=self.feedback_diff_limit,
            humaneness_keyword=self.humaneness_keyword,
            compare_formatting=self.compare_formatting,
            history_summary_limit=self.history_summary_limit,
            repair_operation_limit=self.repair_operation_limit,
            api_down_retry_attempts=self.api_down_retry_attempts,
            api_down_retry_delay=self.api_down_retry_delay,
            log_progress=self.log_progress,
            judge_provider=self.judge_provider or self.provider,
            judge_model=self.judge_model or self.model,
            judge_temperature=self.judge_temperature,
            judge_max_completion_tokens=self.judge_max_completion_tokens,
            judge_reasoning_effort=self.judge_reasoning_effort,
            judge_cache_path=(
                self.judge_cache_path.resolve()
                if self.judge_cache_path
                else (self.cache_path.resolve() if self.cache_path else None)
            ),
            judge_use_cache=self.judge_use_cache,
            judge_keyword=self.judge_keyword,
            compress_inputs=self.compress_inputs,
            input_compression_threshold=self.input_compression_threshold,
            input_string_truncate_threshold=self.input_string_truncate_threshold,
            metadata=self.metadata.copy(),
        )
        resolved.output_dir.mkdir(parents=True, exist_ok=True)
        resolved.capture_dir.mkdir(parents=True, exist_ok=True)
        return resolved

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for key in [
            "step_file",
            "final_workbook",
            "output_dir",
            "sheet_image_path",
            "capture_dir",
            "cache_path",
            "judge_cache_path",
        ]:
            if payload.get(key) is not None:
                payload[key] = str(payload[key])
        return payload
