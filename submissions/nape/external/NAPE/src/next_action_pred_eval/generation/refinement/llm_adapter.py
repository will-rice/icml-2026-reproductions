"""
LLM adapter for the sequence refinement pipeline.

Wraps the package's ``LLMAdapter`` interface to expose the simple shape
that the refinement pipeline expects:
  - complete(messages, request_overrides) -> response with .text and .usage

This module provides:
  - Message / Role dataclasses that match the pipeline's expectations
  - LLMServiceUnavailableError for outage detection
  - RefinementLLMAdapter that wraps the base LLMAdapter
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from next_action_pred_eval.utils.llm.base import LLMAdapter as BaseLLMAdapter

logger = logging.getLogger(__name__)


class LLMServiceUnavailableError(RuntimeError):
    """Raised when the backing LLM service returns a non-JSON outage payload."""


class Role(Enum):
    """Chat message roles."""
    System = "system"
    User = "user"
    Assistant = "assistant"


@dataclass
class Message:
    """Chat message compatible with the pipeline's expectations."""
    role: Role
    content: str
    image: Optional[str] = None


@dataclass
class TokenUsage:
    """Token usage information returned from LLM calls."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def model_dump(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class LLMResponse:
    """Response object matching the pipeline's expected interface."""
    text: str
    usage: Optional[TokenUsage] = None


class RefinementLLMAdapter:
    """
    Wraps next_action_pred_eval's LLMAdapter for use in the refinement pipeline.

    The pipeline calls `adapter.complete(messages, request_overrides=...)` and expects
    a response with `.text` (str) and `.usage` (with prompt_tokens, completion_tokens,
    total_tokens attributes).

    This adapter translates between the pipeline's Message objects and the base
    LLMAdapter's dict-based message format.
    """

    supports_reasoning: bool = False

    def __init__(
        self,
        base_adapter: BaseLLMAdapter,
        *,
        temperature: float = 0.0,
        max_completion_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        supports_reasoning: bool = False,
    ) -> None:
        self._adapter = base_adapter
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.reasoning_effort = reasoning_effort
        self.supports_reasoning = supports_reasoning

    def complete(
        self,
        messages: List[Message],
        request_overrides: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """
        Send a chat completion request through the base adapter.

        Args:
            messages: List of Message objects (with role, content, optional image).
            request_overrides: Optional dict to override temperature, max_tokens, etc.

        Returns:
            LLMResponse with .text and .usage fields.
        """
        # Convert Message objects to dicts expected by BaseLLMAdapter.complete()
        dict_messages = []
        for msg in messages:
            entry: Dict[str, Any] = {
                "role": msg.role.value,
                "content": msg.content,
            }
            # If the message has an image, add it as an image_url content block
            if msg.image:
                entry["image"] = msg.image
            dict_messages.append(entry)

        # Merge default params with overrides
        kwargs: Dict[str, Any] = {}
        temperature = self.temperature
        max_tokens = self.max_completion_tokens

        if request_overrides:
            temperature = request_overrides.get("temperature", temperature)
            max_tokens = request_overrides.get("max_completion_tokens", max_tokens)
            # Pass through any extra kwargs (reasoning_effort, etc.)
            for key, val in request_overrides.items():
                if key not in ("temperature", "max_completion_tokens") and val is not None:
                    kwargs[key] = val

        try:
            response_text = self._adapter.complete(
                dict_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        except ValueError as exc:
            message = str(exc)
            if "Failed to decode JSON response" in message or "<!DOCTYPE" in message:
                raise LLMServiceUnavailableError(message) from exc
            raise
        except Exception as exc:
            message = str(exc)
            if "<!DOCTYPE" in message or "Service Unavailable" in message:
                raise LLMServiceUnavailableError(message) from exc
            raise

        # The base adapter returns a plain string; we don't get token usage
        # from it directly. Return zeroed usage (callers handle this gracefully).
        return LLMResponse(
            text=response_text or "",
            usage=TokenUsage(),
        )
