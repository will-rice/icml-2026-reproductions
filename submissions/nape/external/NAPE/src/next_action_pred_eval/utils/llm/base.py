"""
LLM Adapter Base Class.

Abstract base class for LLM adapters.  Users can implement this interface
to plug in their preferred LLM provider (OpenAI, Azure, Anthropic, etc.).

The adapter exposes **two modes** of interaction:

Chat mode (``chat`` / ``chat_with_response``)
    Takes a list of :class:`Message` objects (system, user, assistant turns)
    and returns the model's response.

Completion mode (``complete_text`` / ``complete_text_with_response``)
    Takes a **single prompt string** and returns the model's continuation.
    This mirrors the legacy *completions* API (``/completions`` endpoint).

An adapter may support one or both modes.  The default implementation of
each raises ``NotImplementedError``; subclasses override whichever mode(s)
they support.
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel

from next_action_pred_eval.utils.llm.messages import Message

T = TypeVar('T', bound=BaseModel)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error classification helpers (adapter-agnostic)
# ---------------------------------------------------------------------------

def is_content_filter_error(error: Exception) -> bool:
    """Check if an exception is due to content filtering (jailbreak detection, etc.)."""
    error_str = str(error).lower()
    return (
        "content_filter" in error_str
        or "content management policy" in error_str
        or "jailbreak" in error_str
        or "responsibleaipolicyviolation" in error_str
    )


def is_context_length_error(error: Exception) -> bool:
    """Check if an exception is due to context length overflow."""
    error_str = str(error).lower()
    return (
        "maximum context length" in error_str
        or "context length" in error_str
        or "token limit" in error_str
        or ("requested" in error_str and "tokens" in error_str)
    )


def _coerce_messages(messages: Union[List[Message], List[Dict[str, str]]]) -> List[Message]:
    """Accept either ``Message`` objects or plain dicts and return ``Message`` list."""
    if not messages:
        return []
    if isinstance(messages[0], dict):
        return [Message.from_dict(m) for m in messages]
    return list(messages)


def _messages_to_dicts(messages: List[Message]) -> List[Dict[str, Any]]:
    """Convert ``Message`` list to OpenAI-compatible dicts."""
    return [m.to_dict() for m in messages]


# ---------------------------------------------------------------------------
# Retry helper (shared by chat_with_retry / complete_text_with_retry)
# ---------------------------------------------------------------------------

def _retry_loop(call_fn, max_retries: int, mode_label: str) -> str:
    """Generic retry loop for both chat and completion modes.

    *call_fn* is a zero-arg callable that performs one attempt and returns
    the generated text (or raises).
    """
    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            return call_fn()
        except Exception as e:
            last_error = e

            # Non-retryable errors — return empty immediately
            if is_content_filter_error(e):
                logger.warning(f"[{mode_label}] Content filter triggered, returning empty: {e}")
                return ""
            if is_context_length_error(e):
                logger.warning(f"[{mode_label}] Context length exceeded, returning empty: {e}")
                return ""

            if attempt >= max_retries:
                break

            # Classify and handle retryable errors
            if isinstance(e, (ConnectionError, ConnectionResetError, ConnectionAbortedError)):
                wait = 2 ** (attempt + 1)
                logger.warning(
                    f"[{mode_label}] Connection error (attempt {attempt + 1}/{max_retries + 1}), "
                    f"waiting {wait}s: {e}"
                )
                time.sleep(wait)
            elif isinstance(e, (ValueError, json.JSONDecodeError)):
                logger.warning(
                    f"[{mode_label}] Decode error (attempt {attempt + 1}/{max_retries + 1}), "
                    f"retrying immediately: {e}"
                )
            else:
                wait = 2 ** attempt
                logger.warning(
                    f"[{mode_label}] Error (attempt {attempt + 1}/{max_retries + 1}), "
                    f"waiting {wait}s: {e}"
                )
                time.sleep(wait)

    raise last_error  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════════
# LLMAdapter
# ═══════════════════════════════════════════════════════════════════════════

class LLMAdapter(ABC):
    """
    Abstract base class for LLM adapters.

    Subclasses implement **chat mode**, **completion mode**, or both by
    overriding the corresponding methods.  The default implementation
    raises ``NotImplementedError`` so callers get a clear error when
    they use an unsupported mode.

    Args:
        cache_enabled: Whether to enable response caching.
        cache_path: Optional path for cache file storage.

    Example::

        adapter = OpenAIAdapter(api_key="...", model="gpt-4")
        response = adapter.chat([Message.user("Hello!")])
    """

    def __init__(
        self,
        cache_enabled: bool = True,
        cache_path: Optional[str] = None,
    ):
        self.cache_enabled = cache_enabled
        self.cache_path = cache_path

    # ------------------------------------------------------------------
    # Chat mode
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: Union[List[Message], List[Dict[str, str]]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """Send a chat conversation and return the assistant's text.

        Args:
            messages: Conversation as :class:`Message` objects **or** plain
                      dicts (``{"role": ..., "content": ...}``).
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in the response.
            stop: Stop sequences.
            **kwargs: Provider-specific parameters.

        Returns:
            Generated text.

        Raises:
            NotImplementedError: If the adapter does not support chat mode.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support chat mode.  "
            "Override the chat() method to add support."
        )

    def chat_with_response(
        self,
        messages: Union[List[Message], List[Dict[str, str]]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> "LLMResponse":
        """Like :meth:`chat` but returns an :class:`LLMResponse` with metadata."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support chat mode.  "
            "Override the chat_with_response() method to add support."
        )

    def chat_with_schema(
        self,
        messages: Union[List[Message], List[Dict[str, str]]],
        schema: Type[T],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> T:
        """Generate a chat completion conforming to a Pydantic *schema*.

        Raises:
            NotImplementedError: If the adapter does not support chat mode.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support chat_with_schema().  "
            "Override the method to add support."
        )

    def chat_with_retry(
        self,
        messages: Union[List[Message], List[Dict[str, str]]],
        max_retries: int = 3,
        **kwargs,
    ) -> str:
        """Chat with automatic retry and error classification.

        Subclasses can override this to add provider-specific error
        handling (e.g., rate-limit back-off, token refresh).
        """
        return _retry_loop(
            lambda: self.chat(messages, **kwargs),
            max_retries=max_retries,
            mode_label="chat",
        )

    # ------------------------------------------------------------------
    # Completion (text-in → text-out) mode
    # ------------------------------------------------------------------

    def complete_text(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """Send a plain text prompt and return the continuation.

        This corresponds to the legacy ``/completions`` API — a single
        string goes in, a single string comes out.

        Raises:
            NotImplementedError: If the adapter does not support
                completion mode.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support completion mode.  "
            "Override the complete_text() method to add support."
        )

    def complete_text_with_response(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> "LLMResponse":
        """Like :meth:`complete_text` but returns an :class:`LLMResponse`."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support completion mode.  "
            "Override the complete_text_with_response() method to add support."
        )

    def complete_text_with_retry(
        self,
        prompt: str,
        max_retries: int = 3,
        **kwargs,
    ) -> str:
        """Completion with automatic retry and error classification."""
        return _retry_loop(
            lambda: self.complete_text(prompt, **kwargs),
            max_retries=max_retries,
            mode_label="completion",
        )

    # ------------------------------------------------------------------
    # Deprecated shims — old interface forwarded to chat()
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: Union[List[Message], List[Dict[str, str]]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """*Deprecated* — use :meth:`chat` instead.

        Forwards to :meth:`chat` for backward compatibility.
        """
        warnings.warn(
            "LLMAdapter.complete(messages) is deprecated; use chat(messages) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens, stop=stop, **kwargs)

    def complete_with_response(
        self,
        messages: Union[List[Message], List[Dict[str, str]]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> "LLMResponse":
        """*Deprecated* — use :meth:`chat_with_response` instead."""
        warnings.warn(
            "LLMAdapter.complete_with_response(messages) is deprecated; "
            "use chat_with_response(messages) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.chat_with_response(
            messages, temperature=temperature, max_tokens=max_tokens, stop=stop, **kwargs
        )

    def complete_with_schema(
        self,
        messages: Union[List[Message], List[Dict[str, str]]],
        schema: Type[T],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> T:
        """*Deprecated* — use :meth:`chat_with_schema` instead."""
        warnings.warn(
            "LLMAdapter.complete_with_schema() is deprecated; "
            "use chat_with_schema() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.chat_with_schema(
            messages, schema=schema, temperature=temperature, max_tokens=max_tokens, **kwargs
        )

    def complete_with_retry(
        self,
        messages: Union[List[Message], List[Dict[str, str]]],
        max_retries: int = 3,
        **kwargs,
    ) -> str:
        """*Deprecated* — use :meth:`chat_with_retry` instead."""
        warnings.warn(
            "LLMAdapter.complete_with_retry(messages) is deprecated; "
            "use chat_with_retry(messages) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.chat_with_retry(messages, max_retries=max_retries, **kwargs)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the model being used."""
        pass

    @property
    def supports_structured_output(self) -> bool:
        """Return True if the adapter supports structured/JSON output natively."""
        return True

    @property
    def supported_generation_params(self) -> frozenset:
        """Extra generation parameters this adapter supports.

        The standard set (``temperature``, ``max_tokens``, ``stop``) is
        handled by every adapter.  This property advertises *additional*
        parameters (e.g. ``repetition_penalty``) so that solvers can
        validate configuration at init time.

        Returns:
            A frozenset of parameter name strings.
        """
        return frozenset()


# ═══════════════════════════════════════════════════════════════════════════
# LLMResponse
# ═══════════════════════════════════════════════════════════════════════════

class LLMResponse:
    """
    Wrapper class for LLM responses with metadata.

    Attributes:
        content: The text content of the response
        model: The model that generated the response
        usage: Token usage information (if available)
        raw_response: The raw response from the provider (for debugging)
        token_logprobs: Per-token log probabilities (if requested and available)
        tokens: Token strings corresponding to *token_logprobs* (if available)
        top_logprobs: Per-token top-k logprob dicts (if available).
                      List of dicts mapping token strings to their logprobs.
    """

    def __init__(
        self,
        content: str,
        model: str,
        usage: Optional[Dict[str, int]] = None,
        raw_response: Optional[Any] = None,
        token_logprobs: Optional[List[float]] = None,
        tokens: Optional[List[str]] = None,
        top_logprobs: Optional[List[Dict[str, float]]] = None,
    ):
        self.content = content
        self.model = model
        self.usage = usage or {}
        self.raw_response = raw_response
        self.token_logprobs = token_logprobs
        self.tokens = tokens
        self.top_logprobs = top_logprobs

    @property
    def prompt_tokens(self) -> int:
        """Number of tokens in the prompt."""
        return self.usage.get('prompt_tokens', 0)

    @property
    def completion_tokens(self) -> int:
        """Number of tokens in the completion."""
        return self.usage.get('completion_tokens', 0)

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.usage.get('total_tokens', self.prompt_tokens + self.completion_tokens)

    def __str__(self) -> str:
        return self.content
