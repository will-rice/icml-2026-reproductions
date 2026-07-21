"""
OpenAI Adapter for the LLM abstraction layer.

Provides integration with OpenAI's API (and compatible APIs like Azure OpenAI).
Supports both **chat** mode (``/chat/completions``) and **completion** mode
(``/completions``).
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel

from next_action_pred_eval.utils.llm.base import LLMAdapter, LLMResponse, _coerce_messages, _messages_to_dicts
from next_action_pred_eval.utils.llm.messages import Message

T = TypeVar('T', bound=BaseModel)
logger = logging.getLogger(__name__)


class OpenAIAdapter(LLMAdapter):
    """
    LLM adapter for OpenAI's API.

    Can also work with Azure OpenAI by providing the appropriate base_url.

    Example::

        # Standard OpenAI
        adapter = OpenAIAdapter(api_key="sk-...", model="gpt-4")

        # Chat mode
        response = adapter.chat([Message.user("Hello!")])

        # Completion mode (legacy /completions endpoint)
        response = adapter.complete_text("Once upon a time")

        # Azure OpenAI
        adapter = OpenAIAdapter(
            api_key="...",
            model="gpt-4",
            base_url="https://your-resource.openai.azure.com/openai/deployments/your-deployment"
        )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        api_version: Optional[str] = None,
        timeout: float = 60.0,
    ):
        """
        Initialize the OpenAI adapter.

        Args:
            api_key: OpenAI API key. If not provided, reads from OPENAI_API_KEY env var.
            model: Model name to use (e.g., "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo")
            base_url: Optional base URL for API calls (for Azure or proxies)
            organization: Optional OpenAI organization ID
            api_version: API version (required for Azure OpenAI)
            timeout: Request timeout in seconds
        """
        self._model = model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._base_url = base_url
        self._organization = organization
        self._api_version = api_version
        self._timeout = timeout
        self._client = None

        if not self._api_key:
            raise ValueError(
                "OpenAI API key is required. Provide api_key parameter or set OPENAI_API_KEY env var."
            )

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model

    @property
    def supported_generation_params(self) -> frozenset:
        """OpenAI supports frequency/presence penalty but not repetition_penalty."""
        return frozenset({"frequency_penalty", "presence_penalty", "logprobs", "top_logprobs"})

    @property
    def client(self):
        """Lazy initialization of the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI, AzureOpenAI
            except ImportError:
                raise ImportError(
                    "OpenAI package is required. Install with: pip install openai"
                )

            # Use Azure client if api_version is specified
            if self._api_version:
                self._client = AzureOpenAI(
                    api_key=self._api_key,
                    api_version=self._api_version,
                    azure_endpoint=self._base_url,
                    timeout=self._timeout,
                )
            else:
                client_kwargs = {
                    "api_key": self._api_key,
                    "timeout": self._timeout,
                }
                if self._base_url:
                    client_kwargs["base_url"] = self._base_url
                if self._organization:
                    client_kwargs["organization"] = self._organization

                self._client = OpenAI(**client_kwargs)

        return self._client

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
        """Generate a chat completion via OpenAI ``/chat/completions``."""
        msgs = _coerce_messages(messages)
        msg_dicts = _messages_to_dicts(msgs)

        request_kwargs = {
            "model": self._model,
            "messages": msg_dicts,
            "temperature": temperature,
        }
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        if stop:
            request_kwargs["stop"] = stop
        request_kwargs.update(kwargs)

        logger.debug(f"OpenAI chat: model={self._model}, messages={len(msgs)}")
        response = self.client.chat.completions.create(**request_kwargs)
        content = response.choices[0].message.content

        logger.debug(f"OpenAI chat response: {len(content)} chars")
        return content

    def chat_with_response(
        self,
        messages: Union[List[Message], List[Dict[str, str]]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Chat and return full response metadata."""
        msgs = _coerce_messages(messages)
        msg_dicts = _messages_to_dicts(msgs)

        request_kwargs = {
            "model": self._model,
            "messages": msg_dicts,
            "temperature": temperature,
        }
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        if stop:
            request_kwargs["stop"] = stop
        request_kwargs.update(kwargs)

        response = self.client.chat.completions.create(**request_kwargs)

        # Extract per-token logprobs when available (chat format)
        token_logprobs = None
        tokens = None
        choice = response.choices[0]
        if choice.logprobs and getattr(choice.logprobs, "content", None):
            token_logprobs = [t.logprob for t in choice.logprobs.content]
            tokens = [t.token for t in choice.logprobs.content]

        return LLMResponse(
            content=choice.message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            raw_response=response,
            token_logprobs=token_logprobs,
            tokens=tokens,
        )

    def chat_with_schema(
        self,
        messages: Union[List[Message], List[Dict[str, str]]],
        schema: Type[T],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> T:
        """Generate a chat completion conforming to a Pydantic schema.

        Uses OpenAI's JSON mode (``response_format={"type": "json_object"}``).
        """
        schema_json = schema.model_json_schema()
        system_instruction = (
            f"You must respond with valid JSON that matches this schema:\n"
            f"```json\n{json.dumps(schema_json, indent=2)}\n```"
        )

        msgs = _coerce_messages(messages)
        msg_dicts = _messages_to_dicts(msgs)

        if msg_dicts and msg_dicts[0].get("role") == "system":
            msg_dicts[0] = {
                "role": "system",
                "content": msg_dicts[0]["content"] + "\n\n" + system_instruction,
            }
        else:
            msg_dicts.insert(0, {"role": "system", "content": system_instruction})

        request_kwargs = {
            "model": self._model,
            "messages": msg_dicts,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        request_kwargs.update(kwargs)

        response = self.client.chat.completions.create(**request_kwargs)
        content = response.choices[0].message.content

        try:
            data = json.loads(content)
            return schema.model_validate(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM response was not valid JSON: {content[:200]}...") from e
        except Exception as e:
            raise ValueError(f"LLM response did not match schema: {e}") from e

    # ------------------------------------------------------------------
    # Completion mode
    # ------------------------------------------------------------------

    def complete_text(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """Generate a text completion via OpenAI ``/completions``."""
        request_kwargs: Dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "temperature": temperature,
        }
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        if stop:
            request_kwargs["stop"] = stop
        request_kwargs.update(kwargs)

        logger.debug(f"OpenAI completion: model={self._model}, prompt_len={len(prompt)}")
        response = self.client.completions.create(**request_kwargs)
        text = response.choices[0].text

        logger.debug(f"OpenAI completion response: {len(text)} chars")
        return text

    def complete_text_with_response(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Completion and return full response metadata."""
        request_kwargs: Dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "temperature": temperature,
        }
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        if stop:
            request_kwargs["stop"] = stop
        request_kwargs.update(kwargs)

        response = self.client.completions.create(**request_kwargs)

        # Extract per-token logprobs when available (completion format)
        token_logprobs = None
        tokens = None
        choice = response.choices[0]
        if choice.logprobs and getattr(choice.logprobs, "token_logprobs", None):
            token_logprobs = choice.logprobs.token_logprobs
            tokens = choice.logprobs.tokens

        return LLMResponse(
            content=choice.text,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            raw_response=response,
            token_logprobs=token_logprobs,
            tokens=tokens,
        )
