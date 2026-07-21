"""
LLM module — LLM adapter abstraction layer.

Provides an abstraction over LLM providers. Implement ``LLMAdapter`` to
plug in your own provider.

Two modes of interaction are exposed by every adapter:

* **Chat mode** — ``adapter.chat(messages)`` / ``adapter.chat_with_response(messages)``
* **Completion mode** — ``adapter.complete_text(prompt)`` / ``adapter.complete_text_with_response(prompt)``

Example::

    from next_action_pred_eval.utils.llm import OpenAIAdapter, Message

    adapter = OpenAIAdapter(api_key="sk-...", model="gpt-4o-mini")
    response = adapter.chat([Message.user("Hello!")])

Built-in adapters: ``OpenAIAdapter``, ``LocalModelAdapter``.

To plug in any other provider, write a class that inherits from
``LLMAdapter`` and load it via the ``custom`` adapter type in YAML
config (see ``create_adapter`` below).
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any

from next_action_pred_eval.utils.llm.base import LLMAdapter, LLMResponse
from next_action_pred_eval.utils.llm.messages import Message, image_to_data_url
from next_action_pred_eval.utils.llm.openai_adapter import OpenAIAdapter
from next_action_pred_eval.utils.llm.local_adapter import LocalModelAdapter, MODEL_ALIASES

__all__ = [
    "LLMAdapter",
    "LLMResponse",
    "Message",
    "image_to_data_url",
    "OpenAIAdapter",
    "LocalModelAdapter",
    "MODEL_ALIASES",
    "create_adapter",
]


def create_adapter(
    adapter: str,
    *,
    model: str | None = None,
    adapter_class: str | None = None,
    adapter_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> LLMAdapter:
    """Construct an ``LLMAdapter`` from a config-friendly spec.

    Parameters
    ----------
    adapter:
        One of ``"openai"``, ``"local"``, or ``"custom"``.
    model:
        Model identifier (HuggingFace ID for ``local``, OpenAI model
        name for ``openai``). Ignored for ``"custom"``.
    adapter_class:
        Dotted import path to an ``LLMAdapter`` subclass. Required
        when ``adapter == "custom"``.
    adapter_kwargs:
        Keyword args passed to the custom adapter's constructor.
    **kwargs:
        Forwarded to the constructor of the chosen built-in adapter.
    """
    if adapter == "openai":
        return OpenAIAdapter(model=model, **kwargs)
    if adapter == "local":
        return LocalModelAdapter(model=model, **kwargs)
    if adapter == "custom":
        if not adapter_class:
            raise ValueError(
                "adapter='custom' requires `adapter_class` (a dotted import path "
                "to an LLMAdapter subclass)."
            )
        module_path, _, class_name = adapter_class.rpartition(".")
        if not module_path:
            raise ValueError(
                f"adapter_class={adapter_class!r} is not a dotted path "
                "(expected e.g. 'my_pkg.my_module.MyAdapter')."
            )
        # Make CWD importable so user-provided adapters under the working
        # directory resolve in worker subprocesses too.
        cwd = os.getcwd()
        if cwd not in sys.path:
            sys.path.insert(0, cwd)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        if not issubclass(cls, LLMAdapter):
            raise TypeError(
                f"{adapter_class} is not a subclass of LLMAdapter."
            )
        return cls(**(adapter_kwargs or {}), **kwargs)
    raise ValueError(
        f"Unknown adapter type: {adapter!r}. Expected one of: openai, local, custom."
    )
