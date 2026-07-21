"""
Local Model Adapter for the LLM abstraction layer.

Provides integration with locally hosted open-source models via HuggingFace
Transformers. Supports auto-downloading from HuggingFace Hub or loading from
a local path (e.g., for finetuned models).

Supports both **chat** mode (messages → apply_chat_template → generate) and
**completion** mode (raw prompt string → tokenize → generate).

Supported models include (but are not limited to):
- microsoft/phi-4-mini-instruct
- HuggingFaceTB/SmolLM2-1.7B-Instruct
- Qwen/Qwen2.5-3B-Instruct
- Any HuggingFace-compatible causal LM or chat model
"""

import json
import logging
import os
import threading
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

from pydantic import BaseModel

from next_action_pred_eval.utils.llm.base import (
    LLMAdapter,
    LLMResponse,
    _coerce_messages,
    _messages_to_dicts,
)
from next_action_pred_eval.utils.llm.messages import Message

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)

# ── Well-known model aliases for convenience ────────────────────────────────
MODEL_ALIASES: Dict[str, str] = {
    # Phi family
    "phi-4-mini": "microsoft/phi-4-mini-instruct",
    "phi-4-mini-instruct": "microsoft/phi-4-mini-instruct",
    "phi-3.5-mini": "microsoft/Phi-3.5-mini-instruct",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    # SmolLM family
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "smollm2-360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "smollm3": "HuggingFaceTB/SmolLM2-1.7B-Instruct",  # update when SmolLM3 ships
    # Qwen family
    "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    # Llama family
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    # Gemma family
    "gemma-2-2b": "google/gemma-2-2b-it",
}


def _resolve_model_id(model: str) -> str:
    """Resolve a model alias or path to a full HuggingFace model ID or local path."""
    if os.path.isdir(model):
        return model
    return MODEL_ALIASES.get(model.lower(), model)


class LocalModelAdapter(LLMAdapter):
    """
    LLM adapter for locally hosted HuggingFace models.

    Loads a causal language model (``AutoModelForCausalLM``) either from:
      - A HuggingFace Hub model ID (auto-downloaded if not cached)
      - A local directory path (for finetuned / custom models)

    Supports optional 4-bit / 8-bit quantization via ``bitsandbytes`` and
    Flash Attention 2 when available.

    Example::

        # From HuggingFace Hub (auto-downloads on first use)
        adapter = LocalModelAdapter(model="microsoft/phi-4-mini-instruct")

        # Using a convenient alias
        adapter = LocalModelAdapter(model="phi-4-mini")

        # From a local finetuned checkpoint
        adapter = LocalModelAdapter(model="D:/models/my-finetuned-model")

        # With 4-bit quantization for lower memory usage
        adapter = LocalModelAdapter(model="phi-4-mini", quantization="4bit")

        # Chat mode
        response = adapter.chat([Message.user("Hello!")])

        # Completion mode (raw prompt, no chat template)
        response = adapter.complete_text("Once upon a time")
    """

    # Class-level cache: avoids reloading the same model when multiple
    # LocalModelAdapter instances share the same (model_id, device, dtype,
    # quantization) tuple.  Thread-safe via _cache_lock.
    _model_cache: ClassVar[Dict[Tuple, Dict]] = {}
    _cache_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        model: str,
        device: Optional[str] = None,
        quantization: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        max_model_len: Optional[int] = None,
        trust_remote_code: bool = True,
        use_flash_attention: bool = False,
        cache_dir: Optional[str] = None,
        cache_enabled: bool = False,
        cache_path: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **model_kwargs,
    ):
        """
        Initialize the local model adapter.

        Args:
            model: HuggingFace model ID, alias (see ``MODEL_ALIASES``),
                   or absolute/relative path to a local model directory.
            device: Device to load the model on. ``"auto"`` lets
                    ``device_map="auto"`` handle multi-GPU placement.
                    Defaults to ``"cuda"`` if available, else ``"cpu"``.
            quantization: Optional quantization mode —
                          ``"4bit"`` / ``"bnb4"`` for 4-bit (bitsandbytes),
                          ``"8bit"`` / ``"bnb8"`` for 8-bit (bitsandbytes),
                          ``None`` for full precision / torch_dtype.
            torch_dtype: Data type string (``"float16"``, ``"bfloat16"``,
                         ``"float32"``, ``"auto"``). Defaults to ``"auto"``.
            max_model_len: Maximum context length override. ``None`` uses
                           the model's default.
            trust_remote_code: Whether to trust remote code in model repos
                               (required by some models like Phi).
            use_flash_attention: Attempt to use Flash Attention 2 if
                                 installed. Silently falls back otherwise.
            cache_dir: HuggingFace cache directory for downloaded models.
                       ``None`` uses the default ``~/.cache/huggingface``.
            cache_enabled: Whether to enable *response* caching (disk).
            cache_path: Path for the response cache file.
            generation_kwargs: Default extra kwargs forwarded to
                               ``model.generate()`` on every call.
            **model_kwargs: Additional keyword arguments forwarded to
                            ``AutoModelForCausalLM.from_pretrained()``.
        """
        super().__init__(cache_enabled=cache_enabled, cache_path=cache_path)

        self._raw_model_name = model
        self._model_id = _resolve_model_id(model)
        self._device = device
        self._quantization = quantization
        self._torch_dtype = torch_dtype or "auto"
        self._max_model_len = max_model_len
        self._trust_remote_code = trust_remote_code
        self._use_flash_attention = use_flash_attention
        self._cache_dir = cache_dir
        self._generation_kwargs = generation_kwargs or {}
        self._model_kwargs = model_kwargs

        # Lazily initialised
        self._model = None
        self._tokenizer = None

        logger.info(
            f"LocalModelAdapter initialised: model_id={self._model_id}, "
            f"device={self._device}, quantization={self._quantization}"
        )

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        """Return the resolved model identifier."""
        return self._model_id

    @property
    def supports_structured_output(self) -> bool:
        """Local models don't natively support structured JSON mode."""
        return False

    @property
    def supported_generation_params(self) -> frozenset:
        """HuggingFace generate() supports these extra parameters."""
        return frozenset({
            "repetition_penalty", "no_repeat_ngram_size",
            "top_k", "num_beams", "length_penalty",
            "logprobs",
        })

    # ── Lazy loading ────────────────────────────────────────────────────────

    def _ensure_loaded(self):
        """Load model + tokenizer on first use (lazy init)."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers and torch are required for LocalModelAdapter. "
                "Install with:  pip install transformers torch"
            )

        # ---- Check class-level model cache -----------------------------------
        cache_key = (
            self._model_id,
            self._device or ("cuda" if torch.cuda.is_available() else "cpu"),
            self._torch_dtype,
            self._quantization,
        )
        with self._cache_lock:
            if cache_key in type(self)._model_cache:
                cached = type(self)._model_cache[cache_key]
                self._model = cached["model"]
                self._tokenizer = cached["tokenizer"]
                logger.info(
                    f"Reusing cached model '{self._model_id}' "
                    f"(dtype={self._model.dtype})"
                )
                return

        logger.info(f"Loading model '{self._model_id}' …")

        # ---- Resolve device --------------------------------------------------
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # ---- Resolve torch dtype ---------------------------------------------
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "auto": "auto",
        }
        torch_dtype = dtype_map.get(self._torch_dtype, "auto")

        # ---- Build from_pretrained kwargs ------------------------------------
        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": self._trust_remote_code,
            "dtype": torch_dtype,
        }

        if self._cache_dir:
            load_kwargs["cache_dir"] = self._cache_dir

        # Device placement
        if self._device == "auto" or self._quantization:
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = {"": self._device}

        # ---- Quantization config ---------------------------------------------
        if self._quantization in ("4bit", "bnb4"):
            try:
                from transformers import BitsAndBytesConfig
            except ImportError:
                raise ImportError(
                    "bitsandbytes is required for 4-bit quantization. "
                    "Install with:  pip install bitsandbytes"
                )
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        elif self._quantization in ("8bit", "bnb8"):
            try:
                from transformers import BitsAndBytesConfig
            except ImportError:
                raise ImportError(
                    "bitsandbytes is required for 8-bit quantization. "
                    "Install with:  pip install bitsandbytes"
                )
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        # Flash Attention 2
        if self._use_flash_attention:
            load_kwargs["attn_implementation"] = "flash_attention_2"

        # Merge user-supplied model kwargs
        load_kwargs.update(self._model_kwargs)

        # ---- Load tokenizer + model ------------------------------------------
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_id,
            trust_remote_code=self._trust_remote_code,
            cache_dir=self._cache_dir,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            **load_kwargs,
        )
        self._model.eval()

        # Ensure pad token is set (many models don't ship with one)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model.config.pad_token_id = self._model.config.eos_token_id

        param_count = sum(p.numel() for p in self._model.parameters()) / 1e6
        logger.info(
            f"Model loaded on {self._device} | "
            f"dtype={self._model.dtype} | "
            f"params={param_count:.1f}M"
        )

        # Store in class-level cache for reuse
        with self._cache_lock:
            type(self)._model_cache[cache_key] = {
                "model": self._model,
                "tokenizer": self._tokenizer,
            }

    # ── Core: chat ───────────────────────────────────────────────────────

    def chat(
        self,
        messages: Union[List[Message], List[Dict[str, str]]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """
        Generate a chat completion from the local model.

        The tokenizer's chat template is applied automatically when
        available; otherwise messages are joined with simple role prefixes.

        Args:
            messages: List of :class:`Message` objects or dicts with
                      ``role`` and ``content``.
            temperature: Sampling temperature (0.0 = greedy).
            max_tokens: Maximum *new* tokens to generate (default 512).
            stop: Stop strings — generation is truncated at the first
                  occurrence of any of these.
            **kwargs: Extra kwargs forwarded to ``model.generate()``.

        Returns:
            The generated text (assistant turn only).
        """
        import torch

        self._ensure_loaded()
        max_tokens = max_tokens or 512

        msgs = _coerce_messages(messages)
        msg_dicts = _messages_to_dicts(msgs)

        # ---- Format prompt ---------------------------------------------------
        prompt_text = self._apply_chat_template(msg_dicts)

        inputs = self._tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_model_len,
        ).to(self._model.device)

        input_len = inputs["input_ids"].shape[-1]

        # ---- Generation params -----------------------------------------------
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self._tokenizer.pad_token_id,
        }

        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = kwargs.pop("top_p", 0.9)
        else:
            gen_kwargs["do_sample"] = False

        # Stop sequences → extra EOS token IDs
        extra_eos = self._resolve_stop_tokens(stop)
        if extra_eos:
            eos = self._tokenizer.eos_token_id
            eos_list = [eos] if isinstance(eos, int) else (eos or [])
            gen_kwargs["eos_token_id"] = list(set(eos_list + extra_eos))

        # Merge default + per-call kwargs
        gen_kwargs.update(self._generation_kwargs)
        gen_kwargs.update(kwargs)

        # ---- Generate --------------------------------------------------------
        logger.debug(
            f"Local generate: input_len={input_len}, "
            f"max_new_tokens={max_tokens}, temp={temperature}"
        )

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        # Decode only the newly generated tokens
        generated_ids = output_ids[0, input_len:]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Manual stop-sequence truncation
        if stop:
            for seq in stop:
                idx = text.find(seq)
                if idx != -1:
                    text = text[:idx]

        logger.debug(f"Local model response: {len(text)} chars")
        return text

    # ── Core: chat_with_schema ────────────────────────────────────────────

    def chat_with_schema(
        self,
        messages: Union[List[Message], List[Dict[str, str]]],
        schema: Type[T],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> T:
        """
        Generate a chat completion that conforms to a Pydantic schema.

        Injects schema instructions into the prompt and attempts to parse
        the JSON response.
        """
        schema_json = schema.model_json_schema()
        instruction = (
            "You must respond with valid JSON that matches this schema:\n"
            f"```json\n{json.dumps(schema_json, indent=2)}\n```\n"
            "Respond ONLY with the JSON object, no additional text."
        )

        msgs = _coerce_messages(messages)
        msg_dicts = _messages_to_dicts(msgs)

        if msg_dicts and msg_dicts[0].get("role") == "system":
            msg_dicts[0] = {
                "role": "system",
                "content": msg_dicts[0]["content"] + "\n\n" + instruction,
            }
        else:
            msg_dicts.insert(0, {"role": "system", "content": instruction})

        response_text = self.chat(
            [Message.from_dict(d) for d in msg_dicts],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        # ---- Parse (handle markdown fences + surrounding text) ---------------
        text = response_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            json_lines, in_block = [], False
            for line in lines:
                if line.startswith("```") and not in_block:
                    in_block = True
                    continue
                elif line.startswith("```") and in_block:
                    break
                elif in_block:
                    json_lines.append(line)
            text = "\n".join(json_lines)

        # Extract first JSON object
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]

        try:
            data = json.loads(text)
            return schema.model_validate(data)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"LLM response was not valid JSON: {response_text[:300]}…"
            ) from e
        except Exception as e:
            raise ValueError(f"LLM response did not match schema: {e}") from e

    # ── Optional: chat_with_response ──────────────────────────────────────

    def chat_with_response(
        self,
        messages: Union[List[Message], List[Dict[str, str]]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Chat and return full response metadata.

        Pass ``logprobs=True`` via *kwargs* to populate
        :attr:`LLMResponse.token_logprobs` and :attr:`LLMResponse.tokens`.

        Returns:
            LLMResponse with content, token-usage metadata, and optionally
            per-token log probabilities.
        """
        import torch

        self._ensure_loaded()
        max_tokens = max_tokens or 512

        # Pop logprobs before it reaches model.generate() (not a HF param)
        request_logprobs = kwargs.pop("logprobs", None)

        msgs = _coerce_messages(messages)
        msg_dicts = _messages_to_dicts(msgs)

        prompt_text = self._apply_chat_template(msg_dicts)
        inputs = self._tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_model_len,
        ).to(self._model.device)

        input_len = inputs["input_ids"].shape[-1]

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = kwargs.pop("top_p", 0.9)
        else:
            gen_kwargs["do_sample"] = False

        extra_eos = self._resolve_stop_tokens(stop)
        if extra_eos:
            eos = self._tokenizer.eos_token_id
            eos_list = [eos] if isinstance(eos, int) else (eos or [])
            gen_kwargs["eos_token_id"] = list(set(eos_list + extra_eos))

        gen_kwargs.update(self._generation_kwargs)
        gen_kwargs.update(kwargs)

        # Request score tensors when logprobs are needed
        if request_logprobs:
            gen_kwargs["output_scores"] = True
            gen_kwargs["return_dict_in_generate"] = True

        with torch.no_grad():
            output = self._model.generate(**inputs, **gen_kwargs)

        if request_logprobs:
            output_ids = output.sequences
        else:
            output_ids = output

        generated_ids = output_ids[0, input_len:]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        if stop:
            for seq in stop:
                idx = text.find(seq)
                if idx != -1:
                    text = text[:idx]

        # Compute per-token logprobs from generation scores
        token_logprobs = None
        tokens = None
        top_logprobs = None
        if request_logprobs and hasattr(output, "scores") and output.scores:
            token_logprobs, tokens, top_logprobs = self._compute_token_logprobs(
                output.scores, generated_ids,
            )

        return LLMResponse(
            content=text,
            model=self._model_id,
            usage={
                "prompt_tokens": int(input_len),
                "completion_tokens": int(generated_ids.shape[0]),
                "total_tokens": int(input_len + generated_ids.shape[0]),
            },
            token_logprobs=token_logprobs,
            tokens=tokens,
            top_logprobs=top_logprobs,
        )

    # ── Completion mode ───────────────────────────────────────────────────

    def complete_text(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """
        Generate a text completion from a raw prompt string.

        Unlike :meth:`chat`, this does **not** apply the tokenizer's chat
        template.  The prompt is tokenized directly and the model generates
        a continuation.

        Args:
            prompt: Raw prompt string.
            temperature: Sampling temperature (0.0 = greedy).
            max_tokens: Maximum *new* tokens to generate (default 512).
            stop: Stop strings.
            **kwargs: Extra kwargs forwarded to ``model.generate()``.

        Returns:
            The generated continuation text.
        """
        import torch

        self._ensure_loaded()
        max_tokens = max_tokens or 512

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_model_len,
        ).to(self._model.device)

        input_len = inputs["input_ids"].shape[-1]

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = kwargs.pop("top_p", 0.9)
        else:
            gen_kwargs["do_sample"] = False

        extra_eos = self._resolve_stop_tokens(stop)
        if extra_eos:
            eos = self._tokenizer.eos_token_id
            eos_list = [eos] if isinstance(eos, int) else (eos or [])
            gen_kwargs["eos_token_id"] = list(set(eos_list + extra_eos))

        gen_kwargs.update(self._generation_kwargs)
        gen_kwargs.update(kwargs)

        logger.debug(
            f"Local completion: input_len={input_len}, "
            f"max_new_tokens={max_tokens}, temp={temperature}"
        )

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        generated_ids = output_ids[0, input_len:]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        if stop:
            for seq in stop:
                idx = text.find(seq)
                if idx != -1:
                    text = text[:idx]

        logger.debug(f"Local completion response: {len(text)} chars")
        return text

    def complete_text_with_response(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Completion and return full response metadata.

        Pass ``logprobs=True`` via *kwargs* to populate
        :attr:`LLMResponse.token_logprobs` and :attr:`LLMResponse.tokens`.

        Returns:
            LLMResponse with content, token-usage metadata, and optionally
            per-token log probabilities.
        """
        import torch

        self._ensure_loaded()
        max_tokens = max_tokens or 512

        # Pop logprobs before it reaches model.generate() (not a HF param)
        request_logprobs = kwargs.pop("logprobs", None)

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_model_len,
        ).to(self._model.device)

        input_len = inputs["input_ids"].shape[-1]

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = kwargs.pop("top_p", 0.9)
        else:
            gen_kwargs["do_sample"] = False

        extra_eos = self._resolve_stop_tokens(stop)
        if extra_eos:
            eos = self._tokenizer.eos_token_id
            eos_list = [eos] if isinstance(eos, int) else (eos or [])
            gen_kwargs["eos_token_id"] = list(set(eos_list + extra_eos))

        gen_kwargs.update(self._generation_kwargs)
        gen_kwargs.update(kwargs)

        # Request score tensors when logprobs are needed
        if request_logprobs:
            gen_kwargs["output_scores"] = True
            gen_kwargs["return_dict_in_generate"] = True

        with torch.no_grad():
            output = self._model.generate(**inputs, **gen_kwargs)

        if request_logprobs:
            output_ids = output.sequences
        else:
            output_ids = output

        generated_ids = output_ids[0, input_len:]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        if stop:
            for seq in stop:
                idx = text.find(seq)
                if idx != -1:
                    text = text[:idx]

        # Compute per-token logprobs from generation scores
        token_logprobs = None
        tokens = None
        top_logprobs = None
        if request_logprobs and hasattr(output, "scores") and output.scores:
            token_logprobs, tokens, top_logprobs = self._compute_token_logprobs(
                output.scores, generated_ids,
            )

        return LLMResponse(
            content=text,
            model=self._model_id,
            usage={
                "prompt_tokens": int(input_len),
                "completion_tokens": int(generated_ids.shape[0]),
                "total_tokens": int(input_len + generated_ids.shape[0]),
            },
            token_logprobs=token_logprobs,
            tokens=tokens,
            top_logprobs=top_logprobs,
        )

    # ── Internal helpers ────────────────────────────────────────────────────

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """Format messages using the tokenizer's chat template (with fallback)."""
        if hasattr(self._tokenizer, "chat_template") and self._tokenizer.chat_template:
            try:
                return self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                logger.debug(f"chat_template failed ({e}), using manual format")

        # Simple fallback
        parts = [f"{m.get('role', 'user').capitalize()}: {m['content']}" for m in messages]
        parts.append("Assistant:")
        return "\n".join(parts)

    def _compute_token_logprobs(
        self,
        scores: tuple,
        generated_ids,
        top_k: int = 5,
    ) -> tuple:
        """Compute per-token log probabilities from generation scores.

        Args:
            scores: Tuple of logit tensors (one per generated token),
                    each of shape ``(batch, vocab_size)``.
            generated_ids: Token IDs actually generated, shape ``(num_tokens,)``.
            top_k: Number of top log-probs to return per token.

        Returns:
            ``(token_logprobs, tokens, top_logprobs)`` — parallel lists of
            log-probs, decoded token strings, and per-token top-k dicts.
        """
        import torch
        import torch.nn.functional as F

        token_logprobs = []
        tokens = []
        top_logprobs = []
        for logits, token_id in zip(scores, generated_ids):
            log_probs = F.log_softmax(logits[0], dim=-1)
            token_logprobs.append(log_probs[token_id].item())
            tokens.append(self._tokenizer.decode([token_id.item()]))
            # Top-k logprobs
            topk_vals, topk_ids = torch.topk(log_probs, min(top_k, log_probs.shape[0]))
            top_dict = {
                self._tokenizer.decode([tid.item()]): lp.item()
                for tid, lp in zip(topk_ids, topk_vals)
            }
            top_logprobs.append(top_dict)
        return token_logprobs, tokens, top_logprobs

    def _resolve_stop_tokens(self, stop: Optional[List[str]]) -> List[int]:
        """Convert stop strings to single-token IDs (best-effort)."""
        if not stop:
            return []
        ids: List[int] = []
        for seq in stop:
            encoded = self._tokenizer.encode(seq, add_special_tokens=False)
            if len(encoded) == 1:
                ids.append(encoded[0])
        return ids

    # ── Resource management ─────────────────────────────────────────────────

    def unload(self):
        """Free GPU / CPU memory by deleting the model and tokenizer."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("Model unloaded and memory freed.")

    def __del__(self):
        try:
            self.unload()
        except Exception:
            pass

    def __repr__(self) -> str:
        loaded = "loaded" if self._model is not None else "not loaded"
        return (
            f"LocalModelAdapter(model={self._model_id!r}, "
            f"device={self._device}, quant={self._quantization}, {loaded})"
        )


__all__ = ["LocalModelAdapter", "MODEL_ALIASES"]
