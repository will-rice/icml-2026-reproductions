"""
Message types for the LLM abstraction layer.

Provides a lightweight ``Message`` dataclass that can be constructed without
any provider-specific dependency. The ``to_dict()`` method returns an
OpenAI-compatible message dict that all adapters can consume.

Image support
-------------
Pass a file path (or list of paths) to ``images`` and the corresponding
base64-encoded ``image_url`` content parts will be emitted automatically::

    msg = Message(
        role="user",
        content="Describe this chart.",
        images=["chart.png"],
    )
"""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def image_to_data_url(image_path: Union[str, Path]) -> str:
    """Encode a local image file as a ``data:`` URL (base64)."""
    image_path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type is None:
        mime_type = "application/octet-stream"
    raw = image_path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


# ---------------------------------------------------------------------------
# Message dataclass
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """A single chat message.

    Mirrors the shape expected by OpenAI / Azure / Anthropic / local
    models, but is provider-agnostic.

    Attributes:
        role: One of ``"system"``, ``"user"``, ``"assistant"``, ``"tool"``,
              ``"developer"``, ``"function"``.
        content: Text content, or a list of OpenAI-style content parts
                 (``{"type": "text", "text": ...}`` /
                 ``{"type": "image_url", ...}``).  When *images* are also
                 supplied, the text is prepended to the image parts.
        images: Optional path(s) to local image files.  Each is
                base64-encoded into an ``image_url`` content part.
        tool_call_id: For tool-result messages (``role="tool"``).
    """

    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    images: Optional[Union[str, Path, List[Union[str, Path]]]] = None
    tool_call_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str, images: Optional[Union[str, Path, List[Union[str, Path]]]] = None) -> "Message":
        return cls(role="user", content=content, images=images)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        return cls(role="assistant", content=content)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return an OpenAI-compatible message dict.

        * Text-only messages → ``{"role": ..., "content": "..."}``
        * Multimodal (images) → ``{"role": ..., "content": [parts...]}``
        * Tool messages       → ``{"role": "tool", "content": "...",
                                    "tool_call_id": "..."}``
        """
        # Tool messages
        if self.tool_call_id is not None:
            if not isinstance(self.content, str):
                raise ValueError("Tool messages must have string content.")
            return {
                "role": self.role,
                "content": self.content,
                "tool_call_id": self.tool_call_id,
            }

        # Normalize images list
        images: List[Union[str, Path]] = []
        if isinstance(self.images, (str, Path)):
            images = [self.images]
        elif isinstance(self.images, list):
            images = self.images

        # No images → simple message
        if not images:
            if self.content is None:
                raise ValueError("Message must have content.")
            body = self.content if isinstance(self.content, str) else list(self.content)
            return {"role": self.role, "content": body}

        # Multimodal → content-parts list
        parts: List[Dict[str, Any]] = []
        if self.content is not None:
            if isinstance(self.content, str):
                parts.append({"type": "text", "text": self.content})
            else:
                parts.extend(self.content)

        for img in images:
            data_url = image_to_data_url(img)
            parts.append({
                "type": "image_url",
                "image_url": {"url": data_url},
            })

        return {"role": self.role, "content": parts}

    # ------------------------------------------------------------------
    # Convenience for constructing from a plain dict
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Message":
        """Create a ``Message`` from an OpenAI-style dict."""
        return cls(
            role=d["role"],
            content=d.get("content"),
            tool_call_id=d.get("tool_call_id"),
        )


__all__ = ["Message", "image_to_data_url"]
