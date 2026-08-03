"""Safetensors-only CPU inference for the audited TimeRewarder architecture."""

import hashlib
import json
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn

from timerewarder_repro.approval import validate_approval_record
from timerewarder_repro.method import logits_to_scalar


class LayerNorm(nn.LayerNorm):
    """CLIP layer normalization with float32 accumulation."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return super().forward(value.float()).to(value.dtype)


class QuickGELU(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * torch.sigmoid(1.702 * value)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads)
        self.ln_1 = LayerNorm(width)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(width, width * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(width * 4, width)),
                ]
            )
        )
        self.ln_2 = LayerNorm(width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = self.ln_1(value)
        attended = self.attn(
            normalized, normalized, normalized, need_weights=False
        )[0]
        value = value + attended
        return value + self.mlp(self.ln_2(value))


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int) -> None:
        super().__init__()
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads) for _ in range(layers)]
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.resblocks(value)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        resolution: int = 224,
        patch_size: int = 16,
        width: int = 768,
        layers: int = 12,
        heads: int = 12,
        output_dim: int = 512,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            3, width, kernel_size=patch_size, stride=patch_size, bias=False
        )
        self.class_embedding = nn.Parameter(torch.empty(width))
        self.positional_embedding = nn.Parameter(
            torch.empty((resolution // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(torch.empty(width, output_dim))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        value = self.conv1(frames)
        value = value.reshape(value.shape[0], value.shape[1], -1).permute(0, 2, 1)
        token = self.class_embedding.to(value.dtype).expand(value.shape[0], 1, -1)
        value = torch.cat((token, value), dim=1)
        value = self.ln_pre(value + self.positional_embedding.to(value.dtype))
        value = self.transformer(value.permute(1, 0, 2)).permute(1, 0, 2)
        return self.ln_post(value[:, 0]) @ self.proj


class CrossAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = nn.Linear(512, 512, bias=False)
        self.to_k = nn.Linear(768, 512, bias=False)
        self.to_v = nn.Linear(768, 512, bias=False)
        self.to_out = nn.Sequential(nn.Linear(512, 512), nn.Dropout(0.0))


class TimeRewarderModel(nn.Module):
    """The exact state layout needed for released ViT-B/16 checkpoints."""

    def __init__(self) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(49408, 512)
        self.positional_embedding = nn.Parameter(torch.empty(77, 512))
        self.transformer = Transformer(512, 12, 8)
        self.ln_final = LayerNorm(512)
        self.text_projection = nn.Parameter(torch.empty(512, 512))
        self.visual = VisionTransformer()
        self.order_attn = CrossAttention()
        self.order_ln = LayerNorm(512)
        self.order_mlp = nn.Sequential(
            LayerNorm(1024), nn.Linear(1024, 20), nn.Dropout(0.2)
        )

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        return self.visual(frames)

    def pair_logits(
        self, features: torch.Tensor, pairs: Sequence[tuple[int, int]]
    ) -> torch.Tensor:
        _validate_pairs(features.shape[0], pairs)
        pair_features = torch.stack(
            [torch.cat((features[start], features[end])) for start, end in pairs]
        )
        return self.order_mlp(pair_features)


class _TensorContainer(nn.Module):
    """Small schema fixture container used only for loader boundary tests."""

    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.values = nn.ParameterList(
            [nn.Parameter(tensor, requires_grad=False) for tensor in tensors.values()]
        )


def load_approved_model(
    safetensors_path: Path,
    approval_path: Path,
    receipt_path: Path,
    schema_path: Path,
) -> nn.Module:
    """Validate all identities and construct a CPU model from tensor-only bytes."""
    if (
        safetensors_path.suffix != ".safetensors"
        or safetensors_path.is_symlink()
        or not safetensors_path.is_file()
    ):
        raise ValueError("approved regular safetensors file required")
    receipt = _read_mapping(receipt_path, "receipt")
    approval = _read_mapping(approval_path, "approval")
    schema = _read_mapping(schema_path, "schema")
    schema_hash = _sha256_file(schema_path)
    validate_approval_record(
        approval,
        receipt=receipt,
        output_path=safetensors_path,
        expected_schema_sha256=schema_hash,
    )
    expected = _schema_tensors(schema)
    with safe_open(safetensors_path, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(expected):
            raise ValueError("safetensors tensor names do not match schema")
        for name in handle.keys():
            specification = expected[name]
            view = handle.get_slice(name)
            if (
                list(view.get_shape()) != specification["shape"]
                or view.get_dtype() != _safe_dtype(specification["dtype"])
            ):
                raise ValueError(f"safetensors tensor schema mismatch: {name}")

    torch.set_num_threads(1)
    state = load_file(safetensors_path, device="cpu")
    if _is_released_schema(expected):
        model: nn.Module = TimeRewarderModel()
        model.load_state_dict(state, strict=True)
        del state
    else:
        model = _TensorContainer(state)
    model.to("cpu")
    model.eval()
    return model


def preprocess_rgb(frame: np.ndarray) -> torch.Tensor:
    """Apply the pinned RGB resize, center crop, and CLIP normalization."""
    if (
        not isinstance(frame, np.ndarray)
        or frame.dtype != np.uint8
        or frame.ndim != 3
        or frame.shape[2] != 3
        or min(frame.shape[:2]) < 1
    ):
        raise ValueError("frame must be a nonempty uint8 RGB array")
    value = torch.from_numpy(frame.copy()).permute(2, 0, 1).float().unsqueeze(0)
    height, width = frame.shape[:2]
    scale = 256.0 / min(height, width)
    resized = (
        max(256, round(height * scale)),
        max(256, round(width * scale)),
    )
    value = functional.interpolate(
        value, size=resized, mode="bicubic", align_corners=False, antialias=True
    )[0]
    top = (resized[0] - 224) // 2
    left = (resized[1] - 224) // 2
    value = value[:, top : top + 224, left : left + 224]
    mean = torch.tensor([123.675, 116.28, 103.53]).reshape(3, 1, 1)
    standard_deviation = torch.tensor([58.395, 57.12, 57.375]).reshape(3, 1, 1)
    return (value - mean) / standard_deviation


def predict_distances(
    model: nn.Module,
    frames: torch.Tensor,
    ordered_pairs: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Encode unique frames once and decode all requested ordered pairs."""
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4:
        raise ValueError("frames must be a four-dimensional tensor")
    _validate_pairs(frames.shape[0], ordered_pairs)
    with torch.inference_mode():
        features = model.encode_frames(frames.to("cpu"))
        logits = model.pair_logits(features, ordered_pairs)
    return logits_to_scalar(
        logits.detach().cpu().numpy().astype(np.float64), bins=20
    ).reshape(-1)


def _validate_pairs(
    frame_count: int, pairs: Sequence[tuple[int, int]]
) -> None:
    if not pairs:
        raise ValueError("ordered pairs must not be empty")
    for pair in pairs:
        if (
            len(pair) != 2
            or any(type(index) is not int for index in pair)
            or pair[0] == pair[1]
            or min(pair) < 0
            or max(pair) >= frame_count
        ):
            raise ValueError("invalid ordered frame pair")


def _read_mapping(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _schema_tensors(schema: dict[str, object]) -> dict[str, dict[str, object]]:
    tensors = schema.get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError("schema tensors")
    result: dict[str, dict[str, object]] = {}
    for name, specification in tensors.items():
        if (
            not isinstance(name, str)
            or not isinstance(specification, dict)
            or not isinstance(specification.get("shape"), list)
            or specification.get("dtype") != "float32"
        ):
            raise ValueError("schema tensor specification")
        result[name] = specification
    return result


def _safe_dtype(dtype: str) -> str:
    if dtype != "float32":
        raise ValueError(f"unsupported tensor dtype: {dtype}")
    return "F32"


def _is_released_schema(expected: dict[str, dict[str, object]]) -> bool:
    return {
        "visual.conv1.weight",
        "visual.proj",
        "order_mlp.1.weight",
        "token_embedding.weight",
    } <= set(expected)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
