from __future__ import annotations

import json
import re
import ast
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


GITHUB_REPO = "z-lab/dflash"
GITHUB_REVISION = "94e4abc5e0c31b67bc1a9d30f1cc34ece28a8756"
ARCHIVE_URL = f"https://github.com/{GITHUB_REPO}/archive/{GITHUB_REVISION}.zip"


CLAIM_HASHES = {
    "mechanism": "2637cad1833ecb87f838786ba8aee2364688f5c9a645ece89d4cf1fddbb26f68",
    "qwen_non_thinking": "83c2b07156c4deb43c365e160990f0cb39c9190c14d7ba862aac88079bce1551",
    "thinking": "03c7bbcd902aee67b70a33a633fb2d90d249e2dceeca4971cd1771969db65076",
    "sglang_fa4": "1c22abd30516043736ec10d93a157de5fc2216ef069661335d1c04d49c919ddf",
    "long_context": "a591774b2149b43146f956d59c5df39775bd15ff9090f19e15a8bd91d141bdd1",
    "ablations": "4e4cbcaf9b48d19f5fb2e11c9f92b04b30bcd8d97de43a3c95f4a3a20da1561e",
}

TARGET_CLAIMS = {
    "mechanism": "DFlash uses a lightweight block diffusion draft model to generate speculative draft tokens in parallel, conditioned on hidden features extracted from the target LLM (Figure 2).",
    "qwen_non_thinking": "On Qwen3 models with thinking disabled, DFlash consistently outperforms EAGLE-3 and reaches about 4.9x average greedy-decoding speedup over autoregressive decoding (Table 1).",
    "thinking": "With thinking mode enabled, DFlash maintains high acceptance length and roughly 4.5x speedups on reasoning-model settings (Table 2).",
    "sglang_fa4": "On SGLang with the FA4 backend, DFlash provides throughput speedups across Qwen3-4B, Qwen3-8B, and Qwen3-Coder-30B-A3B-Instruct up to 5.1x (Table 3).",
    "long_context": "Long-context fine-tuning preserves or improves DFlash acceptance length as LongBench context length increases beyond 4K (Table 4).",
    "ablations": "Ablations show that draft depth, number of target hidden features, block-size choice, and KV injection all materially affect DFlash acceptance length and speedup (Table 6; Table 7; Table 8; Table 9).",
}


@dataclass(frozen=True)
class SourceSnapshot:
    files: dict[str, str]


@lru_cache(maxsize=1)
def load_source_snapshot() -> SourceSnapshot:
    with tempfile.TemporaryDirectory(prefix="dflash-src-") as tmp:
        archive = Path(tmp) / "source.zip"
        urllib.request.urlretrieve(ARCHIVE_URL, archive)
        with zipfile.ZipFile(archive) as zf:
            wanted = {
                "README.md",
                "dflash/model.py",
                "dflash/benchmark.py",
                "dflash/model_mlx.py",
                "pyproject.toml",
            }
            files: dict[str, str] = {}
            for name in zf.namelist():
                relative = "/".join(Path(name).parts[1:])
                if relative in wanted:
                    files[relative] = zf.read(name).decode("utf-8")
    return SourceSnapshot(files=files)


def summarize_source(source: SourceSnapshot) -> dict[str, Any]:
    model = source.files["dflash/model.py"]
    benchmark = source.files["dflash/benchmark.py"]
    readme = source.files["README.md"]

    return {
        "github_repo": GITHUB_REPO,
        "github_revision": GITHUB_REVISION,
        "archive_url": ARCHIVE_URL,
        "files_audited": sorted(source.files),
        "has_dflash_generate": "def dflash_generate(" in model,
        "has_context_feature_extraction": "def extract_context_feature" in model
        and "torch.cat(selected_states" in model,
        "has_target_hidden_conditioning": "target_hidden" in model
        and "k_ctx = self.k_proj(target_hidden)" in model,
        "has_noise_embedding_draft_block": "noise_embedding" in model
        and "block_output_ids[:, 1:] = sample(draft_logits)" in model,
        "has_noncausal_draft_attention": "self.is_causal = False" in model
        and "is_causal=False" in model,
        "has_parallel_block_acceptance": "cumprod(dim=1)" in model
        and "acceptance_lengths.append" in model,
        "benchmark_backends": _benchmark_backends(benchmark),
        "benchmark_datasets": _benchmark_datasets(benchmark),
        "readme_mentions_fa4": "fa4" in readme.lower(),
        "readme_mentions_qwen3_drafts": "Qwen3-4B-DFlash-b16" in readme
        and "Qwen3-8B-DFlash-b16" in readme,
    }


def build_evidence_bundle(source: SourceSnapshot | None = None) -> dict[str, Any]:
    summary = summarize_source(load_source_snapshot() if source is None else source)
    return {
        "paper_id": "Oz335dV48X",
        "attempt_id": "9bdedd5e-7e42-43cc-a356-bab8f5d9c344",
        "snapshot_id": "db4e589075a6516afc06f7748ed6fdf9614a97cc6172a1eb9cbc91996483d48d",
        "challenge_revision": "81166abbeb76e5f79ff87e51061b5a0306507203",
        "upstream": {
            "paper": "arxiv:2602.06036v2",
            "openreview": "https://openreview.net/forum?id=Oz335dV48X",
            "github": f"github:{GITHUB_REPO}@{GITHUB_REVISION}",
            "model_cards": [
                "z-lab/Qwen3-4B-DFlash-b16@b74e3a329c4d963783143b1e970d95b002be72bd",
                "z-lab/Qwen3-8B-DFlash-b16@9b41424b7109f9c5413454f481b09a82b85333f4",
                "z-lab/Qwen3-Coder-30B-A3B-DFlash@98ca0e3e2e6a372f2789d3a5e146566194084317",
            ],
        },
        "source_audit": summary,
        "claims": _claim_records(),
        "reproduced_speedup_measurements": [],
        "reproduced_ablation_measurements": [],
        "cost_usd": 0.0,
        "cpu_only": True,
    }


def _claim_records() -> list[dict[str, str]]:
    return [
        _claim(
            "mechanism",
            "verified",
            "The pinned source implements target hidden-state extraction, non-causal DFlash draft attention, mask/noise-token block drafting, and speculative prefix verification.",
        ),
        _claim(
            "qwen_non_thinking",
            "inconclusive",
            "No GPU serving benchmark was run for non-thinking Qwen3 speedup or EAGLE-3 comparison.",
        ),
        _claim(
            "thinking",
            "inconclusive",
            "No thinking-mode reasoning benchmark was run.",
        ),
        _claim(
            "sglang_fa4",
            "inconclusive",
            "The README exposes SGLang FA4 commands, but no SGLang FA4 throughput run was executed.",
        ),
        _claim(
            "long_context",
            "inconclusive",
            "No LongBench or long-context fine-tuning acceptance-length run was executed.",
        ),
        _claim(
            "ablations",
            "inconclusive",
            "No draft-depth, hidden-feature-count, block-size, or KV-injection ablation runs were executed.",
        ),
    ]


def _claim(key: str, status: str, observation: str) -> dict[str, str]:
    return {
        "target_claim": TARGET_CLAIMS[key],
        "challenge_claim": TARGET_CLAIMS[key],
        "challenge_claim_sha256": CLAIM_HASHES[key],
        "status": status,
        "observation": observation,
    }


def _benchmark_datasets(source: str) -> list[str]:
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "DATASETS":
                    if not isinstance(node.value, ast.Dict):
                        return []
                    return sorted(
                        key.value
                        for key in node.value.keys
                        if isinstance(key, ast.Constant) and isinstance(key.value, str)
                    )
    return []


def _benchmark_backends(source: str) -> list[str]:
    module = ast.parse(source)
    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        if not (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--backend"
        ):
            continue
        for keyword in node.keywords:
            if keyword.arg == "choices":
                return sorted(ast.literal_eval(keyword.value))
    return []
