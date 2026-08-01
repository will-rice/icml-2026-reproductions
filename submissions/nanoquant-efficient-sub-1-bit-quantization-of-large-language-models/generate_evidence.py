from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PAPER_ID = "qiZDlnvWTR"
PAPER_TITLE = "NanoQuant: Efficient Sub-1-bit Quantization of Large Language Models"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_evidence() -> dict:
    observations = {
        "source_indicators": {
            "admm_nanoquant": {
                "path": "src/nanoquant/core/admm_nq.py",
                "function": "factorize_admm_nanoquant",
                "has_binary_terms": True,
                "has_low_rank_factor_matrices": True,
                "evidence": (
                    "The pinned source decomposes W into A/B factor matrices, "
                    "uses rank-1 sign approximations, and exports A/B latent "
                    "matrices plus scale_pre/scale_post."
                ),
            },
            "linear_module": {
                "path": "src/nanoquant/modules/linear.py",
                "class": "NanoQuantLinear",
                "has_binary_terms": True,
                "has_learned_scales": True,
                "evidence": (
                    "NanoQuantLinear registers V/U binary factors or V_latent/"
                    "U_latent trainable factors together with scale_pre, "
                    "scale_mid, and scale_post parameters."
                ),
            },
            "quant_config": {
                "path": "src/nanoquant/modules/quant_config.py",
                "supports_fractional_bits": True,
                "default_bits": 1.0,
                "evidence": (
                    "The quantization configuration exposes bits as a float, "
                    "ADMM type selection, calibration-only PTQ options, and "
                    "device_map for offloaded large-model loading."
                ),
            },
            "hub_wrapper": {
                "path": "src/nanoquant/modules/hub.py",
                "class": "NanoQuantConfigDataclass",
                "supports_hub_checkpoint_config": True,
            },
        },
        "project_metadata": {
            "method_type": "post-training quantization",
            "compression_floor": "sub-1-bit",
            "supported_model_families": ["OPT", "Llama", "Qwen", "Gemma", "Rnj-1"],
            "large_model_usage": (
                "README documents --device_map auto for very large models (>70B)."
            ),
            "kernel_paths": ["CUDA GEMV decode", "CUDA GEMM prefill"],
        },
    }
    claim_results = {
        "claim-1": {
            "claim": (
                "NanoQuant formulates post-training LLM quantization as low-rank "
                "binary factorization with binary matrices and learned scales "
                "(Section 3)."
            ),
            "challenge_claim_sha256": (
                "8a314e988f0e1500fc5b0873afa8fa2049ed5adcb88ca71073a01e5fe32111dc"
            ),
            "status": "verified",
            "observation": (
                "Pinned Apache-2.0 source implements factorize_admm_nanoquant, "
                "NanoQuantLinear binary U/V factor storage, latent binary training "
                "parameters, and learned scale_pre/scale_mid/scale_post parameters."
            ),
        },
        "claim-2": {
            "claim": (
                "NanoQuant is the only compared PTQ method marked as supporting "
                "both 70B+ LLMs and sub-1-bit compression (Table 1)."
            ),
            "challenge_claim_sha256": (
                "8846fdf5a25a7229ec7beb71997bff9feb838614ccd376e900573ee70f3d6aad"
            ),
            "status": "toy",
            "observation": (
                "The released project metadata supports the NanoQuant side of "
                "the claim: PTQ, sub-1-bit operation, and >70B offload usage. "
                "This reproduction does not independently audit every Table 1 "
                "baseline, so the exclusivity portion is not treated as fully "
                "verified."
            ),
        },
    }
    metadata_json = json.dumps(observations, sort_keys=True)
    return {
        "paper_id": PAPER_ID,
        "paper_title": PAPER_TITLE,
        "upstream": {
            "arxiv": "2602.06694",
            "github": "SamsungLabs/NanoQuant@a9e0a430881ff80d83b622c3129e330dc33c04f5",
            "github_default_branch": "main",
            "code_license": "Apache-2.0",
            "project_readme": "https://github.com/SamsungLabs/NanoQuant",
            "paper_page": "https://huggingface.co/papers/2602.06694",
        },
        "claim_results": claim_results,
        "observations": observations,
        "provenance": {
            "source_urls": [
                "https://arxiv.org/abs/2602.06694",
                "https://github.com/SamsungLabs/NanoQuant",
                "https://huggingface.co/papers/2602.06694",
            ],
            "local_metadata_sha256": _sha256_text(metadata_json),
            "commands": [
                "git ls-remote --symref https://github.com/SamsungLabs/NanoQuant.git HEAD refs/heads/main",
                "inspect pinned source files under src/nanoquant/core, modules, and kernel",
                "pytest -q submissions/nanoquant-efficient-sub-1-bit-quantization-of-large-language-models/tests",
            ],
        },
        "unreplicated": [
            "70B compression and 8GB GPU fit were not rerun.",
            "WikiText perplexity tables were not rerun.",
            "Zero-shot commonsense evaluation was not rerun.",
            "CUDA GEMV/GEMM kernel benchmarks were not compiled or executed.",
            "Table 1 baseline exclusivity was not independently audited across all compared methods.",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("evidence/bundle.json"))
    args = parser.parse_args(argv)
    bundle = build_evidence()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote evidence bundle to {args.output}")


if __name__ == "__main__":
    main()
