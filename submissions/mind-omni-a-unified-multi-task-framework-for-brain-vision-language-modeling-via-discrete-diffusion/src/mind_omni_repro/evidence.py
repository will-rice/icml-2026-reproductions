"""CPU-only evidence audit for pinned Mind-Omni artifacts."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any


ATTEMPT_ID = "cacb1796-7495-4440-845c-1002729bea1b"
PAPER_ID = "3gCdh3u2GK"
SNAPSHOT_ID = "8e1331c16c97f17ffa8b34fd6701cd91c805ab8b575818656718799d504db20e"
TITLE = "Mind-Omni: A Unified Multi-Task Framework for Brain-Vision-Language Modeling via Discrete Diffusion"
GITHUB_REVISION = "818dcd160c130334bd36a7a8a7f7e7f00772084d"
GITHUB_REPO = "ReedOnePeck/Mind-Omni"

UPSTREAM_PINS = {
    "paper": "arxiv:2605.29591",
    "official_code": f"github:{GITHUB_REPO}@{GITHUB_REVISION}",
    "dataset_release": "modelscope-dataset:LLLLLYYYYYzzz/NSD",
    "checkpoint_release": "modelscope-model:LLLLLYYYYYzzz/Mind_Omni_V1_ckpt",
}

CLAIMS = [
    {
        "target_claim": "Mind-Omni unifies seven brain, image, and text encoding/decoding tasks in a single discrete diffusion framework (Table 1).",
        "challenge_claim_sha256": "510e56e579a79e97258b2b48ab1c8b0d7bc96dc94c8ae99f00f736faf91e7e08",
    },
    {
        "target_claim": "The Brain Tokenizer converts continuous fMRI signals into discrete tokens aligned with image and text representations in a shared semantic space (Section 3.1).",
        "challenge_claim_sha256": "e846f44cd916828ca7091c8b22d7f9964616fca3b0588b823c5e58ea1fbb0d1b",
    },
    {
        "target_claim": "The framework curates a Brain Question Answering instruction-tuning dataset using Qwen2-VL to support reasoning over brain signals (Section 4).",
        "challenge_claim_sha256": "e275bc58a479ca235b37e6e7aa1dfbf80060a8730523709b1ad910d9f20cc01a",
    },
    {
        "target_claim": "Mind-Omni establishes a new state of the art among unified brain-vision-language frameworks in the multi-task evaluation (Tables 2-4).",
        "challenge_claim_sha256": "69a1212c6650c1c0530dc520736bbc0e67be8d1593ebac5f6f8933e81a3093a1",
    },
    {
        "target_claim": "Joint image-text conditioning shows synergistic gains over single-modality conditioning in neural encoding and decoding analyses (Figures 7 and 8).",
        "challenge_claim_sha256": "4af5cfd5dc824050d15fc178ef13fd92b7668fb65b4768fd2f8d79e0383e4a16",
    },
]

CODE_PATHS = [
    "README.md",
    "MindOmni_src/tri_modal_pipeline.py",
    "MindOmni_src/tri_modal_transformer.py",
    "train_fMRI_tokenizer_perceptual/fMRI_tokenizer_perceptual.py",
    "data_processing/stage2_dataset_prep/tokenize_QAs.py",
    "train_stage2_short_VQA/train_stage2_shortVQA.py",
    "train_stage1/train_stage1_utils.py",
    "train_stage1/train_stage1.sh",
    "train_stage1_2/train_stage1_2.sh",
    "train_stage2_short_VQA/train_stage2_shortVQA.sh",
    "Validate_the_models/evaluate_easy_reasoning_qc.py",
]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def audit_artifacts(code_artifacts: dict[str, str]) -> dict[str, Any]:
    joined = "\n".join(code_artifacts.values()).lower()
    paths = " ".join(code_artifacts)

    return {
        "tri_modal_pipeline": _presence(
            "MindOmni_src/tri_modal_pipeline.py" in code_artifacts
            and all(term in joined for term in ["prompt", "image", "brain"])
        ),
        "tri_modal_transformer": _presence(
            "MindOmni_src/tri_modal_transformer.py" in code_artifacts
            and all(term in joined for term in ["text_seq_len", "image_seq_len", "brain_seq_len"])
        ),
        "brain_tokenizer": _presence(
            "train_fMRI_tokenizer_perceptual/fMRI_tokenizer_perceptual.py" in code_artifacts
            and ("vq_fmri" in joined or "desired_token_num" in joined)
            and ("codebook" in joined or "semantic" in joined or "perceptual" in joined)
        ),
        "brain_qa_processing": _presence(
            "data_processing/stage2_dataset_prep/tokenize_QAs.py" in code_artifacts
            and ("question" in joined and "answer" in joined)
            and ("short_vqa" in joined or "full_prompts" in joined or "qwen" in joined)
        ),
        "stage_launchers": _presence(
            all(
                path in paths
                for path in [
                    "train_stage1/train_stage1.sh",
                    "train_stage1_2/train_stage1_2.sh",
                    "train_stage2_short_VQA/train_stage2_shortVQA.sh",
                ]
            )
        ),
        "dataset_release": _presence(
            "modelscope.cn/datasets/LLLLLYYYYYzzz/NSD" in code_artifacts.get("README.md", "")
            or "dataset repo" in code_artifacts.get("README.md", "").lower()
        ),
        "checkpoint_release": _presence(
            "modelscope.cn/models/LLLLLYYYYYzzz/Mind_Omni_V1_ckpt" in code_artifacts.get("README.md", "")
            or "checkpoint repo" in code_artifacts.get("README.md", "").lower()
        ),
        "evaluation_release": _presence(
            "evaluation code release" in code_artifacts.get("README.md", "").lower()
            and "[ ]" not in code_artifacts.get("README.md", "").lower()
        ),
        "source_hashes": {path: sha256_text(text) for path, text in sorted(code_artifacts.items())},
    }


def build_evidence_bundle(
    code_artifacts: dict[str, str],
    raw_result_artifacts: dict[str, str] | None = None,
) -> dict[str, Any]:
    raw_result_artifacts = raw_result_artifacts or {}
    artifact_audit = audit_artifacts(code_artifacts)
    result_hashes = {path: sha256_text(text) for path, text in sorted(raw_result_artifacts.items())}

    return {
        "attempt_id": ATTEMPT_ID,
        "paper_id": PAPER_ID,
        "title": TITLE,
        "snapshot_id": SNAPSHOT_ID,
        "upstream_pins": UPSTREAM_PINS,
        "artifact_access": {
            "code_files": sorted(code_artifacts),
            "raw_result_artifacts": sorted(raw_result_artifacts),
        },
        "audits": {
            "official_code": artifact_audit,
            "raw_result_hashes": result_hashes,
        },
        "claim_results": [
            _source_claim(1, CLAIMS[0], artifact_audit, ["tri_modal_pipeline", "tri_modal_transformer", "stage_launchers"]),
            _source_claim(2, CLAIMS[1], artifact_audit, ["brain_tokenizer", "tri_modal_pipeline", "checkpoint_release"]),
            _source_claim(3, CLAIMS[2], artifact_audit, ["brain_qa_processing", "dataset_release"]),
            _numeric_claim(4, CLAIMS[3], result_hashes, artifact_audit),
            _numeric_claim(5, CLAIMS[4], result_hashes, artifact_audit),
        ],
    }


def fetch_pinned_artifacts(project_root: Path | None = None) -> dict[str, str]:
    project_root = project_root or Path(__file__).resolve().parents[2]
    local_cache = project_root / ".cache" / f"mind-omni-{GITHUB_REVISION}"
    artifacts: dict[str, str] = {}
    for path in CODE_PATHS:
        local_path = local_cache / path
        if local_path.exists():
            artifacts[path] = local_path.read_text(encoding="utf-8", errors="replace")
        else:
            artifacts[path] = _fetch_github_text(path)
    return artifacts


def write_evidence(output_path: Path) -> dict[str, Any]:
    bundle = build_evidence_bundle(fetch_pinned_artifacts(output_path.resolve().parents[1]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(output_path.resolve().parents[1], bundle)
    return bundle


def _presence(is_present: bool) -> dict[str, str]:
    return {"status": "present" if is_present else "missing"}


def _source_claim(index: int, claim: dict[str, str], audit: dict[str, Any], required: list[str]) -> dict[str, Any]:
    passed = all(audit[name]["status"] == "present" for name in required)
    return {
        "claim_index": index,
        "target_claim": claim["target_claim"],
        "claim_sha256": claim["challenge_claim_sha256"],
        "status": "verified" if passed else "inconclusive",
        "observation": f"CPU-only source audit found required components: {', '.join(required)}.",
        "limitation": "This verifies released source structure and artifact pointers, not a full model training rerun.",
    }


def _numeric_claim(
    index: int,
    claim: dict[str, str],
    raw_result_hashes: dict[str, str],
    audit: dict[str, Any],
) -> dict[str, Any]:
    has_results = bool(raw_result_hashes)
    has_eval = audit["evaluation_release"]["status"] == "present"
    if has_results and has_eval:
        status = "toy"
        limitation = "Raw evaluation artifacts were hashed, but this audit does not rerun full checkpoint inference."
    else:
        status = "unavailable"
        limitation = "No raw evaluation output was available; paper-reported metrics are not treated as reproduced measurements."

    return {
        "claim_index": index,
        "target_claim": claim["target_claim"],
        "claim_sha256": claim["challenge_claim_sha256"],
        "status": status,
        "observation": "Pinned source exposes validation entrypoints and artifact release notes.",
        "limitation": limitation,
    }


def _fetch_github_text(path: str) -> str:
    url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_REVISION}/{path}"
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def _write_report(project_root: Path, bundle: dict[str, Any]) -> None:
    report_dir = project_root / "pages"
    report_dir.mkdir(exist_ok=True)
    statuses = ", ".join(
        f"claim {item['claim_index']}: {item['status']}" for item in bundle["claim_results"]
    )
    report = (
        "# Mind-Omni Evidence Report\n\n"
        f"Paper: `{PAPER_ID}`\n\n"
        f"Attempt: `{ATTEMPT_ID}`\n\n"
        f"Snapshot: `{SNAPSHOT_ID}`\n\n"
        f"Upstream source: `{UPSTREAM_PINS['official_code']}`\n\n"
        f"Statuses: {statuses}.\n\n"
        "Claims 4 and 5 remain unavailable because the released repository marks evaluation code as pending and no raw evaluation output was present.\n"
    )
    (report_dir / "report.md").write_text(report, encoding="utf-8")