from __future__ import annotations

from dataclasses import dataclass


ATTEMPT_ID = "ca01c0a8-f6cc-4d80-bf3a-c569ba7b4896"
PAPER_ID = "9wpwfSJCp9"
PAPER_TITLE = "SleepLM: Natural-Language Intelligence for Human Sleep"
UPSTREAM_REVISION = (
    "arxiv:2602.23605+github:yang-ai-lab/SleepLM@"
    "f788466b926a9ed95d473c220814c912d5ce6abc+hf-model:"
    "yang-ai-lab/SleepLM-Base@ec0f94ff2be04fe11ff5a2b37ac38e8f40aa5c53"
)

UPSTREAM_PINS = {
    "arxiv": "arxiv:2602.23605",
    "project_page": "https://yang-ai-lab.github.io/SleepLM/",
    "code_repo": (
        "github:yang-ai-lab/SleepLM@"
        "f788466b926a9ed95d473c220814c912d5ce6abc"
    ),
    "hf_model": (
        "hf-model:yang-ai-lab/SleepLM-Base@"
        "ec0f94ff2be04fe11ff5a2b37ac38e8f40aa5c53"
    ),
}

TARGET_CLAIMS = [
    {
        "id": "multimodal_alignment",
        "challenge_claim_sha256": (
            "0a9ab1e42662e1b1e40ded1370179413b554c5a8663d8f3bd293c56ea6f694f8"
        ),
        "text": (
            "SleepLM aligns multimodal polysomnography signals with natural "
            "language to support sleep interpretation and interaction beyond "
            "closed sleep-label spaces (Section 3)."
        ),
    },
    {
        "id": "caption_pipeline",
        "challenge_claim_sha256": (
            "a98161b9f57420109f2d31de27ac2b2d45960406af43b39dc86e0f0b17463d01"
        ),
        "text": (
            "The paper introduces a multilevel sleep caption generation pipeline "
            "for creating sleep-text supervision (Section 3)."
        ),
    },
    {
        "id": "dataset_scale",
        "challenge_claim_sha256": (
            "c0760f7182d3b658fb12eb9409432891beb5e7634a49f10187cea9afcb595666"
        ),
        "text": (
            "The curated sleep-text dataset comprises more than 100K hours of "
            "data from over 10,000 individuals (Abstract)."
        ),
    },
    {
        "id": "unified_objective",
        "challenge_claim_sha256": (
            "142c585a4ac9c87506014c60d24333769cec610fc3b02fa9e464082325b984af"
        ),
        "text": (
            "SleepLM uses a unified pretraining objective combining contrastive "
            "alignment, caption generation, and signal reconstruction "
            "(Section 3)."
        ),
    },
]

SOURCE_FILES = {
    "github_readme": {
        "url": (
            "https://raw.githubusercontent.com/yang-ai-lab/SleepLM/"
            "f788466b926a9ed95d473c220814c912d5ce6abc/README.md"
        ),
        "sha256": "dfba50c6afa5eb7023ed03fec2a5f563b3086853b9b9d53da4226c4087e96518",
        "observed_facts": [
            "sleep-language PSG/text alignment",
            "targeted caption generation",
            "cross-modal retrieval",
            "10-channel 30-second input contract",
            "credentialed training data limitation",
        ],
    },
    "github_license": {
        "url": (
            "https://raw.githubusercontent.com/yang-ai-lab/SleepLM/"
            "f788466b926a9ed95d473c220814c912d5ce6abc/LICENSE"
        ),
        "sha256": "1965530331f9dadeca7674cbd66f1d1c431ad2b7fed05a975c769d9508819f74",
        "observed_facts": ["MIT license"],
    },
    "signal_model_config": {
        "url": (
            "https://raw.githubusercontent.com/yang-ai-lab/SleepLM/"
            "f788466b926a9ed95d473c220814c912d5ce6abc/src/open_clip/"
            "model_configs/sleep_coca_base_dualtransformer.json"
        ),
        "sha256": "907f8e149af7c2e2d8bfb2bd894f62fffb54d6038b9814d88241327522d22dc9",
        "observed_facts": [
            "input_channels=10",
            "signal_length=1920",
            "sampling_rate=64",
            "decoder_type=cross_attention",
        ],
    },
    "biosignals_model_source": {
        "url": (
            "https://raw.githubusercontent.com/yang-ai-lab/SleepLM/"
            "f788466b926a9ed95d473c220814c912d5ce6abc/src/open_clip/"
            "biosignals_coca_model.py"
        ),
        "sha256": "612af897074fb33495b75e8333a3dc71bda5bcccf42bea35d370aab7142d2d34",
        "observed_facts": [
            "biosignals-text contrastive learning",
            "biosignals encoder tower",
            "task-specific pooling for contrastive and generative objectives",
        ],
    },
    "hf_model_card": {
        "url": (
            "https://huggingface.co/yang-ai-lab/SleepLM-Base/raw/"
            "ec0f94ff2be04fe11ff5a2b37ac38e8f40aa5c53/README.md"
        ),
        "sha256": "6fa6e6b1ffaef77a08b106e3c7e507c6a37e1e62e3768ad74739692fbfc63924",
        "observed_facts": [
            "license: mit",
            "100K+ PSG hours and 10,000+ individuals statement",
            "targeted caption generation",
            "cross-modal retrieval",
            "training data are credentialed",
        ],
    },
    "hf_license": {
        "url": (
            "https://huggingface.co/yang-ai-lab/SleepLM-Base/raw/"
            "ec0f94ff2be04fe11ff5a2b37ac38e8f40aa5c53/LICENSE"
        ),
        "sha256": "1965530331f9dadeca7674cbd66f1d1c431ad2b7fed05a975c769d9508819f74",
        "observed_facts": ["MIT license"],
    },
}

OBSERVATIONS = {
    "licenses": {
        "github_repository": {
            "license": "MIT",
            "source": "github_license",
        },
        "hf_model_card": {
            "license": "mit",
            "source": "hf_model_card",
        },
    },
    "repository_tree": {
        "released_files": [
            "LICENSE",
            "README.md",
            "demo.ipynb",
            "requirements.txt",
            "src/open_clip/biosignals_coca_model.py",
            "src/open_clip/coca_model.py",
            "src/open_clip/factory.py",
            "src/open_clip/model.py",
            "src/open_clip/model_configs/sleep_coca_base_dualtransformer.json",
            "src/open_clip/tokenizer.py",
            "src/open_clip/transformer.py",
        ],
        "checkpoint_file": "model_checkpoint.pt",
        "checkpoint_source": "hf_model",
    },
    "signal_input_contract": {
        "tensor_shape": "[N, 10, 1920]",
        "epoch_seconds": 30,
        "sampling_rate_hz": 64,
        "channel_count": 10,
        "channels": [
            "ECG",
            "ABD",
            "THX",
            "AF",
            "EOG_Left",
            "EOG_Right",
            "EEG_C3_A2",
            "EEG_C4_A1",
            "EMG_Chin",
            "POS",
        ],
        "preprocessing": [
            "resample to 64 Hz",
            "z-score each channel",
            "zero-pad missing channels",
            "encode POS with repository integer mapping",
        ],
        "sources": ["github_readme", "hf_model_card", "signal_model_config"],
    },
    "capabilities": {
        "targeted_caption_generation": {
            "supported": True,
            "source": "hf_model_card",
        },
        "cross_modal_retrieval": {
            "supported": True,
            "source": "hf_model_card",
        },
        "modality_tokens": [
            "brain",
            "cardiac",
            "respiration",
            "somatic",
        ],
        "shared_signal_text_embedding": True,
        "open_vocabulary_sleep_understanding": "source-backed",
    },
    "caption_pipeline": {
        "pipeline_claim_source": "project_page",
        "sleep_text_supervision": "source-backed by project page and model card",
        "raw_caption_pipeline_code_released": False,
        "limitation": (
            "The released repository focuses on inference. The assessment verifies "
            "that the caption-generation pipeline is documented and used as "
            "supervision, but does not rerun the closed data-generation pipeline."
        ),
    },
    "dataset_scale": {
        "cohorts": 5,
        "hours": "100K+",
        "individuals": "10,000+",
        "source": "hf_model_card",
        "raw_training_data_available": False,
        "credentialed_data_source": "NSRR",
        "limitation": (
            "The scale is documented in primary artifacts, but the raw NSRR "
            "training cohorts are not redistributed and require credentials."
        ),
    },
    "pretraining_objective": {
        "terms": [
            "contrastive_alignment",
            "caption_generation",
            "signal_reconstruction",
        ],
        "project_page_equation": (
            "L_total = lambda_con * L_con + lambda_rec * L_rec + "
            "lambda_cap * L_cap"
        ),
        "architecture": "Reconstructive Contrastive Captioner",
        "config_support": {
            "decoder_type": "cross_attention",
            "num_caption_channels": 12,
            "biosignals_encoder": "pure_transformer",
        },
        "sources": [
            "project_page",
            "hf_model_card",
            "signal_model_config",
            "biosignals_model_source",
        ],
    },
}


@dataclass(frozen=True)
class ClaimResult:
    status: str
    claim: str
    evidence: str
    limitations: str | None = None

    def as_dict(self) -> dict[str, str]:
        result = {
            "status": self.status,
            "claim": self.claim,
            "evidence": self.evidence,
        }
        if self.limitations:
            result["limitations"] = self.limitations
        return result


def build_bundle() -> dict:
    return {
        "attempt_id": ATTEMPT_ID,
        "paper_id": PAPER_ID,
        "paper_title": PAPER_TITLE,
        "cpu_only": True,
        "estimated_api_cost_usd": 0.0,
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_pins": UPSTREAM_PINS,
        "target_claims": TARGET_CLAIMS,
        "source_files": SOURCE_FILES,
        "observations": OBSERVATIONS,
        "claim_results": {
            key: result.as_dict() for key, result in _claim_results().items()
        },
        "excluded_claims": [
            {
                "reason": (
                    "Requires GPU-scale benchmark execution and access to "
                    "credentialed NSRR cohorts."
                ),
                "claim": (
                    "State-of-the-art zero-shot/few-shot, cross-modal retrieval, "
                    "sleep captioning, event localization, and unseen-task "
                    "generalization performance claims from Section 4."
                ),
            }
        ],
        "commands": [
            "curl pinned GitHub README/LICENSE/config/source URLs and compute sha256",
            "curl pinned Hugging Face model-card/LICENSE URLs and compute sha256",
            "UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run --project submissions/sleeplm-natural-language-intelligence-for-human-sleep pytest submissions/sleeplm-natural-language-intelligence-for-human-sleep/tests -q",
            "UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run --project submissions/sleeplm-natural-language-intelligence-for-human-sleep python submissions/sleeplm-natural-language-intelligence-for-human-sleep/generate_evidence.py",
        ],
        "limitations": [
            "No NSRR raw training cohorts were downloaded or inspected.",
            "No GPU training, benchmark reproduction, or model-checkpoint inference was run.",
            "Dataset-scale evidence is primary-artifact documentation, not a raw-data recount.",
        ],
    }


def _claim_results() -> dict[str, ClaimResult]:
    return {
        "multimodal_alignment": ClaimResult(
            status="verified",
            claim=TARGET_CLAIMS[0]["text"],
            evidence=(
                "Pinned GitHub/HF artifacts describe PSG-to-text alignment, "
                "a shared signal-text embedding space, cross-modal retrieval, "
                "and open-vocabulary sleep understanding. The config fixes a "
                "10-channel, 1920-sample biosignals encoder input."
            ),
        ),
        "caption_pipeline": ClaimResult(
            status="verified",
            claim=TARGET_CLAIMS[1]["text"],
            evidence=(
                "The project page and model card document multilevel "
                "sleep-caption supervision and released inference support for "
                "targeted caption generation with modality tokens."
            ),
            limitations=(
                "The closed cohort caption-generation pipeline itself is not "
                "rerun because training data are credentialed."
            ),
        ),
        "dataset_scale": ClaimResult(
            status="inconclusive",
            claim=TARGET_CLAIMS[2]["text"],
            evidence=(
                "Primary artifacts state five NSRR cohorts, 100K+ PSG hours, "
                "and 10,000+ individuals."
            ),
            limitations=(
                "The raw training cohorts are not redistributed, so the scale "
                "cannot be independently recounted from released files."
            ),
        ),
        "unified_objective": ClaimResult(
            status="verified",
            claim=TARGET_CLAIMS[3]["text"],
            evidence=(
                "The project page exposes the total objective with contrastive, "
                "reconstruction, and captioning terms; released config/source "
                "show cross-attention decoding and biosignals-text CoCa support."
            ),
        ),
        "artifact_release": ClaimResult(
            status="verified",
            claim="MIT-licensed code and Hugging Face model artifacts are released.",
            evidence=(
                "Pinned GitHub and HF artifacts include README, LICENSE, demo, "
                "OpenCLIP source, model config, and model_checkpoint.pt."
            ),
        ),
    }
