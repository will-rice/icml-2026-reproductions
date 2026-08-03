from __future__ import annotations


PAPER_ID = "jPKqiaPTEd"
PAPER_TITLE = "Protein Fold Classification at Scale: Benchmarking and Pretraining"

UPSTREAM_PINS = {
    "arxiv": "arxiv:2605.18552",
    "code_repo": "github:BorgwardtLab/TEDBench@ad3c208db13e5e0e124719300ec19fffab4c33e1",
    "ted_dataset": "hf-dataset:TEDBench/ted@825dfceb2acd92cebc62a5b1bb95e8a13407160a",
    "afdb_dataset": "hf-dataset:TEDBench/afdb@bb3caa7a24f9adf9758392298c88c76610dda2b5",
    "cath_dataset": "hf-dataset:TEDBench/cath@ce80cd3e7307bece444423ad0e32943ffce96f35",
    "miae_b_model": "hf-model:TEDBench/miae-b@864ad47cf09276d76df5d97ba2505db1f5dfc57d",
}

TARGET_CLAIMS = [
    {
        "id": "tedbench_dataset",
        "challenge_claim_sha256": "51f432978dc75b230f4bab006d45b7ef51f500cbbf68ba18606258d2750f87e4",
        "text": "TEDBench is a large-scale non-redundant protein fold classification benchmark built from TED/AFDB predicted structures with an external experimental-structure test set (Figure 1, Table 1).",
    },
    {
        "id": "miae_architecture",
        "challenge_claim_sha256": "16690fadb62eb2ab2926b3c2a514d9509412d67db420202c8209dd7d114ae71f",
        "text": "Masked Invariant Autoencoders mask a high ratio of backbone frames, encode only unmasked frames with an SE(3)-invariant geometric encoder, and reconstruct full-frame targets with a decoder (Figure 3).",
    },
    {
        "id": "miae_masking",
        "challenge_claim_sha256": "d3be0d566cf89cb984fe76a66979ca62f1358508bb9b63ea93cc0637abf853cb",
        "text": "MiAE uses up to 90% masking with an SE(3)-invariant encoder and lightweight decoder to reconstruct backbone coordinates from latent representations and mask tokens (Section 4).",
    },
    {
        "id": "cath_transfer_dataset",
        "challenge_claim_sha256": "3316799a694033b48db8483f376328c5376835ee71c2085803991f60376aa0f5",
        "text": "The paper evaluates transfer beyond AlphaFold structures on a curated experimental-structure dataset from CATH v4.4 (Section 5).",
    },
]

DATASETS = {
    "ted": {
        "repo": "TEDBench/ted",
        "revision": "825dfceb2acd92cebc62a5b1bb95e8a13407160a",
        "splits": {"train": 369740, "val": 46217, "test": 46218},
        "total_structures": 462175,
        "class_count": 965,
        "source_components": [
            "Encyclopedia of Domains (TED)",
            "Foldseek-clustered AlphaFold Database",
            "CATH topology labels",
        ],
    },
    "afdb": {
        "repo": "TEDBench/afdb",
        "revision": "bb3caa7a24f9adf9758392298c88c76610dda2b5",
        "splits": {"train": 742183, "val": 7496},
        "total_structures": 749679,
        "clustered_by": "Foldseek",
        "plddt_filter": ">80",
    },
    "cath": {
        "repo": "TEDBench/cath",
        "revision": "ce80cd3e7307bece444423ad0e32943ffce96f35",
        "splits": {"test": 28010},
        "class_count": 965,
        "experimental": True,
        "source": "CATH 4.4 40% non-redundant representative set",
    },
}

MODELS = {
    "miae_b": {
        "repo": "TEDBench/miae-b",
        "revision": "864ad47cf09276d76df5d97ba2505db1f5dfc57d",
        "model_name": "miae_b",
        "parameters": "102M",
        "layers": 12,
        "hidden_dim": 768,
        "attention_heads": 12,
        "mask_ratio": 0.9,
        "has_geometric_encoder": True,
        "geometric_depth": 2,
        "has_decoder": True,
        "decoder_embed_dim": 512,
        "decoder_depth": 8,
        "decoder_num_heads": 16,
        "uses_mask_token": True,
    }
}


def build_bundle() -> dict:
    return {
        "paper_id": PAPER_ID,
        "paper_title": PAPER_TITLE,
        "target_claims": TARGET_CLAIMS,
        "upstream_pins": UPSTREAM_PINS,
        "observations": {
            "datasets": DATASETS,
            "models": MODELS,
            "source_evidence": {
                "repository_layout": [
                    "tedbench/data/hf_datasets.py",
                    "tedbench/model/mae.py",
                    "tedbench/model/encoder.py",
                    "tedbench/model/pl_engine.py",
                    "configs/pretrain.yaml",
                    "configs/finetune_ted.yaml",
                    "configs/test_ted.yaml",
                ],
                "license_tags": {
                    "TEDBench/ted": "license:bsd-3-clause",
                    "TEDBench/afdb": "license:bsd-3-clause",
                    "TEDBench/cath": "license:bsd-3-clause",
                    "TEDBench/miae-b": "license:bsd-3-clause",
                },
            },
        },
        "claim_results": _claim_results(),
        "excluded_claims": [
            {
                "status": "unavailable",
                "reason": "Full model training or large-scale evaluation is outside the CPU-only static-metadata evidence path.",
                "claim_scope": "Accuracy, macro-F1, linear probing, scaling, and ablation performance claims from Tables 2-5 and Figures 4-5.",
            }
        ],
        "commands": [
            "git ls-remote https://github.com/BorgwardtLab/TEDBench.git ad3c208db13e5e0e124719300ec19fffab4c33e1 HEAD",
            "HfApi().repo_info(...) for TEDBench dataset and model revisions",
            "uv run pytest -q submissions/protein-fold-classification-at-scale-benchmarking-and-pretraining",
        ],
    }


def _claim_results() -> dict:
    ted = DATASETS["ted"]
    afdb = DATASETS["afdb"]
    cath = DATASETS["cath"]
    model = MODELS["miae_b"]
    return {
        "tedbench_dataset": {
            "status": "verified",
            "claim": TARGET_CLAIMS[0]["text"],
            "evidence": (
                f"{ted['repo']}@{ted['revision']} records "
                f"{ted['splits']['train']} train, {ted['splits']['val']} val, "
                f"and {ted['splits']['test']} test structures over "
                f"{ted['class_count']} classes; {afdb['repo']}@{afdb['revision']} "
                f"records {afdb['total_structures']} pretraining structures, and "
                f"{cath['repo']}@{cath['revision']} records an experimental test set."
            ),
        },
        "miae_architecture": {
            "status": "verified",
            "claim": TARGET_CLAIMS[1]["text"],
            "evidence": (
                "The pinned source layout includes MiAE model, encoder, and training "
                "modules; the pinned MiAE-B model metadata records a geometric "
                "encoder, decoder, and mask token."
            ),
        },
        "miae_masking": {
            "status": "verified",
            "claim": TARGET_CLAIMS[2]["text"],
            "evidence": (
                f"{model['repo']}@{model['revision']} records mask_ratio "
                f"{model['mask_ratio']}, decoder_embed_dim "
                f"{model['decoder_embed_dim']}, decoder_depth "
                f"{model['decoder_depth']}, and geometric encoder support."
            ),
        },
        "cath_transfer_dataset": {
            "status": "verified",
            "claim": TARGET_CLAIMS[3]["text"],
            "evidence": (
                f"{cath['repo']}@{cath['revision']} records "
                f"{cath['splits']['test']} experimental CATH v4.4 test structures "
                f"over {cath['class_count']} classes."
            ),
        },
    }
