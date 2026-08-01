import json

from mind_omni_repro.evidence import (
    CLAIMS,
    UPSTREAM_PINS,
    audit_artifacts,
    build_evidence_bundle,
    write_evidence,
)


def test_bundle_records_attempt_claims_and_immutable_source_pin():
    bundle = build_evidence_bundle(
        {
            "README.md": "Dataset repo: https://www.modelscope.cn/datasets/LLLLLYYYYYzzz/NSD\nCheckpoint repo: https://www.modelscope.cn/models/LLLLLYYYYYzzz/Mind_Omni_V1_ckpt/files\n[ ] Evaluation code release\n",
            "MindOmni_src/tri_modal_pipeline.py": "def __call__(prompt=None, image=None, brain_data=None): pass\nprepare_brain_ids\n",
            "MindOmni_src/tri_modal_transformer.py": "text_seq_len = 77\nimage_seq_len = 256\nbrain_seq_len = 64\ntext_decoder = True\n",
            "train_fMRI_tokenizer_perceptual/fMRI_tokenizer_perceptual.py": "class VQ_fMRI: pass\ndesired_token_num: int = 64\ncodebook_size: int = 1024\n",
            "data_processing/stage2_dataset_prep/tokenize_QAs.py": "raw_question = item.get('Question')\nanswer = item.get('Answer')\nCLIPTokenizer.from_pretrained\n",
            "train_stage2_short_VQA/train_stage2_shortVQA.py": "short_VQA\nLoRA\n",
        },
        raw_result_artifacts={},
    )

    assert bundle["attempt_id"] == "cacb1796-7495-4440-845c-1002729bea1b"
    assert bundle["paper_id"] == "3gCdh3u2GK"
    assert bundle["snapshot_id"] == "8e1331c16c97f17ffa8b34fd6701cd91c805ab8b575818656718799d504db20e"
    assert bundle["upstream_pins"] == UPSTREAM_PINS
    assert "main" not in bundle["upstream_pins"]["official_code"]
    assert [result["claim_sha256"] for result in bundle["claim_results"]] == [
        claim["challenge_claim_sha256"] for claim in CLAIMS
    ]
    json.dumps(bundle)


def test_artifact_audit_detects_multimodal_tokenizer_and_vqa_components():
    audit = audit_artifacts(
        {
            "README.md": "Training code\nInference code\nDataset repo\nCheckpoint repo\nEvaluation code release",
            "MindOmni_src/tri_modal_pipeline.py": "prompt image brain_data prepare_brain_ids",
            "MindOmni_src/tri_modal_transformer.py": "text_seq_len image_seq_len brain_seq_len text_decoder",
            "train_fMRI_tokenizer_perceptual/fMRI_tokenizer_perceptual.py": "VQ_fMRI desired_token_num codebook_size semantic perceptual",
            "data_processing/stage2_dataset_prep/tokenize_QAs.py": "Question Answer full_prompts CLIPTokenizer",
            "train_stage2_short_VQA/train_stage2_shortVQA.py": "short_VQA easy_reasoning LoRA",
        }
    )

    assert audit["tri_modal_pipeline"]["status"] == "present"
    assert audit["tri_modal_transformer"]["status"] == "present"
    assert audit["brain_tokenizer"]["status"] == "present"
    assert audit["brain_qa_processing"]["status"] == "present"
    assert audit["dataset_release"]["status"] == "present"
    assert audit["checkpoint_release"]["status"] == "present"


def test_numeric_state_of_art_claims_are_unavailable_without_raw_results():
    bundle = build_evidence_bundle(
        {
            "README.md": "Dataset repo\nCheckpoint repo\n[ ] Evaluation code release\n",
            "MindOmni_src/tri_modal_pipeline.py": "prompt image brain_data",
            "MindOmni_src/tri_modal_transformer.py": "text_seq_len image_seq_len brain_seq_len",
            "train_fMRI_tokenizer_perceptual/fMRI_tokenizer_perceptual.py": "VQ_fMRI desired_token_num",
            "data_processing/stage2_dataset_prep/tokenize_QAs.py": "Question Answer",
            "train_stage2_short_VQA/train_stage2_shortVQA.py": "short_VQA",
        },
        raw_result_artifacts={},
    )

    state_of_art = bundle["claim_results"][3]
    synergy = bundle["claim_results"][4]
    assert state_of_art["status"] == "unavailable"
    assert synergy["status"] == "unavailable"
    assert "raw evaluation output" in state_of_art["limitation"]
    assert "raw evaluation output" in synergy["limitation"]


def test_write_evidence_outputs_pre_commit_clean_json(tmp_path):
    output_path = tmp_path / "evidence" / "bundle.json"

    write_evidence(output_path)

    assert output_path.read_bytes().endswith(b"\n")