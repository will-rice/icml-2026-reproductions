import importlib.util
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path("/tmp/menvagent-upstream-codex03")
ARXIV_SOURCE = Path("/tmp/menvagent-2601.22859-src.tar")
ARXIV_PDF = Path("/tmp/menvagent-2601.22859.pdf")


def load_generate_evidence():
    spec = importlib.util.spec_from_file_location(
        "menvagent_generate_evidence",
        PROJECT / "generate_evidence.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_bundle(tmp_path):
    module = load_generate_evidence()
    output = tmp_path / "bundle.json"
    bundle = module.build_bundle(
        source_root=SOURCE_ROOT,
        arxiv_source=ARXIV_SOURCE,
        arxiv_pdf=ARXIV_PDF,
        output=output,
    )
    assert json.loads(output.read_text()) == bundle
    return bundle


def test_bundle_records_primary_artifact_provenance(tmp_path):
    bundle = build_bundle(tmp_path)

    assert bundle["paper_id"] == "Mkal0hTCnh"
    assert bundle["challenge_snapshot_id"] == "bdb659717fc36b718f037c86e25b07c732c5efdb270b366a2fb8ac135476a0e6"
    assert bundle["provenance"]["code"]["revision"] == "d9e63881f7c4a4670bb536c89add24573459bbee"
    assert bundle["provenance"]["arxiv_source_sha256"] == "1cd2573993bd41a200a225da3f17af2b927dc5c307844166e5b5ec4a37ddd8d0"
    assert bundle["provenance"]["arxiv_pdf_sha256"] == "ea8483bc1ab4e47fb3b0a824cf61f364e07287d5b98f0da6aafc90132cf4341f"
    assert bundle["provenance"]["datasets"]["MEnvBench"]["revision"] == "4e312f11663e2ccdbd11f5cc3421de117ef4e118"
    assert bundle["provenance"]["datasets"]["MEnvData-SWE"]["lfs_oid"] == "e111fa1a8c4565f427d928652fddde7ce36a9d32973bb554beb60fba2c6055aa"
    assert bundle["provenance"]["license"] == "Apache-2.0"


def test_dataset_counts_are_recomputed_from_released_rows(tmp_path):
    bundle = build_bundle(tmp_path)

    menvbench = bundle["observations"]["menvbench"]
    assert menvbench["num_rows"] == 1000
    assert menvbench["unique_repositories"] == 200
    assert menvbench["unique_languages"] == 10
    assert menvbench["language_frequencies"]["Python"] == 100
    assert {"repo", "patch", "test_patch", "language"}.issubset(menvbench["schema_fields"])

    menvdata = bundle["observations"]["menvdata_swe"]
    assert menvdata["num_rows"] == 3005
    assert menvdata["unique_repositories"] == 942
    assert menvdata["unique_languages"] == 10
    assert {"env_setup_script", "original_env_setup_script", "eval_script", "image_name"}.issubset(
        menvdata["schema_fields"]
    )

    trajectories = bundle["observations"]["menvdata_swe_trajectory"]
    assert trajectories["viewer_num_rows"] == 3918
    assert trajectories["claimed_trajectory_count"] == 3872
    assert trajectories["matches_claimed_trajectory_count"] is False


def test_source_release_limitations_are_explicit(tmp_path):
    bundle = build_bundle(tmp_path)

    source = bundle["observations"]["source_release"]
    assert source["p_e_v_terms_present"] is True
    assert source["environment_reuse_terms_present"] is True
    assert source["env_patch_agent_present"] is True
    assert source["curation_scripts_present"] is True
    assert source["core_code_public_release_note"] == "The core code is currently being organized for public release."


def test_claim_statuses_do_not_promote_unreproduced_experiments(tmp_path):
    bundle = build_bundle(tmp_path)
    statuses = {claim["claim_index"]: claim["status"] for claim in bundle["claims"]}

    assert statuses[1] == "toy"
    assert statuses[2] == "verified"
    assert statuses[3] == "unavailable"
    assert statuses[4] == "unavailable"
    assert statuses[5] == "unavailable"
    assert statuses[6] == "falsified"
    assert all("paper_table" not in claim.get("evidence_basis", []) for claim in bundle["claims"])
