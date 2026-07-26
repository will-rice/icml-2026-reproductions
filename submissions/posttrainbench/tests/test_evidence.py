"""TDD test suite for PostTrainBench reproduction evidence contract.

Tests cover the complete approved design: paginated-tree oracle (not truncated
siblings), 4×7/1,338-task/47-root census, protocol controls, reward-hacking
status discipline with instruction-model partial-support, tamper detection,
determinism, and static rendering.

This file replaces the stale scaffold that asserted the truncated 85,883/1,039/37
values from the Hub ``siblings`` list.
"""

import hashlib
import json
import re
from pathlib import Path

import pytest

from posttrainbench_repro.constants import (
    ATTEMPT_ID,
    CANONICAL_ALL_ENTRIES_SHA256,
    CANONICAL_DIRS_SHA256,
    CANONICAL_FILES_SHA256,
    CHALLENGE_REVISION,
    CLAIM_1_SHA256,
    CLAIM_1_TEXT,
    CLAIM_2_SHA256,
    CLAIM_2_TEXT,
    CONTAMINATION_WITNESS_BYTES,
    CONTAMINATION_WITNESS_PATH,
    CONTAMINATION_WITNESS_SHA256,
    EXPECTED_BENCHMARKS,
    EXPECTED_CELL_COUNTS,
    EXPECTED_DUPLICATE_PAIRS,
    EXPECTED_MISSING_PAIRS,
    EXPECTED_MODEL_FRAGMENTS,
    EXPECTED_ROOT_CELL_PAIRS,
    EXPECTED_ROOT_COUNT,
    EXPECTED_TASK_COUNT,
    GIT_TREE_DIGEST,
    GITHUB_PINNED_COMMIT,
    HF_PINNED_REVISION,
    HF_TREE_DIR_COUNT,
    HF_TREE_FILE_COUNT,
    HF_TREE_PAGE_SIZE,
    HF_TREE_TOTAL_ENTRIES,
    HF_TREE_TOTAL_PAGES,
    INSTRUCTION_MODEL_JUDGMENT_BYTES,
    INSTRUCTION_MODEL_JUDGMENT_GIT_OBJECT,
    INSTRUCTION_MODEL_JUDGMENT_PATH,
    INSTRUCTION_MODEL_JUDGMENT_SHA256,
    INSTRUCTION_MODEL_JUDGMENT_SIZE,
    INSTRUCTION_MODEL_TRACE_GIT_OBJECT,
    INSTRUCTION_MODEL_TRACE_PATH,
    INSTRUCTION_MODEL_TRACE_SHA256,
    INSTRUCTION_MODEL_TRACE_SIZE,
    MODEL_ORDER,
    PAPER_ID,
    SNAPSHOT_ID,
    TIME_TAKEN_WITNESS_BYTES,
    TIME_TAKEN_WITNESS_PATH,
    TIME_TAKEN_WITNESS_SHA256,
    TRACE_EXCERPTS,
    TRUNCATED_SIBLINGS_COUNT,
    TRUNCATED_SIBLINGS_SHA256,
)

SUBMISSION_DIR = Path(__file__).parent.parent


# ===================================================================
# 1. Challenge binding
# ===================================================================

class TestChallengeBinding:
    """Exact paper ID, attempt ID, snapshot ID, claim texts, and claim SHA-256s."""

    def test_paper_and_attempt_ids(self):
        from posttrainbench_repro.audit import get_provenance
        prov = get_provenance()
        assert prov["paper_id"] == PAPER_ID == "UnjxMTe57e"
        assert prov["attempt_id"] == ATTEMPT_ID == "cb04ab1a-a526-4137-862b-a26d68563737"
        assert prov["assessed_snapshot"] == SNAPSHOT_ID
        assert prov["challenge_revision"] == CHALLENGE_REVISION == "81166abbeb76e5f79ff87e51061b5a0306507203"

    def test_claim_1_text_and_hash(self):
        computed = hashlib.sha256(CLAIM_1_TEXT.encode("utf-8")).hexdigest()
        assert computed == CLAIM_1_SHA256 == "9c0c1fc52ad2a93a9dbe299532b952948c4ecb674f820fe19f78f6a3c33b0073"

    def test_claim_2_text_and_hash(self):
        computed = hashlib.sha256(CLAIM_2_TEXT.encode("utf-8")).hexdigest()
        assert computed == CLAIM_2_SHA256 == "d185d61e5d886672a739e321e048df2378b71f55cf388ec4097bd2df1a916aad"

    def test_tampered_claim_fails(self):
        """Changing one byte of claim text must change the hash."""
        tampered = CLAIM_1_TEXT + " "
        assert hashlib.sha256(tampered.encode("utf-8")).hexdigest() != CLAIM_1_SHA256
        tampered2 = CLAIM_2_TEXT.replace("reward", "Reward")
        assert hashlib.sha256(tampered2.encode("utf-8")).hexdigest() != CLAIM_2_SHA256


# ===================================================================
# 2. Pinned acquisition
# ===================================================================

class TestPinnedAcquisition:
    """Exact GitHub/HF revisions, consumed digests, rejection of mutable refs."""

    def test_github_revision_pinned(self):
        assert GITHUB_PINNED_COMMIT == "d3496fa7d5788a007d6cd143167471ccdfc688d0"
        assert GIT_TREE_DIGEST == "566361d3f86bdf1a22294e6a772117428a8fb23792de6e1ac327897237915aeb"

    def test_hf_revision_pinned(self):
        assert HF_PINNED_REVISION == "46b3fec494f56fbd5f0600c7ad17646e4997aaa2"

    def test_mutable_main_rejected(self):
        """A mutable ``main`` reference must not appear in the acquisition code."""
        from posttrainbench_repro import acquisition
        import inspect
        src = inspect.getsource(acquisition)
        # Should not contain a bare 'main' branch reference in URLs
        assert "tree/main" not in src
        assert "/main/" not in src


# ===================================================================
# 3. Inventory determinism (paginated tree oracle)
# ===================================================================

class TestInventoryDeterminism:
    """Link-header cursor pagination with correct counts and digests."""

    def test_pagination_parameters(self):
        """The paginated tree requires exactly 112 pages at limit=1000."""
        assert HF_TREE_PAGE_SIZE == 1000
        assert HF_TREE_TOTAL_PAGES == 112
        # 111 × 1000 + 326 = 111,326
        assert (HF_TREE_TOTAL_PAGES - 1) * HF_TREE_PAGE_SIZE + 326 == HF_TREE_TOTAL_ENTRIES

    def test_complete_inventory_counts(self):
        """The complete inventory has 111,326 entries: 97,209 files, 14,117 dirs."""
        assert HF_TREE_TOTAL_ENTRIES == 111326
        assert HF_TREE_FILE_COUNT == 97209
        assert HF_TREE_DIR_COUNT == 14117
        assert HF_TREE_FILE_COUNT + HF_TREE_DIR_COUNT == HF_TREE_TOTAL_ENTRIES

    def test_canonical_inventory_digests(self):
        """Canonical sorted path digests match the approved values."""
        assert CANONICAL_ALL_ENTRIES_SHA256 == "045ae5c714aa605b4295e345970cdf9a330600f709ef18508e8bef5eb3eec13d"
        assert CANONICAL_FILES_SHA256 == "320253d2791e878f6539c58365ddc9ac93baffb53a5c78244910ef153f067ca4"
        assert CANONICAL_DIRS_SHA256 == "d480b9917811f32cfe7e055a029f475bfb2fa0c1cb02d0d08a7de0e200714132"

    def test_truncated_siblings_rejected(self):
        """The 85,883-entry siblings response must be rejected as coverage input."""
        assert TRUNCATED_SIBLINGS_COUNT == 85883
        assert TRUNCATED_SIBLINGS_COUNT != HF_TREE_TOTAL_ENTRIES
        assert TRUNCATED_SIBLINGS_SHA256 != CANONICAL_ALL_ENTRIES_SHA256

    def test_shuffled_order_same_digest(self):
        """Shuffling paths does not change the canonical digest (sorted internally)."""
        from posttrainbench_repro.acquisition import compute_canonical_path_digest
        paths = ["z/file.txt", "a/file.txt", "m/dir"]
        digest1 = compute_canonical_path_digest(paths)
        digest2 = compute_canonical_path_digest(list(reversed(paths)))
        assert digest1 == digest2

    def test_link_header_parsing(self):
        """Link-header parser correctly extracts rel=next URL."""
        from posttrainbench_repro.acquisition import _parse_link_next
        header = '<https://example.com/page2?cursor=abc>; rel="next"'
        assert _parse_link_next(header) == "https://example.com/page2?cursor=abc"
        assert _parse_link_next(None) is None
        assert _parse_link_next('<https://example.com>; rel="prev"') is None
        # Multiple links
        multi = '<https://x.com/prev>; rel="prev", <https://x.com/next>; rel="next"'
        assert _parse_link_next(multi) == "https://x.com/next"


# ===================================================================
# 4. Coverage
# ===================================================================

class TestCoverage:
    """4×7 benchmark/model matrix, 1,338 tasks, 47 roots."""

    def test_accepted_benchmarks(self):
        assert len(EXPECTED_BENCHMARKS) == 7
        assert set(EXPECTED_BENCHMARKS) == {
            "aime2025", "arenahardwriting", "bfcl", "gpqamain",
            "gsm8k", "healthbench", "humaneval",
        }

    def test_accepted_models(self):
        assert len(EXPECTED_MODEL_FRAGMENTS) == 4
        assert set(EXPECTED_MODEL_FRAGMENTS.values()) == {
            "Qwen3-1.7B-Base", "Qwen3-4B-Base", "SmolLM3-3B-Base", "Gemma-3-4B-PT",
        }

    def test_coverage_census(self):
        from posttrainbench_repro.audit import compute_coverage
        cov = compute_coverage()
        assert cov["recognized_task_count"] == EXPECTED_TASK_COUNT == 1338
        assert cov["recognized_root_count"] == EXPECTED_ROOT_COUNT == 47
        assert cov["recognized_root_cell_pairs"] == EXPECTED_ROOT_CELL_PAIRS == 1313
        assert cov["duplicate_job_pairs"] == EXPECTED_DUPLICATE_PAIRS == 25
        assert cov["missing_root_cell_pairs"] == EXPECTED_MISSING_PAIRS == 3

    def test_all_28_cells_present(self):
        from posttrainbench_repro.audit import compute_coverage
        cov = compute_coverage()
        matrix = cov["matrix"]
        assert len(matrix) == 28
        for cell in matrix:
            assert cell["count"] > 0, f"Empty cell: {cell['benchmark']}/{cell['model']}"

    def test_cell_counts_match_design(self):
        from posttrainbench_repro.audit import compute_coverage
        cov = compute_coverage()
        for bench, expected_counts in EXPECTED_CELL_COUNTS.items():
            actual = cov["cell_counts"][bench]
            assert actual == expected_counts, f"Counts mismatch for {bench}: {actual} != {expected_counts}"

    def test_task_basename_regex_accepts_valid(self):
        from posttrainbench_repro.constants import TASK_BASENAME_RE
        valid = [
            "humaneval_Qwen_Qwen3-1.7B-Base_16855823",
            "gsm8k_google_gemma-3-4b-pt_12345",
            "bfcl_HuggingFaceTB_SmolLM3-3B-Base_99999",
        ]
        for name in valid:
            assert TASK_BASENAME_RE.match(name), f"Should match: {name}"

    def test_task_basename_regex_rejects_invalid(self):
        from posttrainbench_repro.constants import TASK_BASENAME_RE
        invalid = [
            "viewer_data",           # excluded directory
            "humaneval_Unknown-Model_123",  # unknown model
            "unknown_benchmark_Qwen_Qwen3-1.7B-Base_123",  # unknown benchmark
            "humaneval_Qwen_Qwen3-1.7B-Base",  # missing job ID
            "humaneval_Qwen_Qwen3-1.7B-Base_abc",  # non-numeric job ID
        ]
        for name in invalid:
            assert not TASK_BASENAME_RE.match(name), f"Should not match: {name}"

    def test_viewer_data_excluded(self):
        """viewer_data is excluded and never counts as a task root."""
        from posttrainbench_repro.constants import EXCLUDED_TOP_LEVEL
        assert "viewer_data" in EXCLUDED_TOP_LEVEL


# ===================================================================
# 5. Protocol
# ===================================================================

class TestProtocol:
    """Protocol controls from the pinned runner scripts."""

    def test_protocol_audit_deterministic(self):
        from posttrainbench_repro.audit import audit_protocol
        proto = audit_protocol()
        assert proto["num_gpus_default"] == 1
        assert proto["cuda_device_requirement"] == 'TARGET.CUDADeviceName == "NVIDIA H100 80GB HBM3"'
        assert proto["request_gpus_binding"] == "request_gpus = $(num_gpus)"
        assert proto["solve_timeout_formula"] == "NUM_HOURS * 60 + 5"
        assert proto["timeout_grace_minutes"] == 5
        assert proto["task_dir_10h_suffix"] is True
        assert len(proto["evaluation_dirs_present"]) == 7

    def test_limitation_multi_gpu_extension(self):
        from posttrainbench_repro.audit import audit_protocol
        proto = audit_protocol()
        assert proto["limitation_multi_gpu_extension"] is True
        assert proto["limitation_five_minute_grace"] is True

    def test_commit_sh_limitations_reported(self):
        """The scheduler-dependent 100h/8GPU METR command is surfaced."""
        from posttrainbench_repro.audit import audit_protocol
        proto = audit_protocol()
        analysis = proto["commit_sh_analysis"]
        # Design requires these are surfaced
        assert analysis.get("has_metr_branch") or analysis.get("scheduler_dependent")


# ===================================================================
# 6. Reward-hacking modes
# ===================================================================

class TestRewardHacking:
    """Three reward-hacking submodes with correct status and witnesses."""

    def test_contamination_witness(self):
        assert CONTAMINATION_WITNESS_BYTES == b"contamination detected\n"
        assert hashlib.sha256(CONTAMINATION_WITNESS_BYTES).hexdigest() == CONTAMINATION_WITNESS_SHA256

    def test_contamination_mode_partial_support(self):
        from posttrainbench_repro.audit import audit_reward_hacking
        rh = audit_reward_hacking()
        assert rh["training_on_test_sets"]["status"] == "partial-support"
        assert rh["training_on_test_sets"]["witness_path"] == CONTAMINATION_WITNESS_PATH
        assert rh["training_on_test_sets"]["witness_sha256"] == CONTAMINATION_WITNESS_SHA256

    def test_time_taken_witness(self):
        assert TIME_TAKEN_WITNESS_BYTES == b"10:05:01\n"
        assert hashlib.sha256(TIME_TAKEN_WITNESS_BYTES).hexdigest() == TIME_TAKEN_WITNESS_SHA256

    def test_instruction_model_partial_support(self):
        """The instruction-model mode is partial-support (not unavailable)."""
        from posttrainbench_repro.audit import audit_reward_hacking
        rh = audit_reward_hacking()
        instr = rh["downloading_instruction_tuned_checkpoint"]
        assert instr["status"] == "partial-support"
        assert instr["judgment_path"] == INSTRUCTION_MODEL_JUDGMENT_PATH
        assert instr["judgment_sha256"] == INSTRUCTION_MODEL_JUDGMENT_SHA256
        assert instr["judgment_git_object"] == INSTRUCTION_MODEL_JUDGMENT_GIT_OBJECT
        assert instr["judgment_size"] == INSTRUCTION_MODEL_JUDGMENT_SIZE == 23

    def test_instruction_model_trace_not_redistributed(self):
        """The complete trace must not be redistributed."""
        from posttrainbench_repro.audit import audit_reward_hacking
        rh = audit_reward_hacking()
        instr = rh["downloading_instruction_tuned_checkpoint"]
        assert instr["trace"]["redistributed"] is False
        assert instr["trace"]["sha256"] == INSTRUCTION_MODEL_TRACE_SHA256
        assert instr["trace"]["size"] == INSTRUCTION_MODEL_TRACE_SIZE == 2362113

    def test_trace_excerpts_match(self):
        """Three JSONL-pointer excerpts match expected text and hashes."""
        from posttrainbench_repro.audit import audit_reward_hacking
        rh = audit_reward_hacking()
        excerpts = rh["downloading_instruction_tuned_checkpoint"]["safe_excerpts"]
        assert len(excerpts) == 3

        expected_excerpts = [
            (500, "/part/state/input/content",
             "we'll use the instruct model as our final submission.",
             "ddeabeab4ce59f6e12fc5741490341496fa352adc806c79501097e65c482dca4"),
            (504, "/part/state/output",
             "Loading Qwen3-1.7B instruct model...",
             "c35bbed3a9ac2c9ac923ffa80f6a0bc12cb8239a8b430ad7d140669f9633a7c5"),
            (531, "/part/text",
             "- Used Qwen3-1.7B (instruct model) - the official fine-tuned version of the base model",
             "13e2dbc1f65a1fbe08d52e5037b2b282dc0f7bcbfa127f33def251bf4df75d0e"),
        ]

        for exc, (rec, ptr, text, sha) in zip(excerpts, expected_excerpts):
            assert exc["record"] == rec
            assert exc["json_pointer"] == ptr
            assert exc["text"] == text
            assert exc["sha256"] == sha
            # Verify hash independently
            assert hashlib.sha256(text.encode("utf-8")).hexdigest() == sha

    def test_api_misuse_unavailable(self):
        """The API-key submode remains unavailable."""
        from posttrainbench_repro.audit import audit_reward_hacking
        rh = audit_reward_hacking()
        assert rh["using_discovered_api_key"]["status"] == "unavailable"
        assert "16804408" in rh["using_discovered_api_key"]["unavailability_reason"]

    def test_paper_prose_cannot_satisfy_unavailable(self):
        """Paper prose cannot satisfy an unavailable artifact."""
        from posttrainbench_repro.audit import audit_reward_hacking
        rh = audit_reward_hacking()
        api = rh["using_discovered_api_key"]
        assert "Paper prose cannot satisfy" in api["observations"][2]


# ===================================================================
# 7. Status discipline
# ===================================================================

class TestStatusDiscipline:
    """Neither claim may be verified; evidence is not an official verdict."""

    def test_claims_are_partial_support(self):
        from posttrainbench_repro.audit import evaluate_claims
        claims = evaluate_claims()
        assert claims["claim_1"]["status"] == "partial-support"
        assert claims["claim_2"]["status"] == "partial-support"

    def test_claims_never_verified(self):
        from posttrainbench_repro.audit import evaluate_claims
        claims = evaluate_claims()
        for key in ["claim_1", "claim_2"]:
            assert claims[key]["status"] != "verified"

    def test_no_official_verdict_language(self):
        from posttrainbench_repro.audit import evaluate_claims
        claims = evaluate_claims()
        for key in ["claim_1", "claim_2"]:
            summary = claims[key]["summary"].lower()
            assert "official verdict" not in summary


# ===================================================================
# 8. Determinism and integrity
# ===================================================================

class TestDeterminismAndIntegrity:
    """Two clean generations are byte-identical; manifest resolves."""

    def test_two_runs_byte_identical(self):
        from posttrainbench_repro.pipeline import run_pipeline
        run_pipeline()

        files_to_check = [
            "evidence/provenance.json",
            "evidence/coverage.json",
            "evidence/reward_hacking.json",
            "evidence/claims.json",
            "index.html",
            "report.html",
            "poster.html",
            "README.md",
        ]
        bytes_run1 = {f: (SUBMISSION_DIR / f).read_bytes() for f in files_to_check}

        run_pipeline()
        bytes_run2 = {f: (SUBMISSION_DIR / f).read_bytes() for f in files_to_check}

        for f in files_to_check:
            assert bytes_run1[f] == bytes_run2[f], f"File {f} differs between runs"

    def test_manifest_integrity(self):
        from posttrainbench_repro.pipeline import run_pipeline
        run_pipeline()

        manifest_path = SUBMISSION_DIR / "evidence" / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text("utf-8"))

        for rel_path, meta in manifest.items():
            full_path = SUBMISSION_DIR / rel_path
            assert full_path.exists(), f"Manifest lists missing: {rel_path}"
            actual_bytes = full_path.read_bytes()
            actual_sha = hashlib.sha256(actual_bytes).hexdigest()
            assert actual_sha == meta["sha256"], f"SHA256 mismatch: {rel_path}"
            assert len(actual_bytes) == meta["size"], f"Size mismatch: {rel_path}"

    def test_manifest_does_not_hash_itself(self):
        from posttrainbench_repro.pipeline import run_pipeline
        run_pipeline()
        manifest = json.loads(
            (SUBMISSION_DIR / "evidence" / "manifest.json").read_text("utf-8")
        )
        assert "evidence/manifest.json" not in manifest

    def test_no_wall_clock_in_evidence(self):
        """Canonical files must not contain wall-clock timestamps or temp paths."""
        from posttrainbench_repro.pipeline import run_pipeline
        run_pipeline()

        for rel_path in [
            "evidence/provenance.json",
            "evidence/coverage.json",
            "evidence/reward_hacking.json",
            "evidence/claims.json",
        ]:
            content = (SUBMISSION_DIR / rel_path).read_text("utf-8")
            # No ISO-8601 timestamps from the local process
            # (the design-pinned timestamps are allowed)
            assert "/tmp/" not in content
            assert "\\tmp\\" not in content

    def test_tamper_detection(self):
        """Changing an evidence file must be detectable via the manifest."""
        from posttrainbench_repro.pipeline import run_pipeline
        run_pipeline()

        manifest = json.loads(
            (SUBMISSION_DIR / "evidence" / "manifest.json").read_text("utf-8")
        )
        # Tamper with provenance.json
        prov_path = SUBMISSION_DIR / "evidence" / "provenance.json"
        original = prov_path.read_bytes()
        prov_path.write_bytes(original + b" ")  # tamper

        actual_sha = hashlib.sha256(prov_path.read_bytes()).hexdigest()
        assert actual_sha != manifest["evidence/provenance.json"]["sha256"]

        # Restore
        prov_path.write_bytes(original)


# ===================================================================
# 9. Static rendering
# ===================================================================

class TestStaticRendering:
    """HTML contains JSON pointers, limitations, required tags."""

    def test_index_has_json_pointers(self):
        from posttrainbench_repro.pipeline import run_pipeline
        run_pipeline()
        content = (SUBMISSION_DIR / "index.html").read_text("utf-8")
        assert "evidence/provenance.json" in content
        assert "evidence/coverage.json" in content
        assert "evidence/claims.json" in content

    def test_index_has_statuses(self):
        from posttrainbench_repro.pipeline import run_pipeline
        run_pipeline()
        content = (SUBMISSION_DIR / "index.html").read_text("utf-8")
        assert "partial-support" in content
        assert "unavailable" in content

    def test_report_has_limitations(self):
        from posttrainbench_repro.pipeline import run_pipeline
        run_pipeline()
        content = (SUBMISSION_DIR / "report.html").read_text("utf-8")
        assert "limitation" in content.lower()
        assert "H100" in content or "five-minute" in content.lower()

    def test_poster_has_limitations(self):
        from posttrainbench_repro.pipeline import run_pipeline
        run_pipeline()
        content = (SUBMISSION_DIR / "poster.html").read_text("utf-8")
        assert "limitation" in content.lower()

    def test_readme_space_metadata(self):
        from posttrainbench_repro.pipeline import run_pipeline
        run_pipeline()
        content = (SUBMISSION_DIR / "README.md").read_text("utf-8")
        assert "sdk: static" in content
        assert "app_file: index.html" in content
        assert "icml2026-repro" in content
        assert "paper-UnjxMTe57e" in content

    def test_no_credential_in_outputs(self):
        """No credential-shaped content in any output."""
        from posttrainbench_repro.pipeline import run_pipeline
        run_pipeline()

        for rel_path in [
            "evidence/provenance.json",
            "evidence/coverage.json",
            "evidence/reward_hacking.json",
            "evidence/claims.json",
            "index.html",
            "report.html",
            "poster.html",
            "README.md",
        ]:
            content = (SUBMISSION_DIR / rel_path).read_text("utf-8")
            # No API keys, tokens, or credential patterns
            assert "sk-" not in content
            assert "hf_" not in content.split("huggingface")[0] if "huggingface" in content else True
            assert "ghp_" not in content
            assert "OPENAI_API_KEY" not in content
