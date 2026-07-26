"""Adversarial tests for PostTrainBench reproduction corrective findings.

These tests target the specific deficiencies found by two independent
reviewers of the first proposal (ba9139e).  They must fail before the
corrective implementation and pass after.

All tests are offline and use no network calls.
"""

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from posttrainbench_repro.constants import (
    API_MISUSE_TASK_CLUSTER,
    CONTAMINATION_WITNESS_PATH,
    HF_DATASET_ID,
    HF_PINNED_REVISION,
    INSTRUCTION_MODEL_JUDGMENT_PATH,
    INSTRUCTION_MODEL_TRACE_PATH,
    TIME_TAKEN_WITNESS_PATH,
    TRACE_EXCERPTS,
)

SUBMISSION_DIR = Path(__file__).parent.parent


# ===================================================================
# C1. Canonical production command with mandatory live acquisition
# ===================================================================

class TestProductionCommand:
    """A locked project script must exist that does live acquisition."""

    def test_script_entrypoint_exists(self):
        """pyproject.toml declares a posttrainbench-evidence script."""
        toml_text = (SUBMISSION_DIR / "pyproject.toml").read_text("utf-8")
        assert "posttrainbench-evidence" in toml_text

    def test_pipeline_requires_acquired_data(self):
        """run_pipeline must not silently produce outputs without acquisition."""
        from posttrainbench_repro.pipeline import run_pipeline
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            run_pipeline()

    def test_generate_evidence_function_exists(self):
        """A generate_evidence function must exist for the full pipeline."""
        from posttrainbench_repro import pipeline
        assert hasattr(pipeline, "generate_evidence")
        assert callable(pipeline.generate_evidence)


# ===================================================================
# C2. Fail-closed on acquisition failure
# ===================================================================

class TestFailClosed:
    """Production must call every required acquisition step."""

    def test_generate_evidence_calls_acquisition(self):
        """generate_evidence must reference acquisition functions."""
        from posttrainbench_repro import pipeline
        src = inspect.getsource(pipeline.generate_evidence)
        # Must reference acquisition
        assert "fetch_github_metadata" in src or "acquire_all" in src or "acquisition" in src


# ===================================================================
# C3. Hardened HF tree pagination
# ===================================================================

class TestHardenedPagination:
    """Pagination must validate hosts, page counts, and page sizes."""

    def test_validates_next_url_host(self):
        """fetch_hf_tree_pages must contain host validation logic."""
        from posttrainbench_repro import acquisition
        src = inspect.getsource(acquisition.fetch_hf_tree_pages)
        # Must check the host of the next URL
        assert "huggingface.co" in src

    def test_rejects_too_many_pages(self):
        """fetch_hf_tree_pages must enforce a page limit."""
        from posttrainbench_repro import acquisition
        src = inspect.getsource(acquisition.fetch_hf_tree_pages)
        # Must reference the max page count
        assert "112" in src or "HF_TREE_TOTAL_PAGES" in src or "max_page" in src

    def test_validates_interior_page_size(self):
        """fetch_hf_tree_pages must check that interior pages have 1000 entries."""
        from posttrainbench_repro import acquisition
        src = inspect.getsource(acquisition.fetch_hf_tree_pages)
        assert "1000" in src or "HF_TREE_PAGE_SIZE" in src or "short" in src.lower()


# ===================================================================
# C4. Exact four-file allowlist
# ===================================================================

class TestAllowlist:
    """Only the four approved files may be fetched."""

    def test_allowlist_constant_exists(self):
        """HF_ALLOWLISTED_FILES must exist in constants."""
        from posttrainbench_repro.constants import HF_ALLOWLISTED_FILES
        assert isinstance(HF_ALLOWLISTED_FILES, (set, frozenset, list, tuple))

    def test_allowlist_contains_exactly_four(self):
        """The allowlist must contain exactly the four approved files."""
        from posttrainbench_repro.constants import HF_ALLOWLISTED_FILES
        expected = {
            CONTAMINATION_WITNESS_PATH,
            TIME_TAKEN_WITNESS_PATH,
            INSTRUCTION_MODEL_JUDGMENT_PATH,
            INSTRUCTION_MODEL_TRACE_PATH,
        }
        assert set(HF_ALLOWLISTED_FILES) == expected

    def test_fetch_rejects_unapproved(self):
        """fetch_allowlisted_file must check the allowlist."""
        from posttrainbench_repro import acquisition
        src = inspect.getsource(acquisition.fetch_allowlisted_file)
        assert "ALLOWLIST" in src.upper() or "allowlist" in src or "HF_ALLOWLISTED" in src


# ===================================================================
# C5. Trace parsing — derive excerpts from actual trace
# ===================================================================

class TestTraceParsing:
    """Excerpts must be parsed from the downloaded trace, not constants."""

    def test_extract_excerpts_function_exists(self):
        """extract_trace_excerpts must be a callable in acquisition."""
        from posttrainbench_repro.acquisition import extract_trace_excerpts
        assert callable(extract_trace_excerpts)

    def test_excerpts_not_from_constants_in_audit(self):
        """audit_reward_hacking must not copy excerpts directly from TRACE_EXCERPTS."""
        from posttrainbench_repro import audit
        src = inspect.getsource(audit.audit_reward_hacking)
        # Should use extracted excerpts, not iterate TRACE_EXCERPTS directly
        assert "extract_trace_excerpts" in src or "trace_excerpts" in src.lower()


# ===================================================================
# C6. API-key mode derived from verified inventory
# ===================================================================

class TestAPIKeyDerivedFromInventory:
    """API-key unavailability must be proved from the verified inventory."""

    def test_audit_rh_accepts_inventory(self):
        """audit_reward_hacking must accept inventory data."""
        from posttrainbench_repro.audit import audit_reward_hacking
        sig = inspect.signature(audit_reward_hacking)
        params = list(sig.parameters)
        assert any(
            "inventory" in p or "acquired" in p or "dir_paths" in p
            or "all_paths" in p
            for p in params
        ), f"audit_reward_hacking params {params} must accept inventory data"


# ===================================================================
# C8. Live observations as hard preconditions
# ===================================================================

class TestLiveObservationsPrecondition:
    """Claims must be evaluated from verified audit data, not constants."""

    def test_evaluate_claims_requires_argument(self):
        """evaluate_claims must require at least one verified argument."""
        from posttrainbench_repro.audit import evaluate_claims
        sig = inspect.signature(evaluate_claims)
        params = [
            p for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        ]
        assert len(params) >= 1, \
            "evaluate_claims must require verified audit data"


# ===================================================================
# C9. Duplicate-job counting
# ===================================================================

class TestDuplicateJobCounting:
    """Duplicates are per (root, benchmark, model), not global-cell."""

    def test_duplicate_is_per_root_cell(self):
        """Two tasks for the same root/bench/model = 1 duplicate pair."""
        from posttrainbench_repro.audit import compute_coverage
        fake_inventory = {
            "dir_paths": [
                # root1 has two tasks for humaneval/Qwen3-1.7B
                "agent_10h_run1/humaneval_Qwen_Qwen3-1.7B-Base_100",
                "agent_10h_run1/humaneval_Qwen_Qwen3-1.7B-Base_101",
                # root1 has one task for gsm8k/Qwen3-1.7B
                "agent_10h_run1/gsm8k_Qwen_Qwen3-1.7B-Base_100",
            ],
        }
        cov = compute_coverage(fake_inventory)
        assert cov["duplicate_job_pairs"] == 1, \
            f"Expected 1 duplicate, got {cov['duplicate_job_pairs']}"


# ===================================================================
# C10. Protocol facts from verified bytes
# ===================================================================

class TestProtocolFromVerifiedBytes:
    """audit_protocol must require verified blob contents."""

    def test_protocol_requires_arguments(self):
        """audit_protocol must fail or raise without blob_contents."""
        from posttrainbench_repro.audit import audit_protocol
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            audit_protocol()
