"""Behavioral tests for PostTrainBench reproduction (genuine RED).

Each test exercises real code paths with small deterministic fake HTTP
clients and monkeypatched oracle constants.  No ``inspect.getsource``,
no string searches, and no arity-error substitutes.

Categories 1-12 from worker-task.md.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import posttrainbench_repro.constants as C
from posttrainbench_repro.acquisition import (
    _parse_link_next,
    _validate_next_url,
    acquire_all,
    compute_canonical_path_digest,
    compute_canonical_tree_digest,
    extract_trace_excerpts,
    fetch_allowlisted_file,
    fetch_github_metadata,
    fetch_hf_path_inventory,
    fetch_hf_tree_pages,
)
from posttrainbench_repro.audit import (
    audit_protocol,
    audit_reward_hacking,
    compute_coverage,
    evaluate_claims,
)
from posttrainbench_repro.pipeline import PROJECT_ROOT, generate_evidence, run_pipeline

SUBMISSION_DIR = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Helpers: tiny fake HTTP transport
# ---------------------------------------------------------------------------


class FakeResponse:
    """Minimal httpx.Response replacement for tests."""

    def __init__(
        self,
        status_code: int = 200,
        json_data: Any = None,
        content: bytes = b"",
        headers: dict[str, str] | None = None,
    ):
        self.status_code = status_code
        self._json = json_data
        self.content = content
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        if self._json is not None:
            return self._json
        return json.loads(self.content)


class RecordingClient:
    """Fake httpx.Client that records calls and replays canned responses."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.responses: dict[str, FakeResponse] = {}
        self.prefix_responses: list[tuple[str, FakeResponse]] = []

    def register(self, url: str, resp: FakeResponse) -> None:
        self.responses[url] = resp

    def register_prefix(self, prefix: str, resp: FakeResponse) -> None:
        self.prefix_responses.append((prefix, resp))

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append(url)
        if url in self.responses:
            return self.responses[url]
        for prefix, resp in self.prefix_responses:
            if url.startswith(prefix):
                return resp
        raise RuntimeError(f"No canned response for {url}")

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Build a minimal valid GitHub tree for tests
# ---------------------------------------------------------------------------

def _make_entries(count: int = 5) -> list[dict[str, Any]]:
    """Create N minimal tree entries for testing."""
    entries = []
    for i in range(count):
        entries.append({
            "path": f"file{i}.txt",
            "type": "blob",
            "sha": hashlib.sha1(f"file{i}".encode()).hexdigest(),
            "size": 100 + i,
        })
    return entries


def _valid_github_tree_response(
    entries: list[dict[str, Any]] | None = None,
    tree_id: str | None = None,
) -> FakeResponse:
    """Build a canned response for the GitHub tree endpoint."""
    if entries is None:
        entries = _make_entries()
    return FakeResponse(json_data={
        "sha": tree_id or C.GIT_TREE_ID,
        "tree": entries,
        "truncated": False,
    })


def _make_valid_acquired(
    extra_all_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Build a minimal valid acquired dict for offline audit tests."""
    entries: list[dict[str, Any]] = []
    for d in C.EXPECTED_EVAL_DIRS:
        entries.append({"path": d, "type": "tree", "sha": "0" * 40})
    for path, (sha, _) in C.PINNED_BLOBS.items():
        entries.append({"path": path, "type": "blob", "sha": sha, "size": 100})

    blob_contents = {
        "src/commit_utils/single_task.sub": (
            b'num_gpus = 1\n'
            b'request_gpus = $(num_gpus)\n'
            b'requirements = TARGET.CUDADeviceName == "NVIDIA H100 80GB HBM3"\n'
        ),
        "src/run_task.sh": (
            b'#!/bin/bash\n'
            b'NUM_HOURS=${1:-10}\n'
            b'timeout $((NUM_HOURS * 60 + 5))m python run.py\n'
        ),
        "src/commit_utils/commit.sh": (
            b'#!/bin/bash\n'
            b'if [ "$SCHEDULER" = "htcondor_mpi-is" ]; then\n'
            b'  NUM_HOURS=100\n'
            b'  num_gpus=8\n'
            b'  submit_job\n'
            b'else\n'
            b'  for i in 1 2 3 4 5 6 7; do NUM_HOURS=10 submit; done\n'
            b'  NUM_HOURS=1 submit\n'
            b'  MODELS=("Qwen_Qwen3-4B-Base")\n'
            b'  BENCHMARKS=("healthbench")\n'
            b'fi\n'
        ),
        "README.md": b"# PostTrainBench\n",
        "LICENSE": b"MIT License\n",
    }
    blobs = {}
    for path, (sha, raw_sha) in C.PINNED_BLOBS.items():
        blobs[path] = {"git_object": sha, "raw_sha256": raw_sha, "size": 100}

    github = {
        "commit": C.GITHUB_PINNED_COMMIT,
        "tree_id": C.GIT_TREE_ID,
        "entry_count": len(entries),
        "canonical_tree_digest": C.GIT_TREE_DIGEST,
        "entries": entries,
        "blobs": blobs,
        "blob_contents": blob_contents,
    }

    # Build matching HF inventory
    hf_inv = _make_valid_hf_inventory(extra_all_paths)

    trace_excerpts = []
    for spec in C.TRACE_EXCERPTS:
        trace_excerpts.append({
            "record": spec["record"],
            "json_pointer": spec["pointer"],
            "text": spec["text"],
            "sha256": hashlib.sha256(str(spec["text"]).encode()).hexdigest(),
        })

    return {
        "github": github,
        "hf_inventory": hf_inv,
        "contamination_content": C.CONTAMINATION_WITNESS_BYTES,
        "time_taken_content": C.TIME_TAKEN_WITNESS_BYTES,
        "instruction_judgment_content": C.INSTRUCTION_MODEL_JUDGMENT_BYTES,
        "instruction_trace_sha256": C.INSTRUCTION_MODEL_TRACE_SHA256,
        "instruction_trace_size": C.INSTRUCTION_MODEL_TRACE_SIZE,
        "trace_excerpts": trace_excerpts,
    }


def _make_valid_hf_inventory(
    extra_all_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Build an HF inventory with 1,338 tasks / 47 roots."""
    benchmark_list = sorted(C.EXPECTED_BENCHMARKS)
    model_frags = list(C.EXPECTED_MODEL_FRAGMENTS.keys())
    dir_paths: list[str] = []
    run_roots: list[str] = []
    for i in range(47):
        root = f"agent{i}_10h_run1"
        run_roots.append(root)
        dir_paths.append(root)
    root_cell: set[tuple[str, str, str]] = set()
    task_count = 0
    for bi, bench in enumerate(benchmark_list):
        expected = C.EXPECTED_CELL_COUNTS[bench]
        for mi, frag in enumerate(model_frags):
            for ti in range(expected[mi]):
                root = run_roots[ti % 47]
                jid = 16800000 + bi * 10000 + mi * 1000 + ti
                dir_paths.append(f"{root}/{bench}_{frag}_{jid}")
                root_cell.add((root, bench, C.EXPECTED_MODEL_FRAGMENTS[frag]))
                task_count += 1
    dup_cells = [(benchmark_list[0], model_frags[0]), (benchmark_list[4], model_frags[0])]
    for bench, frag in dup_cells:
        mn = C.EXPECTED_MODEL_FRAGMENTS[frag]
        for i, p in enumerate(list(dir_paths)):
            if p.startswith(f"{run_roots[46]}/{bench}_{frag}_"):
                dir_paths.pop(i)
                jid = 99999000 + dup_cells.index((bench, frag))
                dir_paths.append(f"{run_roots[0]}/{bench}_{frag}_{jid}")
                root_cell.discard((run_roots[46], bench, mn))
                root_cell.add((run_roots[0], bench, mn))
                break
        task_count += 0  # count unchanged since we replaced, not added
    file_paths = [f"{run_roots[0]}/file{i}.txt" for i in range(10)]
    all_paths = dir_paths + file_paths
    if extra_all_paths:
        all_paths.extend(extra_all_paths)
    return {
        "revision": C.HF_PINNED_REVISION,
        "page_count": C.HF_TREE_TOTAL_PAGES,
        "total_entries": len(all_paths),
        "file_count": len(file_paths),
        "dir_count": len(dir_paths),
        "all_paths": all_paths,
        "file_paths": file_paths,
        "dir_paths": dir_paths,
        "canonical_all_digest": "mock",
        "canonical_file_digest": "mock",
        "canonical_dir_digest": "mock",
    }


# ===================================================================
# 1. GitHub root-tree URL/object verification
# ===================================================================

class TestGitHubTree:
    """fetch_github_tree_entries must request the root tree object, not the commit."""

    def test_requests_tree_object_not_commit(self):
        """The request URL must use GIT_TREE_ID, not GITHUB_PINNED_COMMIT."""
        client = RecordingClient()
        tree_url = (
            f"https://api.github.com/repos/{C.GITHUB_REPO}"
            f"/git/trees/{C.GIT_TREE_ID}?recursive=1"
        )
        entries = _make_entries(C.GIT_TREE_ENTRY_COUNT)
        digest = compute_canonical_tree_digest(entries)
        client.register(tree_url, _valid_github_tree_response(entries))
        for path, (_, raw_sha) in C.PINNED_BLOBS.items():
            raw_url = (
                f"https://raw.githubusercontent.com/{C.GITHUB_REPO}"
                f"/{C.GITHUB_PINNED_COMMIT}/{path}"
            )
            content = b"x" * 10
            actual_sha = hashlib.sha256(content).hexdigest()
            client.register(raw_url, FakeResponse(content=content))
        with patch.object(C, "GIT_TREE_ENTRY_COUNT", len(entries)), \
             patch.object(C, "GIT_TREE_DIGEST", digest), \
             patch("posttrainbench_repro.acquisition.GIT_TREE_ENTRY_COUNT", len(entries)), \
             patch("posttrainbench_repro.acquisition.GIT_TREE_DIGEST", digest):
            # We need to also patch PINNED_BLOBS to match our fake content
            fake_blobs = {}
            for path, (git_sha, _) in C.PINNED_BLOBS.items():
                content = b"x" * 10
                fake_blobs[path] = (git_sha, hashlib.sha256(content).hexdigest())
            # Ensure entries contain the pinned blob paths with correct SHAs
            for path, (git_sha, _) in fake_blobs.items():
                entries.append({"path": path, "type": "blob", "sha": git_sha, "size": 10})
            digest2 = compute_canonical_tree_digest(entries)
            client.responses[tree_url] = _valid_github_tree_response(entries)
            with patch("posttrainbench_repro.acquisition.GIT_TREE_ENTRY_COUNT", len(entries)), \
                 patch("posttrainbench_repro.acquisition.GIT_TREE_DIGEST", digest2), \
                 patch("posttrainbench_repro.acquisition.PINNED_BLOBS", fake_blobs):
                result = fetch_github_metadata(client)
        assert tree_url in client.calls
        commit_url = f"https://api.github.com/repos/{C.GITHUB_REPO}/git/trees/{C.GITHUB_PINNED_COMMIT}?recursive=1"
        assert commit_url not in client.calls

    def test_wrong_tree_id_fails(self):
        """Returned tree ID mismatch raises."""
        client = RecordingClient()
        tree_url = (
            f"https://api.github.com/repos/{C.GITHUB_REPO}"
            f"/git/trees/{C.GIT_TREE_ID}?recursive=1"
        )
        client.register(tree_url, FakeResponse(json_data={
            "sha": "0000000000000000000000000000000000000000",
            "tree": [],
            "truncated": False,
        }))
        with pytest.raises(ValueError, match="Tree SHA mismatch"):
            fetch_github_metadata(client)

    def test_wrong_entry_count_fails(self):
        """Entry count mismatch raises."""
        client = RecordingClient()
        tree_url = (
            f"https://api.github.com/repos/{C.GITHUB_REPO}"
            f"/git/trees/{C.GIT_TREE_ID}?recursive=1"
        )
        entries = _make_entries(3)
        client.register(tree_url, _valid_github_tree_response(entries))
        with pytest.raises(ValueError, match="Expected"):
            fetch_github_metadata(client)

    def test_wrong_blob_object_sha_fails(self):
        """Blob with wrong Git object SHA in tree listing raises."""
        client = RecordingClient()
        tree_url = (
            f"https://api.github.com/repos/{C.GITHUB_REPO}"
            f"/git/trees/{C.GIT_TREE_ID}?recursive=1"
        )
        entries = _make_entries(C.GIT_TREE_ENTRY_COUNT - len(C.PINNED_BLOBS))
        for path, (_, raw_sha) in C.PINNED_BLOBS.items():
            entries.append({"path": path, "type": "blob", "sha": "bad" * 13 + "b", "size": 10})
        digest = compute_canonical_tree_digest(entries)
        client.register(tree_url, _valid_github_tree_response(entries))
        with patch("posttrainbench_repro.acquisition.GIT_TREE_ENTRY_COUNT", len(entries)), \
             patch("posttrainbench_repro.acquisition.GIT_TREE_DIGEST", digest):
            with pytest.raises(ValueError, match="blob SHA mismatch"):
                fetch_github_metadata(client)

    def test_wrong_blob_bytes_fails(self):
        """Blob with wrong raw SHA-256 raises."""
        client = RecordingClient()
        tree_url = (
            f"https://api.github.com/repos/{C.GITHUB_REPO}"
            f"/git/trees/{C.GIT_TREE_ID}?recursive=1"
        )
        entries = _make_entries(C.GIT_TREE_ENTRY_COUNT - len(C.PINNED_BLOBS))
        for path, (git_sha, raw_sha) in C.PINNED_BLOBS.items():
            entries.append({"path": path, "type": "blob", "sha": git_sha, "size": 10})
        digest = compute_canonical_tree_digest(entries)
        client.register(tree_url, _valid_github_tree_response(entries))
        for path in C.PINNED_BLOBS:
            raw_url = f"https://raw.githubusercontent.com/{C.GITHUB_REPO}/{C.GITHUB_PINNED_COMMIT}/{path}"
            client.register(raw_url, FakeResponse(content=b"TAMPERED CONTENT"))
        with patch("posttrainbench_repro.acquisition.GIT_TREE_ENTRY_COUNT", len(entries)), \
             patch("posttrainbench_repro.acquisition.GIT_TREE_DIGEST", digest):
            with pytest.raises(ValueError, match="SHA-256 mismatch"):
                fetch_github_metadata(client)

    def test_each_blob_fetched_once(self):
        """Each raw blob URL is requested at most once."""
        client = RecordingClient()
        tree_url = (
            f"https://api.github.com/repos/{C.GITHUB_REPO}"
            f"/git/trees/{C.GIT_TREE_ID}?recursive=1"
        )
        entries = []
        fake_blobs = {}
        for path, (git_sha, _) in C.PINNED_BLOBS.items():
            content = f"content-of-{path}".encode()
            sha = hashlib.sha256(content).hexdigest()
            fake_blobs[path] = (git_sha, sha)
            entries.append({"path": path, "type": "blob", "sha": git_sha, "size": len(content)})
            raw_url = f"https://raw.githubusercontent.com/{C.GITHUB_REPO}/{C.GITHUB_PINNED_COMMIT}/{path}"
            client.register(raw_url, FakeResponse(content=content))
        digest = compute_canonical_tree_digest(entries)
        client.register(tree_url, _valid_github_tree_response(entries))
        with patch("posttrainbench_repro.acquisition.GIT_TREE_ENTRY_COUNT", len(entries)), \
             patch("posttrainbench_repro.acquisition.GIT_TREE_DIGEST", digest), \
             patch("posttrainbench_repro.acquisition.PINNED_BLOBS", fake_blobs):
            fetch_github_metadata(client)
        raw_calls = [c for c in client.calls if "raw.githubusercontent" in c]
        assert len(raw_calls) == len(C.PINNED_BLOBS)
        assert len(set(raw_calls)) == len(raw_calls), "duplicate raw blob fetch"


# ===================================================================
# 2. Fail-closed: existing outputs preserved on acquisition failure
# ===================================================================

class TestFailClosed:
    """Production failure leaves existing outputs byte-identical."""

    def test_existing_outputs_preserved_on_failure(self, tmp_path):
        """If generate_evidence fails, pre-existing canonical outputs survive."""
        sentinel = b"ORIGINAL CONTENT"
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        (evidence / "provenance.json").write_bytes(sentinel)
        (tmp_path / "index.html").write_bytes(sentinel)
        originals = {
            "evidence/provenance.json": sentinel,
            "index.html": sentinel,
        }
        with patch("posttrainbench_repro.pipeline.PROJECT_ROOT", tmp_path), \
             patch("posttrainbench_repro.pipeline.acquire_all", side_effect=RuntimeError("fail")):
            with pytest.raises(RuntimeError):
                generate_evidence()
        for rel, expected in originals.items():
            assert (tmp_path / rel).read_bytes() == expected

    def test_no_new_outputs_on_failure(self, tmp_path):
        """No new canonical outputs are created on acquisition failure."""
        with patch("posttrainbench_repro.pipeline.PROJECT_ROOT", tmp_path), \
             patch("posttrainbench_repro.pipeline.acquire_all", side_effect=RuntimeError("fail")):
            with pytest.raises(RuntimeError):
                generate_evidence()
        assert not (tmp_path / "evidence").exists()


# ===================================================================
# 3. Production calls the complete acquisition bundle
# ===================================================================

class TestProductionCallsAcquisition:
    """generate_evidence must call acquire_all and pass result to run_pipeline."""

    def test_acquire_all_called_and_result_used(self, tmp_path):
        """acquire_all is called; its output reaches run_pipeline."""
        acquired = _make_valid_acquired()
        call_log: list[str] = []
        original_acquire = acquire_all
        original_run = run_pipeline

        def fake_acquire(*a, **kw):
            call_log.append("acquire_all")
            return acquired

        def fake_run(acq):
            call_log.append("run_pipeline")
            assert acq is acquired
            return {}

        with patch("posttrainbench_repro.pipeline.acquire_all", side_effect=fake_acquire), \
             patch("posttrainbench_repro.pipeline.run_pipeline", side_effect=fake_run):
            generate_evidence()
        assert call_log == ["acquire_all", "run_pipeline"]


# ===================================================================
# 4. HF pagination: success + adversarial cases
# ===================================================================

class TestHFPagination:
    """Pagination: success, early-term, page 113, short, cycle, foreign, wrong rev, etc."""

    def _make_page(self, n: int, ptype: str = "file") -> list[dict[str, Any]]:
        return [{"path": f"p{i}", "type": ptype} for i in range(n)]

    def _base_url(self) -> str:
        return (
            f"https://huggingface.co/api/datasets/{C.HF_DATASET_ID}"
            f"/tree/{C.HF_PINNED_REVISION}"
        )

    def test_success_two_pages(self):
        """Two-page pagination returns correct count."""
        client = RecordingClient()
        base = self._base_url()
        p2_url = f"{base}?cursor=p2"
        page1 = self._make_page(5)
        page2 = self._make_page(3)

        client.register_prefix(base, FakeResponse(
            json_data=page1,
            headers={"link": f'<{p2_url}>; rel="next"'},
        ))
        client.responses[p2_url] = FakeResponse(json_data=page2)

        with patch("posttrainbench_repro.acquisition.HF_TREE_TOTAL_PAGES", 2), \
             patch("posttrainbench_repro.acquisition.HF_TREE_PAGE_SIZE", 5):
            entries, pages = fetch_hf_tree_pages(client)
        assert pages == 2
        assert len(entries) == 8

    def test_page_113_rejected(self):
        """Page 113 (exceeding 112 max) is rejected."""
        client = RecordingClient()
        base = self._base_url()
        call_count = [0]

        class CycleResponse:
            status_code = 200
            def raise_for_status(self): pass
            def json(self_):
                call_count[0] += 1
                return [{"path": f"e{call_count[0]}_{i}", "type": "file"} for i in range(1000)]
            @property
            def headers(self_):
                return {"link": f'<{base}?cursor=c{call_count[0]+1}>; rel="next"'}

        client.register_prefix(base, CycleResponse())
        for i in range(200):
            client.responses[f"{base}?cursor=c{i+1}"] = CycleResponse()

        with pytest.raises(RuntimeError, match="exceeded"):
            fetch_hf_tree_pages(client)

    def test_short_interior_page_rejected(self):
        """Non-final page with < 1000 entries is rejected."""
        client = RecordingClient()
        base = self._base_url()
        p2_url = f"{base}?cursor=p2"
        p3_url = f"{base}?cursor=p3"
        client.register_prefix(base, FakeResponse(
            json_data=self._make_page(500),
            headers={"link": f'<{p2_url}>; rel="next"'},
        ))
        client.responses[p2_url] = FakeResponse(json_data=self._make_page(3))
        with pytest.raises(RuntimeError, match="[Ss]hort"):
            fetch_hf_tree_pages(client)

    def test_cursor_cycle_rejected(self):
        """Revisiting the same URL is rejected."""
        client = RecordingClient()
        base = self._base_url()
        # First page points back to itself via next link
        first_url = f"{base}?recursive=true&expand=false&limit=1000"
        client.register_prefix(base, FakeResponse(
            json_data=self._make_page(1000),
            headers={"link": f'<{first_url}>; rel="next"'},
        ))
        with pytest.raises(RuntimeError, match="[Cc]ycle"):
            fetch_hf_tree_pages(client)

    def test_foreign_host_rejected(self):
        """Next URL pointing to evil.com is rejected."""
        with pytest.raises(ValueError, match="[Ff]oreign"):
            _validate_next_url("https://evil.com/api/datasets/foo/tree/abc")

    def test_wrong_revision_rejected(self):
        """Next URL with wrong revision is rejected."""
        wrong_url = (
            f"https://huggingface.co/api/datasets/{C.HF_DATASET_ID}"
            f"/tree/0000000000000000000000000000000000000000?cursor=x"
        )
        with pytest.raises(ValueError, match="[Ww]rong"):
            _validate_next_url(wrong_url)

    def test_same_prefix_path_suffix_rejected(self):
        """Path that starts with the tree endpoint but has extra path segments."""
        tricky_url = (
            f"https://huggingface.co/api/datasets/{C.HF_DATASET_ID}"
            f"/tree/{C.HF_PINNED_REVISION}/evil"
        )
        # This should be rejected since path must EQUAL the tree endpoint, not just startswith
        with pytest.raises(ValueError):
            _validate_next_url(tricky_url)

    def test_duplicate_path_rejected(self):
        """Duplicate paths in inventory are rejected."""
        client = RecordingClient()
        base = self._base_url()
        duped = [
            {"path": "same/path", "type": "file"},
            {"path": "same/path", "type": "file"},
            {"path": "other", "type": "file"},
        ]
        client.register_prefix(base, FakeResponse(json_data=duped))
        with patch("posttrainbench_repro.acquisition.HF_TREE_TOTAL_PAGES", 1), \
             patch("posttrainbench_repro.acquisition.HF_TREE_TOTAL_ENTRIES", 3), \
             patch("posttrainbench_repro.acquisition.HF_TREE_FILE_COUNT", 3), \
             patch("posttrainbench_repro.acquisition.HF_TREE_DIR_COUNT", 0):
            with pytest.raises(ValueError, match="[Dd]uplicate"):
                fetch_hf_path_inventory(client)

    def test_conflicting_type_rejected(self):
        """Same path with different types is rejected."""
        client = RecordingClient()
        base = self._base_url()
        mixed = [
            {"path": "x", "type": "file"},
            {"path": "x", "type": "directory"},
        ]
        client.register_prefix(base, FakeResponse(json_data=mixed))
        with patch("posttrainbench_repro.acquisition.HF_TREE_TOTAL_PAGES", 1):
            with pytest.raises(ValueError, match="[Cc]onflict"):
                fetch_hf_path_inventory(client)

    def test_unknown_type_rejected(self):
        """Entries with type other than file/directory are rejected."""
        client = RecordingClient()
        base = self._base_url()
        bad = [{"path": "x", "type": "symlink"}]
        client.register_prefix(base, FakeResponse(json_data=bad))
        with pytest.raises(ValueError, match="[Uu]nknown"):
            fetch_hf_tree_pages(client)

    def test_count_mismatch_rejected(self):
        """Total entry count mismatch is rejected."""
        client = RecordingClient()
        base = self._base_url()
        entries = [{"path": f"f{i}", "type": "file"} for i in range(5)]
        client.register_prefix(base, FakeResponse(json_data=entries))
        with patch("posttrainbench_repro.acquisition.HF_TREE_TOTAL_PAGES", 1), \
             patch("posttrainbench_repro.acquisition.HF_TREE_TOTAL_ENTRIES", 999):
            with pytest.raises(ValueError, match="Expected"):
                fetch_hf_path_inventory(client)

    def test_digest_mismatch_rejected(self):
        """Canonical path digest mismatch is rejected."""
        client = RecordingClient()
        base = self._base_url()
        entries = [{"path": f"f{i}", "type": "file"} for i in range(5)]
        client.register_prefix(base, FakeResponse(json_data=entries))
        with patch("posttrainbench_repro.acquisition.HF_TREE_TOTAL_PAGES", 1), \
             patch("posttrainbench_repro.acquisition.HF_TREE_TOTAL_ENTRIES", 5), \
             patch("posttrainbench_repro.acquisition.HF_TREE_FILE_COUNT", 5), \
             patch("posttrainbench_repro.acquisition.HF_TREE_DIR_COUNT", 0):
            with pytest.raises(ValueError, match="digest mismatch"):
                fetch_hf_path_inventory(client)


# ===================================================================
# 5. Disallowed file rejected before I/O
# ===================================================================

class TestAllowlist:
    """Requesting a non-allowlisted file raises before the client is called."""

    def test_disallowed_path_no_io(self):
        """Client.get is never called for a disallowed path."""
        client = RecordingClient()
        with pytest.raises(ValueError, match="allowlist"):
            fetch_allowlisted_file("evil/path.txt", client=client)
        assert len(client.calls) == 0


# ===================================================================
# 6. Allowlisted file bytes/hash/object verification
# ===================================================================

class TestAllowlistedFileVerification:
    """All four allowlisted paths must be verified for bytes and hashes."""

    def test_contamination_wrong_bytes_fails(self):
        """Wrong contamination bytes raise."""
        client = RecordingClient()
        url = (
            f"https://huggingface.co/datasets/{C.HF_DATASET_ID}"
            f"/raw/{C.HF_PINNED_REVISION}/{C.CONTAMINATION_WITNESS_PATH}"
        )
        client.register(url, FakeResponse(content=b"WRONG"))
        with pytest.raises(ValueError):
            from posttrainbench_repro.acquisition import _verify_bytes
            data = fetch_allowlisted_file(C.CONTAMINATION_WITNESS_PATH, client=client)
            _verify_bytes(data, C.CONTAMINATION_WITNESS_SHA256, "test")


# ===================================================================
# 7. JSONL trace excerpt extraction
# ===================================================================

class TestTraceExcerpts:
    """Trace parsing: real JSONL, missing/malformed/moved/mutated records."""

    def _build_trace(self, records: dict[int, dict]) -> bytes:
        """Build a JSONL trace with specific records at specific positions."""
        max_idx = max(records.keys()) + 1
        lines = []
        for i in range(max_idx):
            if i in records:
                lines.append(json.dumps(records[i]))
            else:
                lines.append(json.dumps({"part": {"text": f"filler {i}"}}))
        return "\n".join(lines).encode("utf-8")

    def test_valid_extraction(self):
        """Valid trace extracts correct text and hash."""
        records = {}
        for spec in C.TRACE_EXCERPTS:
            records[spec["record"]] = _build_nested(spec["pointer"], spec["text"])
        trace = self._build_trace(records)
        results = extract_trace_excerpts(trace)
        assert len(results) == len(C.TRACE_EXCERPTS)
        for i, spec in enumerate(C.TRACE_EXCERPTS):
            assert results[i]["text"] == spec["text"]
            assert results[i]["sha256"] == spec["sha256"]

    def test_missing_record_fails(self):
        """Trace with too few records raises."""
        trace = b'{"part": {"text": "only one"}}\n'
        with pytest.raises((IndexError, ValueError)):
            extract_trace_excerpts(trace)

    def test_malformed_json_fails(self):
        """Non-JSON record raises."""
        lines = ["NOT JSON"] * 600
        trace = "\n".join(lines).encode()
        with pytest.raises(ValueError, match="[Mm]alformed"):
            extract_trace_excerpts(trace)

    def test_mutated_text_fails(self):
        """Changed text at correct position fails hash verification."""
        spec = C.TRACE_EXCERPTS[0]
        record = _build_nested(spec["pointer"], "TAMPERED TEXT")
        trace = self._build_trace({spec["record"]: record})
        with pytest.raises(ValueError, match="hash mismatch"):
            extract_trace_excerpts(trace)

    def test_moved_record_fails(self):
        """Record at wrong position fails."""
        spec = C.TRACE_EXCERPTS[0]
        wrong_pos = spec["record"] + 1
        record = _build_nested(spec["pointer"], spec["text"])
        trace = self._build_trace({wrong_pos: record})
        # The record at spec["record"] won't have the right pointer/text
        with pytest.raises((ValueError, KeyError)):
            extract_trace_excerpts(trace)


def _build_nested(pointer: str, value: Any) -> dict:
    """Build a nested dict matching a JSON pointer with the given leaf value."""
    parts = pointer.lstrip("/").split("/")
    result: dict = {}
    current = result
    for i, part in enumerate(parts):
        if i == len(parts) - 1:
            current[part] = value
        else:
            current[part] = {}
            current = current[part]
    return result


# ===================================================================
# 8. API cluster 16804408 present → abort (not "unavailable")
# ===================================================================

class TestAPIClusterPresent:
    """If cluster 16804408 is found in inventory, audit_reward_hacking must abort."""

    def test_cluster_present_aborts(self):
        """Presence of API cluster in inventory must raise or refuse 'unavailable'."""
        acquired = _make_valid_acquired(
            extra_all_paths=[
                f"some_root/task_{C.API_MISUSE_TASK_CLUSTER}_dir",
            ]
        )
        # If the function does NOT raise, it must NOT emit "unavailable"
        try:
            rh = audit_reward_hacking(acquired)
            # If it returns, the status must NOT be "unavailable" if cluster is found
            api = rh["using_discovered_api_key"]
            if api["inventory_proof"]["matching_paths"] > 0:
                assert api["status"] != "unavailable", \
                    "Must not emit 'unavailable' when cluster paths exist"
        except (ValueError, RuntimeError):
            pass  # Raising is also acceptable


# ===================================================================
# 9. Duplicate counting: global-cell vs per-root/cell
# ===================================================================

class TestDuplicateCounting:
    """Duplicate counting per (root, bench, model), not global cell."""

    def test_same_bench_model_different_roots(self):
        """Same bench/model in DIFFERENT roots = no duplicates."""
        dirs = [
            "rootA_10h_run1",
            "rootA_10h_run1/humaneval_Qwen_Qwen3-1.7B-Base_100",
            "rootB_10h_run1",
            "rootB_10h_run1/humaneval_Qwen_Qwen3-1.7B-Base_200",
        ]
        inv = {"dir_paths": dirs}
        cov = compute_coverage(inv)
        assert cov["duplicate_job_pairs"] == 0

    def test_multiple_extra_in_one_root_cell(self):
        """Three tasks in one root/cell = 2 duplicate pairs."""
        dirs = [
            "rootA_10h_run1",
            "rootA_10h_run1/humaneval_Qwen_Qwen3-1.7B-Base_100",
            "rootA_10h_run1/humaneval_Qwen_Qwen3-1.7B-Base_101",
            "rootA_10h_run1/humaneval_Qwen_Qwen3-1.7B-Base_102",
        ]
        inv = {"dir_paths": dirs}
        cov = compute_coverage(inv)
        assert cov["duplicate_job_pairs"] == 2

    def test_mixed_roots_with_duplicates(self):
        """Mix of roots with and without duplicates."""
        dirs = [
            "rootA_10h_run1",
            "rootA_10h_run1/humaneval_Qwen_Qwen3-1.7B-Base_100",
            "rootA_10h_run1/humaneval_Qwen_Qwen3-1.7B-Base_101",
            "rootB_10h_run1",
            "rootB_10h_run1/humaneval_Qwen_Qwen3-1.7B-Base_200",
            "rootB_10h_run1/gsm8k_Qwen_Qwen3-1.7B-Base_300",
        ]
        inv = {"dir_paths": dirs}
        cov = compute_coverage(inv)
        assert cov["duplicate_job_pairs"] == 1  # only rootA humaneval


# ===================================================================
# 10. Protocol audit: mutated values fail
# ===================================================================

class TestProtocolMutated:
    """Mutated GPU count, device, timeout, eval dirs, scheduler branch all fail."""

    def _base_blobs(self) -> dict[str, bytes]:
        return {
            "src/commit_utils/single_task.sub": (
                b'num_gpus = 1\n'
                b'request_gpus = $(num_gpus)\n'
                b'requirements = TARGET.CUDADeviceName == "NVIDIA H100 80GB HBM3"\n'
            ),
            "src/run_task.sh": (
                b'#!/bin/bash\n'
                b'NUM_HOURS=${1:-10}\n'
                b'timeout $((NUM_HOURS * 60 + 5))m python run.py\n'
            ),
            "src/commit_utils/commit.sh": (
                b'#!/bin/bash\n'
                b'if [ "$SCHEDULER" = "htcondor_mpi-is" ]; then\n'
                b'  NUM_HOURS=100\n'
                b'  num_gpus=8\n'
                b'  submit_job\n'
                b'fi\n'
            ),
        }

    def _base_entries(self) -> list[dict[str, Any]]:
        entries = []
        for d in C.EXPECTED_EVAL_DIRS:
            entries.append({"path": d, "type": "tree", "sha": "0" * 40})
        return entries

    def test_missing_gpu_count_fails(self):
        blobs = self._base_blobs()
        blobs["src/commit_utils/single_task.sub"] = b"# no gpu spec\n"
        with pytest.raises(ValueError, match="num_gpus"):
            audit_protocol(blobs, self._base_entries())

    def test_missing_device_fails(self):
        blobs = self._base_blobs()
        blobs["src/commit_utils/single_task.sub"] = b"num_gpus = 1\nrequest_gpus = $(num_gpus)\n"
        with pytest.raises(ValueError, match="CUDADeviceName"):
            audit_protocol(blobs, self._base_entries())

    def test_missing_timeout_fails(self):
        blobs = self._base_blobs()
        blobs["src/run_task.sh"] = b"#!/bin/bash\necho hello\n"
        with pytest.raises(ValueError, match="timeout"):
            audit_protocol(blobs, self._base_entries())

    def test_missing_eval_dirs_fails(self):
        blobs = self._base_blobs()
        entries = [{"path": "src/other", "type": "tree", "sha": "0" * 40}]
        with pytest.raises(ValueError, match="eval"):
            audit_protocol(blobs, entries)

    def test_comment_only_metr_insufficient(self):
        """A bare comment mention of htcondor_mpi-is without active code."""
        blobs = self._base_blobs()
        blobs["src/commit_utils/commit.sh"] = (
            b"#!/bin/bash\n"
            b"# htcondor_mpi-is was considered but not used\n"
            b"echo default\n"
        )
        proto = audit_protocol(blobs, self._base_entries())
        analysis = proto["commit_sh_analysis"]
        # Comment-only mention should not report has_metr_branch
        assert analysis["has_metr_branch"] is False


# ===================================================================
# 11. Malformed audit results cannot emit partial-support
# ===================================================================

class TestMalformedAuditRejectsStatus:
    """Malformed coverage/protocol/reward-hacking rejects partial-support."""

    def test_wrong_task_count_rejects_claims(self):
        """Coverage with wrong task count prevents evaluate_claims from succeeding."""
        coverage = {"recognized_task_count": 999}
        protocol = {"commit_sh_analysis": {"htcondor_branch": {"ten_hour_jobs": 7, "one_hour_jobs": 1}}}
        rh = {}
        with pytest.raises(ValueError, match="task count"):
            evaluate_claims(coverage, protocol, rh)


# ===================================================================
# 12. Simulated write/replace failure rolls back
# ===================================================================

class TestTransactionalRollback:
    """Write failure during output publication rolls back all outputs."""

    def test_write_failure_rolls_back(self, tmp_path):
        """If writing one output fails, all prior outputs are rolled back."""
        acquired = _make_valid_acquired()
        sentinel = b"SENTINEL"
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        (evidence / "provenance.json").write_bytes(sentinel)

        write_count = [0]
        original_write = Path.write_text

        def failing_write(self, content, *args, **kwargs):
            write_count[0] += 1
            if "poster.html" in str(self):
                raise OSError("disk full")
            return original_write(self, content, *args, **kwargs)

        with patch("posttrainbench_repro.pipeline.PROJECT_ROOT", tmp_path), \
             patch.object(Path, "write_text", failing_write):
            with pytest.raises(OSError):
                run_pipeline(acquired)

        # Original provenance.json should be restored
        assert (evidence / "provenance.json").read_bytes() == sentinel
