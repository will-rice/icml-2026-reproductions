# RBench Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the committed RBench artifact auditors into an independently executable, CPU-only evidence bundle and Hugging Face Space without claiming that video generation or human-correlation experiments were rerun.

**Architecture:** Preserve the existing pinned acquisition, prompt-census, leaderboard, and source-tracing modules. Add one evidence renderer that translates their typed outputs into a closed JSON schema, one thin CLI that separates online acquisition from offline audit/validation, and one read-only Gradio viewer over a committed evidence bundle. The paper-era and later leaderboard revisions remain distinct cohorts, and unsupported real-video claims remain explicitly unavailable or partial.

**Tech Stack:** Python 3.11+, `huggingface_hub`, `httpx`, `jsonschema`, `pytest`, `gradio`, `uv`.

## Global Constraints

- Paper ID is `p5QSlnwume`; controller attempt ID is `8c21f2dc-a357-422e-9c1b-79a4d417e3dc`.
- Pin ReVidgen to `b03df27`, RBench data to `6bdccf`, the paper-era leaderboard to `6b66282`, and the later comparison leaderboard to `5dd6d55` using their complete 40-character revisions already recorded by `SOURCE_SPECS`.
- Recompute only artifact structure, formula, cohort, and source-route claims. Do not describe paper-reported values as reproduced measurements.
- Do not claim that any video model ran, that semantic metric quality was validated, or that human correlation was reproduced.
- Acquisition may use the network; `audit` and `validate` must run offline from a hash-verified cache.
- Use red-green-refactor for every production behavior. Do not copy unreviewed files from the dirty `.worktrees/rbench` working tree.
- Worker scope is only `submissions/rethinking-video-generation-model-for-the-embodied-world/`; workers must not mutate controller state, `docs/HANDOFF.md`, the Hub, or verdict APIs.

---

### Task 1: Closed Evidence Schema and Claim Renderer

**Files:**
- Create: `submissions/rethinking-video-generation-model-for-the-embodied-world/schema/evidence-v1.schema.json`
- Create: `submissions/rethinking-video-generation-model-for-the-embodied-world/src/rbench_repro/evidence.py`
- Create: `submissions/rethinking-video-generation-model-for-the-embodied-world/tests/test_evidence.py`

**Interfaces:**
- Consumes: `SourceManifest`, `CensusResult`, `MetricTrace`, `Formula`, `LeaderboardResult`, `CohortComparison`, and `FailureModeResult` from the existing modules.
- Produces: `AuditInputs`, `build_evidence(inputs: AuditInputs, generated_at: str, tool_revision: str) -> dict[str, object]`, `validate_evidence(value: object, schema_path: Path) -> None`, and `resolve_json_pointer(value: object, pointer: str) -> object`.

- [ ] **Step 1: Write the failing evidence tests**

```python
def test_evidence_has_controller_identity_and_honest_claim_statuses(
    complete_audit_inputs, schema_path
):
    evidence = build_evidence(
        complete_audit_inputs, "2026-07-27T00:00:00+00:00", "abc123"
    )
    validate_evidence(evidence, schema_path)
    assert evidence["paper_id"] == "p5QSlnwume"
    assert evidence["attempt_id"] == "8c21f2dc-a357-422e-9c1b-79a4d417e3dc"
    assert [item["status"] for item in evidence["claims"]] == [
        "verified", "verified", "partial"
    ]
    assert "real-video" in evidence["claims"][2]["limitations"][0]


def test_evidence_is_canonical_and_every_pointer_resolves(complete_audit_inputs):
    first = build_evidence(complete_audit_inputs, GENERATED_AT, TOOL_REVISION)
    second = build_evidence(complete_audit_inputs, GENERATED_AT, TOOL_REVISION)
    assert canonical_json(first) == canonical_json(second)
    for artifact in first["artifacts"]:
        pointed = resolve_json_pointer(first, artifact["json_pointer"])
        assert artifact["sha256"] == sha256_bytes(canonical_json(pointed))


def test_claims_downgrade_when_required_artifact_routes_are_missing(
    complete_audit_inputs
):
    complete_audit_inputs.metrics = ()
    assert build_evidence(
        complete_audit_inputs, GENERATED_AT, TOOL_REVISION
    )["claims"][0]["status"] == "partial"

    complete_audit_inputs.category_evidence = {}
    assert build_evidence(
        complete_audit_inputs, GENERATED_AT, TOOL_REVISION
    )["claims"][1]["status"] == "partial"

    complete_audit_inputs.failure_modes = ()
    assert build_evidence(
        complete_audit_inputs, GENERATED_AT, TOOL_REVISION
    )["claims"][2]["status"] == "inconclusive"
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `cd submissions/rethinking-video-generation-model-for-the-embodied-world && uv run pytest tests/test_evidence.py -v`

Expected: FAIL because `rbench_repro.evidence` and the schema do not exist.

- [ ] **Step 3: Implement the minimal renderer and schema**

```python
@dataclass(slots=True)
class AuditInputs:
    sources: tuple[SourceManifest, ...]
    census: CensusResult | None
    metrics: tuple[MetricTrace, ...]
    formula: Formula | None
    leaderboards: tuple[LeaderboardResult, ...]
    comparison: CohortComparison | None
    failure_modes: tuple[FailureModeResult, ...]
    category_evidence: dict[str, tuple[str, ...]]
    package_lock_sha256: str


def build_evidence(
    inputs: AuditInputs, generated_at: str, tool_revision: str
) -> dict[str, object]:
    claims = (
        claim_1_status(inputs.census, inputs.metrics, inputs.leaderboards),
        claim_2_status(
            _cohort(inputs.leaderboards, "paper-era"),
            _cohort(inputs.leaderboards, "later"),
            inputs.comparison,
            inputs.category_evidence,
        ),
        claim_3_status(inputs.failure_modes),
    )
    return _render_bundle(inputs, claims, generated_at, tool_revision)
```

The JSON Schema must set `additionalProperties: false` at every owned object, restrict statuses to `verified`, `partial`, `inconclusive`, `contradicted`, or `unavailable`, reject non-finite numbers, and require sources, claims, environment, artifacts, unavailable results, and contradictions.

- [ ] **Step 4: Run evidence tests and the existing core suite**

Run: `cd submissions/rethinking-video-generation-model-for-the-embodied-world && uv run pytest tests/test_evidence.py tests/test_acquisition.py tests/test_census.py tests/test_leaderboard.py tests/test_source_audit.py -v`

Expected: PASS, with claim three capped at `partial` because no real-video evaluation was run.

- [ ] **Step 5: Commit the schema and renderer**

```bash
git add submissions/rethinking-video-generation-model-for-the-embodied-world/schema/evidence-v1.schema.json submissions/rethinking-video-generation-model-for-the-embodied-world/src/rbench_repro/evidence.py submissions/rethinking-video-generation-model-for-the-embodied-world/tests/test_evidence.py
git commit -m "feat: render honest RBench evidence"
```

### Task 2: Online Acquisition and Offline Audit CLI

**Files:**
- Create: `submissions/rethinking-video-generation-model-for-the-embodied-world/src/rbench_repro/cli.py`
- Create: `submissions/rethinking-video-generation-model-for-the-embodied-world/tests/test_cli.py`
- Create: `submissions/rethinking-video-generation-model-for-the-embodied-world/tests/test_worker_proposal.py`
- Modify: `submissions/rethinking-video-generation-model-for-the-embodied-world/pyproject.toml`

**Interfaces:**
- Consumes: `acquire_all`, `load_acquired`, all existing audit functions, and Task 1’s `AuditInputs`, `build_evidence`, and `validate_evidence`.
- Produces: console script `rbench-repro` with `acquire`, `validate-inputs`, `audit`, `validate`, and `propose` subcommands, plus `build_worker_proposal(evidence_bytes: bytes, source_commit: str, source_tree: str) -> dict[str, object]`.

- [ ] **Step 1: Write failing CLI isolation tests**

```python
def test_audit_is_offline_schema_valid_and_byte_identical(cli_fixture, tmp_path):
    outputs = []
    for index in range(2):
        output = tmp_path / f"result-{index}.json"
        subprocess.run(
            audit_command(cli_fixture, output),
            check=True,
            env=offline_environment(),
        )
        validate_evidence(json.loads(output.read_bytes()), cli_fixture.schema)
        outputs.append(output.read_bytes())
    assert outputs[0] == outputs[1]


def test_invalid_input_preserves_existing_output(cli_fixture, tmp_path):
    output = tmp_path / "results.json"
    output.write_bytes(b"preserve\n")
    result = subprocess.run(
        audit_command(cli_fixture, output, manifest=tmp_path / "missing.json"),
        capture_output=True,
        env=offline_environment(),
        text=True,
    )
    assert result.returncode != 0
    assert output.read_bytes() == b"preserve\n"
    assert "Traceback" not in result.stderr


def test_worker_proposal_requests_controller_validation():
    proposal = build_worker_proposal(b"evidence\n", "a" * 40, "b" * 40)
    assert proposal["requested_action"] == "controller_validation"
    assert proposal["external_mutations"] == []
    assert proposal["source_commit"] == "a" * 40
    assert proposal["source_tree"] == "b" * 40
```

- [ ] **Step 2: Run the CLI tests and verify RED**

Run: `cd submissions/rethinking-video-generation-model-for-the-embodied-world && uv run pytest tests/test_cli.py -v`

Expected: FAIL because the CLI module and entry point do not exist.

- [ ] **Step 3: Implement the four commands and atomic output**

```python
def command_audit(args: argparse.Namespace) -> int:
    sources = load_acquired(args.manifest, args.cache_dir)
    inputs = assemble_audit_inputs(sources, Path(__file__).resolve().parents[2])
    evidence = build_evidence(inputs, args.generated_at, args.tool_revision)
    validate_evidence(evidence, args.schema)
    atomic_write(args.output, canonical_json(evidence))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
```

Add `rbench-repro = "rbench_repro.cli:main"` under `[project.scripts]`. `validate-inputs` must rehash every cached file and optionally verify exact `cohort:path:bytes:sha256` expectations. `audit` must not instantiate a Hub or HTTP client. `propose` must require exact 40-character `--source-commit` and `--source-tree` values and atomically serialize the non-authoritative proposal described in Task 5.

- [ ] **Step 4: Verify CLI behavior and regression tests**

Run: `cd submissions/rethinking-video-generation-model-for-the-embodied-world && uv run pytest -v`

Expected: PASS, including two byte-identical offline audit runs and nonzero sanitized failures.

- [ ] **Step 5: Commit the CLI**

```bash
git add submissions/rethinking-video-generation-model-for-the-embodied-world/pyproject.toml submissions/rethinking-video-generation-model-for-the-embodied-world/src/rbench_repro/cli.py submissions/rethinking-video-generation-model-for-the-embodied-world/tests/test_cli.py submissions/rethinking-video-generation-model-for-the-embodied-world/tests/test_worker_proposal.py
git commit -m "feat: add RBench evidence CLI"
```

### Task 3: Generate and Validate the Evidence Bundle

**Files:**
- Create: `submissions/rethinking-video-generation-model-for-the-embodied-world/evidence/input-manifest.json`
- Create: `submissions/rethinking-video-generation-model-for-the-embodied-world/evidence/results.json`
- Create: `submissions/rethinking-video-generation-model-for-the-embodied-world/evidence/validation.json`
- Create: `submissions/rethinking-video-generation-model-for-the-embodied-world/evidence/commands.json`
- Test: `submissions/rethinking-video-generation-model-for-the-embodied-world/tests/test_bundle.py`

**Interfaces:**
- Consumes: Task 2’s CLI and the exact pinned revisions in `SOURCE_SPECS`.
- Produces: a self-describing, machine-readable evidence directory whose committed results validate offline.

- [ ] **Step 1: Write the failing committed-bundle test**

```python
def test_committed_bundle_validates_and_matches_controller_attempt(project_root):
    results = json.loads((project_root / "evidence/results.json").read_text())
    validation = json.loads((project_root / "evidence/validation.json").read_text())
    validate_evidence(results, project_root / "schema/evidence-v1.schema.json")
    assert results["attempt_id"] == "8c21f2dc-a357-422e-9c1b-79a4d417e3dc"
    assert validation["valid"] is True
    assert validation["results_sha256"] == sha256_bytes(
        (project_root / "evidence/results.json").read_bytes()
    )
```

- [ ] **Step 2: Run the bundle test and verify RED**

Run: `cd submissions/rethinking-video-generation-model-for-the-embodied-world && uv run pytest tests/test_bundle.py -v`

Expected: FAIL because the committed bundle does not exist.

- [ ] **Step 3: Acquire pinned inputs and run the offline audit**

```bash
cd submissions/rethinking-video-generation-model-for-the-embodied-world
uv run rbench-repro acquire --cache-dir .cache/rbench --manifest evidence/input-manifest.json --acquired-at "$(date --utc +%Y-%m-%dT%H:%M:%SZ)"
env ALL_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 NO_PROXY= uv run rbench-repro audit --manifest evidence/input-manifest.json --cache-dir .cache/rbench --schema schema/evidence-v1.schema.json --output evidence/results.json --generated-at 2026-07-27T00:00:00+00:00 --tool-revision "$(git rev-parse HEAD)"
uv run rbench-repro validate evidence/results.json --schema schema/evidence-v1.schema.json
```

Record the literal commands, revisions, Python/platform information, lock-file hash, result hash, and validation outcome in `commands.json` and `validation.json`. Do not include cache paths containing credentials or environment dumps.

- [ ] **Step 4: Run the bundle and full submission tests**

Run: `cd submissions/rethinking-video-generation-model-for-the-embodied-world && uv run pytest -v`

Expected: PASS with the pinned paper-era cohort distinct from the later cohort and with no raw video files committed.

- [ ] **Step 5: Commit the evidence bundle**

```bash
git add submissions/rethinking-video-generation-model-for-the-embodied-world/evidence submissions/rethinking-video-generation-model-for-the-embodied-world/tests/test_bundle.py
git commit -m "data: add RBench evidence bundle"
```

### Task 4: Reviewer-Facing Documentation and Space

**Files:**
- Create: `submissions/rethinking-video-generation-model-for-the-embodied-world/README.md`
- Create: `submissions/rethinking-video-generation-model-for-the-embodied-world/POSTER.md`
- Create: `submissions/rethinking-video-generation-model-for-the-embodied-world/app.py`
- Create: `submissions/rethinking-video-generation-model-for-the-embodied-world/tests/test_space.py`

**Interfaces:**
- Consumes: committed `evidence/results.json`; no network or source cache.
- Produces: a dedicated Gradio Space showing claim status, computed observations, provenance, commands, and limitations.

- [ ] **Step 1: Write failing Space tests**

```python
def test_space_loads_only_committed_evidence(monkeypatch):
    monkeypatch.setattr(socket, "create_connection", fail_network)
    module = load_app_module()
    assert module.EVIDENCE["paper_id"] == "p5QSlnwume"
    assert module.EVIDENCE["attempt_id"] == "8c21f2dc-a357-422e-9c1b-79a4d417e3dc"


def test_readme_and_poster_state_unreproduced_limits(project_root):
    text = (
        (project_root / "README.md").read_text()
        + (project_root / "POSTER.md").read_text()
    ).lower()
    assert "video generation was not rerun" in text
    assert "human correlation was not reproduced" in text


def test_space_metadata_targets_exact_challenge_paper(project_root):
    readme = (project_root / "README.md").read_text()
    assert "sdk: gradio" in readme
    assert "sdk_version: 6.20.0" in readme
    assert "app_file: app.py" in readme
    assert "- paper-p5QSlnwume" in readme
    assert "- icml2026-repro" in readme
```

- [ ] **Step 2: Run the Space tests and verify RED**

Run: `cd submissions/rethinking-video-generation-model-for-the-embodied-world && uv run pytest tests/test_space.py -v`

Expected: FAIL because the app and reviewer documents do not exist.

- [ ] **Step 3: Implement the read-only Space and concise documentation**

```python
ROOT = Path(__file__).resolve().parent
EVIDENCE = json.loads((ROOT / "evidence" / "results.json").read_text())


def claim_rows() -> list[list[str]]:
    return [
        [
            claim["claim"],
            claim["status"],
            claim["observations"][0]["summary"],
            "; ".join(claim["limitations"]),
        ]
        for claim in EVIDENCE["claims"]
    ]


with gr.Blocks() as demo:
    gr.Markdown("# RBench artifact reproduction")
    gr.Dataframe(
        headers=["Claim", "Status", "Computed evidence", "Limitations"],
        value=claim_rows(),
        interactive=False,
    )
```

The README must begin with tested Hugging Face Space YAML specifying `sdk: gradio`, `sdk_version: 6.20.0`, `app_file: app.py`, and tags `paper-p5QSlnwume` and `icml2026-repro`. It must provide exact local acquire/audit/validate commands and distinguish paper context from computed results. The poster must summarize the five-task/four-embodiment census, paper-era versus later cohort counts, source-routed failure-mode evidence, and explicit limits in one screen.

- [ ] **Step 4: Run all submission checks**

Run: `cd submissions/rethinking-video-generation-model-for-the-embodied-world && uv run pytest -v`

Expected: PASS.

Run from repository root:

```bash
mapfile -t RBENCH_FILES < <(git ls-files -co --exclude-standard -- submissions/rethinking-video-generation-model-for-the-embodied-world)
env UV_CACHE_DIR=/tmp/icml-rbench-uv-cache PRE_COMMIT_HOME=/tmp/icml-rbench-pre-commit uv run pre-commit run --files "${RBENCH_FILES[@]}"
```

Expected: PASS without running or formatting `submissions/nape/`.

- [ ] **Step 5: Commit the Space source and reviewer docs**

```bash
git add submissions/rethinking-video-generation-model-for-the-embodied-world/README.md submissions/rethinking-video-generation-model-for-the-embodied-world/POSTER.md submissions/rethinking-video-generation-model-for-the-embodied-world/app.py submissions/rethinking-video-generation-model-for-the-embodied-world/tests/test_space.py
git commit -m "docs: package RBench reproduction Space"
```

### Task 5: Worker Proposal for Controller Validation

**Files:**
- Create: `submissions/rethinking-video-generation-model-for-the-embodied-world/evidence/worker-proposal.json`
- Verify: `submissions/rethinking-video-generation-model-for-the-embodied-world/tests/test_worker_proposal.py`

**Interfaces:**
- Consumes: exact source commit/tree, test results, evidence hash, and Space-source file list.
- Produces: a proposal for controller-owned validation and deployment; it performs no external mutation.

- [ ] **Step 1: Verify the already-committed proposal contract test passes**

```python
def test_worker_proposal_is_scoped_and_non_mutating(project_root):
    proposal = build_worker_proposal(
        (project_root / "evidence/results.json").read_bytes(),
        "a" * 40,
        "b" * 40,
    )
    assert proposal["paper_id"] == "p5QSlnwume"
    assert proposal["attempt_id"] == "8c21f2dc-a357-422e-9c1b-79a4d417e3dc"
    assert proposal["requested_action"] == "controller_validation"
    assert proposal["external_mutations"] == []
    assert len(proposal["source_commit"]) == 40
    assert len(proposal["source_tree"]) == 40
    assert proposal["evidence_sha256"] == sha256_bytes(
        (project_root / "evidence/results.json").read_bytes()
    )
```

- [ ] **Step 2: Run the proposal contract test**

Run: `cd submissions/rethinking-video-generation-model-for-the-embodied-world && uv run pytest tests/test_worker_proposal.py -v`

Expected: PASS before proposal generation, proving the generator is already part of the clean executable source commit.

- [ ] **Step 3: Commit all executable source, then invoke deterministic proposal generation**

```bash
cd submissions/rethinking-video-generation-model-for-the-embodied-world
git status --short
SOURCE_COMMIT="$(git rev-parse HEAD)"
SOURCE_TREE="$(git rev-parse HEAD^{tree})"
uv run rbench-repro propose \
  --evidence evidence/results.json \
  --source-commit "$SOURCE_COMMIT" \
  --source-tree "$SOURCE_TREE" \
  --output evidence/worker-proposal.json
```

`git status --short` must be empty before capturing the source identity; otherwise commit the intended executable changes first and rerun this step. The source commit must contain every executable, test, evidence, documentation, and Space-source file. After proposal generation, verify the proposal's `source_commit` and `source_tree` equal the captured values. The controller will independently validate that clean source commit before any phase transition, deployment, or submission.

- [ ] **Step 4: Run final verification**

Run: `cd submissions/rethinking-video-generation-model-for-the-embodied-world && uv run pytest -v`

Expected: PASS.

Run: `git status --short`

Expected: only intentional RBench files are modified or untracked.

- [ ] **Step 5: Commit the proposal**

```bash
git add submissions/rethinking-video-generation-model-for-the-embodied-world/evidence/worker-proposal.json
git commit -m "chore: propose RBench controller validation"
```
