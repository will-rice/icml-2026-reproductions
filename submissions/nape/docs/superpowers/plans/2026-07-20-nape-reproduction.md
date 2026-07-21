# NAPE Challenge Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, validate, and publish a CPU-only challenge logbook that gives reproducible verdicts for both selected NAPE claims.

**Architecture:** A small `src` package audits the pinned official GitHub and Hugging Face releases and drives the official online evaluator with a deterministic scripted solver. It writes a machine-readable evidence bundle; Trackio records the commands, outputs, verdicts, and bundle in the challenge-generated static logbook.

**Tech Stack:** Python 3.13, `uv`/`uv_build`, pytest, NAPE's `next_action_pred_eval`, `huggingface_hub`, Trackio, Ruff, ty, pre-commit, Hugging Face Spaces.

## Global Constraints

- Target both challenge claims and a maximum judged score of four points.
- Use CPU only and make no paid LLM or API calls.
- Pin the NAPE GitHub release to `ac0d10e4dc345f982a5665a2c4bdb6b752d663f2`.
- Pin `Tej-a55/napeval` to `c7e28fe9b08ee2c0bfc429519cf100197b7e018c`.
- Treat one entry in a trajectory's `operations` list as one symbolic action.
- Write deterministic evidence without credentials or machine-specific absolute paths.
- Preserve the challenge-generated Trackio layout and publish to `wrice/repro-a-benchmark-and-framework-for-evaluating-next-action-predictions-in-spreadsheets`.
- Use test-first development for every new Python behavior.
- Base the package structure and tooling on `will-rice/agent-harness-template` and the shared CI conventions in `will-rice/ml-template`.
- Preserve the template conventions: `uv_build`, author `Will Rice <wrice20@gmail.com>`, Apache-2.0, Google-style docstrings, Ruff rules `C,E,F,I,W,D,N,B,PTH,ANN`, `ty`, pytest, v6 pre-commit hooks, and prettier formatting.
- Do not add the ML template's unrelated Torch, Lightning, torchaudio, torchvision, or W&B dependencies.
- Run `uv run pre-commit run -a`, focused tests, upstream tests, the challenge validator, and live Space smoke checks before completion.

## File Structure

- `pyproject.toml`: package metadata, command entry point, pytest settings, and development dependencies.
- `.pre-commit-config.yaml`: template-derived file hygiene, prettier, Ruff, ty, and pytest hooks.
- `.github/workflows/python-package.yml`: template-derived Python package quality workflow.
- `LICENSE`: Apache License 2.0 from the user's templates.
- `src/icml_2026_repro/__init__.py`: public evidence-generation exports.
- `src/icml_2026_repro/audit.py`: structured parsing and count comparison for the two official releases.
- `src/icml_2026_repro/online_trace.py`: deterministic solver and official online evaluator trace.
- `src/icml_2026_repro/cli.py`: fixed-target bundle orchestration and JSON serialization.
- `tests/test_audit.py`: local and JSONL audit behavior.
- `tests/test_online_trace.py`: accepted/rejected online transition behavior.
- `tests/test_cli.py`: complete bundle schema and relative-path guarantees.
- `repro_bundle/`: generated JSON evidence and rerun manifest.
- `README.md`: exact setup, reproduction, validation, and publication commands.
- `.trackio/logbook/pages/*/page.md`: outcome-first challenge report cells.

---

### Task 1: Package Tooling and Local Release Audit

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Create: `.pre-commit-config.yaml`
- Create: `.github/workflows/python-package.yml`
- Create: `LICENSE`
- Create: `src/icml_2026_repro/__init__.py`
- Create: `src/icml_2026_repro/audit.py`
- Create: `tests/test_audit.py`

**Interfaces:**
- Consumes: trajectory JSON objects with `name: str` and `operations: list[str]`.
- Produces: `ReleaseCounts(source: str, revision: str, trajectories: int, actions: int)` and `audit_trajectory_directory(path: Path, revision: str) -> ReleaseCounts`.

- [ ] **Step 1: Apply the user's template foundation**

Adapt `will-rice/agent-harness-template` to this package: retain its author metadata, `uv_build`, src layout, Ruff/Google-docstring configuration, `ty`, pytest, and v6 pre-commit structure. Reuse the Apache-2.0 `LICENSE` and adapt `will-rice/ml-template/.github/workflows/python-package.yml` to Python 3.13 and current action versions. Preserve `.worktrees/` in the template-derived `.gitignore`.

The pre-commit file must retain the template's `check-ast`, `end-of-file-fixer`, `trailing-whitespace`, `check-merge-conflict`, `requirements-txt-fixer`, `will-rice/prettier-pre-commit`, Ruff format, Ruff fix, ty, and pytest hooks. Scope Python hooks to `src/` and `tests/`; exclude `.trackio/`, `external/`, `repro_bundle/`, and `.superpowers/` from repository-wide formatting.

Expected configuration includes:

```toml
[build-system]
requires = ["uv_build>=0.9.2"]
build-backend = "uv_build"

[tool.uv.build-backend]
module-name = ["icml_2026_repro"]

[tool.ruff.lint]
select = ["C", "E", "F", "I", "W", "D", "N", "B", "PTH", "ANN"]
ignore = ["D107"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Write failing local-audit tests**

Create `tests/test_audit.py` with real temporary JSON files:

```python
import json

import pytest

from icml_2026_repro.audit import audit_trajectory_directory


def test_audit_trajectory_directory_counts_files_and_operations(tmp_path):
    for name, operations in {"a": ["one", "two"], "b": ["three"]}.items():
        (tmp_path / f"{name}.json").write_text(
            json.dumps({"name": name, "operations": operations}), encoding="utf-8"
        )

    result = audit_trajectory_directory(tmp_path, revision="abc123")

    assert result.trajectories == 2
    assert result.actions == 3
    assert result.revision == "abc123"


def test_audit_trajectory_directory_rejects_missing_operations(tmp_path):
    (tmp_path / "bad.json").write_text(json.dumps({"name": "bad"}), encoding="utf-8")

    with pytest.raises(ValueError, match="operations"):
        audit_trajectory_directory(tmp_path, revision="abc123")
```

- [ ] **Step 3: Verify the tests fail for the missing package**

Run: `uv run pytest tests/test_audit.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'icml_2026_repro'`.

- [ ] **Step 4: Implement the minimal local audit**

Create `src/icml_2026_repro/audit.py` with a frozen `ReleaseCounts` dataclass. Parse every sorted `*.json` file with `json.loads`, require `operations` to be a list of strings, and sum `len(operations)`. Set `source` to `github:Tej-55/NAPE`.

```python
@dataclass(frozen=True)
class ReleaseCounts:
    source: str
    revision: str
    trajectories: int
    actions: int


def audit_trajectory_directory(path: Path, revision: str) -> ReleaseCounts:
    trajectory_paths = sorted(path.glob("*.json"))
    action_count = 0
    for trajectory_path in trajectory_paths:
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        operations = trajectory.get("operations")
        if not isinstance(operations, list) or not all(
            isinstance(operation, str) for operation in operations
        ):
            raise ValueError(f"{trajectory_path.name}: operations must be a list of strings")
        action_count += len(operations)
    return ReleaseCounts(
        source="github:Tej-55/NAPE",
        revision=revision,
        trajectories=len(trajectory_paths),
        actions=action_count,
    )
```

Export `ReleaseCounts` and `audit_trajectory_directory` from `src/icml_2026_repro/__init__.py`.

- [ ] **Step 5: Verify the local audit passes**

Run: `uv run pytest tests/test_audit.py -q`

Expected: `2 passed`.

- [ ] **Step 6: Commit the independently testable audit**

```bash
git add .gitignore .python-version .github/workflows/python-package.yml LICENSE pyproject.toml uv.lock .pre-commit-config.yaml src/icml_2026_repro tests/test_audit.py
git commit -m "feat: make released NAPE counts independently auditable"
```

### Task 2: Pinned Hugging Face Cross-Check and Claim 1 Evidence

**Files:**
- Modify: `src/icml_2026_repro/audit.py`
- Modify: `src/icml_2026_repro/__init__.py`
- Modify: `tests/test_audit.py`

**Interfaces:**
- Consumes: `data/test.jsonl` from pinned dataset revision `c7e28fe9b08ee2c0bfc429519cf100197b7e018c`.
- Produces: `audit_jsonl(path: Path, revision: str) -> ReleaseCounts` and `build_challenge_card_claim_1_audit() -> dict[str, object]`.

- [ ] **Step 1: Write failing JSONL and verdict tests**

Append tests that create two JSONL rows and assert three actions, reject an inconsistent `num_operations`, and require a falsified verdict when both official releases report 52 trajectories and 11,907 actions against 58 and 13,000.

```python
def test_audit_jsonl_counts_rows_and_operations(tmp_path):
    dataset_path = tmp_path / "test.jsonl"
    rows = [
        {"name": "a", "operations": ["one", "two"], "num_operations": 2},
        {"name": "b", "operations": ["three"], "num_operations": 1},
    ]
    dataset_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    result = audit_jsonl(dataset_path, revision="def456")

    assert result.source == "dataset:Tej-a55/napeval"
    assert result.trajectories == 2
    assert result.actions == 3
```

- [ ] **Step 2: Verify the new tests fail**

Run: `uv run pytest tests/test_audit.py -q`

Expected: failure because `audit_jsonl` and claim comparison do not exist.

- [ ] **Step 3: Implement the pinned cross-check**

Add constants for the claimed values, GitHub revision, dataset revision, and repository IDs. `audit_jsonl` must validate each row's schema and that `num_operations == len(operations)`. `build_challenge_card_claim_1_audit` must:

1. verify the vendored NAPE checkout's `HEAD` equals the pinned GitHub revision;
2. audit `external/NAPE/data/trajectories`;
3. obtain pinned `data/test.jsonl` with `huggingface_hub.hf_hub_download`;
4. audit the downloaded JSONL;
5. require both sources to agree; and
6. return `verdict: "falsified"` when the observed counts differ from 58 and 13,000.

The returned dictionary must include `claim`, `counting_definition`, `claimed`, `observed`, `sources`, and `verdict` keys.

- [ ] **Step 4: Verify all claim 1 tests pass**

Run: `uv run pytest tests/test_audit.py -q`

Expected: all tests in `tests/test_audit.py` pass.

- [ ] **Step 5: Run the real pinned audit**

Run: `uv run python -c 'import json; from icml_2026_repro.audit import build_challenge_card_claim_1_audit; print(json.dumps(build_challenge_card_claim_1_audit(), indent=2))'`

Expected evidence: GitHub and dataset both report `52` trajectories and `11907` actions; verdict is `falsified`.

- [ ] **Step 6: Commit the official-release cross-check**

```bash
git add src/icml_2026_repro tests/test_audit.py
git commit -m "feat: cross-check NAPE counts against the pinned Hub dataset"
```

### Task 3: Deterministic Online Evaluation Trace

**Files:**
- Create: `src/icml_2026_repro/online_trace.py`
- Modify: `src/icml_2026_repro/__init__.py`
- Create: `tests/test_online_trace.py`

**Interfaces:**
- Consumes: NAPE `Orchestrator`, `ISolver`, `PredictionResult`, `HEURISTIC_STEPS_SAVED`, and a fixed five-action trajectory.
- Produces: `build_claim_2_evidence(output_dir: Path) -> dict[str, object]` with user-step, prediction, decision, and future-update records.

- [ ] **Step 1: Write a failing end-to-end online trace test**

Create `tests/test_online_trace.py`:

```python
from icml_2026_repro.online_trace import build_claim_2_evidence


def test_online_trace_covers_prediction_decisions_and_future_update(tmp_path):
    evidence = build_claim_2_evidence(tmp_path)

    assert evidence["verdict"] == "verified"
    assert evidence["summary"]["accepted"] >= 1
    assert evidence["summary"]["rejected"] >= 1
    assert evidence["summary"]["predictions_after_user_actions"] is True
    assert evidence["summary"]["future_was_updated"] is True
    assert {decision["accepted"] for decision in evidence["decisions"]} == {True, False}
```

- [ ] **Step 2: Verify the online trace test fails**

Run: `uv run pytest tests/test_online_trace.py -q`

Expected: collection fails because `icml_2026_repro.online_trace` does not exist.

- [ ] **Step 3: Implement a deterministic scripted solver**

Create a public `ScriptedSolver(ISolver)` that accepts a sequence of symbolic prediction lists, advances one response per `predict` call, converts strings with `symbolic_to_operations`, and returns a `PredictionResult` with zero tokens and `metadata={"solver": "scripted"}`. `reset()` sets the response index to zero.

Use this fixed ground truth:

```python
GROUND_TRUTH = [
    "VALUE | Sheet1!A1 | 1",
    "VALUE | Sheet1!A2 | 2",
    "VALUE | Sheet1!A3 | 3",
    "VALUE | Sheet1!A4 | 4",
    "VALUE | Sheet1!A5 | 5",
]
```

Provide scripted responses that cause one multi-action correct prediction to be accepted and one wrong prediction to be rejected. Run the official `Orchestrator` with fixed-interval stride 1, `HEURISTIC_STEPS_SAVED`, `online_mode=True`, and the caller-provided output directory.

- [ ] **Step 4: Build evidence from official recorder output**

Read the generated `timeline.jsonl` rather than reconstructing events from assumptions. Extract user steps and prediction events, require every prediction to follow a user event, require both acceptance outcomes, and require an accepted event whose `future_after_count` differs from `future_before_count`. Return source revision, fixture, decisions, summary, and `verdict: "verified"`; raise `RuntimeError` if any required transition is absent.

- [ ] **Step 5: Verify the trace and upstream suite**

Run: `uv run pytest tests/test_online_trace.py -q`

Expected: `1 passed`.

Run: `uv run pytest external/NAPE/tests -q`

Expected: `244 passed`.

- [ ] **Step 6: Commit the claim 2 trace**

```bash
git add src/icml_2026_repro tests/test_online_trace.py
git commit -m "feat: demonstrate NAPE online evaluation state transitions"
```

### Task 4: Reproduction Bundle and Operator Documentation

**Files:**
- Create: `src/icml_2026_repro/cli.py`
- Modify: `src/icml_2026_repro/__init__.py`
- Create: `tests/test_cli.py`
- Modify: `README.md`
- Delete: `main.py`
- Generate: `repro_bundle/claim_1_audit.json`
- Generate: `repro_bundle/claim_2_trace.json`
- Generate: `repro_bundle/environment.json`
- Generate: `repro_bundle/README.md`

**Interfaces:**
- Consumes: `build_challenge_card_claim_1_audit()` and `build_claim_2_evidence(output_dir)`.
- Produces: `build_bundle(output_dir: Path = Path("repro_bundle")) -> Path` and console command `nape-repro`.

- [ ] **Step 1: Write a failing bundle test**

Create `tests/test_cli.py` using monkeypatch only to replace the two already-tested evidence builders with deterministic dictionaries. Assert that `build_bundle(tmp_path)` creates all four files, each JSON file parses, no serialized value contains the absolute temporary path, and the returned path equals `tmp_path`.

- [ ] **Step 2: Verify the bundle test fails**

Run: `uv run pytest tests/test_cli.py -q`

Expected: collection fails because `icml_2026_repro.cli` does not exist.

- [ ] **Step 3: Implement bundle generation**

Create `build_bundle` and `main` at the top of `cli.py`. Configure `logging.basicConfig(level=logging.INFO, format="%(message)s")`. Write sorted, indented JSON with a trailing newline. Include Python version, platform, pinned revisions, and exact rerun commands in `environment.json`; write a short bundle `README.md` defining every artifact and the action-count convention. Do not include cache paths or access tokens.

Delete the untracked `main.py` starter because the package entry point replaces it.

Register the command only after `cli.py` exists:

```toml
[project.scripts]
nape-repro = "icml_2026_repro.cli:main"
```

- [ ] **Step 4: Verify the bundle behavior and generate real evidence**

Run: `uv run pytest tests/test_cli.py -q`

Expected: all tests pass.

Run: `uv run nape-repro`

Expected: exit 0 and four files under `repro_bundle/`; claim 1 is `falsified` and claim 2 is `verified`.

- [ ] **Step 5: Replace the generated starter README**

Document prerequisites, `uv sync`, `uv run nape-repro`, focused tests, upstream tests, pre-commit, challenge validation, and the target Space URL. State observed counts, both proposed verdicts, CPU-only cost, pinned source links, and the distinction between reproducing evaluator behavior and benchmarking model quality.

- [ ] **Step 6: Run all local quality checks**

Run: `uv run pytest -q`

Expected: all root tests pass.

Run: `uv run pre-commit run -a`

Expected: every hook passes.

Run: `uv run ty check`

Expected: no type errors.

- [ ] **Step 7: Commit the rerunnable bundle**

```bash
git add README.md pyproject.toml uv.lock src tests repro_bundle .pre-commit-config.yaml
git commit -m "feat: package the two-claim NAPE reproduction evidence"
```

### Task 5: Complete and Validate the Trackio Logbook

**Files:**
- Modify: `.trackio/logbook/pages/executive-summary/page.md`
- Modify: `.trackio/logbook/pages/claim-1-benchmark-generates-58-symbolic-action-sequences-consisting-of-13k-actions-from-publicly-available-spreadsheets/page.md`
- Modify: `.trackio/logbook/pages/claim-2-online-evaluation-methodology-predicts-actions-after-each-user-action-accepts-or-rejects-prediction-and-updates-future-actions/page.md`
- Modify: `.trackio/logbook/pages/conclusion/page.md`
- Modify: `.trackio/logbook/logbook.json`
- Modify: `.trackio/metadata.json`

**Interfaces:**
- Consumes: fresh command outputs and `repro_bundle/`.
- Produces: a challenge-valid static Trackio logbook and Trackio artifact reference.

- [ ] **Step 1: Record fresh claim commands through Trackio**

Run the focused claim 1 audit, claim 2 trace test, and full upstream suite with `uv run trackio logbook run --page ... --title ... -- <command>`. Use descriptive cell titles and retain exact exit status and wall time.

- [ ] **Step 2: Attach the reproduction bundle**

Log `repro_bundle/` with `trackio.log_artifact(..., name="repro-bundle", type="dataset")`, then add its artifact cell to the Conclusion page with `uv run trackio logbook cell artifact ... --type dataset`.

- [ ] **Step 3: Write outcome-first report cells**

Use `uv run trackio logbook cell markdown` for authored prose rather than editing generated metadata by hand. State:

- Claim 1 proposed verdict: **falsified**, because both pinned author releases contain 52 trajectories and 11,907 symbolic operations rather than 58 and approximately 13,000.
- Claim 2 proposed verdict: **verified**, because the official online orchestrator predicts after user actions, records both acceptance outcomes, and updates future operations after acceptance.
- Compute: CPU only; paid API cost: `$0.00`.
- Limitation: this reproduces the released benchmark cardinality and evaluator mechanics, not model quality tables.

Replace the scaffold's placeholder poster figure with a real `poster_embed.html` generated from the two verdicts, observed counts, pinned revisions, methods, compute, and limitations. Keep the poster self-contained and legible in the Trackio figure frame.

- [ ] **Step 4: Fetch and run the official validator**

Download `scripts/validate_icml_logbook.py` from the challenge Space at its current revision and run it against `.trackio`. Treat warnings about missing verdict language, source links, artifacts, or required tags as failures to fix.

- [ ] **Step 5: Review the rendered logbook locally**

Run `uv run trackio logbook serve --no-browser` long enough to request the root page and each hash-routed page. Confirm all four pages, command cells, verdicts, the completed poster, and artifact link render; then stop the server.

- [ ] **Step 6: Commit the validated logbook**

```bash
git add .trackio README.md repro_bundle
git commit -m "docs: present the NAPE reproduction evidence for judging"
```

### Task 6: Publish and Verify the Challenge Space

**Files:**
- Publish repository content to the configured Hugging Face Space.

**Interfaces:**
- Consumes: validated `.trackio` metadata and static logbook.
- Produces: public Space `wrice/repro-a-benchmark-and-framework-for-evaluating-next-action-predictions-in-spreadsheets`.

- [ ] **Step 1: Reconfirm authentication and target metadata**

Run: `uv run hf auth whoami`

Expected: authenticated user `wrice`.

Run: `jq '{space_id, tags, paper}' .trackio/metadata.json`

Expected: target Space, `icml2026-repro`, `paper-NvPgRwURDC`, and arXiv `2606.13802`.

- [ ] **Step 2: Publish with Trackio**

Run `uv run trackio logbook publish wrice/repro-a-benchmark-and-framework-for-evaluating-next-action-predictions-in-spreadsheets`. Do not upload `.venv`, caches, or the vendored `.git` directory.

- [ ] **Step 3: Verify runtime and logs**

Run: `uv run hf spaces info wrice/repro-a-benchmark-and-framework-for-evaluating-next-action-predictions-in-spreadsheets --expand runtime`

Expected: the Space reaches `RUNNING` on CPU hardware.

Run: `uv run hf spaces logs wrice/repro-a-benchmark-and-framework-for-evaluating-next-action-predictions-in-spreadsheets --tail 200`

Expected: no build or runtime errors.

- [ ] **Step 4: Smoke-test the public report**

Request the public Space URL and verify HTTP 200 plus the logbook title, both claim page names, both verdict terms, and the reproduction artifact link. Re-run the official validator against the published Space if supported by the helper.

- [ ] **Step 5: Run the final verification gate**

Run fresh, in order:

```bash
uv run pytest -q
uv run pytest external/NAPE/tests -q
uv run nape-repro
uv run pre-commit run -a
uv run ty check
```

Then inspect `git status --short`, confirm generated evidence matches the report, and confirm the live Space still serves the same revision before reporting completion.

- [ ] **Step 6: Commit any publication metadata refresh**

If Trackio updates local publication metadata, commit only those generated changes:

```bash
git add .trackio
git commit -m "chore: record the published reproduction revision"
```
