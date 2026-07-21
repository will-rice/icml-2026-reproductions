# NAPE Judge-Aligned Score Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current two-card audit with reproducible, judge-aligned evidence for paper Claims 1-3, explicitly document Claims 4-6 as unreplicated, and republish the canonical challenge Space for rejudging.

**Architecture:** An OpenResearch project records the currently judged revision as the baseline and the score-improvement revision as a child experiment with an exact local CPU run and archived evidence. Three focused evidence modules read the pinned NAPE checkout: one audits benchmark construction, one aggregates released oracle predictability outputs, and one exercises official future-adaptation code across all 52 trajectories plus a residual-patch fixture. The existing CLI serializes those reports, the existing Trackio project presents one page per judge claim, and the same Hugging Face Space remains the canonical submission.

**Tech Stack:** Python 3.13, `uv`, pytest, NAPE `next_action_pred_eval`, OpenResearch CLI `orx` 0.1.69, Trackio, Ruff, ty, pre-commit, GitHub, Hugging Face Hub/Spaces.

## Global Constraints

- Use only released NAPE artifacts and CPU execution; make no paid model or API calls.
- Pin NAPE to `ac0d10e4dc345f982a5665a2c4bdb6b752d663f2` and reject a dirty checkout before producing evidence.
- Report human annotation as release provenance, not as an independently rerunnable stage.
- Describe Claim 2 as a recomputation from released oracle outputs, not a rerun of frontier-model generations.
- Claims 4-6 must say `not replicated`; do not infer or fabricate model-performance verdicts.
- Preserve the current canonical Space: `wrice/repro-a-benchmark-and-framework-for-evaluating-next-action-predictions-in-spreadsheets`.
- Use the installed `orx` 0.1.69 `/icml-repro` workflow as the provenance and formal-run layer; Trackio remains the challenge's judged publication format.
- Keep `orx` out of the Python package dependencies. Its experiment metadata, exact commands, logs, and artifacts are external orchestration records.
- Run `uv run pre-commit run -a`, root tests, all upstream NAPE tests, the official logbook validator, and local page smoke tests before publication.

## File Structure

- `src/icml_2026_repro/benchmark_evidence.py`: benchmark statistics, raw-artifact correspondence, and construction-source audit for Claim 1.
- `tests/test_benchmark_evidence.py`: exact release aggregates and malformed/missing-input failures for Claim 1.
- `src/icml_2026_repro/predictability_evidence.py`: schema validation and aggregation of the 52 released predictability outputs for Claim 2.
- `tests/test_predictability_evidence.py`: exact aggregate and invalid arithmetic/schema tests for Claim 2.
- `src/icml_2026_repro/future_adaptation_evidence.py`: one deterministic official future-edit case per release trajectory and a fixed residual-patch mechanism case for Claim 3.
- `tests/test_future_adaptation_evidence.py`: target preservation, mechanism counts, malformed trajectory, and determinism tests for Claim 3.
- `src/icml_2026_repro/cli.py`: serialize the six judge-aligned claim artifacts and environment guide while retaining existing safe artifact writes.
- `tests/test_cli.py`: assert the new bundle contract and portable serialization.
- `README.md`: explain the judge-aligned commands, observed scope, and known limitations.
- `repro_bundle/*`: regenerated portable evidence consumed by the logbook.
- `.trackio/logbook/pages/*/page.md`: six claim pages plus revised executive summary and conclusion.
- `.trackio/logbook/logbook.json`, `.trackio/metadata.json`: generated Trackio navigation and challenge metadata.
- `.trackio/logbook/assets/poster_embed.html`: self-contained six-claim poster.
- `.openresearch/artifacts/repro_bundle/*`: text-readable copy of the formal child experiment's evidence bundle.

---

### Task 1: OpenResearch Baseline and Child Experiment

**Files:**
- Create external GitHub repository: `will-rice/icml-2026-repro`
- Create local OpenResearch project: `NAPE ICML 2026 Reproduction`
- Create OpenResearch baseline branch: `orx/current-judged-submission`
- Create OpenResearch child branch: `orx/judge-aligned-claims-1-3`

**Interfaces:**
- Consumes: current clean commit `2651386`, installed `orx` 0.1.69, GitHub authentication, and local `orx up` HTTP API.
- Produces: shell variables `PROJECT_ID`, `BASELINE_ID`, and `IMPROVEMENT_ID`; an immutable baseline experiment; and a child experiment branch used by Tasks 2-5.

- [ ] **Step 1: Complete authentication and publish the template-derived repository**

Run `gh auth login -h github.com -p https -w` and verify:

```bash
gh auth status
gh api user --jq .login
```

Expected: GitHub reports account `will-rice`. Then run:

```bash
gh repo create will-rice/icml-2026-repro --public --source=. --remote=origin --push
git remote get-url origin
```

Expected: `origin` is `https://github.com/will-rice/icml-2026-repro.git` and public `main` contains commit `2651386`. This repository preserves the current template-derived history; do not seed a blank replacement.

- [ ] **Step 2: Start local OpenResearch and create the project**

Run `orx up --no-browser --no-agent` in a retained terminal session. Verify `curl --fail http://127.0.0.1:4791/api/health` succeeds. Then run:

```bash
PROJECT_ID=$(curl --fail --silent --show-error \
  -H 'content-type: application/json' \
  --data '{"name":"NAPE ICML 2026 Reproduction","githubOwner":"will-rice","githubRepo":"icml-2026-repro","baselineBranch":"main","runCommand":"uv sync && uv run nape-repro && mkdir -p .openresearch/artifacts && cp -R repro_bundle .openresearch/artifacts/repro_bundle && uv run pytest -q","paperId":"2606.13802"}' \
  http://127.0.0.1:4791/api/projects | jq -er '.project.id')
orx projects
```

Expected: `orx projects` lists `NAPE ICML 2026 Reproduction` under `Local (orx up)` with ID equal to `$PROJECT_ID`.

- [ ] **Step 3: Create and run the current judged baseline**

```bash
BASELINE_ID=$(orx create-experiment "$PROJECT_ID" \
  --title "Current judged submission" \
  --description "Published two-card audit at 1/12 before judge-aligned evidence expansion." \
  --run-command "uv sync && uv run nape-repro && mkdir -p .openresearch/artifacts && cp -R repro_bundle .openresearch/artifacts/repro_bundle && uv run pytest -q" \
  | sed -n 's/^  id: *//p')
orx exp run "$BASELINE_ID" --backend local
orx exp wait "$BASELINE_ID" --timeout 1800
```

Capture and inspect the exact run ID:

```bash
BASELINE_RUN_ID=$(orx exp status "$BASELINE_ID" | sed -n 's/^  last run: \([^ ]*\).*/\1/p')
orx logs "$BASELINE_RUN_ID"
orx artifacts "$BASELINE_RUN_ID"
```

Expected: the run finishes successfully, logs show bundle generation and passing root tests, and artifacts list text files under `repro_bundle/`.

- [ ] **Step 4: Create the score-improvement child**

```bash
IMPROVEMENT_ID=$(orx create-experiment "$PROJECT_ID" \
  --title "Judge-aligned Claims 1-3" \
  --parent "$BASELINE_ID" \
  --description "Expand released-artifact evidence to the six claims used by the Logbook Judge; rerun Claims 1-3 without paid model APIs." \
  | sed -n 's/^  id: *//p')
orx experiments "$PROJECT_ID"
```

Expected: the tree has one root and one child; the child inherits the exact baseline run command and uses branch `orx/judge-aligned-claims-1-3`.

- [ ] **Step 5: Switch implementation work to the child branch checkout**

Use `orx project "$PROJECT_ID"` and `orx exp status "$IMPROVEMENT_ID"` to locate the local OpenResearch cache checkout. Verify:

```bash
git branch --show-current
git status --short
```

Expected: branch `orx/judge-aligned-claims-1-3` and an empty status. Copy the already committed design and this implementation plan into the child only if the branch point predates their commits, then commit them before Task 2. All subsequent code tasks run in this checkout.

### Task 2: Benchmark and Construction Evidence

**Files:**
- Create: `src/icml_2026_repro/benchmark_evidence.py`
- Create: `tests/test_benchmark_evidence.py`
- Reuse: `src/icml_2026_repro/audit.py`

**Interfaces:**
- Consumes: `GITHUB_REPOSITORY`, `GITHUB_REVISION`, `REPOSITORY_ROOT`, `read_git_head(repository_path: Path) -> str`, and `read_git_worktree_status(repository_path: Path) -> str`.
- Produces: `audit_benchmark_release(nape_path: Path) -> dict[str, object]` and `build_benchmark_evidence() -> dict[str, object]`.

- [ ] **Step 1: Write failing tests for the exact pinned release**

```python
from pathlib import Path

import pytest

from icml_2026_repro.benchmark_evidence import audit_benchmark_release


NAPE_PATH = Path(__file__).resolve().parents[1] / "external" / "NAPE"


def test_audit_benchmark_release_reproduces_paper_statistics():
    evidence = audit_benchmark_release(NAPE_PATH)

    assert evidence["observed"] == {
        "trajectories": 52,
        "operations": 11907,
        "minimum_sequence_length": 35,
        "maximum_sequence_length": 821,
        "mean_sequence_length": pytest.approx(228.98076923076923),
        "paper_rounded_mean": 229,
        "median_sequence_length": 164,
    }
    assert evidence["artifact_audit"]["matched_trajectory_ids"] == 52
    assert evidence["artifact_audit"]["required_files_per_trajectory"] == [
        "operations.txt",
        "predictable_state.json",
        "sheet_image.png",
        "spreadsheet.xlsx",
    ]
    assert evidence["construction_pipeline"]["human_annotation"]["evidence_scope"] == (
        "release_provenance_only"
    )
    assert evidence["verdict"] == "reproduced"
```

Add fixtures that copy two minimal trajectory JSON files and raw directories into `tmp_path`; assert duplicate JSON `name` values, a filename/name mismatch, unequal trajectory/raw ID sets, missing required files, non-string operations, and empty releases each raise a concise `ValueError` containing the offending condition.

- [ ] **Step 2: Run the focused tests and confirm the module is missing**

Run: `uv run pytest tests/test_benchmark_evidence.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'icml_2026_repro.benchmark_evidence'`.

- [ ] **Step 3: Implement the benchmark audit**

Create constants for the paper values, required raw filenames, and these executable source paths:

```python
PIPELINE_SOURCES = {
    "symbolic_sequencing": "src/next_action_pred_eval/generation/sequencing/engine.py",
    "region_annotation": "src/next_action_pred_eval/generation/regions/analyzer.py",
    "llm_refinement": "src/next_action_pred_eval/generation/refinement/pipeline.py",
}
```

In `audit_benchmark_release`, parse every `data/trajectories/*.json`, require `name == path.stem`, unique names, and a non-empty `list[str]` `operations`. Use `statistics.mean` and `statistics.median`, compare trajectory IDs exactly with directories under `data/raw`, and require all four raw files. Require each `PIPELINE_SOURCES` path to be a regular file. Return:

```python
{
    "claim": "NAPE contains 52 trajectories and 11,907 actions with sequence lengths 35-821 (mean 229, median 164), produced by symbolic sequencing, LLM refinement, and human annotation.",
    "source_revision": GITHUB_REVISION,
    "input_paths": ["external/NAPE/data/trajectories", "external/NAPE/data/raw"],
    "counting_definition": "One trajectory is one JSON file; one action is one string in its operations array.",
    "observed": observed,
    "artifact_audit": artifact_audit,
    "construction_pipeline": {
        "symbolic_sequencing": {"source_path": PIPELINE_SOURCES["symbolic_sequencing"], "evidence_scope": "executable_source"},
        "region_annotation": {"source_path": PIPELINE_SOURCES["region_annotation"], "evidence_scope": "executable_source"},
        "llm_refinement": {"source_path": PIPELINE_SOURCES["llm_refinement"], "evidence_scope": "executable_source"},
        "human_annotation": {"source_path": None, "evidence_scope": "release_provenance_only"},
    },
    "evidence_scope": "released_artifact_recomputation_and_source_audit",
    "verdict": "reproduced",
}
```

`build_benchmark_evidence` must verify exact Git HEAD and full NAPE worktree cleanliness before calling `audit_benchmark_release`; share the existing worktree-status helper by making it public as `read_git_worktree_status` in `audit.py` and updating `online_trace.py` to import it.

- [ ] **Step 4: Run Claim 1 tests and regressions**

Run: `uv run pytest tests/test_benchmark_evidence.py tests/test_audit.py tests/test_online_trace.py -q`

Expected: all tests pass and the release test reports 52 trajectories and 11,907 operations.

- [ ] **Step 5: Commit Claim 1 evidence**

```bash
git add src/icml_2026_repro/audit.py src/icml_2026_repro/benchmark_evidence.py src/icml_2026_repro/online_trace.py tests/test_benchmark_evidence.py
git commit -m "feat: reproduce NAPE benchmark construction evidence"
```

### Task 3: Released Predictability Aggregate

**Files:**
- Create: `src/icml_2026_repro/predictability_evidence.py`
- Create: `tests/test_predictability_evidence.py`

**Interfaces:**
- Consumes: pinned raw directories under `external/NAPE/data/raw`, `GITHUB_REVISION`, and the clean-checkout helpers from `audit.py`.
- Produces: `aggregate_predictability(raw_path: Path) -> dict[str, object]` and `build_predictability_evidence() -> dict[str, object]`.

- [ ] **Step 1: Write failing aggregate and validation tests**

```python
from pathlib import Path

import pytest

from icml_2026_repro.predictability_evidence import aggregate_predictability


RAW_PATH = Path(__file__).resolve().parents[1] / "external" / "NAPE" / "data" / "raw"


def test_aggregate_predictability_reproduces_released_ceiling():
    evidence = aggregate_predictability(RAW_PATH)

    assert evidence["observed"]["trajectories"] == 52
    assert evidence["observed"]["predictable_properties"] == 126940
    assert evidence["observed"]["final_state_properties"] == 186574
    assert evidence["observed"]["weighted_coverage_pct"] == pytest.approx(
        68.03734711160227
    )
    assert evidence["observed"]["mean_coverage_pct"] == pytest.approx(65.99230769230769)
    assert evidence["observed"]["median_coverage_pct"] == pytest.approx(66.34)
    assert evidence["observed"]["trajectories_above_50_pct"] == 44
    assert evidence["evidence_scope"] == "released_oracle_output_recomputation"
```

Create one-file `tmp_path` fixtures and assert failures for a missing `predictable_state.json`, a non-object root, mismatched `trajectory_name`, boolean/non-integer counts, zero or negative final-state size, non-finite or non-numeric coverage, coverage outside `[0, 100]`, and `coverage_pct` inconsistent with `100 * predictable_count / final_state_size` beyond `1e-2` percentage points.

- [ ] **Step 2: Run the focused tests and confirm the module is missing**

Run: `uv run pytest tests/test_predictability_evidence.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'icml_2026_repro.predictability_evidence'`.

- [ ] **Step 3: Implement strict released-output aggregation**

Read sorted direct child directories only. Validate exact keys needed from each JSON object: `trajectory_name: str`, `predictable_count: int`, `final_state_size: int`, `coverage_pct: int | float`, and `predictable_properties: dict[str, dict[str, list[str]]]` mapping sheet names to cell addresses to predictable property names. Reject booleans as numeric values, require unique matching IDs, require the nested property-entry total to equal `predictable_count`, and use `math.isfinite`, `math.isclose(..., abs_tol=1e-2)`, `statistics.mean`, and `statistics.median`.

Return the exact report shape:

```python
{
    "claim": "Approximately 68% of final spreadsheet properties are empirically predictable by the released frontier-model oracle pipeline.",
    "source_revision": GITHUB_REVISION,
    "input_path": "external/NAPE/data/raw/*/predictable_state.json",
    "counting_definition": "Weighted coverage is 100 times the sum of predictable_count divided by the sum of final_state_size.",
    "observed": {
        "trajectories": len(rows),
        "predictable_properties": total_predictable,
        "final_state_properties": total_final,
        "weighted_coverage_pct": 100 * total_predictable / total_final,
        "mean_coverage_pct": mean(coverages),
        "median_coverage_pct": median(coverages),
        "trajectories_above_50_pct": sum(value > 50 for value in coverages),
    },
    "evidence_scope": "released_oracle_output_recomputation",
    "limitation": "The original paid frontier-model oracle calls were not rerun.",
    "verdict": "reproduced_from_released_outputs",
}
```

`build_predictability_evidence` must reject a mismatched or dirty NAPE checkout, then call `aggregate_predictability`.

- [ ] **Step 4: Run Claim 2 tests twice to establish determinism**

Run: `uv run pytest tests/test_predictability_evidence.py -q && uv run pytest tests/test_predictability_evidence.py -q`

Expected: both runs pass with identical exact aggregates.

- [ ] **Step 5: Commit Claim 2 evidence**

```bash
git add src/icml_2026_repro/predictability_evidence.py tests/test_predictability_evidence.py
git commit -m "feat: recompute released NAPE predictability ceiling"
```

### Task 4: Release-Wide Future Adaptation Evidence

**Files:**
- Create: `src/icml_2026_repro/future_adaptation_evidence.py`
- Create: `tests/test_future_adaptation_evidence.py`
- Modify: `src/icml_2026_repro/online_trace.py`

**Interfaces:**
- Consumes: `build_claim_2_evidence(output_dir: Path) -> dict[str, object]` as the supporting orchestrator trace and official `StateBuilder`, `symbolic_to_operations`, `StepEvaluator`, `FutureEditsManager`, and `StateComparator` APIs.
- Produces: `audit_future_adaptation(nape_path: Path) -> dict[str, object]`, `_audit_fixture_trajectory_directory(trajectory_path: Path) -> dict[str, object]`, `audit_residual_patch_fixture() -> dict[str, object]`, and `build_future_adaptation_evidence(output_dir: Path) -> dict[str, object]`.

- [ ] **Step 1: Write failing release-wide and fixture tests**

```python
from pathlib import Path

from icml_2026_repro.future_adaptation_evidence import (
    audit_future_adaptation,
    audit_residual_patch_fixture,
)


NAPE_PATH = Path(__file__).resolve().parents[1] / "external" / "NAPE"


def test_audit_future_adaptation_preserves_all_release_targets():
    evidence = audit_future_adaptation(NAPE_PATH)

    assert evidence["summary"] == {
        "trajectories": 52,
        "removal_cases": 50,
        "inverse_insertion_cases": 52,
        "residual_patch_cases": 0,
        "target_preserved_cases": 52,
        "skipped_cases": 0,
    }
    assert all(row["target_preserved"] for row in evidence["trajectories"])


def test_residual_patch_fixture_synthesizes_missing_operation():
    evidence = audit_residual_patch_fixture()

    assert evidence["ground_truth"] == [
        "VALUE | Sheet1!A1 | 1",
        "VALUE | Sheet1!A2 | 1",
        "VALUE | Sheet1!A2 | 1",
        "VALUE | Sheet1!A2 | 1",
    ]
    assert evidence["prediction"] == [
        "VALUE | Sheet1!A1 | 1",
        "VALUE | Sheet1!A1 | 2",
    ]
    assert evidence["missing_ops_count"] == 1
    assert evidence["target_preserved"] is True
```

Add tests asserting malformed operations raise with the trajectory filename, an empty release fails, a monkeypatched unequal final-state comparison raises, and two calls return equal dictionaries.

- [ ] **Step 2: Run the focused tests and confirm the module is missing**

Run: `uv run pytest tests/test_future_adaptation_evidence.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'icml_2026_repro.future_adaptation_evidence'`.

- [ ] **Step 3: Implement the 52-trajectory mutation sweep**

For each sorted trajectory, parse all operations, require at least two operations, build the state after operation 0 and the target state after all operations, and predict the exact next operation followed by this deterministic false positive on the first operation's sheet:

```python
predicted = [
    operations[1],
    *symbolic_to_operations(
        [f"VALUE | {operations[0].cell_range.sheet}!ZZ999 | 987654321"]
    ),
]
```

Evaluate with `ground_truth_operations=operations[1:3]`, `all_future_operations=operations[1:]`, cached initial and target states, and `skip_ops_diff=True`. Call `simulate_future_edits(current_gt=operations, start_idx=1, end_idx=1, ...)`, rebuild the returned sequence, and compare it with the target via `StateComparator(ignore_defaults=True).compare(..., skip_ops_diff=True)`. A target is preserved only when false positives, false negatives, and mismatches are all zero; otherwise raise.

Record per-trajectory `name`, counts from `operations_removed`, `inverse_ops_added`, and `metadata["missing_ops_count"]`, plus `target_preserved`. Aggregate exactly the six summary keys asserted above. Keep skipped-case support only for an explicitly unsupported operation that raises during parsing/application, recording its name and reason; the pinned release must have zero skips.

- [ ] **Step 4: Implement and compose the residual fixture and orchestrator trace**

Use the exact four-operation ground truth and two-operation prediction asserted in Step 1. Run the same official evaluator/simulator flow at `start_idx=1`, `end_idx=1`; require `metadata["missing_ops_count"] == 1` and final target equality.

`build_future_adaptation_evidence(output_dir)` must pass the verified NAPE root to the release audit, call the residual audit, call the renamed `build_online_trace_evidence(output_dir)` from `online_trace.py`, and return:

```python
{
    "claim": "NAPE's future-edit implementation removes satisfied future operations, prepends inverses for false positives, and patches residual differences to preserve the target state after accepted predictions.",
    "source_revision": GITHUB_REVISION,
    "verified_input_root": "external/NAPE",
    "input_path": "external/NAPE/data/trajectories/*.json",
    "release_case_scope": "One deterministic adaptation case per each of the 52 released trajectories; this is not a full per-action model rollout.",
    "release_sweep": release_evidence,
    "residual_patch_fixture": residual_evidence,
    "orchestrator_trace": trace_evidence,
    "evidence_scope": "official_evaluator_one_case_per_release_trajectory_and_fixture",
    "verdict": "reproduced",
}
```

Rename the misleading old `build_claim_2_evidence` to `build_online_trace_evidence` and update imports/tests. Do not change its trace behavior.

- [ ] **Step 5: Run Claim 3 and upstream future-edit tests**

Run: `uv run pytest tests/test_future_adaptation_evidence.py tests/test_online_trace.py external/NAPE/tests/test_future_edits.py -q`

Expected: all tests pass; the release summary is `52/50/52/0/52/0`, and the fixture reports one residual patch.

- [ ] **Step 6: Commit Claim 3 evidence**

```bash
git add src/icml_2026_repro/future_adaptation_evidence.py src/icml_2026_repro/online_trace.py tests/test_future_adaptation_evidence.py tests/test_online_trace.py
git commit -m "feat: exercise NAPE future adaptation across the release"
```

### Task 5: Judge-Aligned Evidence Bundle

**Files:**
- Modify: `src/icml_2026_repro/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `README.md`
- Regenerate: `repro_bundle/claim_1_benchmark.json`
- Regenerate: `repro_bundle/claim_2_predictability.json`
- Regenerate: `repro_bundle/claim_3_future_adaptation.json`
- Regenerate: `repro_bundle/claims_4_6_status.json`
- Regenerate: `repro_bundle/environment.json`
- Regenerate: `repro_bundle/README.md`
- Remove: `repro_bundle/claim_1_audit.json`
- Remove: `repro_bundle/claim_2_trace.json`

**Interfaces:**
- Consumes: `build_benchmark_evidence()`, `build_predictability_evidence()`, and `build_future_adaptation_evidence(output_dir: Path)`.
- Produces: `build_bundle(output_dir: Path = Path("repro_bundle")) -> Path` with exactly six portable artifacts.

- [ ] **Step 1: Replace CLI bundle tests with the six-artifact contract**

Define:

```python
expected_files = {
    "claim_1_benchmark.json",
    "claim_2_predictability.json",
    "claim_3_future_adaptation.json",
    "claims_4_6_status.json",
    "environment.json",
    "README.md",
}
```

Monkeypatch all three builders, assert each JSON artifact equals its builder result, assert no output-directory absolute path occurs in any artifact, and assert the status artifact is exactly:

```python
{
    "claim_4": {"status": "not replicated", "reason": "Named model outputs and paid API budget were not available."},
    "claim_5": {"status": "not replicated", "reason": "Named model outputs and paid API budget were not available."},
    "claim_6": {"status": "not replicated", "reason": "Named model outputs and paid API budget were not available."},
}
```

Update every existing symlink, hard-link, staging-failure, and macOS alias parametrization to use `BUNDLE_ARTIFACT_NAMES` rather than old literal filenames. Keep the existing race-safe writer tests intact.

- [ ] **Step 2: Run CLI tests and observe the old artifact mismatch**

Run: `uv run pytest tests/test_cli.py -q`

Expected: failures show old `claim_1_audit.json` and `claim_2_trace.json` instead of the new bundle.

- [ ] **Step 3: Wire the new builders and documentation**

Set `BUNDLE_ARTIFACT_NAMES` to the six exact filenames above. Build all three reports before writing artifacts. Keep the existing temporary directory for Claim 3 recorder files. Add `_unreplicated_model_claims()` returning the exact status object asserted in Step 1.

Update `_environment_manifest()` to schema `icml-2026-repro-bundle/v2`, include `uv run pytest tests/test_benchmark_evidence.py tests/test_predictability_evidence.py tests/test_future_adaptation_evidence.py -q`, and retain root/upstream/pre-commit/ty commands. Rewrite `_bundle_readme()` to define every artifact, state `52`, `11,907`, `35-821`, `229`, `164`, `68.04%`, `50/52` removal cases, `52/52` inverse and target-preservation cases, and the one residual fixture. State that Claims 4-6 were not replicated and that Claim 2 audits released oracle outputs.

Update root `README.md` with the same scope, `uv sync`, `uv run nape-repro`, focused test commands, official validation command, and canonical Space URL.

- [ ] **Step 4: Run tests and regenerate twice**

Run:

```bash
uv run pytest tests/test_cli.py -q
uv run nape-repro
cp -R repro_bundle /private/tmp/nape-repro-first
uv run nape-repro
diff -ru /private/tmp/nape-repro-first repro_bundle
```

Expected: tests pass, both CLI runs succeed, and `diff` prints nothing. Environment platform values are stable on the same host.

- [ ] **Step 5: Run all local and upstream quality gates**

Run:

```bash
uv run pytest -q
uv run pytest external/NAPE/tests -q
uv run pre-commit run -a
uv run ty check src tests
git diff --check
```

Expected: all root tests and all 244 upstream tests pass; every pre-commit hook passes; ty reports no diagnostics; `git diff --check` prints nothing.

- [ ] **Step 6: Commit the bundle revision**

```bash
git add src/icml_2026_repro/cli.py tests/test_cli.py README.md repro_bundle
git commit -m "feat: publish judge-aligned NAPE evidence bundle"
```

### Task 6: Six-Claim Trackio Logbook

**Files:**
- Modify: `.trackio/logbook/pages/executive-summary/page.md`
- Create: `.trackio/logbook/pages/claim-1-benchmark-and-construction/page.md`
- Create: `.trackio/logbook/pages/claim-2-empirical-predictability/page.md`
- Create: `.trackio/logbook/pages/claim-3-online-future-adaptation/page.md`
- Create: `.trackio/logbook/pages/claim-4-gpt-5-r-results/page.md`
- Create: `.trackio/logbook/pages/claim-5-always-accept-ablation/page.md`
- Create: `.trackio/logbook/pages/claim-6-stride-ablation/page.md`
- Modify: `.trackio/logbook/pages/conclusion/page.md`
- Modify: `.trackio/logbook/logbook.json`
- Modify: `.trackio/metadata.json`
- Modify: `.trackio/logbook/assets/poster_embed.html`
- Remove: the two obsolete challenge-card claim page directories.

**Interfaces:**
- Consumes: the six files in `repro_bundle/` and the exact focused/full validation commands.
- Produces: a validator-clean static Trackio logbook with one page corresponding to each judge claim.

- [ ] **Step 1: Record fresh command evidence through Trackio**

Run these exact Trackio commands:

```bash
uv run trackio logbook run --page claim-1-benchmark-and-construction --title "Recompute all benchmark statistics" -- uv run pytest tests/test_benchmark_evidence.py -q
uv run trackio logbook run --page claim-2-empirical-predictability --title "Aggregate all released oracle outputs" -- uv run pytest tests/test_predictability_evidence.py -q
uv run trackio logbook run --page claim-3-online-future-adaptation --title "Exercise official adaptation across all trajectories" -- uv run pytest tests/test_future_adaptation_evidence.py tests/test_online_trace.py -q
uv run trackio logbook run --page conclusion --title "Generate the judge-aligned evidence bundle" -- uv run nape-repro
uv run trackio logbook run --page conclusion --title "Run all upstream NAPE tests" -- uv run pytest external/NAPE/tests -q
uv run trackio logbook run --page conclusion --title "Run repository quality gates" -- uv run pre-commit run -a
```

Expected: every cell records exit code 0 and fresh wall time. Do not paste terminal output into hand-authored markdown.

- [ ] **Step 2: Attach the regenerated bundle**

Run a short `uv run python -c` command that calls `trackio.log_artifact("repro_bundle", name="repro-bundle-v2", type="dataset")`, then add it to the Conclusion using `uv run trackio logbook cell artifact ... --type dataset`.

Expected: the artifact cell links to all six bundle files and contains no machine-specific absolute paths.

- [ ] **Step 3: Author outcome-first six-claim pages**

Use `uv run trackio logbook cell markdown` for prose and generated Trackio commands for page/navigation changes. Include these exact outcomes:

- Claim 1: `REPRODUCED` for released benchmark counts/distribution and executable source audit; state human annotation is provenance-only.
- Claim 2: `REPRODUCED FROM RELEASED OUTPUTS`; report `126,940 / 186,574 = 68.04%`, mean `65.99%`, median `66.34%`, and `44/52`; state the frontier oracle calls were not rerun.
- Claim 3: `REPRODUCED`; report 52 trajectories, 50 removal cases, 52 inverse cases, 52 target-preserving cases, and one fixed residual-patch fixture, with the small orchestrator trace used only for ordering and accept/reject evidence.
- Claim 4: `NOT REPLICATED`; name the paper values `32.7/29.4/41.6` only as the claim under test and state no verdict.
- Claim 5: `NOT REPLICATED`; name `-19.2 UAS` only as the claim under test and state no verdict.
- Claim 6: `NOT REPLICATED`; name `27.4 to 10.6` only as the claim under test and state no verdict.

The executive summary and conclusion must show current public score `1/12`, describe the new evidence as a rejudging submission rather than a guaranteed score, state zero paid API cost, and avoid claiming Claims 4-6. Remove the obsolete 58/13K challenge-card page from navigation.

- [ ] **Step 4: Replace and inspect the poster**

Create a self-contained `poster_embed.html` with a compact six-row evidence table, top-line `Current judged score: 1/12`, full-release statistics, CPU/zero-paid-API method, and limitations. Use no external scripts or network assets. Render it in the pinned Trackio figure cell and verify all text fits at desktop and mobile widths.

- [ ] **Step 5: Validate structure and local rendering**

Run:

```bash
uv run python scripts/validate_icml_logbook.py --space-id wrice/repro-a-benchmark-and-framework-for-evaluating-next-action-predictions-in-spreadsheets
uv run trackio logbook serve --no-browser
```

Expected: validator exits 0 with required `icml2026-repro` and `paper-NvPgRwURDC` metadata; the server starts without errors. Request the root and all eight hash-routed pages, confirm HTTP 200, claim labels, artifact link, command cells, and poster, then stop the server.

- [ ] **Step 6: Commit the validated logbook**

```bash
git add .trackio
git commit -m "docs: align NAPE logbook with judged paper claims"
```

### Task 7: OpenResearch Run, Promotion, Publication, and Rejudging

**Files:**
- Modify: `README.md` with the OpenResearch provenance table.
- Promote the validated child commits to GitHub `main`.
- The canonical Hugging Face Space and public verdict dataset are external outputs.

**Interfaces:**
- Consumes: clean committed `orx/judge-aligned-claims-1-3`, validated `.trackio`, regenerated bundle, and OpenResearch IDs from Task 1.
- Produces: a successful formal OpenResearch child run, updated GitHub `main`, updated canonical Space revision, and a checked rejudge result keyed to that revision.

- [ ] **Step 1: Run the final verification suite from a clean commit**

Run:

```bash
git status --short
uv run pytest -q
uv run pytest external/NAPE/tests -q
uv run pre-commit run -a
uv run ty check src tests
uv run nape-repro
uv run python scripts/validate_icml_logbook.py --space-id wrice/repro-a-benchmark-and-framework-for-evaluating-next-action-predictions-in-spreadsheets
git status --short
```

Expected: both status commands are empty; all tests and validation pass; regeneration changes no tracked bundle file.

- [ ] **Step 2: Run and analyze the formal OpenResearch child experiment**

```bash
git push origin orx/judge-aligned-claims-1-3
orx exp run "$IMPROVEMENT_ID" --backend local
orx exp wait "$IMPROVEMENT_ID" --timeout 1800
orx exp status "$IMPROVEMENT_ID"
orx runs "$PROJECT_ID" --experiment "$IMPROVEMENT_ID"
```

Expected: the child run finishes successfully from the pushed child branch with the inherited exact command. Read the run ID printed by `orx runs`, then run `orx logs`, `orx artifacts`, and `orx artifact` for that ID. Confirm the log records passing root tests and the artifact tree contains all six `repro_bundle` files.

Write this assessment with `orx exp desc "$IMPROVEMENT_ID" --stdin`: Claims 1 and 3 reproduced with official released code/artifacts, Claim 2 reproduced from released oracle outputs with original paid calls not rerun, Claims 4-6 not attempted, local CPU, zero paid API cost. Include exact paper/observed numbers and no claim beyond the tested setup.

- [ ] **Step 3: Add reader-facing provenance and promote the child**

Add a compact `Experiment log` table near the top of `README.md`. Link `orx/current-judged-submission` and `orx/judge-aligned-claims-1-3` on GitHub; include purpose, the exact command copied verbatim from `orx exp status`, outcome, and local CPU compute. Do not publish raw OpenResearch project, experiment, or run IDs.

Commit and rerun `uv run pre-commit run -a`. Push the child, then fast-forward `main` to the validated child tip:

```bash
git push origin orx/judge-aligned-claims-1-3
git checkout main
git merge --ff-only orx/judge-aligned-claims-1-3
git push origin main
```

Expected: no merge commit, GitHub `main` and the child branch point to the same validated commit, and both descriptive branch links work publicly.

- [ ] **Step 4: Publish over the canonical Space**

Run:

```bash
uv run trackio logbook publish wrice/repro-a-benchmark-and-framework-for-evaluating-next-action-predictions-in-spreadsheets
```

Expected: publication succeeds without creating a second Space and reports the canonical Space URL.

- [ ] **Step 5: Verify Space revision and runtime**

Use `huggingface_hub.HfApi().space_info(...)` to record the new Space SHA and runtime stage. Poll only until the Space is `RUNNING` or reports a terminal build error.

Expected: the SHA differs from the previously judged revision, hardware remains CPU, and runtime reaches `RUNNING`.

- [ ] **Step 6: Smoke-test public content**

Request the public Space and its static assets. Confirm HTTP 200 plus the six page names, `68.04%`, `50/52`, `52/52`, `NOT REPLICATED`, and `repro-bundle-v2`.

Expected: public content matches the local validated logbook and references the same canonical Space.

- [ ] **Step 7: Check asynchronous judge output by Space SHA**

Query the challenge's public verdict dataset and select the row whose submitted Space SHA equals the SHA captured in Step 5. If no row exists yet, report `rejudge pending` and poll later without republishing.

Expected when complete: a new verdict row tied to the new SHA. Report its per-claim labels and total score exactly; do not infer success from publication alone.

- [ ] **Step 8: Decide separately whether to run Claims 4-6**

Only if the released-artifact revision remains insufficient and the user approves model/API cost, write a separate design and implementation plan for model reruns. Extend the existing OpenResearch experiment tree with one child per model or ablation, run those children on approved Hugging Face or remote compute, and leave the deterministic Claims 1-3 child unchanged.
