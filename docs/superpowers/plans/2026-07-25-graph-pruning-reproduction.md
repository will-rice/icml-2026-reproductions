# Graph Dataset Pruning Formal-Evidence Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, validate, publish, and submit a deterministic CPU formal-evidence reproduction of the two admitted claims from `arxiv:2606.12913v2` without presenting paper-reported experiments as reproduced measurements.

**Architecture:** An independent Python project under the authoritative long submission slug transcribes and verifies the pinned PDF, then evaluates seven explicitly named model variants through independent exact-arithmetic objective, diminishing-returns, shift, greedy, Algorithm 1, and proof-ledger oracles. A deterministic runner enforces each bounded search ceiling and emits schema-validated canonical JSON, witness fixtures, a report, poster, and CPU Hugging Face Space; external coordinator state is mutated only after the local validation, deployment, and live-refresh gates pass.

**Tech Stack:** Python 3.11+, standard-library `fractions`, `itertools`, `json`, and `hashlib`; `jsonschema`; pytest; Gradio; Hugging Face Hub CLI; repository `icml-repro-loop` state CLI.

## Global Constraints

- Attempt ID is exactly `e485c086-6fa5-4ff6-a3c3-1f31c79bbae6`; challenge paper ID is exactly `a3GdvuPItd`.
- The authoritative submission slug is exactly `selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration`; do not abbreviate it in paths, package metadata, Space metadata, or evidence.
- Target claim 1 is exactly `The paper casts dataset pruning as a graph problem with node weights for intrinsic importance and edge weights for extrinsic diversity/interaction, yielding a Maximum Weight Clique formulation (Section 3.3).`.
- Target claim 2 is exactly `Under mild conditions, the unified objective becomes submodular and admits a greedy approximation guarantee (Section 3.6; Appendix F).`.
- The two `target_claims` strings are immutable, ordered identifiers; reject missing, additional, reordered, or rewritten text before any coordinator mutation.
- Pin the source to `arxiv:2606.12913v2`, `https://arxiv.org/pdf/2606.12913v2`, byte count `683737`, SHA-256 `26ce80e8d347340e0055f2bcf061b6b3e29489fc68a85b8d5711e12cc9da5090`, and CC BY-NC-SA 4.0.
- Retain the approved PDF acquisition command verbatim in provenance and canonical evidence; the PDF is a verified input and is not committed.
- Root-authored software and JSON Schema are MIT; transcriptions, evidence, README, poster, and explanatory Space assets are CC BY-NC-SA 4.0, with boundaries stated in `NOTICE.md`.
- Never copy paper figures, tables, experimental images, or unreleased code.
- Keep `paper_mwcp`, `paper_samplewise_literal`, `single_counted_pairwise`, `half_corrected_samplewise`, `appendix_inline_shift_literal`, `appendix_eq26_score`, and `modular_shift_candidate` distinct in every interface and record.
- Keep `paper_algorithm1_literal`, `paper_eq7_score_greedy`, and `true_marginal_greedy` distinct; literal Algorithm 1 stops on its first undefined read and is never silently repaired.
- `appendix_inline_shift_literal` means exactly `f_lit(S) + alpha * eta * |S|^2`; it is not a modular shift and must carry the minimal two-element `1`-then-`3` diminishing-returns witness.
- The Appendix E witness must be linked by the diminishing-returns result, greedy-guarantee premise result, and every applicable Appendix F proof-ledger row; no repaired variant may use its identifier.
- Each oracle has an independent implementation and must not call another oracle's objective or marginal implementation.
- Use `fractions.Fraction` for all non-integral truth decisions; never use floating point to decide an identity, premise, witness, or approximation result.
- Search only the approved domains; refuse undeclared expansion. The aggregate ceiling is exactly `1_177_835` case, path, subset, marginal, or ledger-row evaluations.
- Label evidence only `symbolic`, `exhaustive_finite`, or `non_exhaustive`; an early-stopped finite search is not exhaustive.
- CPU only, no model training, no paid API use, no network during evidence computation, and total evidence runtime under 30 minutes.
- Paper-reported CIFAR-10/100, ImageNet-1k, segmentation, detection, accuracy, training-time, and acceleration results are context-only and must be marked `unavailable`, never reproduced.
- Write and run a failing test before each production change. Record every red command, timestamp, test ID, and expected missing behavior in `evidence/tdd-log.jsonl`.
- Do not modify, run, test, or format `submissions/nape/`; root validation relies on the repository's existing exclusion.
- Do not copy or mutate coordinator index, attempt, lease, transaction, or snapshot shards in this branch. Lifecycle work runs in `/home/will/projects/icml-2026-reproductions/.worktrees/five-paper-scheduler` with a freshly read current owner and fencing token.
- Never commit credentials, cookies, unredacted environment dumps, downloaded PDF bytes, caches, or mutable source URLs.
- Update `docs/HANDOFF.md` after every material lifecycle transition or blocker, but not during implementation tasks 1-8.

---

## Exact File Map

All paths below are relative to the repository root. Set once in shell commands:

```bash
SLUG=selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration
PROJECT="submissions/$SLUG"
```

- `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/pyproject.toml`: isolated Python 3.11 project, package entry point, runtime and test dependencies, pytest configuration.
- `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/uv.lock`: exact resolved dependency lock generated by `uv lock`.
- `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/LICENSE`: MIT license for original executable code and schema.
- `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/LICENSES/CC-BY-NC-SA-4.0.txt`: paper source license legal code.
- `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/NOTICE.md`: seven-author attribution, exact source/revision, adaptation statement, and file-boundary license map.
- `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/paper_transcriptions/manifest.json`: reviewed source records for Eq. (2)--(8), Eq. (10)--(14), Appendix E inline/Eq. (26)--(27), Appendix F Eq. (28)--(38), and Algorithm 1.
- `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/paper_transcriptions/algorithm1.txt`: literal line transcription from PDF page 5.
- `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/src/graph_pruning_repro/provenance.py`: immutable paper identity, acquisition command, digest verification, transcription validation.
- `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/src/graph_pruning_repro/types.py`: frozen exact-arithmetic instance, search, witness, greedy, and ledger record types.
- `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/src/graph_pruning_repro/objectives.py`: seven named objective/score variants and the independently traversed Eq. (3)/Eq. (4)--(5) evaluators.
- `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/src/graph_pruning_repro/equivalence.py`: symbolic coefficient comparison and 26-case finite objective witness search.
- `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/src/graph_pruning_repro/diminishing_returns.py`: direct set-difference oracle, independent closed forms, symmetric/asymmetric bounded controls, and Appendix E witness.
- `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/src/graph_pruning_repro/shifts.py`: monotonicity, Eq. (26)--(27), Appendix-inline, modular-shift, and boundary controls.
- `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/src/graph_pruning_repro/greedy.py`: Eq. (7) score greedy, objective-difference greedy, all-tie paths, exhaustive optimum, and ratio classification.
- `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/src/graph_pruning_repro/algorithm1.py`: line-indexed literal Algorithm 1 state-machine audit only.
- `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/src/graph_pruning_repro/proof_ledger.py`: Appendix F Eq. (28)--(38) premise rows and finite controls.
- `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/src/graph_pruning_repro/minimize.py`: deterministic property-preserving witness minimization and canonical IDs.
- `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/src/graph_pruning_repro/evidence.py`: ceiling-enforcing orchestration, stable serialization, acceptance checks, and artifact hashes.
- `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/src/graph_pruning_repro/render.py`: report/poster generation solely from accepted evidence JSON.
- `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/src/graph_pruning_repro/cli.py`: `recompute`, `validate`, and `render` commands.
- `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/evidence/schema.json`: canonical evidence schema version 1.
- `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/evidence/evidence.json`: canonical deterministic computed artifact.
- `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/evidence/witnesses/*.json`: canonical minimized witnesses generated from evidence.
- `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/evidence/tdd-log.jsonl`: red-phase command ledger.
- `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/README.md`: provenance, exact commands, findings boundaries, unavailable claims, and licensing.
- `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/report.md`: generated human report with no numeric claim absent from `evidence.json`.
- `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/poster.html` and `poster_embed.html`: generated static theorem-audit presentation.
- `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/app.py`: Gradio CPU Space exposing rendering, downloads, and deterministic recomputation.
- `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/tests/`: one focused functional test module per production module plus end-to-end acceptance and rendering tests.

## Pinned Cross-Task Interfaces

```python
from fractions import Fraction
from pathlib import Path
from typing import Literal, Mapping, Sequence

Vertex = str
Edge = tuple[Vertex, Vertex]
ModelVariant = Literal[
    "paper_mwcp",
    "paper_samplewise_literal",
    "single_counted_pairwise",
    "half_corrected_samplewise",
    "appendix_inline_shift_literal",
    "appendix_eq26_score",
    "modular_shift_candidate",
]
GreedyPath = Literal[
    "paper_algorithm1_literal",
    "paper_eq7_score_greedy",
    "true_marginal_greedy",
]

@dataclass(frozen=True)
class Instance:
    vertices: tuple[Vertex, ...]
    vertex_weights: Mapping[Vertex, Fraction]
    interactions: Mapping[tuple[Vertex, Vertex], Fraction]
    alpha: Fraction = Fraction(1)
    eta: Fraction = Fraction(0)

def evaluate_objective(instance: Instance, selected: frozenset[Vertex], model_variant: ModelVariant) -> Fraction: ...
def direct_marginal(instance: Instance, selected: frozenset[Vertex], candidate: Vertex, model_variant: ModelVariant) -> Fraction: ...
def closed_form_marginal(instance: Instance, selected: frozenset[Vertex], candidate: Vertex, model_variant: ModelVariant) -> Fraction: ...
def enumerate_greedy_paths(instance: Instance, budget: int, model_variant: ModelVariant, greedy_path: GreedyPath) -> tuple[tuple[Vertex, ...], ...]: ...
def exhaustive_optima(instance: Instance, budget: int, model_variant: ModelVariant) -> tuple[frozenset[Vertex], ...]: ...
def build_evidence(output_dir: Path, code_revision: str) -> dict[str, object]: ...
def validate_evidence(evidence: Mapping[str, object], schema_path: Path) -> None: ...
```

`appendix_eq26_score` is rejected by `evaluate_objective` because it is a score, not a set function. `paper_algorithm1_literal` is rejected by executable greedy dispatch and routed only to `audit_literal_algorithm1()`.

### Task 1: Isolated Project, Licensing, PDF Provenance, and Transcriptions

**Files:**
- Create: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/pyproject.toml`
- Create: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/uv.lock`
- Create: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/LICENSE`
- Create: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/LICENSES/CC-BY-NC-SA-4.0.txt`
- Create: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/NOTICE.md`
- Create: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/paper_transcriptions/manifest.json`
- Create: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/paper_transcriptions/algorithm1.txt`
- Create: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/src/graph_pruning_repro/__init__.py`
- Create: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/src/graph_pruning_repro/provenance.py`
- Test: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/tests/test_provenance.py`

**Interfaces:**
- Produces: `PAPER`, `TARGET_CLAIMS`, `PDF_ACQUISITION_COMMAND`, `verify_pdf(path: Path) -> None`, and `load_transcriptions(root: Path) -> tuple[dict[str, object], ...]`.
- Produces: manifest records with exact keys `equation`, `pdf_page`, `section`, `normalized_expression`, `source_excerpt_sha256`, and `reviewed_by`.
- Consumes: no production interfaces.

- [ ] **Step 1: Create project metadata and write the failing provenance tests**

Use this project core in `pyproject.toml` and generate the lock with `uv lock --project "$PROJECT"`:

```toml
[project]
name = "graph-pruning-formal-evidence-reproduction"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["gradio>=5.0", "jsonschema>=4.25"]

[project.scripts]
graph-pruning-repro = "graph_pruning_repro.cli:main"

[dependency-groups]
dev = ["pytest>=8.4", "pre-commit>=4.2"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Write `tests/test_provenance.py`:

```python
import hashlib
from pathlib import Path

import pytest

from graph_pruning_repro.provenance import PAPER, TARGET_CLAIMS, load_transcriptions, verify_pdf

def test_pdf_identity_and_digest_rejection(tmp_path: Path) -> None:
    assert PAPER["revision"] == "arxiv:2606.12913v2"
    assert PAPER["pdf_byte_count"] == 683737
    bad = tmp_path / "paper.pdf"
    bad.write_bytes(b"not the pinned PDF")
    with pytest.raises(ValueError, match="pinned PDF byte count"):
        verify_pdf(bad)

def test_transcriptions_are_complete_and_checksummed() -> None:
    records = load_transcriptions(Path(__file__).parents[1])
    equations = {record["equation"] for record in records}
    assert {"2", "3", "4", "5", "6", "7", "8", "10-11", "12-14", "Appendix E inline", "26", "27", "28-38", "Algorithm 1"} <= equations
    for record in records:
        excerpt = record["source_excerpt"].encode()
        assert record["source_excerpt_sha256"] == hashlib.sha256(excerpt).hexdigest()
        assert record["reviewed_by"] == ["codex-graph-pruning-writer", "independent-design-reviewer"]

def test_target_claims_are_exact() -> None:
    assert TARGET_CLAIMS == (
        "The paper casts dataset pruning as a graph problem with node weights for intrinsic importance and edge weights for extrinsic diversity/interaction, yielding a Maximum Weight Clique formulation (Section 3.3).",
        "Under mild conditions, the unified objective becomes submodular and admits a greedy approximation guarantee (Section 3.6; Appendix F).",
    )
```

- [ ] **Step 2: Run the red test and record it**

Run:

```bash
uv run --project "$PROJECT" pytest tests/test_provenance.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'graph_pruning_repro'`. Append that command, UTC timestamp, the three test IDs, and `expected_missing_behavior: provenance module and transcription manifest absent` to `evidence/tdd-log.jsonl`.

- [ ] **Step 3: Add the minimal provenance implementation and exact transcriptions**

Implement the immutable constants and validation:

```python
PAPER = {
    "challenge_id": "a3GdvuPItd",
    "revision": "arxiv:2606.12913v2",
    "source_url": "https://arxiv.org/pdf/2606.12913v2",
    "pdf_byte_count": 683737,
    "pdf_sha256": "26ce80e8d347340e0055f2bcf061b6b3e29489fc68a85b8d5711e12cc9da5090",
    "license": "CC BY-NC-SA 4.0",
}

def verify_pdf(path: Path) -> None:
    data = path.read_bytes()
    if len(data) != PAPER["pdf_byte_count"]:
        raise ValueError("pinned PDF byte count mismatch")
    if hashlib.sha256(data).hexdigest() != PAPER["pdf_sha256"]:
        raise ValueError("pinned PDF SHA-256 mismatch")
```

Populate `manifest.json` from every exact expression and every exact Algorithm 1 line in the approved design, preserving PDF pages and distinctions between Appendix E inline, Eq. (26), literal-derived marginal, single-counted-derived marginal, Eq. (27), and Eq. (28)--(38). Compute each `source_excerpt_sha256` from the committed `source_excerpt`; do not hash normalized text or the whole PDF in its place. Copy the Algorithm 1 text exactly from design lines 129-149 into `algorithm1.txt` and assert its manifest excerpt is byte-identical.

Acquire and verify the source bytes once:

```bash
curl --fail --location --proto '=https' --tlsv1.2 --output /tmp/2606.12913v2.pdf https://export.arxiv.org/pdf/2606.12913v2 && test "$(wc -c < /tmp/2606.12913v2.pdf)" -eq 683737 && printf '%s  %s\n' 26ce80e8d347340e0055f2bcf061b6b3e29489fc68a85b8d5711e12cc9da5090 /tmp/2606.12913v2.pdf | sha256sum --check --strict
```

Expected: `/tmp/2606.12913v2.pdf: OK`. Independently compare each committed transcription to the pinned PDF before setting the second `reviewed_by` value.

- [ ] **Step 4: Run the green test**

Run: `uv run --project "$PROJECT" pytest tests/test_provenance.py -q`

Expected: `3 passed`.

- [ ] **Step 5: Commit the provenance boundary**

```bash
git add "$PROJECT/pyproject.toml" "$PROJECT/uv.lock" "$PROJECT/LICENSE" "$PROJECT/LICENSES" "$PROJECT/NOTICE.md" "$PROJECT/paper_transcriptions" "$PROJECT/src/graph_pruning_repro/__init__.py" "$PROJECT/src/graph_pruning_repro/provenance.py" "$PROJECT/tests/test_provenance.py" "$PROJECT/evidence/tdd-log.jsonl"
git commit -m "feat: pin graph pruning source transcriptions"
```

### Task 2: Exact Types, Objective Variants, and Equivalence Oracles

**Files:**
- Create: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/src/graph_pruning_repro/types.py`
- Create: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/src/graph_pruning_repro/objectives.py`
- Create: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/src/graph_pruning_repro/equivalence.py`
- Test: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/tests/test_objectives.py`
- Test: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/tests/test_equivalence.py`

**Interfaces:**
- Produces: pinned `Instance`, `ModelVariant`, and `Witness` types from the cross-task block.
- Produces: `evaluate_mwcp_edges()`, `evaluate_samplewise_literal()`, `evaluate_objective()`, `symbolic_coefficients()`, and `run_equivalence_audit(code_revision: str) -> dict[str, object]`.
- Consumes: only `types.py`; the edge and samplewise evaluators may not call one another.

- [ ] **Step 1: Write exact failing tests for independent totals and variant rejection**

```python
from fractions import Fraction
import pytest

from graph_pruning_repro.equivalence import compare_objectives
from graph_pruning_repro.objectives import evaluate_objective
from graph_pruning_repro.types import Instance

INSTANCE = Instance(
    vertices=("x", "y"),
    vertex_weights={"x": Fraction(1), "y": Fraction(2)},
    interactions={("x", "y"): Fraction(-1), ("y", "x"): Fraction(-1)},
)

def test_two_vertex_literal_objective_mismatch() -> None:
    result = compare_objectives(INSTANCE, frozenset({"x", "y"}))
    assert result == {
        "paper_mwcp": "2/1",
        "paper_samplewise_literal": "1/1",
        "samplewise_minus_mwcp": "-1/1",
        "mwcp_edge_coefficient": 1,
        "samplewise_edge_coefficient": 2,
    }

def test_half_correction_matches_mwcp() -> None:
    selected = frozenset({"x", "y"})
    assert evaluate_objective(INSTANCE, selected, "half_corrected_samplewise") == Fraction(2)

def test_eq26_is_not_misrepresented_as_objective() -> None:
    with pytest.raises(ValueError, match="score, not a set function"):
        evaluate_objective(INSTANCE, frozenset(), "appendix_eq26_score")
```

- [ ] **Step 2: Verify red**

Run: `uv run --project "$PROJECT" pytest tests/test_objectives.py tests/test_equivalence.py -q`

Expected: collection fails for missing `graph_pruning_repro.equivalence`; log this red phase.

- [ ] **Step 3: Implement minimal independent objective traversals**

```python
def evaluate_mwcp_edges(instance: Instance, selected: frozenset[str]) -> Fraction:
    total = sum((instance.vertex_weights[v] for v in selected), start=Fraction())
    for left, right in combinations(sorted(selected), 2):
        total += instance.interactions.get((left, right), Fraction())
    return total

def evaluate_samplewise_literal(instance: Instance, selected: frozenset[str]) -> Fraction:
    total = Fraction()
    for left in sorted(selected):
        total += instance.vertex_weights[left]
        for right in sorted(selected - {left}):
            total += instance.interactions.get((left, right), Fraction())
    return total

def evaluate_objective(instance: Instance, selected: frozenset[str], model_variant: ModelVariant) -> Fraction:
    if model_variant == "paper_mwcp" or model_variant == "single_counted_pairwise":
        return evaluate_mwcp_edges(instance, selected)
    if model_variant == "paper_samplewise_literal":
        return evaluate_samplewise_literal(instance, selected)
    if model_variant == "half_corrected_samplewise":
        vertices = sum((instance.vertex_weights[v] for v in selected), start=Fraction())
        return vertices + (evaluate_samplewise_literal(instance, selected) - vertices) / 2
    if model_variant == "appendix_inline_shift_literal":
        return evaluate_samplewise_literal(instance, selected) + instance.alpha * instance.eta * len(selected) ** 2
    if model_variant == "modular_shift_candidate":
        return evaluate_mwcp_edges(instance, selected) + instance.eta * len(selected)
    raise ValueError("appendix_eq26_score is a score, not a set function")
```

Implement symbolic coefficient maps directly, then enumerate the approved `n in {1, 2}` domain in deterministic size/vertex/weight order. Assert `cases_examined == 26`, persist the smallest nonzero-edge mismatch, and separately state that the arbitrary-weight result is symbolic.

- [ ] **Step 4: Verify green and finite accounting**

Run: `uv run --project "$PROJECT" pytest tests/test_objectives.py tests/test_equivalence.py -q`

Expected: all tests pass and `run_equivalence_audit("test-revision")["search"]["cases_examined"] == 26`.

- [ ] **Step 5: Commit**

```bash
git add "$PROJECT/src/graph_pruning_repro/types.py" "$PROJECT/src/graph_pruning_repro/objectives.py" "$PROJECT/src/graph_pruning_repro/equivalence.py" "$PROJECT/tests/test_objectives.py" "$PROJECT/tests/test_equivalence.py" "$PROJECT/evidence/tdd-log.jsonl"
git commit -m "feat: audit graph objective equivalence exactly"
```

### Task 3: Diminishing Returns, Shift Boundaries, and Canonical Witnesses

**Files:**
- Create: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/src/graph_pruning_repro/diminishing_returns.py`
- Create: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/src/graph_pruning_repro/shifts.py`
- Create: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/src/graph_pruning_repro/minimize.py`
- Test: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/tests/test_diminishing_returns.py`
- Test: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/tests/test_shifts.py`
- Test: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/tests/test_minimize.py`

**Interfaces:**
- Produces: `direct_marginal()` by set-function subtraction and `closed_form_marginal()` by independent formulas.
- Produces: `enumerate_diminishing_returns(instance, model_variant)`, `run_diminishing_returns_audit(code_revision)`, `run_shift_audit(code_revision)`, and `minimize_witness(predicate, witness)`.
- Consumes: `Instance` and `evaluate_objective` only in `direct_marginal`; the closed-form implementation may not call either.

- [ ] **Step 1: Write failing exact Appendix E and independence tests**

```python
from fractions import Fraction

from graph_pruning_repro.diminishing_returns import closed_form_marginal, direct_marginal, appendix_shift_witness
from graph_pruning_repro.types import Instance

def test_appendix_inline_shift_has_minimal_one_then_three_witness() -> None:
    instance = Instance(
        vertices=("x", "y"),
        vertex_weights={"x": Fraction(), "y": Fraction()},
        interactions={("x", "y"): Fraction(), ("y", "x"): Fraction()},
        alpha=Fraction(1), eta=Fraction(1),
    )
    assert direct_marginal(instance, frozenset(), "x", "appendix_inline_shift_literal") == 1
    assert direct_marginal(instance, frozenset({"y"}), "x", "appendix_inline_shift_literal") == 3
    witness = appendix_shift_witness()
    assert witness["model_variant"] == "appendix_inline_shift_literal"
    assert witness["minimality_checks"] == {"one_vertex_strict_chain_exists": False, "two_vertices_required": True}

def test_direct_and_closed_forms_agree_without_sharing_implementations() -> None:
    instance = Instance(("x", "y"), {"x": Fraction(2), "y": Fraction()}, {("x", "y"): Fraction(-1), ("y", "x"): Fraction(-1)})
    for variant in ("paper_samplewise_literal", "single_counted_pairwise"):
        assert direct_marginal(instance, frozenset({"y"}), "x", variant) == closed_form_marginal(instance, frozenset({"y"}), "x", variant)
```

Add shift tests asserting the Eq. (27) boundary values immediately below/at/above the deduplicated threshold, `6_459` maximum marginal cases, and 256 labeled non-exhaustive rational-alpha controls.

- [ ] **Step 2: Verify red**

Run: `uv run --project "$PROJECT" pytest tests/test_diminishing_returns.py tests/test_shifts.py tests/test_minimize.py -q`

Expected: collection fails for missing `diminishing_returns`; log it.

- [ ] **Step 3: Implement exact marginals and bounded searches**

```python
def direct_marginal(instance, selected, candidate, model_variant):
    return evaluate_objective(instance, selected | {candidate}, model_variant) - evaluate_objective(instance, selected, model_variant)

def closed_form_marginal(instance, selected, candidate, model_variant):
    incident = sum((instance.interactions.get((candidate, other), Fraction()) for other in selected), start=Fraction())
    if model_variant == "paper_samplewise_literal":
        return instance.vertex_weights[candidate] + 2 * incident
    if model_variant == "single_counted_pairwise":
        return instance.vertex_weights[candidate] + incident
    if model_variant == "appendix_inline_shift_literal":
        return instance.vertex_weights[candidate] + 2 * incident + instance.alpha * instance.eta * (2 * len(selected) + 1)
    raise ValueError(f"no closed-form marginal for {model_variant}")
```

Enumerate all `A subseteq B`, `x not in B` directly. Implement the exact symmetric ceiling formula `sum(n * 3**(n-1) * 3**comb(n, 2) for n in range(1, 5)) == 79_480` and asymmetric diagnostic ceiling `19_738`; stop before iteration if the declared domain computes a larger ceiling. Implement minimization in the approved order: vertex deletion, selected/cardinality reduction, zero weights, absolute magnitude reduction, lexicographic canonicalization. Derive witness IDs as `sha256(canonical_json).hexdigest()[:16]`.

- [ ] **Step 4: Verify green and exact ceilings**

Run: `uv run --project "$PROJECT" pytest tests/test_diminishing_returns.py tests/test_shifts.py tests/test_minimize.py -q`

Expected: all pass; audit records declare ceilings `79_480`, `19_738`, `6_459`, and `256`, and the Appendix witness marginals are serialized as `"1/1"` and `"3/1"`.

- [ ] **Step 5: Commit**

```bash
git add "$PROJECT/src/graph_pruning_repro/diminishing_returns.py" "$PROJECT/src/graph_pruning_repro/shifts.py" "$PROJECT/src/graph_pruning_repro/minimize.py" "$PROJECT/tests/test_diminishing_returns.py" "$PROJECT/tests/test_shifts.py" "$PROJECT/tests/test_minimize.py" "$PROJECT/evidence/tdd-log.jsonl"
git commit -m "feat: bound graph submodularity and shift audits"
```

### Task 4: Eq. (7) Greedy, True-Marginal Greedy, Optimum, and Literal Algorithm 1

**Files:**
- Create: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/src/graph_pruning_repro/greedy.py`
- Create: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/src/graph_pruning_repro/algorithm1.py`
- Test: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/tests/test_greedy.py`
- Test: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/tests/test_algorithm1.py`

**Interfaces:**
- Produces: `enumerate_eq7_paths()`, `enumerate_true_marginal_paths()`, `exhaustive_optima()`, `classify_ratio()`, and `run_greedy_audit(code_revision)`.
- Produces: `audit_literal_algorithm1(transcription: Sequence[str]) -> dict[str, object]` returning the first undefined read, exact line, state snapshot, and no executable selection.
- Consumes: `Instance`; true-marginal greedy may call `evaluate_objective`, while Eq. (7) greedy implements its score independently and may not call an objective or marginal.

- [ ] **Step 1: Write failing path-separation, tie, ratio, and Algorithm 1 tests**

```python
def test_literal_algorithm1_stops_before_first_selection() -> None:
    algorithm_lines = (Path(__file__).parents[1] / "paper_transcriptions" / "algorithm1.txt").read_text().splitlines()
    result = audit_literal_algorithm1(algorithm_lines)
    assert result["greedy_path"] == "paper_algorithm1_literal"
    assert result["status"] == "undefined_read"
    assert result["line"] == 8
    assert result["symbol"] == "x*"
    assert result["selected"] is None
    assert result["repairs"] == []

def test_all_ties_are_retained() -> None:
    instance = Instance(("x", "y"), {"x": Fraction(), "y": Fraction()}, {})
    paths = enumerate_eq7_paths(instance, budget=1)
    assert paths == (("x",), ("y",))

def test_zero_optimum_ratio_is_one() -> None:
    assert classify_ratio(Fraction(), Fraction()) == {"status": "defined_zero_equality", "ratio": "1/1"}

def test_negative_optimum_has_no_ratio() -> None:
    assert classify_ratio(Fraction(-1), Fraction(-1))["status"] == "negative_objective_regime"
```

Also assert that Appendix-inline true-marginal selection ties match the unshifted literal path at each fixed iteration while objective ratios are recomputed, never transferred.

- [ ] **Step 2: Verify red**

Run: `uv run --project "$PROJECT" pytest tests/test_greedy.py tests/test_algorithm1.py -q`

Expected: collection fails for missing modules; log it.

- [ ] **Step 3: Implement the three non-interchangeable paths**

```python
def eq7_score(instance: Instance, selected: frozenset[str], candidate: str) -> Fraction:
    return instance.vertex_weights[candidate] + sum(
        (instance.interactions.get((candidate, other), Fraction()) for other in selected),
        start=Fraction(),
    )

def exhaustive_optima(instance, budget, model_variant):
    candidates = tuple(frozenset(items) for items in combinations(instance.vertices, budget))
    values = {selected: evaluate_objective(instance, selected, model_variant) for selected in candidates}
    best = max(values.values())
    return tuple(selected for selected in candidates if values[selected] == best)
```

Branch on every tied maximum in lexicographic order and retain all terminal paths. For literal Algorithm 1, execute transcription lines in order and return at line 8 because `x*` is absent from state; separately record later static ambiguities at lines 10-11, 14, and candidate carry-forward without executing through the first undefined read. Enumerate exactly `16_239` weighted-cardinality instances, at most `389_736` terminal paths per executable greedy implementation, and at most `97_434` optimum subsets.

- [ ] **Step 4: Verify green and accounting**

Run: `uv run --project "$PROJECT" pytest tests/test_greedy.py tests/test_algorithm1.py -q`

Expected: all pass; no record names an executable resolution `paper_algorithm1_literal` and no Appendix-inline record claims ratio transfer.

- [ ] **Step 5: Commit**

```bash
git add "$PROJECT/src/graph_pruning_repro/greedy.py" "$PROJECT/src/graph_pruning_repro/algorithm1.py" "$PROJECT/tests/test_greedy.py" "$PROJECT/tests/test_algorithm1.py" "$PROJECT/evidence/tdd-log.jsonl"
git commit -m "feat: separate literal and executable greedy audits"
```

### Task 5: Appendix F Proof Ledger and Search Accounting

**Files:**
- Create: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/src/graph_pruning_repro/proof_ledger.py`
- Test: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/tests/test_proof_ledger.py`
- Test: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/tests/test_search_accounting.py`

**Interfaces:**
- Produces: `build_symbolic_ledger(model_variant, witness_ids) -> tuple[dict[str, object], ...]`, `run_finite_ledger_control(instances)`, and `declared_aggregate_ceiling() -> int`.
- Consumes: canonical Appendix witness ID, greedy-domain instances, and exact arithmetic records; it does not infer proof status from final claim verdicts.

- [ ] **Step 1: Write failing proof-premise tests**

```python
def test_appendix_literal_ledger_links_shift_witness_everywhere() -> None:
    appendix_witness_id = appendix_shift_witness()["id"]
    rows = build_symbolic_ledger("appendix_inline_shift_literal", {"appendix_shift": appendix_witness_id})
    submodularity_rows = [row for row in rows if "submodular" in row["required_premises"]]
    assert submodularity_rows
    assert all(row["status"] == "contradicted" for row in submodularity_rows)
    assert all(row["witness_ids"] == [appendix_witness_id] for row in submodularity_rows)
    assert next(row for row in rows if row["equation"] == "28")["status"] == "supported"
    assert next(row for row in rows if row["check"] == "ratio_transfer_from_repaired_objective")["status"] == "contradicted"

def test_paper_cardinality_and_product_steps_are_audited() -> None:
    rows = build_symbolic_ledger("paper_samplewise_literal", {})
    assert any(row["check"] == "optimum_remainder_at_most_b_not_b_minus_t" for row in rows)
    assert any(row["check"] == "product_includes_k_equals_one_zero_factor" for row in rows)

def test_aggregate_ceiling_is_frozen() -> None:
    assert declared_aggregate_ceiling() == 1_177_835
```

- [ ] **Step 2: Verify red**

Run: `uv run --project "$PROJECT" pytest tests/test_proof_ledger.py tests/test_search_accounting.py -q`

Expected: collection fails for missing `proof_ledger`; log it.

- [ ] **Step 3: Implement explicit Eq. (28)--(38) rows**

Represent all 11 numbered equation transitions explicitly. Each row contains `equation`, `model_variant`, `statement`, `required_premises`, `check`, `evidence_kind`, `status`, and `witness_ids`. Use `not_applicable` for symbolic transitions without an instance predicate. Reuse, rather than regenerate, the 16,239 greedy instances; enforce at most `11 * 16_239 == 178_629` finite rows.

Compute the aggregate from named constants:

```python
def declared_aggregate_ceiling() -> int:
    return sum((26, 79_480, 19_738, 6_459, 256, 16_239, 389_736, 389_736, 97_434, 178_629, 2))
```

- [ ] **Step 4: Verify green**

Run: `uv run --project "$PROJECT" pytest tests/test_proof_ledger.py tests/test_search_accounting.py -q`

Expected: all pass and the sum is exactly `1_177_835`.

- [ ] **Step 5: Commit**

```bash
git add "$PROJECT/src/graph_pruning_repro/proof_ledger.py" "$PROJECT/tests/test_proof_ledger.py" "$PROJECT/tests/test_search_accounting.py" "$PROJECT/evidence/tdd-log.jsonl"
git commit -m "feat: audit graph pruning proof premises"
```

### Task 6: Canonical Evidence Schema, Runner, and Acceptance Gates

**Files:**
- Create: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/evidence/schema.json`
- Create: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/src/graph_pruning_repro/evidence.py`
- Create: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/src/graph_pruning_repro/cli.py`
- Test: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/tests/test_evidence.py`
- Test: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/tests/test_cli.py`

**Interfaces:**
- Produces: `build_evidence()`, `validate_evidence()`, `canonical_json_bytes()`, and CLI commands `recompute OUTPUT_DIR`, `validate EVIDENCE`, `render EVIDENCE OUTPUT_DIR`.
- Consumes: all prior audit interfaces; the runner orchestrates but does not duplicate their mathematics.

- [ ] **Step 1: Write failing schema, linkage, determinism, and ceiling tests**

```python
def test_required_appendix_witness_has_three_audit_linkages(tmp_path: Path) -> None:
    evidence = build_evidence(tmp_path, code_revision="0" * 40)
    witness = next(w for w in evidence["witnesses"] if w["property"] == "appendix_inline_shift_diminishing_returns")
    assert witness["model_variant"] == "appendix_inline_shift_literal"
    assert witness["inputs"]["alpha"] == "1/1"
    assert witness["inputs"]["eta"] == "1/1"
    assert witness["intermediate_values"]["marginal_empty"] == "1/1"
    assert witness["intermediate_values"]["marginal_y"] == "3/1"
    links = {result["audit"] for result in evidence["claim_results"] if witness["id"] in result["witness_ids"]}
    assert {"diminishing_returns", "greedy_guarantee_premise", "appendix_f_proof_ledger"} <= links

def test_repaired_variants_cannot_claim_literal_witness(tmp_path: Path) -> None:
    evidence = build_evidence(tmp_path, code_revision="0" * 40)
    literal_id = next(w["id"] for w in evidence["witnesses"] if w["model_variant"] == "appendix_inline_shift_literal")
    assert not any(literal_id in result["witness_ids"] for result in evidence["claim_results"] if result["model_variant"] == "modular_shift_candidate")

def test_canonical_build_is_byte_identical(tmp_path: Path) -> None:
    first = build_evidence(tmp_path / "first", code_revision="0" * 40)
    second = build_evidence(tmp_path / "second", code_revision="0" * 40)
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
```

Add subprocess tests that reject altered claim text, an undeclared domain, an actual count above its ceiling, missing completion status, merged variants, a non-Fraction rational, or an incomplete search presented as a pass.

- [ ] **Step 2: Verify red**

Run: `uv run --project "$PROJECT" pytest tests/test_evidence.py tests/test_cli.py -q`

Expected: collection fails for missing `evidence`; log it.

- [ ] **Step 3: Implement schema version 1 and stable serialization**

Require top-level `schema_version`, `attempt_id`, `paper`, `target_claims`, `environment`, `transcriptions`, `searches`, `witnesses`, `proof_ledger`, `claim_results`, `unavailable_claims`, `commands`, and `artifacts`. Serialize with:

```python
def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()
```

Sort records by stable IDs, omit timestamps and wall time from canonical truth content, and place measured runtime only in a separately hashed command record that is excluded from byte-comparison input or normalized to a declared deterministic field. Write files atomically via a sibling temporary file plus `Path.replace`. Validate with `jsonschema.Draft202012Validator`, then run semantic acceptance checks for exact claims, exact provenance, ceilings, witness links, disjoint variants, reportable ratio regimes, and all unavailable claims.

- [ ] **Step 4: Verify focused green**

Run: `uv run --project "$PROJECT" pytest tests/test_evidence.py tests/test_cli.py -q`

Expected: all pass.

- [ ] **Step 5: Run the complete evidence command twice**

```bash
revision=$(git rev-parse HEAD)
rm -rf /tmp/graph-pruning-evidence-a /tmp/graph-pruning-evidence-b
uv run --project "$PROJECT" graph-pruning-repro recompute /tmp/graph-pruning-evidence-a --code-revision "$revision"
uv run --project "$PROJECT" graph-pruning-repro recompute /tmp/graph-pruning-evidence-b --code-revision "$revision"
cmp /tmp/graph-pruning-evidence-a/evidence.json /tmp/graph-pruning-evidence-b/evidence.json
uv run --project "$PROJECT" graph-pruning-repro validate /tmp/graph-pruning-evidence-a/evidence.json
```

Expected: each recomputation reports `completed 1177835/1177835 declared evaluations in under 1800 seconds`; `cmp` is silent with exit 0; validation reports `schema and semantic acceptance: PASS`.

- [ ] **Step 6: Commit code and generated canonical evidence**

Copy the accepted first output into `"$PROJECT/evidence/evidence.json"` and `"$PROJECT/evidence/witnesses/"`, rerun validation there, then commit:

```bash
git add "$PROJECT/evidence" "$PROJECT/src/graph_pruning_repro/evidence.py" "$PROJECT/src/graph_pruning_repro/cli.py" "$PROJECT/tests/test_evidence.py" "$PROJECT/tests/test_cli.py"
git commit -m "feat: generate canonical graph pruning evidence"
```

### Task 7: Evidence-Only Report, Poster, and README

**Files:**
- Create: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/src/graph_pruning_repro/render.py`
- Create: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/README.md`
- Create: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/report.md`
- Create: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/poster.html`
- Create: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/poster_embed.html`
- Test: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/tests/test_render.py`

**Interfaces:**
- Produces: `render_report(evidence) -> str`, `render_poster(evidence) -> str`, and `assert_render_agreement(evidence, report, poster) -> None`.
- Consumes: schema- and semantic-accepted evidence only; renderers may not invoke oracles or introduce computed numbers.

- [ ] **Step 1: Write failing rendering and unavailable-claim tests**

```python
def test_render_leads_with_two_target_claims_and_boundaries() -> None:
    evidence = json.loads((Path(__file__).parents[1] / "evidence" / "evidence.json").read_text())
    report = render_report(evidence)
    assert report.index(evidence["target_claims"][0]) < report.index("Unavailable empirical claims")
    assert "appendix_inline_shift_literal" in report
    assert "modular_shift_candidate" in report
    assert "1 then 3" in report
    assert "CIFAR-10/100" in report and "unavailable" in report
    assert "https://arxiv.org/pdf/2606.12913v2" in report

def test_report_numbers_all_come_from_evidence() -> None:
    evidence = json.loads((Path(__file__).parents[1] / "evidence" / "evidence.json").read_text())
    report = render_report(evidence)
    assert_render_agreement(evidence, report, render_poster(evidence))
```

- [ ] **Step 2: Verify red**

Run: `uv run --project "$PROJECT" pytest tests/test_render.py -q`

Expected: collection fails for missing `render`; log it.

- [ ] **Step 3: Implement deterministic rendering**

Render provenance and equations, literal-versus-repaired table, independent oracle statuses, canonical witnesses, exact greedy/optimum values and ratios, proof-ledger rows, exhaustive domains and limitations, and the prominent unavailable panel. Every displayed numeric token must resolve to an evidence JSON pointer recorded in an embedded `data-evidence-path` attribute or report footnote. State explicitly that bounded enumeration can refute but cannot prove arbitrary-real universal claims and that no released implementation resolves edge counting or the shift.

- [ ] **Step 4: Render and verify**

Run:

```bash
uv run --project "$PROJECT" graph-pruning-repro render "$PROJECT/evidence/evidence.json" "$PROJECT"
uv run --project "$PROJECT" pytest tests/test_render.py -q
```

Expected: generated `report.md`, `poster.html`, and `poster_embed.html`; all rendering tests pass.

- [ ] **Step 5: Commit**

```bash
git add "$PROJECT/src/graph_pruning_repro/render.py" "$PROJECT/tests/test_render.py" "$PROJECT/README.md" "$PROJECT/report.md" "$PROJECT/poster.html" "$PROJECT/poster_embed.html" "$PROJECT/evidence/tdd-log.jsonl"
git commit -m "docs: render graph pruning formal evidence"
```

### Task 8: CPU Space and Local End-to-End Validation

**Files:**
- Create: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/app.py`
- Test: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/tests/test_app.py`
- Modify only if evidence changed: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/evidence/evidence.json`
- Modify only if rendering changed: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/report.md`, `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/poster.html`, `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/poster_embed.html`

**Interfaces:**
- Produces: Gradio `demo` with summary, variant table, witness/proof ledger, unavailable panel, canonical downloads, and a CPU recomputation action.
- Consumes: committed accepted evidence and CLI; imports have no network or evidence mutation side effects.

- [ ] **Step 1: Write failing Space smoke tests**

```python
def test_space_import_is_offline_and_exposes_downloads(monkeypatch) -> None:
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network used")))
    from app import EVIDENCE_PATH, demo
    assert EVIDENCE_PATH.name == "evidence.json"
    assert demo is not None

def test_space_recompute_uses_bounded_cli(tmp_path: Path) -> None:
    from app import recompute
    status, evidence_path = recompute(tmp_path)
    assert status.startswith("PASS: 1177835/1177835")
    assert evidence_path.exists()
```

- [ ] **Step 2: Verify red**

Run: `uv run --project "$PROJECT" pytest tests/test_app.py -q`

Expected: import fails because `app.py` is absent; log it.

- [ ] **Step 3: Implement the minimal offline Space**

Load and validate committed evidence at startup, render from it, expose `gr.DownloadButton` for canonical JSON and witness files, and invoke the same `build_evidence` path for recomputation. Do not fetch the PDF or call Hub APIs from `app.py`.

- [ ] **Step 4: Create a clean environment and run all local gates**

Verify this worktree has its own environment before running:

```bash
test "$(realpath .venv)" = "$(realpath "$PWD/.venv")" || uv venv .venv
uv sync --project "$PROJECT" --frozen
uv run --project "$PROJECT" pytest -q
uv run --project "$PROJECT" graph-pruning-repro validate "$PROJECT/evidence/evidence.json"
uv run pytest -q
uv run pre-commit run --files "docs/superpowers/plans/2026-07-25-graph-pruning-reproduction.md" $(git ls-files "$PROJECT")
git diff --check
git status --short
```

Expected: clean install succeeds; submission tests all pass; semantic acceptance is `PASS`; root tests pass without collecting NAPE; targeted pre-commit passes; `git diff --check` is silent. `git status --short` may list only the intended Task 8 files before commit.

- [ ] **Step 5: Review security, provenance, and scope**

Run:

```bash
git diff --check
git diff -- "$PROJECT"
git status --short
```

Confirm no credential-like values, downloaded PDF, cache, mutable unversioned paper URL, environment dump, unrelated submission change, or NAPE change appears. Confirm `evidence.json` names exact code revision and each artifact hash verifies.

- [ ] **Step 6: Invoke completion verification and commit validated source**

Invoke `superpowers:verification-before-completion`, rerun every command it requires, then:

```bash
git add "$PROJECT/app.py" "$PROJECT/tests/test_app.py" "$PROJECT/evidence/tdd-log.jsonl"
git commit -m "feat: expose graph pruning evidence Space"
```

Do not transition lifecycle state until the committed revision is rerun and accepted. If the commit changed the evidence `code_revision`, rerun Task 6 with the committed source revision, commit only regenerated evidence/rendering as `chore: pin validated graph pruning evidence`, and rerun all Task 8 gates.

### Task 9: Fenced Validation, Deployment, Submission, and Judging Lifecycle

**Files:**
- Modify in coordinator worktree only: `state/repro-loop.json`
- Modify in coordinator worktree only: named shards under `state/repro-loop/` through the state CLI
- Modify in coordinator worktree only: `docs/HANDOFF.md`
- External mutation: paper-specific Hugging Face Space and challenge submission

**Interfaces:**
- Consumes: exact validated Git commit, accepted evidence, current external attempt owner/token, and fresh immutable live snapshots.
- Produces: fenced `validated`, `deployed`, `submitted`, and bounded `judging` states; exact deployed Space SHA; post-submission presence proof; HANDOFF milestones.

- [ ] **Step 1: Re-read authoritative attempt state and verify design approval**

Run from the coordinator worktree:

```bash
COORDINATOR=/home/will/projects/icml-2026-reproductions/.worktrees/five-paper-scheduler
uv run python skills/icml-repro-loop/scripts/state.py show-attempt state/repro-loop.json --attempt-id e485c086-6fa5-4ff6-a3c3-1f31c79bbae6
```

Expected: one JSON attempt with `design_approved: true`, phase `implementing`, and the exact target claims. Read the separate authoritative attempt lease shard in Step 2; the attempt record is not a lease source.

- [ ] **Step 2: Extract the current fence and persist validated only after revalidation**

```bash
lease_json=$(uv run python -c 'import json,sys; sys.path.insert(0,"skills/icml-repro-loop/scripts"); import store; paths=store.StatePaths("state/repro-loop.json"); print(json.dumps(store.read_json(paths.resource_lease("attempt:e485c086-6fa5-4ff6-a3c3-1f31c79bbae6"))))')
owner=$(printf '%s' "$lease_json" | uv run python -c 'import json,sys; print(json.load(sys.stdin)["owner"])')
token=$(printf '%s' "$lease_json" | uv run python -c 'import json,sys; print(json.load(sys.stdin)["fencing_token"])')
validated_sha=$(git -C /home/will/projects/icml-2026-reproductions/.worktrees/graph-pruning rev-parse HEAD)
uv run python skills/icml-repro-loop/scripts/state.py transition-attempt state/repro-loop.json validated --attempt-id e485c086-6fa5-4ff6-a3c3-1f31c79bbae6 --owner "$owner" --fencing-token "$token" --updates-json "{\"validated_commit\":\"$validated_sha\",\"validation\":\"submission pytest, canonical double-run cmp, schema/semantic acceptance, root pytest, targeted pre-commit, verification-before-completion\"}"
```

Expected: attempt phase `validated`. If the lease is expired or released, do not run the transition: call `claim-attempt` with the shard's predecessor token and explicit successor owner `graph-pruning-lifecycle-writer`, then use the returned owner/token. If it is live but near expiry, call `renew-attempt` first. Update `docs/HANDOFF.md` with attempt ID, exact validated SHA, passed gates, and deployment as next action; commit coordinator-generated shards plus HANDOFF with `chore: validate graph pruning reproduction`.

- [ ] **Step 3: Create and push the dedicated Space**

Use `hf-cli` and `huggingface-spaces` skills. Set the paper-specific Space ID once and do not reuse another submission Space:

```bash
SPACE_ID=wrice/repro-selecting-samples-on-graphs-a-unified-dataset-pruning-framework
SOURCE_PROJECT=/home/will/projects/icml-2026-reproductions/.worktrees/graph-pruning/submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration
hf repos create "$SPACE_ID" --type space --sdk gradio --public --exist-ok
hf upload "$SPACE_ID" "$SOURCE_PROJECT" . --type space --exclude "**/__pycache__/**" --commit-message "Deploy validated graph pruning reproduction" --format json
```

Expected: upload returns an exact Space commit SHA. The Space README metadata must include `paper-a3GdvuPItd`, `icml2026-repro`, CPU hardware, and the authoritative long submission slug in title/description even though the Hub repository ID is shortened to fit Hub naming limits.

- [ ] **Step 4: Verify exact deployed SHA and live behavior before state mutation**

```bash
hf spaces wait "$SPACE_ID" --timeout 1800
deployed_sha=$(hf spaces info "$SPACE_ID" --expand sha --format json | uv run python -c 'import json,sys; print(json.load(sys.stdin)["sha"])')
test -n "$deployed_sha"
```

Run `hf spaces info "$SPACE_ID" --expand runtime --format json`, inspect `hf spaces logs "$SPACE_ID" --tail 200`, discover the public endpoint with `gradio info "$SPACE_ID"`, invoke it with `gradio predict`, download `evidence.json`, validate it locally, and compare its artifact hash and code revision to the intended validated commit. Expected: runtime is healthy, logs contain no fallback or traceback, live recomputation reports all `1_177_835` declared evaluations complete under 30 minutes, and downloaded canonical evidence passes schema/semantic acceptance.

- [ ] **Step 5: Persist deployed and HANDOFF milestone with the refreshed fence**

Re-read attempt JSON and extract owner/token exactly as Step 2, then:

```bash
uv run python skills/icml-repro-loop/scripts/state.py transition-attempt state/repro-loop.json deployed --attempt-id e485c086-6fa5-4ff6-a3c3-1f31c79bbae6 --owner "$owner" --fencing-token "$token" --updates-json "{\"space_id\":\"$SPACE_ID\",\"deployed_sha\":\"$deployed_sha\",\"validated_commit\":\"$validated_sha\"}"
```

Expected: phase `deployed`. Update HANDOFF with exact Space ID/SHA and fresh-refresh/submission as next action; commit as `chore: deploy graph pruning reproduction`.

- [ ] **Step 6: Perform the mandatory fresh assessed live refresh**

Use the coordinator's current assessment workflow, preserving the admitted assessment and exact target claims:

```bash
raw_id=$(uv run python skills/icml-repro-loop/scripts/state.py refresh-live state/repro-loop.json | uv run python -c 'import json,sys; print(json.load(sys.stdin)["snapshot_id"])')
uv run python skills/icml-repro-loop/scripts/state.py show-snapshot state/repro-loop.json --snapshot-id "$raw_id"
assessed_snapshot_id=$(uv run python skills/icml-repro-loop/scripts/state.py refresh-live state/repro-loop.json --assessments-json state/candidate-assessments.json | uv run python -c 'import json,sys; print(json.load(sys.stdin)["snapshot_id"])')
uv run python skills/icml-repro-loop/scripts/state.py show-snapshot state/repro-loop.json --snapshot-id "$assessed_snapshot_id"
```

Expected: a new immutable assessed snapshot still shows this paper eligible and the Space unsubmitted/unjudged. Cancel submission and persist a blocker if eligibility, claim text, challenge revision, or existing verdict changed.

- [ ] **Step 7: Submit exact claims and persist submitted**

Generate the submission payload only from `evidence.json.target_claims`, preserving byte order; use the existing challenge submission mechanism documented in `skills/icml-repro-loop/references/submission-checklist.md`. Re-read owner/token, then persist the returned submission ID:

```bash
uv run python skills/icml-repro-loop/scripts/state.py transition-attempt state/repro-loop.json submitted --attempt-id e485c086-6fa5-4ff6-a3c3-1f31c79bbae6 --owner "$owner" --fencing-token "$token" --updates-json "{\"submission_id\":\"$submission_id\",\"space_id\":\"$SPACE_ID\",\"deployed_sha\":\"$deployed_sha\",\"pre_submission_snapshot_id\":\"$assessed_snapshot_id\"}"
```

Expected: phase `submitted`. HANDOFF names attempt, submission ID, Space ID, deployed SHA, phase, and post-submission refresh as next action; commit `chore: submit graph pruning reproduction`.

- [ ] **Step 8: Verify post-submission presence before judging**

```bash
post_id=$(uv run python skills/icml-repro-loop/scripts/state.py refresh-live state/repro-loop.json | uv run python -c 'import json,sys; print(json.load(sys.stdin)["snapshot_id"])')
uv run python skills/icml-repro-loop/scripts/state.py show-snapshot state/repro-loop.json --snapshot-id "$post_id"
```

Expected: exact Space ID and deployed SHA occur in queued/live state for `a3GdvuPItd`. If absent, stale, or terminal, persist a blocker and do not enter judging.

- [ ] **Step 9: Enter bounded judging and finish through the existing exact-claim flow**

Re-read owner/token and transition with an explicit finite polling bound:

```bash
uv run python skills/icml-repro-loop/scripts/state.py transition-attempt state/repro-loop.json judging --attempt-id e485c086-6fa5-4ff6-a3c3-1f31c79bbae6 --owner "$owner" --fencing-token "$token" --updates-json "{\"post_submission_snapshot_id\":\"$post_id\",\"poll_limit\":12,\"poll_interval_seconds\":300}"
```

Expected: phase `judging`. Update HANDOFF with polling bounds and next action. Poll at most 12 times, preserve verdicts for exactly the two immutable claims, improve at most once only when eligible, and otherwise complete or persist a nonempty blocker with `blocked_from`. Add a HANDOFF milestone and focused coordinator tests after every transition; never auto-abandon this attempt.

- [ ] **Step 10: Validate and commit each coordinator lifecycle mutation**

After each external mutation and state transition:

```bash
uv run pytest -q
uv run pre-commit run --files docs/HANDOFF.md state/repro-loop.json $(git diff --name-only -- state/repro-loop)
git diff --check
git status --short
```

Expected: root tests and targeted hooks pass, no NAPE path appears, and only intended coordinator shards/HANDOFF are staged. Commit each material phase separately; never combine validated, deployed, submitted, and judging into one unverifiable commit.

## Final Implementation Checklist

- [ ] Every production behavior first appeared behind an observed failing test and a `tdd-log.jsonl` red record.
- [ ] PDF acquisition command, byte count, digest, revision, pages, equations, excerpts, checksums, and two-person review are complete.
- [ ] Seven objective/score variants and three greedy paths remain distinct.
- [ ] Eq. (3) and literal Eq. (4)--(5) are independently traversed and symbolically compared.
- [ ] Objective witness search examines its exact 26-case finite domain.
- [ ] Symmetric and asymmetric diminishing-returns controls enforce ceilings 79,480 and 19,738.
- [ ] Shift controls enforce ceilings 6,459 and 256 and distinguish exhaustive from non-exhaustive evidence.
- [ ] Appendix-inline literal witness is exactly two vertices, zero weights, `alpha=eta=1`, and marginals `1/1` then `3/1`.
- [ ] The Appendix witness links diminishing returns, greedy-guarantee premise, and every applicable proof-ledger row, and never links a repaired variant.
- [ ] Eq. (7) and true-marginal greedy retain all ties independently; Algorithm 1 stops at line 8 without repair.
- [ ] Greedy enumeration has 16,239 instances, bounded paths/subsets, exact ratios, and flagged invalid regimes.
- [ ] Appendix F ledger covers Eq. (28)--(38), normalization, non-negativity, monotonicity, submodularity, cardinality, product indices, constants, and ratio transfer.
- [ ] Declared aggregate ceiling is exactly 1,177,835 and runtime is under 30 minutes without domain shrinkage.
- [ ] Canonical evidence passes JSON Schema plus semantic acceptance and is byte-identical across two clean runs.
- [ ] Report, poster, and Space derive numeric statements only from canonical JSON and prominently mark empirical claims unavailable.
- [ ] License boundaries, attribution, seven authors, source URL, and adaptation status are visible locally and on Space.
- [ ] Submission pytest, root pytest, targeted pre-commit, diff/security review, and `verification-before-completion` pass.
- [ ] Validated, deployed, submitted, and judging transitions use freshly read owner/token and receive separate HANDOFF milestones.
- [ ] Exact deployed Space SHA and live recomputation/download are verified before `deployed`.
- [ ] Fresh assessed pre-submission and post-submission snapshots prove eligibility and queued/live presence before bounded judging.
