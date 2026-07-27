# Graph Dataset Pruning Formal-Evidence Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, validate, publish, and submit a deterministic CPU formal-evidence reproduction of the two admitted claims from `arxiv:2606.12913v2` without presenting paper-reported experiments as reproduced measurements.

**Architecture:** An independent Python project under the authoritative long submission slug transcribes and verifies the pinned PDF, then evaluates seven explicitly named model variants through independent exact-arithmetic objective, diminishing-returns, shift, greedy, Algorithm 1, and proof-ledger oracles. A deterministic runner enforces each bounded search ceiling and emits schema- and semantic-validated canonical JSON, file-backed witness fixtures, a report, poster, and CPU Hugging Face Space proposal. The paper worker is an untrusted proposal producer; only the controller validates proposals, mutates external services, records observations, or changes lifecycle state.

**Tech Stack:** Python 3.11+, standard-library `fractions`, `itertools`, `json`, and `hashlib`; `jsonschema`; pytest; Gradio; Hugging Face Hub CLI; repository `icml-repro-loop` state CLI.

## Global Constraints

- Attempt ID is exactly `64bfe193-333b-4b37-9683-9ac25ca5ac27`; challenge paper ID is exactly `a3GdvuPItd`; assessed snapshot is exactly `35d2104cb8462a652d933aa5a776f9b166e8c2724df12da7b35f54cbe19c883d`.
- Design author is `codex-graph-pruning-design-author-v2`; independent design review must use a different identity.
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
- `modular_shift_candidate` means exactly
  `f_lit(S) + eta_mod * |S|`, using the literal double-counted base. Its
  greedy/proof-domain coefficient is fixed to
  `eta_mod = 2 * (n - 1) * M`; no task may substitute a single-counted base
  or the Appendix-inline quadratic term.
- For every canonical graph in the symmetric diminishing-returns, premise,
  greedy, optimum, and finite proof-ledger domains, compute
  `M = max(abs(a_ij))` (or zero for no edges) before evaluation. The only
  allowed parameter tuples are `alpha=1, eta=0` for the four unshifted
  variants, `alpha=1, eta=M` for `appendix_inline_shift_literal`, and
  `alpha=1, eta=2*(n-1)*M` for `modular_shift_candidate`. Encode that tuple in
  every parameterized case ID; it is not an additional enumeration axis and
  changes no ceiling.
- The Appendix E witness must be linked by the diminishing-returns result, greedy-guarantee premise result, and every applicable Appendix F proof-ledger row; no repaired variant may use its identifier.
- Each oracle has an independent implementation and must not call another oracle's objective or marginal implementation.
- Use `fractions.Fraction` for all non-integral truth decisions; never use floating point to decide an identity, premise, witness, or approximation result.
- Search only the approved domains; refuse undeclared expansion. The generation ceiling is `13_833_860` named work units and the independent full replay has the same ceiling, for a controller generation-plus-validation ceiling of `27_667_720`. Every component records `actual <= declared_ceiling`; no command requires actual work to equal a ceiling. The former `1_177_735` figure is withdrawn because it hid per-variant premise subset/marginal work.
- Label evidence only `symbolic`, `exhaustive_finite`, or `non_exhaustive`; an early-stopped finite search is not exhaustive.
- CPU only, no model training, no paid API use, and no network during evidence
  computation. The controller externally measures generation and replay and
  withholds validation if either exceeds 1,800 seconds; it never shrinks a
  domain. Measured time exists only in a controller attestation or
  noncanonical `/tmp` log, never in canonical evidence or witnesses.
- Paper-reported CIFAR-10/100, ImageNet-1k, segmentation, detection, accuracy, training-time, and acceleration results are context-only and must be marked `unavailable`, never reproduced.
- Write and run a failing test before each production change. Record every red command, timestamp, test ID, and expected missing behavior in `evidence/tdd-log.jsonl`.
- Do not modify, run, test, or format `submissions/nape/`; root validation relies on the repository's existing exclusion.
- Do not copy or mutate coordinator index, attempt, lease, judgment, transaction, or snapshot shards in the paper branch. The worker stops at a validated local proposal; the controller alone performs `attest-validation`, `publish-deployment`, `attest-submission`, `watch-attempt`, and `sync-verdict` with a freshly read current owner and fencing token.
- Never commit credentials, cookies, unredacted environment dumps, downloaded PDF bytes, caches, or mutable source URLs.
- The controller updates `docs/HANDOFF.md` after every material lifecycle transition or blocker; the worker does not.

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
- `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/paper_transcriptions/excerpts/*.txt`: exact UTF-8 source-excerpt bytes, one uniquely referenced file per equation record.
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
def canonical_variant_parameters(instance: Instance, model_variant: ModelVariant) -> tuple[Fraction, Fraction]: ...
def canonical_parameterized_instance_id(instance: Instance, model_variant: ModelVariant) -> str: ...
def direct_marginal(instance: Instance, selected: frozenset[Vertex], candidate: Vertex, model_variant: ModelVariant) -> Fraction: ...
def closed_form_marginal(instance: Instance, selected: frozenset[Vertex], candidate: Vertex, model_variant: ModelVariant) -> Fraction: ...
def enumerate_greedy_paths(instance: Instance, budget: int, model_variant: ModelVariant, greedy_path: GreedyPath) -> tuple[tuple[Vertex, ...], ...]: ...
def exhaustive_optima(instance: Instance, budget: int, model_variant: ModelVariant) -> tuple[frozenset[Vertex], ...]: ...
def build_evidence(output_dir: Path, source_revision: str) -> dict[str, object]: ...
def validate_evidence(evidence_path: Path, schema_path: Path, evidence_root: Path) -> None: ...
```

`appendix_eq26_score` is rejected by `evaluate_objective` because it is a score, not a set function. `paper_algorithm1_literal` is rejected by executable greedy dispatch and routed only to `audit_literal_algorithm1()`.
Validation takes paths because acceptance must open and hash the actual transcription and witness files; validating only an in-memory mapping is forbidden.

### Task 1: Isolated Project, Licensing, PDF Provenance, and Transcriptions

**Files:**
- Create: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/pyproject.toml`
- Create: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/uv.lock`
- Create: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/LICENSE`
- Create: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/LICENSES/CC-BY-NC-SA-4.0.txt`
- Create: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/NOTICE.md`
- Create: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/paper_transcriptions/manifest.json`
- Create: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/paper_transcriptions/excerpts/*.txt`
- Create: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/paper_transcriptions/algorithm1.txt`
- Create: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/src/graph_pruning_repro/__init__.py`
- Create: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/src/graph_pruning_repro/provenance.py`
- Test: `submissions/selecting-samples-on-graphs-a-unified-dataset-pruning-framework-for-lossless-training-acceleration/tests/test_provenance.py`

**Interfaces:**
- Produces: `PAPER`, `TARGET_CLAIMS`, `PDF_ACQUISITION_COMMAND`, `verify_pdf(path: Path) -> None`, and `load_transcriptions(root: Path) -> tuple[dict[str, object], ...]`.
- Produces: manifest records with exactly `record_id`, `equation`, `pdf_page`, `section`, `normalized_expression`, `source_excerpt_path`, `source_excerpt_byte_count`, `source_excerpt_sha256`, and `reviewed_by`; plus `TRANSCRIPTION_SET_SHA256`.
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

from graph_pruning_repro.provenance import (
    PAPER,
    TARGET_CLAIMS,
    TRANSCRIPTION_SET_SHA256,
    load_transcriptions,
    transcription_set_sha256,
    verify_pdf,
)

def test_pdf_identity_and_digest_rejection(tmp_path: Path) -> None:
    assert PAPER["revision"] == "arxiv:2606.12913v2"
    assert PAPER["pdf_byte_count"] == 683737
    bad = tmp_path / "paper.pdf"
    bad.write_bytes(b"not the pinned PDF")
    with pytest.raises(ValueError, match="pinned PDF byte count"):
        verify_pdf(bad)

def test_transcriptions_are_complete_and_checksummed() -> None:
    root = Path(__file__).parents[1]
    records = load_transcriptions(root)
    equations = {record["equation"] for record in records}
    assert {"2", "3", "4", "5", "6", "7", "8", "10-11", "12-14", "Appendix E inline", "26", "27", "28-38", "Algorithm 1"} <= equations
    expected_keys = {
        "record_id", "equation", "pdf_page", "section",
        "normalized_expression", "source_excerpt_path",
        "source_excerpt_byte_count", "source_excerpt_sha256", "reviewed_by",
    }
    seen_paths = set()
    for record in records:
        assert set(record) == expected_keys
        path = root / record["source_excerpt_path"]
        assert path.is_relative_to(root / "paper_transcriptions")
        assert record["source_excerpt_path"] not in seen_paths
        seen_paths.add(record["source_excerpt_path"])
        excerpt = path.read_bytes()
        excerpt.decode("utf-8")
        assert record["source_excerpt_byte_count"] == len(excerpt)
        assert record["source_excerpt_sha256"] == hashlib.sha256(excerpt).hexdigest()
        assert record["reviewed_by"][0] == "codex-graph-pruning-design-author-v2"
        assert len(record["reviewed_by"]) == 2
        assert record["reviewed_by"][1] != record["reviewed_by"][0]
    assert transcription_set_sha256(records) == TRANSCRIPTION_SET_SHA256

def test_target_claims_are_exact() -> None:
    assert TARGET_CLAIMS == (
        "The paper casts dataset pruning as a graph problem with node weights for intrinsic importance and edge weights for extrinsic diversity/interaction, yielding a Maximum Weight Clique formulation (Section 3.3).",
        "Under mild conditions, the unified objective becomes submodular and admits a greedy approximation guarantee (Section 3.6; Appendix F).",
    )
```

- [ ] **Step 2: Run the red test and record it**

Run:

```bash
uv run --project "$PROJECT" python -m pytest tests/test_provenance.py -q
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

Populate `manifest.json` from every exact expression in the approved design,
preserving PDF pages and distinctions between Appendix E inline, Eq. (26),
literal-derived marginal, single-counted-derived marginal, Eq. (27), and Eq.
(28)--(38). Give every equation record one unique canonical
`paper_transcriptions/excerpts/<record_id>.txt` path. Give the single Algorithm
1 record the unique path `paper_transcriptions/algorithm1.txt`, containing all
of its exact lines. Resolve each path under `paper_transcriptions/`, reject
absolute paths, traversal, duplicate paths, unreferenced excerpt files, and
files referenced more than once. Read raw bytes with `Path.read_bytes()` and
verify the stored byte count, UTF-8 decodability, and SHA-256 without newline,
Unicode, or whitespace normalization. The manifest must not contain an inline
`source_excerpt` key. Compute `TRANSCRIPTION_SET_SHA256` from the ordered
tuples `(record_id, source_excerpt_path, source_excerpt_byte_count,
source_excerpt_sha256)`, and pin that aggregate in `provenance.py`. Store Eq.
(28)'s union/submodular conclusion and its separate \((b-t)\)-maximum
conclusion in the proof ledger, not as an inconsistent manifest-only schema.

Acquire and verify the source bytes once:

```bash
curl --fail --location --proto '=https' --tlsv1.2 --output /tmp/2606.12913v2.pdf https://export.arxiv.org/pdf/2606.12913v2 && test "$(wc -c < /tmp/2606.12913v2.pdf)" -eq 683737 && printf '%s  %s\n' 26ce80e8d347340e0055f2bcf061b6b3e29489fc68a85b8d5711e12cc9da5090 /tmp/2606.12913v2.pdf | sha256sum --check --strict
```

Expected: `/tmp/2606.12913v2.pdf: OK`. Independently compare each committed transcription to the pinned PDF before setting the second `reviewed_by` value.

- [ ] **Step 4: Run the green test**

Run: `uv run --project "$PROJECT" python -m pytest tests/test_provenance.py -q`

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
- Produces: `evaluate_mwcp_edges()`, `evaluate_samplewise_literal()`, `evaluate_objective()`, `symbolic_coefficients()`, and `run_equivalence_audit(source_revision: str) -> dict[str, object]`.
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

Run: `uv run --project "$PROJECT" python -m pytest tests/test_objectives.py tests/test_equivalence.py -q`

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
        # For this variant eta is the fixed modular coefficient eta_mod.
        return evaluate_samplewise_literal(instance, selected) + instance.eta * len(selected)
    raise ValueError("appendix_eq26_score is a score, not a set function")
```

Implement symbolic coefficient maps directly, then enumerate the approved `n in {1, 2}` domain in deterministic size/vertex/weight order. Assert `cases_examined == 26`, persist the smallest nonzero-edge mismatch, and separately state that the arbitrary-weight result is symbolic.
The `modular_shift_candidate` definition above is authoritative in every later
task: it always uses the literal double-counted base. Task 3's sole canonical
parameter builder constructs its later greedy and finite-proof instances with
`eta = 2 * (len(vertices) - 1) * max_abs_interaction`; never dispatch it to
`evaluate_mwcp_edges`.

- [ ] **Step 4: Verify green and finite accounting**

Run: `uv run --project "$PROJECT" python -m pytest tests/test_objectives.py tests/test_equivalence.py -q`

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
- Produces: `canonical_variant_parameters()` and
  `canonical_parameterized_instance_id()` as the sole parameter/ID builders
  for all six set-function variants.
- Produces: `direct_marginal()` by set-function subtraction and `closed_form_marginal()` by independent formulas.
- Produces: `enumerate_diminishing_returns(instance, model_variant)`, `run_diminishing_returns_audit(source_revision)`, `run_shift_audit(source_revision)`, and `minimize_witness(predicate, witness)`.
- Consumes: `Instance` and `evaluate_objective` only in `direct_marginal`; the closed-form implementation may not call either.

- [ ] **Step 1: Write failing exact Appendix E and independence tests**

```python
from fractions import Fraction

from graph_pruning_repro.diminishing_returns import (
    SET_FUNCTION_VARIANTS,
    appendix_shift_witness,
    canonical_parameterized_instance_id,
    canonical_variant_parameters,
    closed_form_marginal,
    direct_marginal,
)
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
    instance = Instance(
        ("x", "y"),
        {"x": Fraction(2), "y": Fraction()},
        {("x", "y"): Fraction(-1), ("y", "x"): Fraction(-2)},
        alpha=Fraction(1),
        eta=Fraction(3),
    )
    expected = {
        "paper_mwcp": Fraction(1),
        "paper_samplewise_literal": Fraction(-1),
        "single_counted_pairwise": Fraction(1),
        "half_corrected_samplewise": Fraction(1, 2),
        "appendix_inline_shift_literal": Fraction(8),
        "modular_shift_candidate": Fraction(2),
    }
    assert set(expected) == set(SET_FUNCTION_VARIANTS)
    for variant, value in expected.items():
        assert direct_marginal(
            instance, frozenset({"y"}), "x", variant
        ) == value
        assert closed_form_marginal(
            instance, frozenset({"y"}), "x", variant
        ) == value

def test_greedy_domain_modular_coefficient_is_fixed_from_graph() -> None:
    vertices = ("x", "y", "z")
    eta_mod = 2 * (len(vertices) - 1) * Fraction(1)
    instance = Instance(
        vertices=vertices,
        vertex_weights={vertex: Fraction() for vertex in vertices},
        interactions={
            ("x", "y"): Fraction(-1),
            ("y", "x"): Fraction(-1),
        },
        eta=eta_mod,
    )
    assert instance.eta == eta_mod
    selected = frozenset({"y"})
    literal = direct_marginal(
        instance, selected, "x", "paper_samplewise_literal"
    )
    modular = direct_marginal(
        instance, selected, "x", "modular_shift_candidate"
    )
    assert modular == literal + instance.eta

def test_canonical_variant_parameters_cover_all_six_variants() -> None:
    graph = Instance(
        vertices=("x", "y"),
        vertex_weights={"x": Fraction(), "y": Fraction()},
        interactions={
            ("x", "y"): Fraction(-1),
            ("y", "x"): Fraction(-1),
        },
    )
    assert {
        variant: canonical_variant_parameters(graph, variant)
        for variant in SET_FUNCTION_VARIANTS
    } == {
        "paper_mwcp": (Fraction(1), Fraction()),
        "paper_samplewise_literal": (Fraction(1), Fraction()),
        "single_counted_pairwise": (Fraction(1), Fraction()),
        "half_corrected_samplewise": (Fraction(1), Fraction()),
        "appendix_inline_shift_literal": (Fraction(1), Fraction(1)),
        "modular_shift_candidate": (Fraction(1), Fraction(2)),
    }
    edgeless = Instance(("x",), {"x": Fraction()}, {})
    assert canonical_variant_parameters(
        edgeless, "appendix_inline_shift_literal"
    ) == (Fraction(1), Fraction())
    expected_suffixes = {
        "paper_mwcp": "::variant=paper_mwcp::alpha=1/1::eta=0/1",
        "paper_samplewise_literal":
            "::variant=paper_samplewise_literal::alpha=1/1::eta=0/1",
        "single_counted_pairwise":
            "::variant=single_counted_pairwise::alpha=1/1::eta=0/1",
        "half_corrected_samplewise":
            "::variant=half_corrected_samplewise::alpha=1/1::eta=0/1",
        "appendix_inline_shift_literal":
            "::variant=appendix_inline_shift_literal::alpha=1/1::eta=1/1",
        "modular_shift_candidate":
            "::variant=modular_shift_candidate::alpha=1/1::eta=2/1",
    }
    for variant, suffix in expected_suffixes.items():
        assert canonical_parameterized_instance_id(graph, variant).endswith(
            suffix
        )
    assert canonical_parameterized_instance_id(
        graph, "appendix_inline_shift_literal"
    ) == (
        "graph=n=2;vw=0/1,0/1;ew=-1/1"
        "::variant=appendix_inline_shift_literal::alpha=1/1::eta=1/1"
    )
```

Add shift tests asserting the Eq. (27) boundary values immediately below/at/above the deduplicated threshold, `6_459` maximum marginal cases, and 256 labeled non-exhaustive rational-alpha controls. The exact modular channel always uses `eta_mod = 2 * (n - 1) * M`; below-threshold fixed-shift diagnostics have separate identifiers and may not be serialized as `modular_shift_candidate`.

- [ ] **Step 2: Verify red**

Run: `uv run --project "$PROJECT" python -m pytest tests/test_diminishing_returns.py tests/test_shifts.py tests/test_minimize.py -q`

Expected: collection fails for missing `diminishing_returns`; log it.

- [ ] **Step 3: Implement exact marginals and bounded searches**

Implement one source-constant table in `diminishing_returns.py`:

| variant | canonical `alpha` | canonical `eta` |
| --- | ---: | ---: |
| `paper_mwcp` | `1` | `0` |
| `paper_samplewise_literal` | `1` | `0` |
| `single_counted_pairwise` | `1` | `0` |
| `half_corrected_samplewise` | `1` | `0` |
| `appendix_inline_shift_literal` | `1` | `M` |
| `modular_shift_candidate` | `1` | `2 * (n - 1) * M` |

Here `M` is the maximum absolute canonical unordered-edge weight and is zero
for an edgeless graph. `canonical_variant_parameters()` computes the tuple
before any objective result is available. The canonical graph ID is exactly
`n=<n>;vw=<w0>,...,<w(n-1)>;ew=<a01>,<a02>,...`, using normalized fractions,
canonical vertex indices, and lexicographic unordered-pair order; use `ew=-`
when there are no edges. The symmetric diminishing-returns graph ID uses an
explicit all-zero vertex vector after symbolic cancellation.
`canonical_parameterized_instance_id()` emits exactly
`graph=<graph-id>::variant=<variant>::alpha=1/1::eta=<p>/<q>`, with a
normalized fraction. Later case IDs append their subset, candidate, budget,
path, or ledger suffix to this complete base ID. Reject a duplicate ID or any
instance whose stored tuple differs from its reconstructed tuple.

The fixed two-vertex Appendix \(1\)-then-\(3\) falsification remains a
separate symbolic diagnostic rather than a member of the canonical per-graph
finite enumeration. Its immutable ID ends in
`alpha=1/1::eta=1/1::diagnostic=appendix-minimal`; this preserves its
zero-edge construction without outcome-dependent parameter selection.

```python
def direct_marginal(instance, selected, candidate, model_variant):
    return evaluate_objective(instance, selected | {candidate}, model_variant) - evaluate_objective(instance, selected, model_variant)

def closed_form_marginal(instance, selected, candidate, model_variant):
    directed_incident = sum(
        (
            instance.interactions.get((candidate, other), Fraction())
            + instance.interactions.get((other, candidate), Fraction())
            for other in selected
        ),
        start=Fraction(),
    )
    unordered_incident = sum(
        (
            instance.interactions.get(
                tuple(sorted((candidate, other))),
                Fraction(),
            )
            for other in selected
        ),
        start=Fraction(),
    )
    vertex = instance.vertex_weights[candidate]
    if model_variant in {"paper_mwcp", "single_counted_pairwise"}:
        return vertex + unordered_incident
    if model_variant == "paper_samplewise_literal":
        return vertex + directed_incident
    if model_variant == "half_corrected_samplewise":
        return vertex + directed_incident / 2
    if model_variant == "appendix_inline_shift_literal":
        return (
            vertex
            + directed_incident
            + instance.alpha * instance.eta * (2 * len(selected) + 1)
        )
    if model_variant == "modular_shift_candidate":
        return vertex + directed_incident + instance.eta
    raise ValueError(f"no closed-form marginal for {model_variant}")
```

Define `SET_FUNCTION_VARIANTS` as exactly the six keys in the test. Reject
`appendix_eq26_score` before enumeration. Enumerate all `A subseteq B`,
`x not in B` directly for each of those six variants. For every symmetric
triple/assignment/variant, count four independent subset objective
evaluations for the two direct differences and two closed-form marginal
evaluations; assert both marginals agree before comparing diminishing returns.
This is the exact
`79_480 * 6 * (4 subset + 2 marginal) == 2_861_280` charged component.
The `19_738 * (4 + 2)` asymmetric component remains a literal-only diagnostic
and is not silently multiplied by six. Stop before iteration if a declared
domain computes a larger ceiling. Implement minimization in the approved
order: vertex deletion, selected/cardinality reduction, zero weights,
absolute magnitude reduction, lexicographic canonicalization. Derive witness
IDs as `sha256(canonical_json).hexdigest()[:16]`.

- [ ] **Step 4: Verify green and exact ceilings**

Run: `uv run --project "$PROJECT" python -m pytest tests/test_diminishing_returns.py tests/test_shifts.py tests/test_minimize.py -q`

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
- Produces: `enumerate_eq7_paths()`, `enumerate_true_marginal_paths()`, `exhaustive_optima()`, `classify_ratio()`, and `run_greedy_audit(source_revision)`.
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

def test_failed_global_premise_is_not_a_guarantee_violation() -> None:
    result = audit_adversarial_poor_ratio_instance()
    assert result["premise_evaluation"]["theorem_eligible"] is False
    assert result["guarantee_violations"] == []
    assert len(result["out_of_premise_diagnostics"]) == 1

def test_no_eligible_instances_is_not_evaluated() -> None:
    result = summarize_guarantee([ineligible_audit_record()])
    assert result["status"] == "not_evaluated"
```

Also assert that Appendix-inline true-marginal selection ties match the unshifted literal path at each fixed iteration while objective ratios are recomputed, never transferred. Add adversarial tests that mutate each of global non-negativity, normalization, global monotonicity, and global submodularity independently and prove the failing record cannot enter `guarantee_violations`.

- [ ] **Step 2: Verify red**

Run: `uv run --project "$PROJECT" python -m pytest tests/test_greedy.py tests/test_algorithm1.py -q`

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

Branch on every tied maximum in lexicographic order and retain all terminal
paths. For literal Algorithm 1, execute transcription lines in order and
return at line 8 because `x*` is absent from state; separately record later
static ambiguities at lines 10-11, 14, and candidate carry-forward without
executing through the first undefined read.

The theorem-premise and ratio audit has exactly six set-function variants:
`paper_mwcp`, `paper_samplewise_literal`, `single_counted_pairwise`,
`half_corrected_samplewise`, `appendix_inline_shift_literal`, and
`modular_shift_candidate`. `appendix_eq26_score` emits one explicit
`not_applicable` premise record and performs no finite premise work. For every
graph, call the Task 3 canonical parameter builder: the four unshifted
variants receive `(alpha, eta)=(1, 0)`,
`appendix_inline_shift_literal` receives `(1, M)`, and the already-defined
literal-base modular objective receives `(1, 2*(n-1)*M)`. Assert the complete
tuple for all six variants, not only the modular coefficient. Every premise,
greedy, optimum, ratio, and later proof record begins with Task 3's exact
parameterized base ID and appends only its canonical budget/path/result
suffix. No path may reconstruct parameters locally or select one after
observing an outcome.

Premises are budget-independent. Enumerate exactly
\[
G=\sum_{n=1}^{4}3^n2^{\binom n2}=5{,}421
\]
weighted graphs and, for every one of the six variants, certify the entire
power set. The exact per-variant work ceilings are:

- \(84{,}750=\sum_n3^n2^{\binom n2}2^n\) subset objective values for global
  non-negativity and normalization;
- \(168{,}555=\sum_n3^n2^{\binom n2}n2^{n-1}\) marginal values for global
  monotonicity; and
- \(565{,}815=\sum_n3^n2^{\binom n2}n3^{n-1}\) diminishing-return
  comparisons for global submodularity.

Thus premise certification is `819_120` work units per variant and
`4_914_720` overall. Store the four booleans plus canonical failing witness
IDs. Only all-true records are theorem-eligible and may be compared with
\(1-1/e\) or placed in `guarantee_violations`; all others go only to
`out_of_premise_diagnostics`. If there are no eligible records, emit
`not_evaluated`. Compare rational \(\rho\) to \(1-1/e\) using rigorously
shrinking rational upper/lower enclosures for \(e\), never binary floating
point.

Enumerate exactly `16_239` weighted-cardinality instances. Replace loose
per-instance maxima with the exact domain sums:

```text
O = 74_145   # optimum subset objective values per variant
P = 210_675  # terminal paths per greedy selector
C = 316_983  # candidate score/look-up operations per selector
```

Compute them from the formulas in the approved design and assert the formulas
before traversal. Compute the optimum table once per variant (`O * 6`) and
reuse it for both greedy families' terminal values. Compute Eq. (7) candidate
scores and paths once (`C` and `P`) because that selector is
variant-independent, then classify its terminal objectives for every variant.
Reuse the global premise marginal table for true-marginal selection: count
each cache lookup/selection comparison in `C * 6` and each path in `P * 6`,
but do not hide a recomputation. Record best, worst, and canonical
classifications for both greedy families and all variants
(`16_239 * 6 * 2 * 3`). Every component records an actual count at or below
its ceiling; ties may make actual path counts smaller, and equality to a
ceiling is never required.

- [ ] **Step 4: Verify green and accounting**

Run: `uv run --project "$PROJECT" python -m pytest tests/test_greedy.py tests/test_algorithm1.py -q`

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
    conclusions = [conclusion for row in rows for conclusion in row["conclusions"]]
    submodularity_conclusions = [
        conclusion
        for conclusion in conclusions
        if "global_submodularity" in conclusion["required_premise_ids"]
    ]
    assert submodularity_conclusions
    assert all(
        conclusion["status"] in {"contradicted", "not_applicable"}
        for conclusion in submodularity_conclusions
    )
    assert all(
        appendix_witness_id in conclusion["witness_ids"]
        for conclusion in submodularity_conclusions
    )
    eq28 = next(row for row in rows if row["equation"] == "28")
    assert set(eq28) == {"row_id", "equation", "model_variant", "instance_id", "conclusions"}
    assert {part["conclusion_id"] for part in eq28["conclusions"]} == {
        "eq28_union_submodular_bound",
        "eq28_b_minus_t_bound",
    }
    assert conclusion_by_check(rows, "ratio_transfer_from_repaired_objective")["status"] == "contradicted"

def test_paper_cardinality_and_product_steps_are_audited() -> None:
    rows = build_symbolic_ledger("paper_samplewise_literal", {})
    cardinality = conclusion_by_check(rows, "optimum_remainder_at_most_b_not_b_minus_t")
    assert cardinality["witness_ids"] == ["cardinality-b-minus-t"]
    assert any(
        conclusion["check_id"] == "product_includes_k_equals_one_zero_factor"
        for row in rows
        for conclusion in row["conclusions"]
    )

def test_failed_prerequisite_blocks_downstream_rows() -> None:
    rows = build_symbolic_ledger("paper_samplewise_literal", {})
    assert status(rows, "28", "eq28_b_minus_t_bound") == "contradicted"
    assert status(rows, "30") == "not_applicable"
    assert (
        "paper_samplewise_literal/symbolic/eq28/eq28_b_minus_t_bound"
        in blocked_by(rows, "30")
    )

def test_equation_36_handles_log_domain_exactly() -> None:
    assert equation_36(0)["status"] == "not_applicable"
    assert equation_36(1)["conclusion_status"] == "supported"
    assert equation_36(1)["log_derivation_status"] == "not_applicable"
    assert equation_36(2)["log_derivation_status"] == "supported"

def test_uniform_conclusion_schema_and_ceiling_are_frozen() -> None:
    rows = build_symbolic_ledger("paper_samplewise_literal", {})
    expected = {
        "conclusion_id", "statement", "required_premise_ids",
        "prerequisite_conclusion_refs", "check_id", "evidence_kind", "status",
        "blocked_by", "witness_ids",
    }
    assert len(rows) == 11
    assert sum(len(row["conclusions"]) for row in rows) == 12
    assert all(set(conclusion) == expected for row in rows for conclusion in row["conclusions"])
    assert declared_aggregate_ceiling() == 13_833_860

def test_prerequisite_graph_rejects_missing_duplicate_replaced_and_cycles() -> None:
    rows = build_symbolic_ledger("paper_samplewise_literal", {})
    for mutation in ("missing", "duplicate", "replaced", "unknown", "cycle"):
        with pytest.raises(ValueError):
            validate_prerequisite_graph(mutate_prerequisites(rows, mutation))
```

- [ ] **Step 2: Verify red**

Run: `uv run --project "$PROJECT" python -m pytest tests/test_proof_ledger.py tests/test_search_accounting.py -q`

Expected: collection fails for missing `proof_ledger`; log it.

- [ ] **Step 3: Implement explicit Eq. (28)--(38) rows**

Represent all 11 numbered equation rows explicitly. Every row has exactly
`row_id`, `equation`, `model_variant`, `instance_id`, and `conclusions`.
Every conclusion, including single-conclusion rows, has exactly
`conclusion_id`, `statement`, `required_premise_ids`,
`prerequisite_conclusion_refs`, `check_id`, `evidence_kind`, `status`,
`blocked_by`, and `witness_ids`. No row-level shadow status, premises,
blockers, check, or witness fields are permitted. Eq. (28) contains two
independently statused conclusions; Eq. (29)--(38) each contain one, for 12
conclusions per variant/instance. A failed prerequisite makes downstream
conclusions `not_applicable`, not automatically contradicted.

Use canonical prerequisite references
`<model_variant>/<instance_id>/<row_id>/<conclusion_id>`. Reconstruct the
complete expected acyclic adjacency map from source constants and reject a
missing, duplicate, unknown, replaced, cross-instance, or cyclic reference.
For finite `modular_shift_candidate` rows, call the same Task 2 literal-base
objective and Task 3 `vertex + directed_incident + eta` marginal, with
`(alpha, eta) = (1, 2 * (n - 1) * M)` asserted from the complete Task 3
parameterized instance ID. Assert the corresponding Task 3 tuple and ID for
the other five finite variants as well. The proof ledger may not construct
parameters locally or infer a single-counted modular alternative.

Gate Eq. (29) on true-marginal argmax and a defined \(S_{t+1}\); Eq. (30) on both Eq. (28) conclusions plus Eq. (29); Eq. (31) on defined algebraic quantities; Eq. (32) on Eq. (30) and \(b-t>0\); Eq. (33) on Eq. (32); Eq. (34) on the recurrence for every \(t=0,\ldots,b-1\) and exact product indices; Eq. (35) on a well-defined product and positive integer \(b\); Eq. (36) on its conclusion separately from the logarithmic derivation; Eq. (37) on Eq. (34)--(36) and all theorem premises; and Eq. (38) on Eq. (37), the exact objective relation, and a nonnegative optimum. For Eq. (36), \(b=1\) supports \(0\leq1/e\) while the `ln(0)` derivation is `not_applicable`; only integer \(b>1\) enters that log derivation, and \(b=0\) is `not_applicable`.

Persist the fixed symbolic \(V=\{a,b,c\}, b=2, t=1, S_t=\{a\},
S^\star=\{b,c\}\) witness showing \(2>b-t=1\). It is a cardinality witness,
independent of weights, and cannot be labeled an Eq. (28a) or theorem
counterexample. Reuse, rather than regenerate, the 16,239 greedy instances.
Evaluate 12 conclusions for all six set-function variants, for a ceiling of
`16_239 * 6 * 12 == 1_169_208` finite conclusion operations. Evaluate 12
symbolic conclusions for all seven named variants, including explicit
`not_applicable` conclusions for `appendix_eq26_score`, for a ceiling of 84.

Compute the aggregate from named constants:

```python
def declared_aggregate_ceiling() -> int:
    return sum((
        52, 2_861_280, 118_428, 45_213, 1_792,
        508_500, 1_011_330, 3_394_890,
        316_983, 210_675, 1_901_898, 1_264_050,
        444_870, 584_604, 1_169_208, 84, 1, 2,
    ))
```

- [ ] **Step 4: Verify green**

Run: `uv run --project "$PROJECT" python -m pytest tests/test_proof_ledger.py tests/test_search_accounting.py -q`

Expected: all pass and the sum is exactly `13_833_860`. The test asserts every
named component and every premise formula, not only the final sum. Every
actual component must be nonnegative and no greater than its declared
ceiling; no equality assumption or smoke allowance exists.

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
- Produces: `build_evidence()`, `validate_evidence(evidence_path, schema_path, evidence_root)`, `canonical_json_bytes()`, and CLI commands `recompute OUTPUT_DIR`, `validate EVIDENCE`, `render EVIDENCE OUTPUT_DIR`.
- Consumes: all prior audit interfaces; the runner orchestrates but does not duplicate their mathematics.

- [ ] **Step 1: Write failing schema, linkage, determinism, and ceiling tests**

```python
def test_required_appendix_witness_has_three_audit_linkages(tmp_path: Path) -> None:
    evidence = build_evidence(tmp_path, source_revision="0" * 40)
    witness = next(w for w in evidence["witnesses"] if w["property"] == "appendix_inline_shift_diminishing_returns")
    assert witness["model_variant"] == "appendix_inline_shift_literal"
    assert witness["inputs"]["alpha"] == "1/1"
    assert witness["inputs"]["eta"] == "1/1"
    assert witness["intermediate_values"]["marginal_empty"] == "1/1"
    assert witness["intermediate_values"]["marginal_y"] == "3/1"
    links = {result["audit"] for result in evidence["claim_results"] if witness["id"] in result["witness_ids"]}
    assert {"diminishing_returns", "greedy_guarantee_premise", "appendix_f_proof_ledger"} <= links

def test_repaired_variants_cannot_claim_literal_witness(tmp_path: Path) -> None:
    evidence = build_evidence(tmp_path, source_revision="0" * 40)
    literal_id = next(w["id"] for w in evidence["witnesses"] if w["model_variant"] == "appendix_inline_shift_literal")
    assert not any(literal_id in result["witness_ids"] for result in evidence["claim_results"] if result["model_variant"] == "modular_shift_candidate")

def test_canonical_build_is_byte_identical(tmp_path: Path) -> None:
    first = build_evidence(tmp_path / "first", source_revision="0" * 40)
    second = build_evidence(tmp_path / "second", source_revision="0" * 40)
    assert canonical_json_bytes(first) == canonical_json_bytes(second)

def test_required_top_level_classification_arrays_are_present(tmp_path: Path) -> None:
    evidence = build_evidence(tmp_path, source_revision="0" * 40)
    assert "guarantee_violations" in evidence
    assert "out_of_premise_diagnostics" in evidence
    assert isinstance(evidence["guarantee_violations"], list)
    assert isinstance(evidence["out_of_premise_diagnostics"], list)

def test_canonical_evidence_contains_no_measured_clock_fields(tmp_path: Path) -> None:
    evidence = build_evidence(tmp_path, source_revision="0" * 40)
    forbidden = {
        "runtime", "runtime_seconds", "wall_time", "wall_time_seconds",
        "duration", "duration_seconds", "elapsed", "elapsed_seconds",
        "started_at", "finished_at",
    }
    def assert_no_clock_keys(value: object) -> None:
        if isinstance(value, dict):
            assert forbidden.isdisjoint(value)
            for child in value.values():
                assert_no_clock_keys(child)
        elif isinstance(value, list):
            for child in value:
                assert_no_clock_keys(child)
    assert_no_clock_keys(evidence)

@pytest.mark.parametrize(
    "mutation",
    [
        "stored_status",
        "transcription_text",
        "witness_id",
        "witness_bytes",
        "witness_hash",
        "witness_link",
        "missing_witness_file",
        "extra_witness_file",
        "replaced_witness_file",
        "missing_middle_graph",
        "duplicate_middle_graph_same_length",
        "replace_middle_graph_under_original_id",
        "missing_middle_subset",
        "duplicate_middle_subset_same_length",
        "replace_middle_subset_under_original_id",
        "wrong_alpha",
        "wrong_unshifted_eta",
        "wrong_appendix_eta",
        "wrong_modular_eta",
        "replaced_parameterized_case_id",
        "missing_guarantee_violations",
        "missing_out_of_premise_diagnostics",
        "remove_out_of_premise_diagnostic",
        "move_ineligible_result_to_guarantee_violations",
        "insert_measured_runtime_field",
        "nested_eq28_conclusion",
        "missing_prerequisite_edge",
        "duplicate_prerequisite_edge",
        "redirected_prerequisite_edge",
        "unknown_prerequisite_edge",
        "cyclic_prerequisite_edge",
        "manifest_missing_key",
        "manifest_replaced_key",
        "excerpt_bytes",
    ],
)
def test_full_replay_rejects_self_consistent_tampering(tmp_path: Path, mutation: str) -> None:
    evidence_path, schema_path = build_fixture_tree(tmp_path)
    apply_adversarial_mutation_and_repair_candidate_metadata(tmp_path, mutation)
    with pytest.raises(ValueError):
        validate_evidence(evidence_path, schema_path, tmp_path)
```

Add subprocess tests that reject altered claim text, an undeclared domain, an
actual count above its ceiling, missing completion status, merged variants, a
non-Fraction rational, an incomplete search presented as a pass, an ineligible
record listed as a guarantee violation, an out-of-premise failure labeled a
counterexample, missing/reordered classification entries, altered canonical
parameters or parameterized IDs, any measured clock field, path traversal,
duplicate IDs, and a pseudo-pointer such as `/witnesses/{id}`. The graph/subset tests must preserve the candidate array
length and update its stored counts, statuses, and hashes so they demonstrate
that validation derives the canonical domain from source rather than trusting
candidate-controlled metadata.

- [ ] **Step 2: Verify red**

Run: `uv run --project "$PROJECT" python -m pytest tests/test_evidence.py tests/test_cli.py -q`

Expected: collection fails for missing `evidence`; log it.

- [ ] **Step 3: Implement schema version 1 and stable serialization**

Require top-level `schema_version`, `attempt_id`, `source_revision`, `paper`,
`target_claims`, `environment`, `transcriptions`, `searches`, `witnesses`,
`guarantee_violations`, `out_of_premise_diagnostics`, `proof_ledger`,
`claim_results`, `unavailable_claims`, `commands`, and `artifacts`. Reject
unknown top-level keys. Serialize with:

```python
def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()
```

Sort records by stable IDs. Canonical evidence, witness files, and command
records contain no measured wall time, start/end timestamp, duration, elapsed
seconds, or host-clock value. Canonical command records may contain only
deterministic argv, return/completion facts, and completion status. Every
canonical byte participates in clean-run and full-replay comparison; no field
is excluded or normalized. A controller wrapper measures runtime externally
and writes it only to its validation attestation or a noncanonical `/tmp` log,
never beneath the project or evidence root. Write canonical files atomically
via a sibling temporary file plus `Path.replace`.

Each witness record includes a canonical relative `artifact_path` under
`evidence/witnesses/` and `artifact_sha256`. After
`jsonschema.Draft202012Validator`, acceptance performs a full deterministic
replay, never a selection of checks named by the candidate:

1. Open the actual transcription manifest. Require the exact record IDs and
   key sets, unique canonical safe paths, and exact referenced-file set. Read
   every excerpt/Algorithm file as bytes and authenticate its stored byte
   count, SHA-256, UTF-8 decoding, reviewed expression, and aggregate
   `TRANSCRIPTION_SET_SHA256`.
2. Reconstruct graph, `M`, all six canonical `(alpha, eta)` tuples,
   parameterized base IDs, budget, subset, marginal, greedy, witness, and
   12-conclusion proof domains entirely from source constants. Do not consume
   candidate arrays, search counts, parameters, IDs, or prerequisite links to
   define expected work.
3. Rerun the complete `13_833_860`-ceiling generation with the candidate's
   `source_revision` into an isolated temporary root. Require each replay and
   candidate component to satisfy `0 <= actual <= declared_ceiling`.
4. Compare the complete canonical `evidence.json` bytes and the exact relative
   witness-file set and bytes. Recompute IDs/hashes from semantic payloads and
   reject missing, extra, replaced, tampered, stale, duplicate, or
   cross-variant files or links.
5. Validate every numeric RFC 6901 render pointer against replayed evidence,
   using a deterministic ID-to-index map and rejecting ID-as-index
   pseudo-pointers.

The replay reconstructs the exact acyclic proof prerequisite graph and all
nested conclusion objects. It therefore rejects missing, duplicate, unknown,
redirected, replaced, cross-instance, or cyclic edges even if the candidate
changes its status and blocker metadata consistently.

Replay also independently regenerates the complete top-level
`guarantee_violations` and `out_of_premise_diagnostics` arrays from canonical
premise and exact-ratio results. Compare their canonical order and bytes;
reject a missing, extra, reordered, removed, or misclassified entry. In
particular, no candidate-controlled classification may move a failed-premise
result into `guarantee_violations`.

- [ ] **Step 4: Verify focused green**

Run: `uv run --project "$PROJECT" python -m pytest tests/test_evidence.py tests/test_cli.py -q`

Expected: all pass.

- [ ] **Step 5: Run the complete evidence command twice**

```bash
source_revision=$(git rev-parse HEAD)
rm -rf /tmp/graph-pruning-evidence-a /tmp/graph-pruning-evidence-b
/usr/bin/time -f '%e' -o /tmp/graph-pruning-recompute-a.seconds uv run --project "$PROJECT" python -m graph_pruning_repro.cli recompute /tmp/graph-pruning-evidence-a --source-revision "$source_revision"
/usr/bin/time -f '%e' -o /tmp/graph-pruning-recompute-b.seconds uv run --project "$PROJECT" python -m graph_pruning_repro.cli recompute /tmp/graph-pruning-evidence-b --source-revision "$source_revision"
cmp /tmp/graph-pruning-evidence-a/evidence.json /tmp/graph-pruning-evidence-b/evidence.json
diff -qr /tmp/graph-pruning-evidence-a/witnesses /tmp/graph-pruning-evidence-b/witnesses
/usr/bin/time -f '%e' -o /tmp/graph-pruning-replay.seconds uv run --project "$PROJECT" python -m graph_pruning_repro.cli validate /tmp/graph-pruning-evidence-a/evidence.json
```

Expected: each recomputation reports the deterministic
`completed actual=<N> ceiling=13833860` with `0 <= N <= 13_833_860`; neither
command requires equality and neither emits measured time into its evidence.
`cmp` and `diff` are silent with exit 0. Validation reports
`schema and full-replay semantic acceptance: PASS`, performs a second pass
whose actual counts remain under the same component ceilings, and reads the
actual transcription and witness files. Generation plus replay therefore has
the declared controller ceiling `27_667_720`. The three `.seconds` files are
noncanonical controller inputs under `/tmp`: each must parse as at most
`1800`, is recorded only in the controller validation attestation, and is
never copied into, hashed with, rendered from, or uploaded with evidence.

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

def test_all_rendered_json_pointers_are_canonical_and_resolve() -> None:
    evidence = load_accepted_evidence()
    for pointer, displayed_value in rendered_pointer_values(evidence):
        assert "{id}" not in pointer
        assert pointer.startswith("/")
        assert resolve_rfc6901(evidence, pointer) == displayed_value

def test_notice_and_both_licenses_are_surfaced() -> None:
    assets = render_distribution_assets(load_accepted_evidence())
    assert {"NOTICE.md", "LICENSE", "LICENSES/CC-BY-NC-SA-4.0.txt"} <= set(assets)
```

- [ ] **Step 2: Verify red**

Run: `uv run --project "$PROJECT" python -m pytest tests/test_render.py -q`

Expected: collection fails for missing `render`; log it.

- [ ] **Step 3: Implement deterministic rendering**

Render provenance and equations, literal-versus-repaired table, independent oracle statuses, canonical witnesses, exact greedy/optimum values and ratios, proof-ledger rows, exhaustive domains and limitations, and the prominent unavailable panel. Every displayed numeric token must resolve to an RFC 6901 evidence JSON pointer recorded in an embedded `data-evidence-path` attribute or report footnote. Because records are arrays, resolve stable IDs through a deterministic ID-to-numeric-index map and emit pointers such as `/witnesses/0/intermediate_values/marginal_empty`, never `/witnesses/{id}`. State explicitly that bounded enumeration can refute but cannot prove arbitrary-real universal claims and that no released implementation resolves edge counting or the shift. Surface the seven-author `NOTICE.md` and both license boundaries in README/report/poster, and include visible/downloadable `NOTICE.md`, `LICENSE`, and `LICENSES/CC-BY-NC-SA-4.0.txt` in the Space distribution.

- [ ] **Step 4: Render and verify**

Run:

```bash
uv run --project "$PROJECT" python -m graph_pruning_repro.cli render "$PROJECT/evidence/evidence.json" "$PROJECT"
uv run --project "$PROJECT" python -m pytest tests/test_render.py -q
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
    actual, ceiling = parse_pass_counts(status)
    assert ceiling == 13_833_860
    assert 0 <= actual <= ceiling
    assert evidence_path.exists()

def test_space_exposes_notice_and_license_downloads() -> None:
    from app import DOWNLOAD_PATHS
    assert {"NOTICE.md", "LICENSE", "LICENSES/CC-BY-NC-SA-4.0.txt"} <= {
        path.as_posix() for path in DOWNLOAD_PATHS
    }
```

- [ ] **Step 2: Verify red**

Run: `uv run --project "$PROJECT" python -m pytest tests/test_app.py -q`

Expected: import fails because `app.py` is absent; log it.

- [ ] **Step 3: Implement the minimal offline Space**

Load and validate committed evidence at startup, render from it, expose `gr.DownloadButton` for canonical JSON, witness files, `NOTICE.md`, MIT `LICENSE`, and `LICENSES/CC-BY-NC-SA-4.0.txt`, and invoke the same `build_evidence` path for recomputation. Show the attribution/license boundary in the UI. Do not fetch the PDF or call Hub APIs from `app.py`.

- [ ] **Step 4: Create a clean environment and run all local gates**

Verify this worktree has its own environment before running:

```bash
test "$(realpath .venv)" = "$(realpath "$PWD/.venv")" || uv venv .venv
uv sync --project "$PROJECT" --frozen
uv run --project "$PROJECT" python -m pytest -q
uv run --project "$PROJECT" python -m graph_pruning_repro.cli validate "$PROJECT/evidence/evidence.json"
uv run python -m pytest -q
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

Confirm no credential-like values, downloaded PDF, cache, mutable unversioned paper URL, environment dump, unrelated submission change, or NAPE change appears. Confirm acceptance reads and hashes each actual witness file and rejects a missing, tampered, or mislinked copy.

- [ ] **Step 6: Invoke completion verification and commit validated source**

Invoke `superpowers:verification-before-completion`, rerun every command it requires, then:

```bash
git add "$PROJECT/app.py" "$PROJECT/tests/test_app.py" "$PROJECT/evidence/tdd-log.jsonl"
git commit -m "feat: expose graph pruning evidence Space"
```

Do not transition lifecycle state. First commit/integrate every executable
source file, JSON Schema, transcription and excerpt, test, application file,
dependency/lock/configuration file, README, notice/license, and render
template. Treat that exact SHA as `source_revision` and embed it in the next
canonical evidence build. Rerun Task 6 and regenerate evidence, witnesses,
report, and both poster files from that source revision, then rerun every Task
8 gate. Commit only those generated files as the artifact-only commit.

The controller-observed current `HEAD` is `artifact_revision`; it is recorded
in the validation attestation and is not embedded in generated files because
its own SHA would be self-referential. Require `source_revision` to be an
ancestor of `artifact_revision`, and require every path changed in that range
to be exactly one of:

```text
evidence/evidence.json
evidence/witnesses/*.json
report.md
poster.html
poster_embed.html
```

within this submission. No source, schema, transcription, excerpt, test,
lockfile, app, config, README, notice, or license may intervene. Any later
executable or template change creates a new `source_revision`, requires a full
regeneration, and starts a new artifact-only range. Deliver the branch as an
untrusted worker proposal for controller validation.

### Task 9: Controller-Only Validation, Deployment, Submission, and Verdict Lifecycle

Task 9 is not worker work. The paper worker returns an untrusted branch proposal
after Task 8 and must not write coordinator state, deploy, submit, poll, import
verdicts, or claim external phases.

**Controller inputs:**
- attempt `64bfe193-333b-4b37-9683-9ac25ca5ac27`;
- the independently approved design and assessed snapshot
  `35d2104cb8462a652d933aa5a776f9b166e8c2724df12da7b35f54cbe19c883d`;
- the embedded final-source `source_revision` and controller-observed current
  `artifact_revision`;
- schema/semantic-accepted evidence rooted in actual transcription and witness
  files; and
- freshly read current owner and fencing token for every mutation.

- [ ] **Step 1: Controller revalidates and attests validation**

Inspect the current schema-v6 attempt and lease, confirm independent design
approval, rerun the complete local gates, and verify the embedded
`source_revision`, observed `artifact_revision`, ancestry, exact artifact-only
diff allowlist, actual file hashes, `13_833_860` generation ceiling,
`13_833_860` replay ceiling, every `actual <= declared_ceiling`, and the
NOTICE/license assets. Measure generation and replay outside the canonical
commands, require each duration to be at most 1,800 seconds, and put those
measurements only in the controller attestation input or a noncanonical
`/tmp` log. Only the controller may record the immutable result with
`attest-validation`. Do not substitute a generic phase transition.

- [ ] **Step 2: Controller deploys, verifies, and publishes deployment**

Use the current `hf-cli` and `huggingface-spaces` skills to deploy one
paper-specific CPU Space. Verify the exact remote SHA, healthy runtime, clean
logs, live recomputation with reported `ceiling=13_833_860` and
`actual <= ceiling`, downloaded evidence and witness hashes, target claims,
`source_revision`, observed `artifact_revision`, `NOTICE.md`, MIT `LICENSE`,
and CC BY-NC-SA legal text. Only then record the immutable observation with
`publish-deployment`.

- [ ] **Step 3: Controller refreshes and submits exact claims**

Run `refresh-live` without assessments, inspect its immutable raw snapshot,
produce a revision-bound independent assessment, rerun `refresh-live` with that
assessment, and cancel on eligibility, claim, source, or verdict drift. Submit
only the two byte-exact `evidence.json.target_claims` and record the returned
external identity with `attest-submission`.

- [ ] **Step 4: Controller proves presence and watches bounded judging**

Run a post-submission refresh and verify the exact Space/SHA in queued or live
state. If absent, stale, or terminal, persist a blocker. Otherwise invoke
`watch-attempt` with explicit finite polling bounds; do not synthesize
observations or let a worker poll.

- [ ] **Step 5: Controller imports the official verdict**

Use `sync-verdict` for the official exact-claim verdict and preserve immutable
provenance. Validate coordinator tests, targeted pre-commit, diff scope, and a
HANDOFF milestone after every material controller phase. The controller never
auto-abandons this attempt, and each attestation uses the explicit attempt ID,
fresh owner, and current fencing token.

## Final Implementation Checklist

- [ ] Every production behavior first appeared behind an observed failing test and a `tdd-log.jsonl` red record.
- [ ] PDF acquisition command, byte count, digest, revision, pages, equations, exact excerpt paths/raw byte counts/SHA-256 values, aggregate transcription pin, and two-person review are complete.
- [ ] Seven objective/score variants and three greedy paths remain distinct.
- [ ] `modular_shift_candidate` is the literal-base objective `f_lit + eta_mod|S|` in objective, marginal, shift, greedy, optimum, premise, proof-ledger, and accounting paths, with greedy/proof `eta_mod=2(n-1)M`.
- [ ] Every canonical graph derives all six tuples from one table: `alpha=1`,
  unshifted `eta=0`, Appendix `eta=M`, and modular `eta=2(n-1)M`; exact
  fractions appear in reconstructed parameterized case IDs without adding an
  enumeration dimension.
- [ ] Eq. (3) and literal Eq. (4)--(5) are independently traversed and symbolically compared.
- [ ] Objective witness search examines its exact 26-case finite domain.
- [ ] Independent direct and closed-form marginals agree at formula-specific values for all six charged set-function variants; symmetric and asymmetric diminishing-returns controls enforce ceilings 79,480 and 19,738.
- [ ] Shift controls enforce ceilings 6,459 and 256 and distinguish exhaustive from non-exhaustive evidence.
- [ ] Appendix-inline literal witness is exactly two vertices, zero weights, `alpha=eta=1`, and marginals `1/1` then `3/1`.
- [ ] The Appendix witness links diminishing returns, greedy-guarantee premise, and every applicable proof-ledger row, and never links a repaired variant.
- [ ] Eq. (7) and true-marginal greedy retain all ties independently; Algorithm 1 stops at line 8 without repair.
- [ ] All six set-function variants certify `84,750` subset values, `168,555` marginals, and `565,815` submodularity comparisons per variant; the score-only variant is explicitly not applicable.
- [ ] Greedy enumeration has 16,239 instances, exact `O=74,145`, `P=210,675`, and `C=316,983` ceilings, counted cache lookups/classifications, and exact ratios; only globally nonnegative, normalized, monotone, and submodular instances can enter `guarantee_violations`.
- [ ] The required top-level `guarantee_violations` and
  `out_of_premise_diagnostics` arrays are independently replayed; out-of-premise
  failures are stored separately, never called counterexamples, and a domain
  with no eligible instance is `not_evaluated`.
- [ ] Appendix F uses the one normalized nested conclusion schema for every row, splits Eq. (28), persists the symbolic \(b-t\) witness, reconstructs an exact acyclic prerequisite graph, gates Eq. (29)--(38), and handles Eq. (36) at \(b=0\), \(b=1\), and \(b>1\).
- [ ] The generation ceiling is exactly 13,833,860 and full-replay validation
  has the same ceiling; all component actuals are at or below ceilings and the
  withdrawn 1,177,735 total is never an equality target. The controller's
  external, noncanonical measurements put generation and replay at or below
  1,800 seconds each without domain shrinkage.
- [ ] Canonical evidence passes JSON Schema plus source-derived full deterministic replay and is byte-identical, including witness file sets/bytes, across two clean runs.
- [ ] Canonical evidence and command records contain no measured clock data;
  every canonical byte is compared without exclusion or normalization, while
  runtime exists only in controller attestation or a noncanonical `/tmp` log.
- [ ] Acceptance reconstructs every canonical domain, reads/hashes actual transcription and witness bytes, and rejects missing, duplicate, replaced, tampered, or mislinked domain records, nested conclusions, prerequisite edges, and files even after candidate metadata is made self-consistent.
- [ ] Report, poster, and Space derive numeric statements only from canonical JSON; every RFC 6901 pointer resolves, and empirical claims are prominently unavailable.
- [ ] `NOTICE.md`, MIT `LICENSE`, CC BY-NC-SA legal text, attribution, seven authors, source URL, and adaptation status are visible and downloadable locally and on Space.
- [ ] Submission `python -m pytest`, root `python -m pytest`, targeted pre-commit, diff/security review, and `verification-before-completion` pass.
- [ ] Evidence embeds the final executable `source_revision`; controller attestation records the non-self-referential `artifact_revision`; their intervening diff contains only the five allowed generated artifact path classes.
- [ ] Controller attestations use freshly read owner/token and receive separate HANDOFF milestones; the worker claims no lifecycle phase.
- [ ] Exact deployed Space SHA and live recomputation/download are verified before `publish-deployment`.
- [ ] Fresh assessed pre-submission and post-submission snapshots prove eligibility and queued/live presence before bounded judging.
