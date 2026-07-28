# Learning Randomized Reductions Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic CPU-only reproduction of all five admitted Learning Randomized Reductions claim paths from pinned papers, released raw results, and bounded executable audits, including an honest falsification of the conflated nonlinear-invariant claim.

**Architecture:** A verified immutable-input layer feeds five independent claim auditors: finite correlated-sampling theory, the 80-function census, Vanilla Bitween plus sigmoid, Agentic Bitween, and cross-version claim-scope consistency. One closed evidence renderer supplies canonical JSON, root `pages/*.md`, and a read-only Gradio Space; acquisition is the only networked operation and no paper value is relabeled as a reproduced measurement.

**Tech Stack:** Python 3.12, standard library (`ast`, `csv`, `dataclasses`, `fractions`, `hashlib`, `json`, `pathlib`), SymPy 1.14, pypdf 6.1, jsonschema 4.25, Gradio 6.20, pytest 8.4, `uv`.

## Global Constraints

- Paper ID is `hCAEcqig2C`; attempt ID is `eb10c79b-fc26-47c4-88c1-6f45cb592833`.
- Work only in `submissions/learning-randomized-reductions/`. Every implementation commit is a worker proposal.
- Pin arXiv `2412.18134v1`, arXiv `2412.18134v5`, and repository commit `e13d4b59f6d23051c73e07cfc447336da84e7bd2`.
- Use no GPU, paid API, LLM inference, Gurobi run, training, or remote Agentic Bitween rerun. Estimated API cost remains USD 0.00.
- Paper text and reported values are `paper_context`, never reproduced measurements. Reproduced values must come from raw CSV aggregation, source census, symbolic algebra, finite enumeration, or source-location consistency checks.
- Preserve all five exact challenge claims and SHA-256 bindings from the approved design. Do not soften the fifth claim; mark it falsified when the pinned sources establish the table/benchmark conflation.
- For claim three, verify `43/80` and the sigmoid identity, but state that historical “first known” priority was not exhaustively reproduced.
- Network access is allowed only for `lrr-repro acquire`. `audit`, `validate`, tests, page rendering, and app import must work offline.
- Canonical evidence excludes credentials, environment dumps, absolute paths, host identity, random values, and wall-clock timestamps.
- Use a failing test before each production behavior. Keep the complete evidence run CPU-only and under 30 minutes.
- Root reviewer pages must live at `pages/*.md`; `app.py` reads them and committed evidence without calculating independent results.
- README Space metadata must include `sdk: gradio`, `sdk_version: 6.20.0`, `app_file: app.py`, `paper-hCAEcqig2C`, and `icml2026-repro`.
- Do not mutate coordinator state, `docs/HANDOFF.md`, skill source, another submission, NAPE, Hub resources, submissions, verdicts, or controller attestations.

---

### Task 1: Project skeleton and immutable provenance

**Files:**
- Create: `submissions/learning-randomized-reductions/pyproject.toml`
- Create: `submissions/learning-randomized-reductions/uv.lock`
- Create: `submissions/learning-randomized-reductions/src/lrr_repro/__init__.py`
- Create: `submissions/learning-randomized-reductions/src/lrr_repro/provenance.py`
- Create: `submissions/learning-randomized-reductions/scripts/acquire_upstream.py`
- Create: `submissions/learning-randomized-reductions/evidence/inputs/upstream_manifest.json`
- Create: `submissions/learning-randomized-reductions/evidence/inputs/paper_context.json`
- Create: `submissions/learning-randomized-reductions/evidence/inputs/upstream/`
- Test: `submissions/learning-randomized-reductions/tests/test_provenance.py`

**Interfaces:**
- Consumes: the exact URLs, hashes, sizes, Git blobs, and licenses in the approved design.
- Produces:

```python
class IntegrityError(ValueError):
    """A pinned input or manifest failed verification."""

@dataclass(frozen=True)
class VerifiedInput:
    artifact_id: str
    relative_path: str | None
    sha256: str
    size_bytes: int
    git_blob: str | None

git_blob_id(payload: bytes) -> str
load_verified_inputs(project_root: Path, cache_dir: Path) -> tuple[VerifiedInput, ...]
load_paper_context(project_root: Path) -> Mapping[str, object]
```

- [ ] **Step 1: Write failing provenance tests**

```python
def test_pins_exact_upstreams(project_root, cache_dir):
    items = {item.artifact_id: item for item in load_verified_inputs(project_root, cache_dir)}
    assert items["paper-v1"].sha256 == "abaac08eabec2e77c8af7ae3ca028691b9cd862e21bfa779452b9fd729e3222f"
    assert items["paper-v5"].sha256 == "93cab4aa8cec06434b704e639bab87dd15ea95ac46a335961138a94fc1bae2b8"
    assert items["results-csv"].git_blob == "0432241ef42d1be06179546c7b96d6bf6f598986"


def test_tampered_input_fails_closed(project_root, populated_cache):
    path = populated_cache / "results.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(IntegrityError, match="results-csv.*SHA-256"):
        load_verified_inputs(project_root, populated_cache)


def test_manifest_rejects_unsafe_or_duplicate_paths(project_root, cache_dir):
    manifest = read_manifest(project_root)
    manifest["artifacts"][0]["relative_path"] = "../escape"
    with pytest.raises(IntegrityError, match="safe relative path"):
        validate_manifest(manifest)
```

- [ ] **Step 2: Run RED**

Run: `cd submissions/learning-randomized-reductions && uv run pytest -q tests/test_provenance.py`

Expected: collection fails because `lrr_repro.provenance` and the manifest do not exist.

- [ ] **Step 3: Implement the pinned manifest and verifier**

The manifest must contain the two PDF records and these repository records:

| Path | Bytes | Blob | SHA-256 |
|---|---:|---|---|
| `LICENSE` | 1,091 | `2eb9ad588c5b6e720b168588a640d7a653265c96` | `04601314559ab36aa7403fbaa56ccba106be0de6671190497e6835bbd3107bdb` |
| `results/Bitween-Results(Sheet1-ICML).csv` | 502,974 | `0432241ef42d1be06179546c7b96d6bf6f598986` | `7198413f93830f7903bf3b670b718f2ccfbab1a41496a1fc3fe085850af0df0b` |
| `src/bitween/evaluation/evaluation_rsr_bench_paper.py` | 47,780 | `1aa8de34e60dcdbf77c0af53e1d5af25a673522f` | `6afe05589eeb08f34d63f98ed55fc38a3856a84ce7fc1d21e47327baad54ffbf` |
| `src/bitween/evaluation/evaluation_rsr_bench_paper_extended.py` | 29,907 | `88f1ef6b3c1b280afd8d2754509a1f6f0b30df7c` | `02fcfb7805e1704e040ae9e854b22ab199ae7ea95a18b5e45f2f9c886c0f40e2` |
| `src/bitween/evaluation/evaluation_rsr_bench_agentic_paper.py` | 34,555 | `e660ab8d39b8083117097641983fe69c5efaeffb` | `c99842f831d6bf0296452e632a5b0eb24f8ae9438acc42bad92604ac61c64bb0` |
| `src/bitween/evaluation/rsr_checker.py` | 22,751 | `417f5f0a8ef6789be4f01c3108f4427f2580c9d0` | `28e9dc80cec82d8e11dee4e867b55cba707b295c8f9d2065c7e9d286967fd3aa` |
| `src/bitween/pac.py` | 1,395 | `4dd47da9cde937e7f7074424a9cddb5f3aa523ab` | `69da63ffce26b7f85f306a69e13eb345a8c12be7257f60297b96c5122c2274a4` |

Implement Git blobs as:

```python
def git_blob_id(payload: bytes) -> str:
    return hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
```

`acquire_upstream.py` downloads to a temporary directory, verifies all bytes,
then atomically installs repository files under `evidence/inputs/upstream/`
and PDFs under ignored `.cache/upstream/`. `paper_context.json` records exact
page/table locators and labels as context bound to the two PDF hashes.

- [ ] **Step 4: Run GREEN and commit**

```bash
uv lock
uv run pytest -q tests/test_provenance.py
git add pyproject.toml uv.lock src/lrr_repro/__init__.py src/lrr_repro/provenance.py scripts/acquire_upstream.py evidence/inputs tests/test_provenance.py
git commit -m "evidence: pin randomized-reductions inputs"
```

Expected: provenance tests pass; no PDF is staged.

---

### Task 2: Correlated-sampling theory lane

**Files:**
- Create: `submissions/learning-randomized-reductions/src/lrr_repro/theory.py`
- Test: `submissions/learning-randomized-reductions/tests/test_theory.py`

**Interfaces:**
- Consumes: verified v5 paper context plus finite domains/functions supplied by tests.
- Produces:

```python
@dataclass(frozen=True)
class FiniteRSR:
    domain: tuple[int, ...]
    randomness: tuple[int, ...]
    queries: tuple[Callable[[int, int], int], ...]
    recovery: Callable[[int, int, tuple[int, ...]], int]

@dataclass(frozen=True)
class TheoryAudit:
    marginal_uniform: bool
    perfect_for_hypothesis: bool
    epsilon: Fraction
    good_input_fraction: Fraction
    minimum_recovery_probability: Fraction
    implication_holds: bool

audit_claim_a1(rsr: FiniteRSR, truth: Mapping[int, int], hypothesis: Mapping[int, int], rho: Fraction, xi: Fraction) -> TheoryAudit
verify_theory_locators(context: Mapping[str, object]) -> None
```

- [ ] **Step 1: Write the failing finite-model tests**

```python
def test_correlated_queries_need_only_uniform_marginals():
    rsr = modular_addition_rsr(4)
    audit = audit_claim_a1(
        rsr,
        truth={0: 1, 1: 1, 2: 2, 3: 3},
        hypothesis={0: 0, 1: 1, 2: 2, 3: 3},
        rho=Fraction(1, 2),
        xi=Fraction(1, 4),
    )
    assert audit.marginal_uniform is True
    assert audit.epsilon == Fraction(1, 4)
    assert audit.good_input_fraction >= Fraction(3, 4)
    assert audit.minimum_recovery_probability >= Fraction(1, 2)
    assert audit.implication_holds is True


def test_nonuniform_marginal_is_rejected():
    with pytest.raises(ValueError, match="marginal uniformity"):
        audit_claim_a1(nonuniform_rsr(), IDENTITY, IDENTITY, Fraction(1, 2), Fraction(1, 4))
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest -q tests/test_theory.py`

Expected: collection fails because `lrr_repro.theory` does not exist.

- [ ] **Step 3: Implement exact enumeration**

Use `q1(x,r)=(x+r) mod 4`, `q2(x,r)=r`, and
`p(x,r,(y1,y2))=(y1-y2) mod 4` for the canonical test model. Enumerate every
`x`, `r`, and query marginal with `Fraction`; compute
`epsilon=min(rho/k, xi)` and never use floats. Verify paper-context locators for
Definitions 4.1, 4.3, 4.5 and Claims A.1–A.2 before returning a verified lane.

- [ ] **Step 4: Run GREEN and commit**

```bash
uv run pytest -q tests/test_theory.py tests/test_provenance.py
git add src/lrr_repro/theory.py tests/test_theory.py
git commit -m "evidence: audit correlated-sampling theory"
```

---

### Task 3: 80-function census lane

**Files:**
- Create: `submissions/learning-randomized-reductions/src/lrr_repro/benchmark.py`
- Test: `submissions/learning-randomized-reductions/tests/test_benchmark.py`

**Interfaces:**
- Consumes: verified base and extended benchmark scripts and results CSV.
- Produces:

```python
@dataclass(frozen=True)
class BenchmarkRecord:
    benchmark_id: int
    source_name: str
    csv_name: str

extract_test_ids(source: str) -> tuple[tuple[int, str], ...]
read_primary_csv_rows(path: Path) -> tuple[dict[str, str], ...]
build_census(base_source: str, extended_source: str, csv_path: Path) -> tuple[BenchmarkRecord, ...]
```

- [ ] **Step 1: Write failing census tests**

```python
def test_pinned_sources_and_csv_reconcile_to_exactly_80(project_root):
    records = build_census(base_text(project_root), extended_text(project_root), csv_path(project_root))
    assert [record.benchmark_id for record in records] == list(range(1, 81))
    assert len({record.csv_name for record in records}) == 80
    assert records[32].csv_name == "sigmoid"


def test_continuation_rows_do_not_inflate_count(csv_fixture):
    assert len(read_primary_csv_rows(csv_fixture)) == 80


def test_duplicate_or_missing_id_fails(csv_fixture):
    corrupt_primary_id(csv_fixture, old=80, new=79)
    with pytest.raises(ValueError, match="exactly 1..80"):
        build_census(BASE, EXTENDED, csv_fixture)
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest -q tests/test_benchmark.py`

Expected: collection fails because `lrr_repro.benchmark` does not exist.

- [ ] **Step 3: Implement syntax-only registration extraction**

Parse each script with `ast.parse`. Walk calls to `evaluate(...)`, extract only
literal `test_id="<integer>_<name>"` keyword values, and select IDs 1–40 from
the base script plus 41–80 from the extended script. Do not count all
`test_*` definitions: the files contain helpers and unused definitions.
Parse CSV rows only when column zero is an integer in `1..80`; attach blank-ID
rows as continuation data.

- [ ] **Step 4: Run GREEN and commit**

```bash
uv run pytest -q tests/test_benchmark.py tests/test_provenance.py
git add src/lrr_repro/benchmark.py tests/test_benchmark.py
git commit -m "evidence: census the 80 RSR benchmarks"
```

---

### Task 4: Vanilla, sigmoid, and Agentic result lanes

**Files:**
- Create: `submissions/learning-randomized-reductions/src/lrr_repro/results.py`
- Test: `submissions/learning-randomized-reductions/tests/test_results.py`

**Interfaces:**
- Consumes: Task 3 primary rows and released property continuation rows.
- Produces:

```python
@dataclass(frozen=True)
class BackendSummary:
    backend: str
    rsr_total: int
    covered_benchmarks: int
    coverage: Fraction
    runtime_min: Decimal
    runtime_mean: Decimal
    runtime_max: Decimal

summarize_backend(rows: Sequence[Mapping[str, str]], rsr_column: int, time_column: int, name: str) -> BackendSummary
verify_sigmoid_identity() -> bool
extract_function_arguments(expression: str) -> tuple[str, ...]
canonical_query(argument: str) -> str
novel_agentic_queries(rows: Sequence[Mapping[str, str]]) -> tuple[str, ...]
```

- [ ] **Step 1: Write failing aggregate and algebra tests**

```python
def test_recomputes_vanilla_and_agentic_coverage(primary_rows):
    lr = summarize_backend(primary_rows, 18, 21, "vanilla-lr")
    agentic = summarize_backend(primary_rows, 53, 56, "agentic-opus")
    assert (lr.rsr_total, lr.covered_benchmarks, lr.coverage) == (87, 43, Fraction(43, 80))
    assert (lr.runtime_min, lr.runtime_mean, lr.runtime_max) == (
        Decimal("0.13"), Decimal("4.791"), Decimal("19.12")
    )
    assert (agentic.rsr_total, agentic.covered_benchmarks, agentic.coverage) == (
        793, 64, Fraction(4, 5)
    )


def test_sigmoid_reduction_is_an_identity():
    assert verify_sigmoid_identity() is True


def test_agentic_outputs_contain_queries_outside_fixed_set(all_rows):
    queries = novel_agentic_queries(all_rows)
    assert "x+log(k)" in queries
    assert not {"x+r", "x-r", "x*r", "x", "r"}.intersection(queries)
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest -q tests/test_results.py`

Expected: collection fails because `lrr_repro.results` does not exist.

- [ ] **Step 3: Implement exact aggregation and expression checks**

Use `Decimal` for times and `Fraction` for coverage. Count coverage as
`int(row[rsr_column]) > 0`, never from the spreadsheet's summary row.
Verify:

```python
x, r = sympy.symbols("x r", real=True)
sigma = lambda value: 1 / (1 + sympy.exp(-value))
rhs = sigma(x + r) * (sigma(r) - 1) / (
    2 * sigma(x + r) * sigma(r) - sigma(x + r) - sigma(r)
)
assert sympy.simplify(sigma(x) - rhs) == 0
```

Extract `f(...)` arguments with a balanced-parenthesis scanner. Canonicalize
whitespace and multiplication symbols, compare against the exact fixed set,
and sort/deduplicate novel arguments. A malformed expression is an explicit
parse error, not an ignored row.

- [ ] **Step 4: Run GREEN and commit**

```bash
uv run pytest -q tests/test_results.py tests/test_benchmark.py
git add src/lrr_repro/results.py tests/test_results.py
git commit -m "evidence: recompute Bitween result claims"
```

---

### Task 5: Honest nonlinear-invariant falsification lane

**Files:**
- Create: `submissions/learning-randomized-reductions/src/lrr_repro/claim_scope.py`
- Test: `submissions/learning-randomized-reductions/tests/test_claim_scope.py`

**Interfaces:**
- Consumes: verified paper context and locally cached pinned PDF text.
- Produces:

```python
@dataclass(frozen=True)
class SourceLocator:
    version: str
    section: str
    table: str | None
    benchmark: str
    compared_methods: tuple[str, ...]
    metrics: tuple[str, ...]

@dataclass(frozen=True)
class ScopeAudit:
    exact_claim_supported: bool
    status: Literal["verified", "falsified", "inconclusive"]
    contradictions: tuple[str, ...]
    locators: tuple[SourceLocator, ...]

audit_nonlinear_invariant_claim(context: Mapping[str, object]) -> ScopeAudit
```

- [ ] **Step 1: Write failing source-scope tests**

```python
def test_exact_live_claim_is_falsified(paper_context):
    audit = audit_nonlinear_invariant_claim(paper_context)
    assert audit.status == "falsified"
    assert audit.exact_claim_supported is False
    assert set(audit.contradictions) == {
        "v1 Table 2 is a learned post-condition example, not a backend comparison",
        "LR versus MILP sample/runtime results are for RSR-Bench, not NLA-DigBench",
        "NLA-DigBench compares Bitween with DIG and SymInfer, not MILP",
        "v5 Table 2 reports novel Agentic Bitween query functions",
    }


def test_missing_locator_is_inconclusive(paper_context):
    del paper_context["versions"]["v1"]["nla_digbench"]
    assert audit_nonlinear_invariant_claim(paper_context).status == "inconclusive"
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest -q tests/test_claim_scope.py`

Expected: collection fails because `lrr_repro.claim_scope` does not exist.

- [ ] **Step 3: Implement conjunctive source matching**

Require one source locator to match all four live-claim dimensions:
`benchmark=nonlinear invariant`, `methods=(regression,MILP)`,
`metrics=(sample count,runtime)`, and `table=2`. Emit `verified` only if one
locator matches every dimension, `falsified` when verified locators establish
the four contradictions above, and `inconclusive` when required source text or
hash verification is absent. Preserve the narrower RSR-Bench and NLA-DigBench
statements as context, not replacement claims.

- [ ] **Step 4: Run GREEN and commit**

```bash
uv run pytest -q tests/test_claim_scope.py tests/test_provenance.py
git add src/lrr_repro/claim_scope.py tests/test_claim_scope.py
git commit -m "evidence: audit the nonlinear-invariant claim"
```

---

### Task 6: Closed evidence schema, CLI, and committed bundle

**Files:**
- Create: `submissions/learning-randomized-reductions/schema/evidence-v1.schema.json`
- Create: `submissions/learning-randomized-reductions/src/lrr_repro/evidence.py`
- Create: `submissions/learning-randomized-reductions/src/lrr_repro/cli.py`
- Create: `submissions/learning-randomized-reductions/tests/test_evidence.py`
- Create: `submissions/learning-randomized-reductions/tests/test_cli.py`
- Create: `submissions/learning-randomized-reductions/evidence/results.json`
- Create: `submissions/learning-randomized-reductions/evidence/commands.json`
- Create: `submissions/learning-randomized-reductions/evidence/validation.json`

**Interfaces:**
- Consumes: Tasks 1–5 audit records.
- Produces:

```python
build_evidence(project_root: Path, cache_dir: Path) -> dict[str, object]
canonical_json(value: object) -> bytes
validate_evidence(value: object, schema_path: Path) -> None
build_worker_proposal(evidence_bytes: bytes, source_commit: str, source_tree: str) -> dict[str, object]
main(argv: list[str] | None = None) -> int
```

The console script is `lrr-repro` with `acquire`, `audit`, `validate`, and
`propose` subcommands.

- [ ] **Step 1: Write failing evidence and CLI tests**

```python
def test_evidence_binds_all_claims_and_honest_outcomes(project_root, cache_dir):
    evidence = build_evidence(project_root, cache_dir)
    assert evidence["attempt_id"] == "eb10c79b-fc26-47c4-88c1-6f45cb592833"
    assert [claim["challenge_claim_sha256"] for claim in evidence["claims"]] == EXPECTED_HASHES
    assert evidence["claims"][4]["status"] == "falsified"
    assert "historical priority" in " ".join(evidence["claims"][2]["limitations"])
    assert evidence["unavailable_operations"] == [
        "agentic_rerun", "gpu_training", "gurobi_rerun", "paid_api"
    ]


def test_offline_audits_are_byte_identical(cli_fixture, tmp_path):
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    run_offline_audit(cli_fixture, first)
    run_offline_audit(cli_fixture, second)
    assert first.read_bytes() == second.read_bytes()


def test_worker_proposal_has_no_external_mutation():
    proposal = build_worker_proposal(b"{}\n", "a" * 40, "b" * 40)
    assert proposal["requested_action"] == "controller_validation"
    assert proposal["external_mutations"] == []
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest -q tests/test_evidence.py tests/test_cli.py`

Expected: collection fails because the evidence renderer, schema, and CLI do not exist.

- [ ] **Step 3: Implement canonical rendering and atomic CLI output**

```python
def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def command_audit(args: argparse.Namespace) -> int:
    evidence = build_evidence(args.project_root, args.cache_dir)
    validate_evidence(evidence, args.schema)
    atomic_write(args.output, canonical_json(evidence))
    return 0
```

Set `additionalProperties: false` at every owned schema object. Require source
hashes, five ordered claims, observations, limitations, commands, environment,
contradictions, unavailable operations, and artifact JSON pointers. Map lanes
1, 2, and 4 to `verified`, lane 3 to `partial` because historical priority is
unreplicated, and lane 5 to `falsified`.

- [ ] **Step 4: Generate and validate the committed bundle**

```bash
uv run lrr-repro acquire --manifest evidence/inputs/upstream_manifest.json --cache-dir .cache/upstream
env ALL_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 NO_PROXY= uv run lrr-repro audit --project-root . --cache-dir .cache/upstream --schema schema/evidence-v1.schema.json --output evidence/results.json
uv run lrr-repro validate evidence/results.json --schema schema/evidence-v1.schema.json --validation-output evidence/validation.json
uv run pytest -q tests/test_evidence.py tests/test_cli.py
```

Record literal commands and exit codes in `commands.json`; bind
`validation.json` to the exact `results.json` SHA-256.

- [ ] **Step 5: Commit the renderer and bundle**

```bash
git add pyproject.toml uv.lock schema src/lrr_repro/evidence.py src/lrr_repro/cli.py tests/test_evidence.py tests/test_cli.py evidence/results.json evidence/commands.json evidence/validation.json
git commit -m "evidence: render randomized-reductions bundle"
```

---

### Task 7: Root pages, documentation, Space, and final proposal

**Files:**
- Create: `submissions/learning-randomized-reductions/pages/00-summary.md`
- Create: `submissions/learning-randomized-reductions/pages/01-correlated-sampling-theory.md`
- Create: `submissions/learning-randomized-reductions/pages/02-rsr-bench-census.md`
- Create: `submissions/learning-randomized-reductions/pages/03-vanilla-and-sigmoid.md`
- Create: `submissions/learning-randomized-reductions/pages/04-agentic-bitween.md`
- Create: `submissions/learning-randomized-reductions/pages/05-nonlinear-invariant-falsification.md`
- Create: `submissions/learning-randomized-reductions/pages/06-methods-and-provenance.md`
- Create: `submissions/learning-randomized-reductions/README.md`
- Create: `submissions/learning-randomized-reductions/app.py`
- Create: `submissions/learning-randomized-reductions/tests/test_pages.py`
- Create: `submissions/learning-randomized-reductions/tests/test_space.py`
- Create: `submissions/learning-randomized-reductions/evidence/worker-proposal.json`

**Interfaces:**
- Consumes: committed `evidence/results.json`; no cache and no network.
- Produces: ordered root Markdown pages, a read-only Gradio app, exact local reproduction instructions, and a controller-validation proposal.

- [ ] **Step 1: Write failing page and Space tests**

```python
def test_root_pages_cover_every_claim(project_root):
    pages = sorted((project_root / "pages").glob("*.md"))
    assert [page.name for page in pages] == EXPECTED_PAGE_NAMES
    text = "\n".join(page.read_text() for page in pages)
    for digest in EXPECTED_HASHES:
        assert digest in text
    assert "falsified" in (project_root / "pages/05-nonlinear-invariant-falsification.md").read_text().lower()


def test_space_is_offline_and_evidence_backed(monkeypatch):
    monkeypatch.setattr(socket, "create_connection", fail_network)
    module = load_app()
    assert module.EVIDENCE["paper_id"] == "hCAEcqig2C"
    assert len(module.PAGE_TEXT) == 7


def test_readme_has_exact_space_metadata(project_root):
    readme = (project_root / "README.md").read_text()
    assert "sdk_version: 6.20.0" in readme
    assert "- paper-hCAEcqig2C" in readme
    assert "- icml2026-repro" in readme
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest -q tests/test_pages.py tests/test_space.py`

Expected: failures because the pages, README, and app do not exist.

- [ ] **Step 3: Implement reviewer surfaces**

```python
ROOT = Path(__file__).resolve().parent
EVIDENCE = json.loads((ROOT / "evidence/results.json").read_text())
PAGE_PATHS = sorted((ROOT / "pages").glob("*.md"))
PAGE_TEXT = tuple(path.read_text() for path in PAGE_PATHS)

with gr.Blocks() as demo:
    gr.Markdown("# Learning Randomized Reductions — artifact reproduction")
    gr.Dataframe(value=claim_rows(EVIDENCE), interactive=False)
    for path, body in zip(PAGE_PATHS, PAGE_TEXT, strict=True):
        with gr.Tab(path.stem.split("-", 1)[-1].replace("-", " ").title()):
            gr.Markdown(body)
```

Each claim page must include the exact challenge text and hash, status,
reproduced observation, source pins, command, and limitation. The README must
give exact acquire/audit/validate/test commands and state that Agentic Bitween,
Gurobi, GPU training, and paid APIs were not rerun.

- [ ] **Step 4: Run complete verification and create the proposal**

```bash
uv sync --frozen
uv run pytest -q
uv run lrr-repro validate evidence/results.json --schema schema/evidence-v1.schema.json
env ALL_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 NO_PROXY= uv run lrr-repro audit --project-root . --cache-dir .cache/upstream --schema schema/evidence-v1.schema.json --output /tmp/lrr-a.json
env ALL_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 NO_PROXY= uv run lrr-repro audit --project-root . --cache-dir .cache/upstream --schema schema/evidence-v1.schema.json --output /tmp/lrr-b.json
cmp /tmp/lrr-a.json /tmp/lrr-b.json
uv run lrr-repro propose --evidence evidence/results.json --source-commit "$(git rev-parse HEAD)" --source-tree "$(git rev-parse HEAD^{tree})" --output evidence/worker-proposal.json
git diff --check
```

From the repository root run:

```bash
env UV_CACHE_DIR=/tmp/icml-repro-uv-cache PRE_COMMIT_HOME=/tmp/icml-repro-pre-commit uv run pre-commit run -a
```

Expected: all tests and validation pass, offline audits are byte-identical,
and the proposal requests controller validation with no external mutation.

- [ ] **Step 5: Commit and hand off**

```bash
git add README.md app.py pages tests/test_pages.py tests/test_space.py evidence/worker-proposal.json
git commit -m "docs: publish randomized-reductions evidence"
git status --short
```

Expected: the assigned submission is clean. Report the branch, commit, source
tree, test counts, evidence SHA-256, and proposal path to the controller.
Do not deploy, submit, poll, or edit coordinator state.
