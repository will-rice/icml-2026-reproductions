# Learning Randomized Reductions Reproduction Design

## Approval and scope

This design implements the controller-approved five-claim assessment for ICML
2026 paper `hCAEcqig2C`, **Learning Randomized Reductions**, under attempt
`eb10c79b-fc26-47c4-88c1-6f45cb592833`.

The reproduction is an independently executable, CPU-only artifact audit under
`submissions/learning-randomized-reductions/`. It will not rerun Agentic
Bitween with an LLM, call a paid API, invoke a GPU, or require Gurobi. It will
instead recompute claims from immutable released artifacts, execute bounded
mathematical checks, and distinguish reproduced observations from
paper-reported context.

The five challenge claims and their immutable bindings are:

| Lane | Challenge claim SHA-256 | Intended outcome |
|---|---|---|
| Correlated-sampling theory | `5f0d21d91c0ae1d2877563e7115e804db60361304db4aea72b97596300e60f57` | Verify the formal definitions and sample-complexity implication with source-location and finite-model audits. |
| RSR-Bench census | `79d94d106cfded95104c54624068a07dc9ae16dca681a6ad5370bbb648e8c7de` | Verify exactly 80 distinct benchmark rows and definitions. |
| Vanilla Bitween | `4b8bfdf084cb0038acc0a589837dc4379ba1fb079f30f4be8edf839a21d23a51` | Verify 43/80 coverage and the sigmoid identity; explicitly limit the historical “first known” wording. |
| Agentic Bitween | `9b35061b3b4e2873f1b7a4fffc6fa22d659f281c096d990706ebd805303c4c00` | Verify 64/80 coverage and released novel-query examples without rerunning an LLM. |
| Nonlinear-invariant comparison | `13999601811ffe2bb8e9526ed601e9d59480b217d6d1917787db2a9c7dbc8372` | Falsify the exact live wording by demonstrating its benchmark and table conflation. |

## Immutable inputs

The upstream identity is:

```text
arxiv:2412.18134v5
arxiv:2412.18134v1
github:ferhaterata/learning-randomized-reductions@e13d4b59f6d23051c73e07cfc447336da84e7bd2
```

The acquisition manifest will record source URLs, exact byte sizes, SHA-256
hashes, Git blob IDs where applicable, and license provenance. The core pins
known at design time are:

| Artifact | Bytes | Git blob | SHA-256 |
|---|---:|---|---|
| arXiv v1 PDF | 2,490,520 | n/a | `abaac08eabec2e77c8af7ae3ca028691b9cd862e21bfa779452b9fd729e3222f` |
| arXiv v5 PDF | 2,004,773 | n/a | `93cab4aa8cec06434b704e639bab87dd15ea95ac46a335961138a94fc1bae2b8` |
| `LICENSE` | 1,091 | `2eb9ad588c5b6e720b168588a640d7a653265c96` | `04601314559ab36aa7403fbaa56ccba106be0de6671190497e6835bbd3107bdb` |
| `results/Bitween-Results(Sheet1-ICML).csv` | 502,974 | `0432241ef42d1be06179546c7b96d6bf6f598986` | `7198413f93830f7903bf3b670b718f2ccfbab1a41496a1fc3fe085850af0df0b` |
| `src/bitween/evaluation/evaluation_rsr_bench_paper.py` | 47,780 | `1aa8de34e60dcdbf77c0af53e1d5af25a673522f` | `6afe05589eeb08f34d63f98ed55fc38a3856a84ce7fc1d21e47327baad54ffbf` |
| `src/bitween/evaluation/evaluation_rsr_bench_paper_extended.py` | 29,907 | `88f1ef6b3c1b280afd8d2754509a1f6f0b30df7c` | `02fcfb7805e1704e040ae9e854b22ab199ae7ea95a18b5e45f2f9c886c0f40e2` |
| `src/bitween/evaluation/evaluation_rsr_bench_agentic_paper.py` | 34,555 | `e660ab8d39b8083117097641983fe69c5efaeffb` | `c99842f831d6bf0296452e632a5b0eb24f8ae9438acc42bad92604ac61c64bb0` |
| `src/bitween/evaluation/rsr_checker.py` | 22,751 | `417f5f0a8ef6789be4f01c3108f4427f2580c9d0` | `28e9dc80cec82d8e11dee4e867b55cba707b295c8f9d2065c7e9d286967fd3aa` |
| `src/bitween/pac.py` | 1,395 | `4dd47da9cde937e7f7074424a9cddb5f3aa523ab` | `69da63ffce26b7f85f306a69e13eb345a8c12be7257f60297b96c5122c2274a4` |

The MIT-licensed repository inputs may be committed beneath
`evidence/inputs/upstream/` so claims 2–4 can be recomputed offline. The arXiv
PDFs use arXiv's non-exclusive distribution license and will not be
redistributed. Their verified hashes and minimal factual transcriptions will
be committed; acquisition places the PDFs only in an ignored local cache.

## Chosen architecture

### Considered approaches

1. **Full upstream rerun.** Re-execute Vanilla and Agentic Bitween across all
   80 functions. This would require an LLM/API for the agentic lane and Gurobi
   for one comparison, would not reproduce the historical model behavior, and
   violates the zero-API design constraint.
2. **Static paper/table mirror.** Copy the reported table values into a Space.
   This is cheap but is self-report, not independent reproduction evidence.
3. **Pinned artifact audit with bounded executable checks.** Recompute row
   counts and aggregates from released raw outputs, verify the sigmoid identity
   symbolically and numerically, audit the theory with finite exhaustive
   models, and test source/table consistency across arXiv versions.

Approach 3 is approved. It maximizes independent evidence while keeping the
complete audit deterministic, CPU-only, and comfortably below 30 minutes.

### Component boundaries

The project uses one focused module per responsibility:

- `provenance.py` verifies manifests, cached PDFs, Git blobs, SHA-256 hashes,
  sizes, safe paths, and exact upstream identity.
- `theory.py` represents finite functions and correlated query distributions,
  checks marginal uniformity and perfect RSR identities, and exhaustively
  verifies the union-bound implication underlying Claim A.1 on bounded models.
- `benchmark.py` parses benchmark definitions and raw CSV benchmark rows and
  checks the exact 1–80 census without importing or executing untrusted
  upstream Python.
- `results.py` parses the ragged multi-row CSV, recomputes per-backend RSR
  totals, coverage, and runtime summaries, extracts balanced function calls,
  classifies fixed versus novel queries, and verifies the sigmoid equation.
- `claim_scope.py` audits exact section/table locators in verified v1/v5 paper
  text and produces the honest falsification record for the fifth claim.
- `evidence.py` is the only component allowed to assign evidence statuses and
  render the closed machine-readable schema.
- `cli.py` separates network acquisition from offline `audit`, `validate`, and
  worker `propose` operations.
- `app.py` and root `pages/*.md` are read-only views over committed evidence.
  They never calculate a result independently.

All public interfaces use immutable dataclasses or canonical JSON-compatible
objects. Failures are fail-closed: an absent file, hash mismatch, malformed
row, duplicate benchmark, unresolved paper locator, or inconsistent aggregate
must make the relevant lane inconclusive or abort generation; it must never
silently substitute a paper value.

## Five evidence lanes

### Lane 1: correlated-sampling theory

The source audit verifies that the pinned v5 paper contains Definitions 4.1,
4.3, and 4.5 and Claims A.1 and A.2 at their recorded pages. The executable
audit then constructs small finite domains and enumerates:

- correlated query tuples whose individual marginals are uniform;
- perfect RSRs for a learned hypothesis;
- ground-truth functions within an exact error fraction `epsilon`;
- the predicted bound
  `m_RSR(rho, xi, delta) <= m_PAC(min(rho/k, xi), delta)`.

For every enumerated valid case, it checks that at least a `1 - xi` fraction
of inputs recover with probability at least `1 - rho`. This does not replace
the general mathematical proof; it independently checks the proof's marginal
uniformity and union-bound mechanism and records that scope.

### Lane 2: RSR-Bench census

The benchmark parser reads, but never imports, the pinned base and extended
benchmark scripts. It extracts literal `test_id` registrations from
`evaluate(...)` calls—IDs 1–40 from the base script and 41–80 from the
extended script—and reconciles them with the 80 numbered CSV benchmark
records. It deliberately does not count every `test_*` definition because the
files contain helpers and unused functions.
The lane verifies:

- IDs are exactly `1..80`;
- names are nonempty and unique after normalization;
- there are exactly 80 primary benchmark rows;
- every parsed benchmark name maps one-to-one to the released results.

Any extra continuation rows in the spreadsheet are attached to their primary
row and cannot inflate the census.

### Lane 3: Vanilla Bitween and sigmoid

The CSV parser recomputes the linear-regression column from all 80 primary
rows. The expected independently computed observations are:

- total RSR count `87`;
- nonzero-RSR coverage `43/80 = 0.5375`, displayed as 54% only as a rounded
  derivative;
- runtime minimum/mean/maximum `0.13 / 4.791 / 19.12` seconds;
- sigmoid row 33 has three LR RSRs.

One released sigmoid equation is parsed into an internal expression and
checked after substituting `sigma(t) = 1/(1+exp(-t))`. A deterministic rational
identity check is preferred; a seeded high-precision numeric grid is retained
as a regression guard. The evidence may verify the 43/80 result and the
identity, but it must state that an exhaustive literature-priority search was
not performed and therefore “first known” remains unreplicated.

### Lane 4: Agentic Bitween

The same raw CSV parser recomputes the Claude-Opus-4.1 Agentic Bitween column:

- total RSR count `793`;
- nonzero-RSR coverage `64/80 = 0.8`;
- all 80 numbered rows are included, including zero-result rows.

The query scanner extracts balanced outer `f(...)` arguments from released
agentic properties, canonicalizes the fixed set `{x+r, x-r, x*r, x, r}`, and
records concrete non-fixed examples such as `x+log(k)`. This verifies that the
released agentic outputs contain query functions beyond the fixed prior set.
It does not claim that the remote Claude run was repeated or that current LLM
behavior matches the released run.

### Lane 5: honest falsification of the nonlinear-invariant wording

This lane treats inconsistency as evidence, not as a reason to rewrite the
claim. It verifies the following source locations against pinned paper bytes:

- arXiv v1 Table 2 is a learned post-condition example with 20 samples, not a
  backend comparison;
- v1 Section 5.3.1 compares LR and MILP on **RSR-Bench**, reporting 594 versus
  1,095 samples and 130.53 versus 187.47 seconds;
- v1 Section 5.3.2 compares Bitween against DIG and SymInfer on
  **NLA-DigBench**, not against the MILP backend;
- arXiv v5 Table 2 concerns novel Agentic Bitween query functions.

Therefore the exact live claim—nonlinear invariant benchmarks, regression
versus MILP, sample count and runtime, Table 2—is marked `falsified`. The
evidence will separately preserve the narrower true paper statements without
using them to rescue the conflated claim.

## Evidence model

`schema/evidence-v1.schema.json` is closed with `additionalProperties: false`
at every owned object. `evidence/results.json` contains:

- paper, attempt, challenge-claim, and upstream identities;
- one ordered record per challenge claim;
- status, expected observation, measured observation, supporting source
  hashes, commands, and limitations for each lane;
- environment and dependency-lock hashes;
- artifact JSON pointers and canonical subobject hashes;
- unavailable operations (`agentic_rerun`, `gpu_training`, `paid_api`,
  `gurobi_rerun`);
- explicit contradictions and the source facts that establish them.

Canonical JSON uses UTF-8, sorted keys, compact separators, a final newline,
and rejects non-finite numbers. Two offline audits over identical verified
inputs must be byte-identical. `commands.json` records literal commands and
exit codes; `validation.json` binds the result hash and schema validation.
Host paths, credentials, environment dumps, and wall-clock values are
excluded.

## Reviewer surfaces

The submission root will contain:

```text
pages/00-summary.md
pages/01-correlated-sampling-theory.md
pages/02-rsr-bench-census.md
pages/03-vanilla-and-sigmoid.md
pages/04-agentic-bitween.md
pages/05-nonlinear-invariant-falsification.md
pages/06-methods-and-provenance.md
```

These are committed reviewer-readable pages, not generated fragments hidden
inside Python. Each claim page states the exact challenge claim, immutable
hash, status, independently computed observations, source pins, command, and
limitations. `app.py` loads the pages and `evidence/results.json` without
network access and presents a summary table plus the seven pages in order.
`README.md` carries valid Hugging Face Space metadata, exact local commands,
and the paper tag `paper-hCAEcqig2C`.

## Testing and acceptance

Every production behavior begins with a failing pytest. Focused tests cover
manifest tampering, unsafe paths, duplicate CSV IDs, ragged continuation rows,
exact aggregates, sigmoid algebra, balanced query extraction, finite correlated
models, locator mismatches, status downgrade rules, canonical evidence,
committed-page completeness, offline app import, and worker-proposal scope.

The worker's final acceptance commands are:

```bash
cd submissions/learning-randomized-reductions
uv sync --frozen
uv run pytest -q
uv run lrr-repro validate evidence/results.json --schema schema/evidence-v1.schema.json
env ALL_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 NO_PROXY= uv run lrr-repro audit --manifest evidence/inputs/upstream_manifest.json --cache-dir .cache/upstream --output /tmp/lrr-results-a.json
env ALL_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 NO_PROXY= uv run lrr-repro audit --manifest evidence/inputs/upstream_manifest.json --cache-dir .cache/upstream --output /tmp/lrr-results-b.json
cmp /tmp/lrr-results-a.json /tmp/lrr-results-b.json
git diff --check
```

From the repository root, the worker also runs:

```bash
env UV_CACHE_DIR=/tmp/icml-repro-uv-cache PRE_COMMIT_HOME=/tmp/icml-repro-pre-commit uv run pre-commit run -a
```

The implementation worker may commit only its assigned
`submissions/learning-randomized-reductions/` project. It may not edit state,
controller documents, skills, another submission, or Hub resources. Its
result is a proposal for separate controller validation, publication,
submission, and judgment.
