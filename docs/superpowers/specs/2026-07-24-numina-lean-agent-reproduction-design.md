# Reproduction Design: Numina-Lean-Agent (Approved)

**Paper**: Numina-Lean-Agent: An Open and General Agentic Reasoning System for Formal Mathematics
**OpenReview ID**: `0bTEd4LpQr`
**arXiv**: `2601.14027`
**Design date**: 2026-07-24 (revised after two rejected reviews)
**Phase gate**: `design-pending` — independently `APPROVED` for the root agent
to advance; this document does not itself transition authoritative state (see §19).

---

## 1. Live Challenge Status (observed 2026-07-24T11:49–11:54Z)

| Item | Observed value |
|---|---|
| Challenge Space | RUNNING at SHA `5bbcad2e9a7e8a7479f3563ac1fc6c768d4bb050` |
| Challenge dataset `challenge.json` | 200 indexed papers; `0bTEd4LpQr` present as `orid`, paper index 2276 |
| Official claims for `0bTEd4LpQr` | **4 claims, all `unverified`** (see §6 for exact text) |
| Spaces tagged `paper-0bTEd4LpQr` | **0** (HF API confirmed) |
| Verdicts for `0bTEd4LpQr` | **0** (verdicts dataset search empty) |

**Correction**: The prior design stated the paper was "not found" in the catalog.
This was wrong — `challenge.json` on the dataset has 200 papers and `0bTEd4LpQr` is
present. The HF datasets-server search API returned 0 rows because the underlying
dataset does not use a standard Parquet/CSV format; the paper is found by direct JSON
key lookup.

**Eligibility verdict**: Paper is **in the active catalog, unclaimed, and eligible**.

---

## 2. Candidate-Pool Record and Non-Selection

| Candidate | orid | Base | Penalties | Final | Testable claims | Expected points | CPU estimate | API estimate | Disposition |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| Numina-Lean-Agent | `0bTEd4LpQr` | 16 | -2 | **14** | 2 | **3.15** | 50–80 min local CPU | $0.00 | design-pending; independently approved |
| TerminalTraj | `PeFSCRulgy` | — | — | — | — | — | >24 hr GPU training | $0.00 CPU | persisted rejected |
| OXE-AugE | `LcswwEzzX7` | — | — | — | — | — | — | — | persisted rejected; 3 existing Spaces |

**Candidate-pool closure is authoritatively persisted**:
- **dXPP** (`2jpMiRwsrL`): persisted rejected because it is absent from the current
  challenge catalog and its proposed diagnostics do not reproduce its official claims.
- **OXE-AugE** (`LcswwEzzX7`): 3 existing reproduction Spaces (algorise, abhishekkataria16, Edd16); ineligible under the duplicate-selection rubric rule.
- **TerminalTraj** (`PeFSCRulgy`): persisted rejected because released artifacts do
  not provide two independently CPU-testable official claims; the performance claim
  requires impractical GPU-dependent inference and dataset licensing is undeclared.

The active orchestration worktree records all three rejections at
`d7fc300eff937a958eccf886ff088d2b279ddd7f`. This design worktree does not edit
authoritative state or HANDOFF. The independent approval in §19 authorizes the root
agent to perform the later state transition separately.

---

## 3. Upstream Revisions (Immutable, Pinned)

| Repository | Pinned SHA | Commit date | Purpose |
|---|---|---|---|
| `project-numina/numina-lean-agent` | `1c9af8a52e715f22fede766425ba3d3b95526132` | 2026-07-08 | Agent source, problem statements |
| `project-numina/Numina-Putnam2025` | `60d33c8ba19af905bd731e938ebde1c5b8c76519` | 2026-01-20 | Completed Putnam 2025 proofs |
| `project-numina/BrascampLieb` | `413f2bfd31100187eb6c2d632c9cbf12e3115494` | 2026-04-10T15:20:33Z | Brascamp-Lieb formalization |

All confirmed live via GitHub API. After root-serialized candidate closure and an
independent different-agent `APPROVE`, the immutable selection token binds every
released input rather than only the agent repository:

```
github:project-numina/numina-lean-agent@1c9af8a52e715f22fede766425ba3d3b95526132+
project-numina/Numina-Putnam2025@60d33c8ba19af905bd731e938ebde1c5b8c76519+
project-numina/BrascampLieb@413f2bfd31100187eb6c2d632c9cbf12e3115494
```

---

## 4. Lean Toolchain and Build Manifests

### Putnam proofs (`Numina-Putnam2025@60d33c8`)

- Lean: `leanprover/lean4:v4.26.0`
- Mathlib: `leanprover-community/mathlib4` rev `2df2f0150c275ad53cb3c90f7c98ec15a56a1a67` (inputRev `v4.26.0`)
- 12 `.lean` files, total 12 proofs (A1–A6, B1–B6)
- Root module: `NuminaPutnam2025.lean` imports all 12

### Brascamp-Lieb formalization (`BrascampLieb@413f2bf`)

- Lean: `leanprover/lean4:v4.28.0`
- Mathlib: `leanprover-community/mathlib4` rev `8f9d9cff6bd728b17a24e163c9402775d9e6a365` (inputRev `v4.28.0`)
- 21 `.lean` files, total ~100 KB source
- Main theorem: `BrascampLieb.upperBound` in `BrascampLieb/Code/MainTheorems.lean`
- Paper reference: Bénard-He, arXiv:2511.11091, Theorem 1.4

### Two separate `lake build` runs required

The Putnam and Brascamp-Lieb repos use **different Lean/Mathlib versions** (v4.26.0
vs v4.28.0). They must be built independently. Each requires its own Mathlib cache
download and `lake build`.

---

## 5. Sorry / Axiom Audit — Parser-Backed Source Analysis

### Method

All source files were analyzed with a Python block-comment-aware parser that tracks
`/-` … `-/` nesting depth and `--` line comments, identifying `sorry` tokens at word
boundaries only in non-comment code positions. This is strictly more accurate than
`grep -c sorry`, which counted comment mentions as false positives in the prior design.

### Putnam 2025 (12 files, companion repo `@60d33c8`)

| File | Lines | Tactic-level sorry count |
|---|---|---|
| putnam_2025_a1.lean | 445 | **0** |
| putnam_2025_a2.lean | 524 | **0** |
| putnam_2025_a3.lean | 583 | **0** |
| putnam_2025_a4.lean | 906 | **0** |
| putnam_2025_a5.lean | 3937 | **0** |
| putnam_2025_a6.lean | 1588 | **0** |
| putnam_2025_b1.lean | 364 | **0** |
| putnam_2025_b2.lean | 1080 | **0** |
| putnam_2025_b3.lean | 320 | **0** |
| putnam_2025_b4.lean | 1158 | **0** |
| putnam_2025_b5.lean | 1091 | **0** |
| putnam_2025_b6.lean | 4406 | **0** |

**All 12 files are source-sorry-free.** Prior grep hits (A5: 3, A6: 1, B6: 4) were
stale developer comments inside `--` or `/- -/` blocks, not Lean tactic invocations.
Example: `-- The sorry here is within the "suffices" block` (line 1264 of A6) is a
comment describing historical state, not live code.

**Correction**: The prior design's central Claim 2 — that 3 files contain admitted
lemmas — was **false**. All tests and evidence related to sorry detection in the
prior design are removed.

### Brascamp-Lieb (21 files, `@413f2bf`)

All 21 `.lean` files: **0 tactic-level sorry** across the entire repository.

### `#print axioms` evidence plan

Every pinned Putnam proof file A1–A6 and B1–B6 already ends with its committed
`#print axioms` command. The pipeline must execute each committed file with ordinary
`lake env lean NuminaPutnam2025/putnam_2025_<id>.lean` and parse that command's
output; it must not add or invent a Putnam query file. The review observed the Lean
constant `Quot.sound` (not `Quotient.sound`) and did not establish a universal
`funext` requirement. The sole cross-theorem assertion is absence of `sorryAx`;
every observed axiom list is retained as normalized output rather than forced
through an unproved allowlist.

BrascampLieb does not contain the needed query, so the pipeline locally authors only
`axiom_check_bl.lean` and runs it as `lake env lean axiom_check_bl.lean`. Neither
path uses `--run`: `#print axioms` is an elaboration-time command, not a Lean
program entry point.

---

## 6. Official Paper Claims (from live `challenge.json`)

The challenge catalog contains exactly 4 official claims for `0bTEd4LpQr`, all
`unverified`:

| # | Exact official claim text | Status |
|---|---|---|
| 1 | Numina-Lean-Agent combines Claude Code with Numina-Lean-MCP and specialized tools for Lean interaction, theorem retrieval, informal proving, and auxiliary reasoning (Figure 1). | unverified |
| 2 | Using Claude Opus 4.5, Numina-Lean-Agent solves all 12 Putnam 2025 problems, matching AXIOM's 12/12 in the comparison table (Table 1). | unverified |
| 3 | Ablations show the agent solves 4 Putnam 2025 problems without the informal prover, 11 with the informal prover, and all 12 with the subagent setup (Table 2). | unverified |
| 4 | The paper reports successful formalization of the Brascamp-Lieb theorem through interaction with mathematicians (Abstract). | unverified |

### Evidence scope distinction

- **Claims 1, 3**: Describe the agent architecture and ablation experiments.
  Reproducing these would require running the agent with Claude Opus 4.5, which is
  a paid API outside the $10 cap and outside autonomous GPU/API-call scope. These
  claims are **not independently testable from released artifacts alone**.

- **Claim 2**: "Solves all 12 Putnam 2025 problems" — the released companion proofs
  (`Numina-Putnam2025@60d33c8`) are the artifacts supporting this claim. Verifying
  that all 12 proofs kernel-check (type-check without `sorryAx`) under the pinned
  Lean/Mathlib versions is an independent released-proof verification. This does NOT
  re-run the agent; it verifies the proofs the agent reportedly produced.

- **Claim 4**: "Successful formalization of the Brascamp-Lieb theorem" — the released
  formalization (`BrascampLieb@413f2bf`) is the artifact. Verifying `lake build` and
  `#print axioms` confirms the formalization kernel-checks its described scope
  (Theorem 1.4 of arXiv:2511.11091).

### Selected target claims (immutable)

- **Target Claim 1**: "Using Claude Opus 4.5, Numina-Lean-Agent solves all 12 Putnam
  2025 problems, matching AXIOM's 12/12 in the comparison table (Table 1)."
  - Evidence: released formalizations (`Numina-Putnam2025@60d33c8`) cover all 12
    problems (A1–A6, B1–B6) and kernel-check under Lean v4.26.0 / Mathlib v4.26.0
    without `sorryAx`.

- **Target Claim 2**: "The paper reports successful formalization of the Brascamp-Lieb
  theorem through interaction with mathematicians (Abstract)."
  - Evidence: released formalization (`BrascampLieb@413f2bf`) kernel-checks
    `BrascampLieb.upperBound` (Theorem 1.4 of arXiv:2511.11091) under Lean v4.28.0
    / Mathlib v4.28.0 without `sorryAx`.

Both are released-proof verification, not agent re-execution. The evidence
distinguishes "the released proofs are valid Lean 4 kernel-checked formalizations"
from "we independently ran the agent and it produced these proofs."

---

## 7. Rubric Scoring (Revised)

| Dimension | Score | Rationale |
|---|---|---|
| Direct artifacts | **4** | Versioned Lean repos at pinned SHAs directly support both target claims |
| Independently testable claim count | **2** | Two distinct claims with separate `lake build` + `#print axioms` observables |
| CPU feasibility | **3** | Two separate Lean builds with Mathlib cache: ~1–2 CPU hours total |
| Provenance | **5** | Exact SHAs, URLs, manifest hashes, acquisition commands recordable |
| Licensing | **2** | Putnam has MIT terms, but both the agent repository (README-only MIT) and BrascampLieb (no terms) lack a LICENSE file. The latter two may be inspected locally but are not redistributable inputs. |

**Base score**: 4 + 2 + 3 + 5 + 2 = **16**
**Penalty**: -2 (BrascampLieb license is unclear — required artifact with no license)

**Final rubric score**: **14**

> Note: The -2 licensing penalty is applied because BrascampLieb has no LICENSE file,
> no GitHub-detected license, and no README license section. Redistribution/deployment
> of its contents in a Space is not authorized by public GitHub hosting. The evidence
> pipeline may link to its URL/SHA but must not reproduce its contents. The agent repo
> (`numina-lean-agent`) similarly has no LICENSE file: its README text is provenance,
> not a substitute for a file-level redistribution grant.

### Expected official points (conservative)

The conservative probabilities and arithmetic are:

- Putnam: `P(full)=0.75`, `P(toy)=0.15`, so `2×0.75 + 1×0.15 = 1.65`.
- BrascampLieb: `P(full)=0.65`, `P(toy)=0.20`, so
  `2×0.65 + 1×0.20 = 1.50`.
- Total: `1.65 + 1.50 = 3.15` expected official points.

This discounts both claims for the risk that released-proof checking is judged as
toy rather than agent re-execution and for BrascampLieb's licensing limitation. It
is not a promise of judge credit.

---

## 8. CPU Build Time and Resource Estimates (Corrected)

### Putnam 2025 (`Numina-Putnam2025@60d33c8`, Lean v4.26.0)

| Phase | Estimate |
|---|---|
| Mathlib cache download (`lake exe cache get`) | ~62.5 seconds (documented), ~9.4 GB dependencies |
| `lake build` with warm cache | 20–40 minutes; B6 (4406 lines) peaks at ~6.6 GB RAM |
| `#print axioms` extraction | <5 minutes (Lean environment already loaded) |
| **Total wall time** | **~30–50 minutes** |

### Brascamp-Lieb (`BrascampLieb@413f2bf`, Lean v4.28.0)

| Phase | Estimate |
|---|---|
| Mathlib cache download | ~60–90 seconds, ~9–10 GB dependencies |
| `lake build` with warm cache | 10–20 minutes (smaller codebase: 21 files, ~100 KB) |
| `#print axioms` extraction | <5 minutes |
| **Total wall time** | **~20–30 minutes** |

### Combined: ~50–80 minutes total CPU time

### Local CPU default; HF CPU Job only with approval

The default execution path is a local CPU machine with at least 16 GB RAM, `elan`,
and network access for the pinned caches. Its paid cost is **$0.00**. No Job is
needed for the design or for local evidence generation.

An HF CPU Job is an optional paid alternative, not an autonomous fallback. Starting
one requires a separate explicit user approval that names the hardware and maximum
spend; without that approval, the loop remains on the local-CPU path or pauses.

HF Jobs are billed per minute. Costs at listed hourly rates:

| Hardware | vCPU | RAM | Hourly rate | 80-min cost |
|---|---|---|---|---|
| cpu-basic | 2 | 16 GB | $0.01/hr | $0.01 |
| cpu-upgrade | 8 | 32 GB | $0.03/hr | $0.04 |

B6's ~6.6 GB RAM requirement fits within `cpu-basic` (16 GB). However, 2 vCPU may
be slow for Lean. `cpu-upgrade` (8 vCPU, 32 GB) provides headroom.

**Optional approved HF Job cost**: $0.04 (cpu-upgrade, 80 min) per run, up to $0.12
for three separately approved runs (build, evidence, validation). The $10 ceiling
does not grant approval to spend below it.

### Paid API cost

No Claude, Gemini, or other paid model API is called. Evidence is pure Lean
type-checking of released source. **Total paid-API cost: $0.00.**

---

## 9. Safety Analysis

| Risk | Assessment |
|---|---|
| Lean type-checker | Deterministic proof kernel; safe |
| `unsafe` declarations | Not observed in any file across either repo |
| `native_decide` | Not used as a tactic in any proof file |
| Network calls | Only to GitHub/Mathlib cache (pinned SHAs) during `lake build` |
| GPU requirement | None |
| Credential requirement | None |

**Safety verdict**: Safe inside ordinary CPU isolation.

---

## 10. License and Provenance (Corrected)

| Artifact | LICENSE file | GitHub detection | README mention | Treatment |
|---|---|---|---|---|
| `Numina-Putnam2025@60d33c8` | **Yes** (MIT, 1063 bytes) | `mit` | — | licensed input |
| `numina-lean-agent@1c9af8a` | **No** (no LICENSE file at root) | `null` | "MIT License" in README.md footer | inspect/link only |
| `BrascampLieb@413f2bf` | **No** (no LICENSE file) | `null` | **No mention** | inspect/link only; no redistribution |
| Mathlib4 | Yes (Apache 2.0) | `apache-2.0` | — | licensed dependency |
| Lean 4 | Yes (Apache 2.0) | `apache-2.0` | — | licensed dependency |

**Licensing concerns**:
1. **BrascampLieb** has no license at all. The repo is public on GitHub, but public
   availability is not a license grant. Our evidence pipeline references it by URL
   and SHA without redistributing source. The Space logbook can report build results
   and axiom lists without embedding the source code, mitigating redistribution risk.
2. **numina-lean-agent** has no LICENSE file. Its README says "MIT License," but
   that ambiguity does not authorize copying its source into the evidence bundle or
   Space.

**Penalty applied**: -2 for BrascampLieb's unclear license (required artifact).

---

## 11. TDD Plan (Revised)

### Task 1: Putnam build and axiom audit

**Test (write first, observe failure)**:

```python
# tests/test_putnam_build.py
import json, pathlib

COMPANION_SHA = "60d33c8ba19af905bd731e938ebde1c5b8c76519"
FILES = [f"putnam_2025_{x}" for x in
         ["a1","a2","a3","a4","a5","a6","b1","b2","b3","b4","b5","b6"]]

def test_build_result_exists():
    p = pathlib.Path("submissions/numina-lean-agent/evidence/putnam_build.json")
    assert p.exists(), "Run build pipeline first"

def test_build_succeeded():
    data = json.loads(pathlib.Path(
        "submissions/numina-lean-agent/evidence/putnam_build.json").read_text())
    assert data["exit_code"] == 0
    assert data["pinned_sha"] == COMPANION_SHA
    assert data["lean_toolchain"] == "leanprover/lean4:v4.26.0"

def test_all_12_proofs_no_sorry_ax():
    data = json.loads(pathlib.Path(
        "submissions/numina-lean-agent/evidence/putnam_axioms.json").read_text())
    assert len(data) == 12
    for f in FILES:
        assert f in data, f"{f} missing from axiom audit"
        assert "sorryAx" not in data[f]["axioms"], f"{f} has sorryAx"

def test_axiom_audit_is_normalized_and_tracks_quot_sound():
    data = json.loads(pathlib.Path(
        "submissions/numina-lean-agent/evidence/putnam_axioms.json").read_text())
    assert "observed_at" not in data
    assert any("Quot.sound" in item["axioms"] for item in data.values())
    # Do not require funext or an invented universal axiom allowlist.
```

### Task 2: Brascamp-Lieb build and axiom audit

```python
# tests/test_brascamp_lieb_build.py
BL_SHA = "413f2bfd31100187eb6c2d632c9cbf12e3115494"

def test_bl_build_succeeded():
    data = json.loads(pathlib.Path(
        "submissions/numina-lean-agent/evidence/brascamp_lieb_build.json").read_text())
    assert data["exit_code"] == 0
    assert data["pinned_sha"] == BL_SHA
    assert data["lean_toolchain"] == "leanprover/lean4:v4.28.0"

def test_bl_main_theorem_no_sorry_ax():
    data = json.loads(pathlib.Path(
        "submissions/numina-lean-agent/evidence/brascamp_lieb_axioms.json").read_text())
    assert "BrascampLieb.upperBound" in data
    assert "sorryAx" not in data["BrascampLieb.upperBound"]["axioms"]

def test_bl_evidence_is_locally_authored_json_only():
    paths = pathlib.Path("submissions/numina-lean-agent/evidence").glob("**/*")
    assert all(p.suffix == ".json" for p in paths if p.is_file())
```

### Task 3: Evidence CLI, determinism, and logbook

```python
def test_evidence_cli_deterministic():
    # Two runs write byte-identical, sorted-key JSON and no timestamps.
    ...

def test_claims_json_matches_official():
    data = json.loads(pathlib.Path(
        "submissions/numina-lean-agent/evidence/claims.json").read_text())
    # Must reference exactly the two target claims
    claim_ids = {c["claim_id"] for c in data}
    assert "putnam-12-12" in claim_ids
    assert "brascamp-lieb-formalization" in claim_ids
```

### Task 4: Posterly, Trackio, validator, Space

Standard logbook/poster/deployment flow per submission checklist.

---

## 12. Evidence Bundle

```
submissions/numina-lean-agent/
├── pyproject.toml
├── tests/
│   ├── test_putnam_build.py
│   ├── test_brascamp_lieb_build.py
│   └── test_evidence_cli.py
├── src/numina_lean/
│   ├── __init__.py
│   ├── cli.py
│   ├── putnam_audit.py
│   └── brascamp_lieb_audit.py
└── evidence/                        # tracked, normalized, locally authored JSON
    ├── putnam_build.json            # exit status + pinned provenance only
    ├── putnam_axioms.json           # sorted `#print axioms` names per theorem
    ├── brascamp_lieb_build.json     # exit status + pinned provenance only
    ├── brascamp_lieb_axioms.json    # sorted axiom names for `upperBound`
    └── claims.json                  # final claim results
```

Every tracked evidence file is locally authored normalized JSON, rendered with
sorted keys and stable arrays. It contains the composite `upstream_revision`, the
relevant toolchain, command identity, exit status, and parsed axiom names. It omits
wall-clock timestamps, elapsed time, host paths, raw stdout/stderr, and environment
dumps so a clean rerun has byte-identical output. No ignored output is relied on.

Because BrascampLieb has no license, neither the repository source, `.olean`/
binaries, tarballs/cache contents, nor raw build or Lean logs may be committed or
put in the Space. The Space contains only these locally-authored JSON summaries and
links to the upstream URL/SHA; it does not copy any unlicensed upstream content.

---

## 13. Provenance Manifest and Commands

```bash
# === Putnam 2025 proofs ===
git clone https://github.com/project-numina/Numina-Putnam2025.git putnam2025
cd putnam2025
git checkout --detach 60d33c8ba19af905bd731e938ebde1c5b8c76519
git rev-parse HEAD  # → 60d33c8ba19af905bd731e938ebde1c5b8c76519
lake exe cache get  # ~62.5s, ~9.4 GB deps
lake build  # type-check all 12 proofs
# Execute and parse each proof file's committed `#print axioms`, for example:
lake env lean NuminaPutnam2025/putnam_2025_a1.lean

# === Brascamp-Lieb formalization ===
cd ..
git clone https://github.com/project-numina/BrascampLieb.git bl
cd bl
git checkout --detach 413f2bfd31100187eb6c2d632c9cbf12e3115494
git rev-parse HEAD  # → 413f2bfd31100187eb6c2d632c9cbf12e3115494
lake exe cache get  # ~60-90s, ~9-10 GB deps
lake build  # type-check formalization
lake env lean axiom_check_bl.lean
```

The twelve committed Putnam commands and the one locally authored
`axiom_check_bl.lean` command produce stdout that is parsed in the local work
directory into tracked normalized JSON; raw logs are not retained or distributed.

### Reviewer-observed Brascamp-Lieb facts

The reviewer checked the pinned checkout rather than inferring behavior from the
paper: it declares `leanprover/lean4:v4.28.0`; `lakefile.toml` sets
`defaultTargets = ["BrascampLieb"]` and pins Mathlib `v4.28.0`; and the main
declaration is `upperBound` in `BrascampLieb/Code/MainTheorems.lean`. The review
also observed that the cache retrieval and `lake build` complete for that checkout,
and that an ordinary `lake env lean axiom_check_bl.lean` axiom query reports no
`sorryAx` for `BrascampLieb.upperBound`. Those are rerunnable observations, not a
license to retain its cache, build products, or output log.

---

## 14. Controls

| Control | Implementation |
|---|---|
| No agent run | Python agent never invoked; evidence from released proof files only |
| Pinned SHAs | All three repos cloned at exact SHAs, verified with `git rev-parse HEAD` |
| Two separate builds | Putnam (Lean v4.26.0) and BL (Lean v4.28.0) built independently |
| Parser-backed sorry audit | Block-comment-aware parser, not grep; validated against all 33 files |
| Axiom extraction | Execute the 12 committed Putnam commands plus one local BL query through `lake env lean <file>`; retain sorted parsed names, not logs |
| Deterministic evidence | Track normalized locally-authored JSON with no timestamps, host paths, raw output, or ignored dependency |
| No unlicensed redistribution | BL source, caches, binaries, and logs stay local; the Space receives only JSON summaries plus upstream URL/SHA |
| Scope labeling | Evidence explicitly states "released-proof verification" not "agent re-execution" |
| No paper values as evidence | Build success/failure and axiom lists are independently computed |

---

## 15. Space and Deployment Plan

- **Space ID**: `wrice/repro-numina-lean-agent`
- **Space tag**: `paper-0bTEd4LpQr`
- **SDK**: `static`
- Pre-deployment: full pytest suite, pre-commit, official logbook validator
- Commit the normalized JSON evidence before deployment; the exact committed JSON is
  the only evidence copied to the Space.
- BrascampLieb evidence: report locally-authored JSON and URL/SHA reference only.
  Do not copy its source, binaries, cache, or raw build/axiom output.

---

## 16. Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Judge requires agent re-execution, not proof verification | Medium | Claim scored as toy/inconclusive | Scope label is explicit; proofs ARE the claim's evidence |
| BrascampLieb license dispute | Medium | Cannot redistribute source in Space | Local inspection only; Space gets normalized JSON and upstream URL/SHA, never source/binaries/logs |
| Lean v4.28.0 Mathlib cache unavailable | Low | Cannot build BL | Build from source (~8 hr fallback) |
| B6 OOM on cpu-basic (6.6 GB peak) | Low | Build failure | cpu-basic has 16 GB; headroom sufficient |
| Paper becomes claimed during implementation | Medium | Wasted work | Re-check before each milestone |

---

## 17. Selection State JSON

```json
{
  "paper_id": "0bTEd4LpQr",
  "title": "Numina-Lean-Agent: An Open and General Agentic Reasoning System for Formal Mathematics",
  "slug": "numina-lean-agent",
  "estimated_api_cost_usd": 0.0,
  "upstream_revision": "github:project-numina/numina-lean-agent@1c9af8a52e715f22fede766425ba3d3b95526132+project-numina/Numina-Putnam2025@60d33c8ba19af905bd731e938ebde1c5b8c76519+project-numina/BrascampLieb@413f2bfd31100187eb6c2d632c9cbf12e3115494",
  "target_claims": [
    "Using Claude Opus 4.5, Numina-Lean-Agent solves all 12 Putnam 2025 problems, matching AXIOM's 12/12 in the comparison table (Table 1).",
    "The paper reports successful formalization of the Brascamp-Lieb theorem through interaction with mathematicians (Abstract)."
  ]
}
```

---

## 18. Checklist Pre-Entry to `implementing`

- [x] Live challenge status refreshed: paper in catalog, 4 unverified claims, 0 spaces, 0 verdicts
- [x] Authoritative state at `d7fc300eff937a958eccf886ff088d2b279ddd7f` persists dXPP, TerminalTraj, and OXE rejections
- [x] Upstream revisions pinned (3 repos) and GitHub-API-confirmed
- [x] Lean toolchains recorded: v4.26.0 (Putnam) and v4.28.0 (BrascampLieb)
- [x] All 12 Putnam files source-sorry-free (parser-backed, not grep)
- [x] All 21 BrascampLieb files source-sorry-free (parser-backed)
- [x] Official paper claims identified (4); two selected as target claims
- [x] Local CPU default: 50–80 min; paid-API $0.00; an HF Job is optional and requires explicit approval
- [x] License audit: MIT (Putnam), README-MIT/no file (agent), **no license** (BrascampLieb); licensing score 2 and -2 penalty applied
- [x] Safety: no GPU, no paid API, no unsafe code
- [x] TDD plan with 4 tasks, 9 tests
- [x] Evidence bundle and commands specified
- [x] Space and submission plan specified
- [x] **Independent different-agent review approved the corrected design**

---

## 19. Approval Record

The original design was reviewed and **REJECTED** by an independent agent review on
2026-07-24T11:49Z. A subsequent review also left the design **REJECTED** and
`design-pending`. A future independent different-agent review is the approval
authority for the corrected design; this author commit is not.

1. False sorry-count claims (all 12 files are source-sorry-free; grep hit comments)
2. Missing BrascampLieb claim and repo inspection
3. Wrong live catalog statement ("not found" vs present in 200-paper catalog)
4. HF Jobs pricing error (not free)
5. Agent repo license detection error (no LICENSE file)
6. Missing top-three comparison
7. Missing Lean/parser-backed sorry analysis

**APPROVED — 2026-07-24.** An independent different-agent reviewer verified corrected
commit `3200c3e7ddac06633ea05dedd502c7e54adc0742` against authoritative orchestration
state `d7fc300eff937a958eccf886ff088d2b279ddd7f`. The persisted state contains the
dXPP (`2jpMiRwsrL`), TerminalTraj (`PeFSCRulgy`), and OXE-AugE (`LcswwEzzX7`)
rejections. The reviewer also rechecked the composite pins, exact claims, licensing
score and restrictions, Lean commands and axiom assertions, deterministic tracked
JSON plan, local-$0 default, expected-points arithmetic, dates, and whitespace.

This approval authorizes the root agent to transition the authoritative loop
separately. It does not itself change state/HANDOFF, start implementation, launch
paid compute, or mutate any external service.

---

*Design originally authored: 2026-07-24T11:27–11:34Z*
*Revised and independently approved: 2026-07-24*
*Challenge `challenge.json` accessed at dataset revision current as of 2026-07-24T11:53Z*
*Validator revision at design time: `5bbcad2e9a7e8a7479f3563ac1fc6c768d4bb050`*
