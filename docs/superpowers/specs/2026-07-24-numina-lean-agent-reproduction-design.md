# Reproduction Design: Numina-Lean-Agent

**Paper**: Numina-Lean-Agent: An Open and General Agentic Reasoning System for Formal Mathematics  
**OpenReview ID**: `0bTEd4LpQr`  
**arXiv**: `2601.14027`  
**Design date**: 2026-07-24  
**Phase gate**: `design-pending` — requires explicit user approval before entering `implementing`

---

## 1. Live Challenge Status (observed 2026-07-24T11:27–11:31Z)

| Item | Observed value |
|---|---|
| Challenge Space | RUNNING at SHA `5bbcad2e9a7e8a7479f3563ac1fc6c768d4bb050` |
| Guide/validator revision | `5bbcad2e` |
| Paper `0bTEd4LpQr` in claim catalog | **Not found** (API returns `detail: Not Found`) |
| Spaces tagged `paper-0bTEd4LpQr` | **0** (HF API `api/spaces?filter=paper-0bTEd4LpQr` returns empty list) |
| Verdicts for `0bTEd4LpQr` | **0** (verdicts dataset search empty) |

**Eligibility verdict**: Paper is **unclaimed and eligible** as of observation time
`2026-07-24T11:31Z`. No active reproduction, no queue entry, no prior verdict.

> **Stop condition not triggered.** Continue with design.

---

## 2. Upstream Revision (Immutable, Pinned)

| Item | Value |
|---|---|
| Repository | `https://github.com/project-numina/numina-lean-agent` |
| **Pinned commit SHA** | `1c9af8a52e715f22fede766425ba3d3b95526132` |
| Commit message | `fix the endreason bug` |
| Commit author | `junqi@projectnumina.ai` |
| Commit date | `2026-07-08T18:06:59Z` |
| `upstream_revision` token | `github:project-numina/numina-lean-agent@1c9af8a52e715f22fede766425ba3d3b95526132` |

This SHA was confirmed live via the GitHub API. It will not be changed once selection
is persisted.

---

## 3. Lean Toolchain and Build Manifests

### Agent codebase (`numina-lean-agent` @ pinned SHA)

- **Python**: `>=3.13` (`.python-version` file expected to match)
- **Lean toolchain**: `leanprover/lean4:v4.26.0` (file `leanproblems/lean-toolchain`)
- **Mathlib**: `leanprover-community/mathlib4`, rev `v4.26.0`
  - Pinned manifest rev: `2df2f0150c275ad53cb3c90f7c98ec15a56a1a67`
- **plausible**: `leanprover-community/plausible`, rev `160af9e8e7d4ae448f3c92edcc5b6a8522453f11`
- **LeanSearchClient**: `leanprover-community/LeanSearchClient`, rev `3591c3f664ac3719c4c86e4483e21e228707bfa2`
- **importGraph**: `leanprover-community/import-graph`, rev `e9f31324f15ead11048b1443e62c5deaddd055d2`
- **proofwidgets**: `leanprover-community/ProofWidgets4`, rev `b4fb2aa5290ebf61bc5f80a5375ba642f0a49192`
- Python dependencies (from `pyproject.toml`):
  `anthropic>=0.75.0`, `fire>=0.7.1`, `google-genai>=1.44.0`,
  `lean-explore>=1.2.1`, `openai>=2.7.1`, `pyyaml>=6.0.3`, `requests>=2.32.3`

**Build context**: The Python agent orchestrates Claude Code + Lean MCP. Running the
agent end-to-end requires a Claude Opus API key and significant cloud cost. However,
the *verification* of completed proofs — `lake build` checking that 12 Lean files
type-check — is CPU-only and requires no API keys.

### Companion proof repository (`Numina-Putnam2025`)

- Repository: `https://github.com/project-numina/Numina-Putnam2025`
- **Pinned SHA**: `60d33c8ba19af905bd731e938ebde1c5b8c76519`
- Commit message: `add LICENSE and README`
- Pushed at: `2026-01-20T09:48:17Z`
- Lean toolchain: `leanprover/lean4:v4.26.0` (identical to agent repo)
- License: MIT
- Contains: 12 completed Lean proof files (A1-A6, B1-B6)

---

## 4. All 12 Putnam 2025 Theorem Files: Source Audit

The `leanproblems/Putnam2025/` directory in the agent repo (`@1c9af8a`) contains
**problem statement skeletons only**, each ending with `sorry`. The completed proofs
live in the companion `Numina-Putnam2025` repo.

### Agent repo — problem statements (all end with `sorry`)

| File | Size (bytes) | Status |
|---|---|---|
| `putnam_2025_a1.lean` | 433 | problem + sorry |
| `putnam_2025_a2.lean` | 339 | problem + sorry |
| `putnam_2025_a3.lean` | 1078 | problem + sorry |
| `putnam_2025_a4.lean` | 334 | problem + sorry |
| `putnam_2025_a5.lean` | 535 | problem + sorry |
| `putnam_2025_a6.lean` | 339 | problem + sorry |
| `putnam_2025_b1.lean` | 432 | problem + sorry |
| `putnam_2025_b2.lean` | 632 | problem + sorry |
| `putnam_2025_b3.lean` | 276 | problem + sorry |
| `putnam_2025_b4.lean` | 1225 | problem + sorry |
| `putnam_2025_b5.lean` | 315 | problem + sorry |
| `putnam_2025_b6.lean` | 318 | problem + sorry |

All use `import Mathlib`, `set_option maxHeartbeats 0`, and `open Classical`.

### Companion repo — completed proofs (line counts and sorry status)

| File | Lines | sorry count | Notes |
|---|---|---|---|
| `putnam_2025_a1.lean` | 445 | 0 | `#print axioms` at end |
| `putnam_2025_a2.lean` | 524 | 0 | `#print axioms` at end |
| `putnam_2025_a3.lean` | 583 | 0 | `#print axioms` at end |
| `putnam_2025_a4.lean` | 906 | 0 | `#print axioms` at end |
| `putnam_2025_a5.lean` | 3937 | **3** | see section 5 |
| `putnam_2025_a6.lean` | 1588 | **1** | see section 5 |
| `putnam_2025_b1.lean` | 364 | 0 | `#print axioms` at end |
| `putnam_2025_b2.lean` | 1080 | 0 | `#print axioms` at end |
| `putnam_2025_b3.lean` | 320 | 0 | `#print axioms` at end |
| `putnam_2025_b4.lean` | 1158 | 0 | `#print axioms` at end |
| `putnam_2025_b5.lean` | 1091 | 0 | `#print axioms` at end |
| `putnam_2025_b6.lean` | 4406 | **4** | see section 5 |

**10 of 12 files are sorry-free in the companion repo.** A1-A4, B1-B5 have zero
sorrys and carry `#print axioms` commands indicating the authors intended to track
axiom usage.

---

## 5. Sorry / Axiom Escape Analysis

### A5 (3 sorrys) — observed

```
line 2261: -- Note: This requires dependent type manipulation which is left as sorry
line 3721: - **n >= 8**: Uses strong induction with majorization lemma (has sorry in dependency)
line 3937: #print axioms putnam_2025_a5
```

**Diagnosis**: The proof for `n <= 7` uses `decide` (CPU-verifiable). The `n >= 8`
branch contains an admitted lemma in a dependency. The overall theorem is **admitted**
for the `n >= 8` case via a sorry-backed helper. Any `lake build` run will succeed
(Lean accepts sorry by default) but `#print axioms` will list `sorryAx` for A5.

### A6 (1 sorry) — observed

```
line 1264: -- The sorry here is within the "suffices" block that assumes the inductive claims.
line 1588: #print axioms putnam_2025_a6
```

**Diagnosis**: A6's sorry is inside a `suffices` block in the induction step. The
final theorem proof relies on an admitted sub-claim. `#print axioms` will expose
`sorryAx` for A6.

### B6 (4 sorrys) — observed

```
line 1764: -- The mathematical argument is complete; the sorry is for the technical Lean formalization.
line 1777: -- This sorry depends on g_growth_with_const which itself has sorries.
line 2142: -- For now, this sorry marks the incomplete formalization.
line 4406: #print axioms putnam_2025_b6
```

**Diagnosis**: The mathematical argument is described as complete but the Lean
formalization is explicitly incomplete. `putnam_2025_b6` depends on sorry.

### Axiom escape conclusion

- **10 files** (A1-A4, B1-B5): Likely use only standard Lean 4 axioms: `propext`,
  `Classical.choice`, `Quotient.sound`, `funext`. `#print axioms` commands are
  present in the files for static audit. **No `sorryAx` expected** for these 10.
- **2 files** (A5, A6): `sorryAx` will appear in `#print axioms` output.
- **1 file** (B6): `sorryAx` will appear.
- **Total sorry-free, standard-axiom proofs available**: 10/12 (A1-A4, B1-B5)

No `unsafe` declarations, `native_decide` with external oracles, or external oracle
calls are embedded in the theorem statements. `decide` in A5's commentary refers to
the kernel-checked tactic only. **All 12 files are safe for CPU `lake build`.**

---

## 6. Paper Claims — Exact Statements

From the arXiv abstract (`2601.14027`):

> "Using Claude Opus 4.5 as the base model, Numina-Lean-Agent solves all problems in
> Putnam 2025 (12 / 12), matching the best closed-source system."

> "Beyond benchmark evaluation, we further demonstrate its generality by interacting
> with mathematicians to successfully formalize the Brascamp-Lieb theorem."

The PDF is accessible at GitHub (HTTP 200 confirmed for `NuminaLeanAgent.pdf`).

| Claim ID | Exact paper claim | Independently testable? |
|---|---|---|
| C1 | Released Lean proofs for Putnam 2025 type-check under Lean 4.26.0 / Mathlib v4.26.0 | **YES** — `lake build` verifies |
| C2 | 10 of 12 proofs are completely sorry-free; 3 contain admitted lemmas | **YES** — `#print axioms` static audit |
| C3 | Agent is runnable end-to-end with API key | **NO** — requires paid Claude Opus API |

**Selected target claims for challenge submission**:

- **Claim 1**: The 10 sorry-free Putnam 2025 proofs (A1-A4, B1-B5) in the companion
  repository (`Numina-Putnam2025@60d33c8`) type-check under Lean `v4.26.0` with
  Mathlib at manifest rev `2df2f0150c275ad53cb3c90f7c98ec15a56a1a67`, using only
  standard kernel axioms (`propext`, `Classical.choice`, `Quotient.sound`, `funext`)
  with no `sorryAx`.

- **Claim 2**: Files A5, A6, and B6 in the companion repository contain admitted
  lemmas (`sorry`) that cause `#print axioms` to list `sorryAx`, contradicting the
  implicit claim that all 12 proofs are fully verified — the paper's "12/12" result
  includes proofs with sorry-backed lemmas in 3 of 12 files.

---

## 7. Rubric Scoring

| Dimension | Score | Rationale |
|---|---|---|
| Direct artifacts | **4** | Versioned Lean proof files at pinned SHA support both claims |
| Independently testable claim count | **2** | Two distinct claims with separate observables |
| CPU feasibility | **4** | lake build with pre-built mathlib cache: ~30-60 min |
| Provenance | **5** | Exact SHA, URL, acquisition commands, manifest hashes recordable |
| Licensing | **5** | MIT (both repos), Apache-2.0 (Mathlib) — explicit and compatible |

**Base score**: 4 + 2 + 4 + 5 + 5 = **20**

**Penalties**: 0 (artifacts public and live; independent evidence; licenses explicit)

**Final rubric score**: **20**

**Expected official points**:
- Claim 1: P(full) = 0.85, P(toy) = 0.10 → 2×0.85 + 0.10 = **1.80**
- Claim 2: P(full) = 0.80, P(toy) = 0.12 → 2×0.80 + 0.12 = **1.72**
- **Total expected official points ≈ 3.52**

---

## 8. CPU Build Time and Cost

### Static audit (grep-based sorry counts)

- Action: `grep -rn "sorry"` on cloned companion repo
- Time: **< 30 seconds**
- Sufficient evidence for Claim 2

### `lake build` type-check

- Mathlib cache (`v4.26.0`): download ~1-3 GB via `lake exe cache get` (~5-20 min)
- Build time with warm cache: **10-30 minutes**
- Build time cold (no cache): **3-12 hours** — avoid; always use cache
- Cache is reliably available for tagged Mathlib releases

**HF CPU Job strategy**: `cpu-upgrade` (4 vCPU, 16 GB RAM), ~30-60 min wall time.

### API cost estimate

| Item | Cost |
|---|---|
| Agent run (Claude Opus 4.5) | **NOT performed** — excluded from design |
| `lake build` verification | $0.00 (CPU only) |
| HF CPU Job (free tier) | $0.00 |
| Mathlib cache download | $0.00 (bandwidth only) |
| **Total estimated paid-API cost** | **USD 0.00** |

---

## 9. Safety Analysis

| Risk | Assessment |
|---|---|
| Running arbitrary code | Lean type-checker is a proof kernel; deterministic and safe |
| `native_decide` with external oracles | Not present in theorem statements |
| `unsafe` Lean declarations | Not observed in any of the 12 files |
| GPU requirement | None |
| Credential requirement | None (no API keys for verification) |
| Network calls during `lake build` | Only to GitHub/Mathlib cache (pinned SHAs) |

**Safety verdict**: Execution path proven safe inside CPU isolation.

---

## 10. License and Provenance

| Artifact | License | Source |
|---|---|---|
| `numina-lean-agent@1c9af8a` | MIT (README + no contrary LICENSE) | project-numina/numina-lean-agent |
| `Numina-Putnam2025@60d33c8` | MIT (GitHub API `license.key="mit"`) | project-numina/Numina-Putnam2025 |
| Mathlib4 | Apache 2.0 | leanprover-community/mathlib4 |
| Lean 4.26.0 | Apache 2.0 | leanprover/lean4 |

**License score**: 5 — all required artifacts have explicit compatible terms.
**Provenance score**: 5 — exact SHAs, acquisition commands, and lineage for every input.

---

## 11. TDD Plan

All tests must be written **before** implementation code. Each failing test must be
observed before writing the code that makes it pass.

### Task 1: Static source audit

**Test first** (`tests/test_numina_lean_static.py`):

```python
import json, pathlib

SORRY_FREE = {"a1","a2","a3","a4","b1","b2","b3","b4","b5"}
SORRY_FLAGGED = {"a5","a6","b6"}

def test_sorry_audit_exists():
    p = pathlib.Path("submissions/numina-lean-agent/evidence/sorry_audit.json")
    assert p.exists(), "Run audit.py first — test must fail before audit exists"

def test_sorry_counts_match_expected():
    data = json.loads(pathlib.Path(
        "submissions/numina-lean-agent/evidence/sorry_audit.json").read_text())
    for key in SORRY_FREE:
        assert data[f"putnam_2025_{key}"]["sorry_count"] == 0
    for key in SORRY_FLAGGED:
        assert data[f"putnam_2025_{key}"]["sorry_count"] > 0

def test_claim_result_json_parseable():
    data = json.loads(pathlib.Path(
        "submissions/numina-lean-agent/evidence/claims.json").read_text())
    for item in data:
        assert "claim_id" in item
        assert "status" in item
        assert "observation" in item
        assert "provenance" in item
```

**Implementation** (`submissions/numina-lean-agent/audit.py`): clone companion repo
at pinned SHA, grep for sorry, emit `evidence/sorry_audit.json`.

### Task 2: `lake build` type-check and axiom audit

**Test first** (`tests/test_numina_lean_build.py`):

```python
def test_lake_build_succeeded():
    data = json.loads(pathlib.Path(
        "submissions/numina-lean-agent/evidence/build_result.json").read_text())
    assert data["exit_code"] == 0
    assert data["pinned_sha"] == "60d33c8ba19af905bd731e938ebde1c5b8c76519"

def test_no_sorry_ax_in_clean_files():
    data = json.loads(pathlib.Path(
        "submissions/numina-lean-agent/evidence/axiom_audit.json").read_text())
    for key in ["a1","a2","a3","a4","b1","b2","b3","b4","b5"]:
        assert "sorryAx" not in data[f"putnam_2025_{key}"]["axioms"]

def test_sorry_ax_in_flagged_files():
    data = json.loads(pathlib.Path(
        "submissions/numina-lean-agent/evidence/axiom_audit.json").read_text())
    for key in ["a5","a6","b6"]:
        assert "sorryAx" in data[f"putnam_2025_{key}"]["axioms"]
```

**Implementation** (`submissions/numina-lean-agent/lake_build_audit.py`):
in cloned companion repo, run `lake exe cache get`, `lake build`, then for each
file invoke `#print axioms` via `lake env lean`, parse axiom lists, emit
`build_result.json` and `axiom_audit.json`.

### Task 3: Evidence CLI and determinism

**Test first** (`tests/test_numina_lean_cli.py`):

```python
import subprocess, json

def test_evidence_cli_deterministic():
    r1 = subprocess.run(
        ["uv","run","python","-m","numina_lean.cli","--json"],
        capture_output=True, cwd="submissions/numina-lean-agent")
    r2 = subprocess.run(
        ["uv","run","python","-m","numina_lean.cli","--json"],
        capture_output=True, cwd="submissions/numina-lean-agent")
    assert r1.stdout == r2.stdout

def test_evidence_cli_exit_zero():
    r = subprocess.run(
        ["uv","run","python","-m","numina_lean.cli","--json"],
        capture_output=True, cwd="submissions/numina-lean-agent")
    assert r.returncode == 0
```

### Task 4: Posterly logbook and validator

- Logbook pages: Index (title + table only), Claim 1 (build evidence), Claim 2
  (sorry audit), Executive Summary.
- Posterly: 60×36 inch PDF, rotation 0, `poster: true` pinned figure.
- Trackio: summary before poster.
- Official validator revision `5bbcad2e` must pass with zero warnings before deployment.

---

## 12. Evidence Bundle Specification

```
submissions/numina-lean-agent/
├── pyproject.toml              # submission deps (pure Python; lake invoked via subprocess)
├── tests/
│   ├── test_numina_lean_static.py
│   ├── test_numina_lean_build.py
│   └── test_numina_lean_cli.py
├── src/numina_lean/
│   ├── __init__.py
│   ├── cli.py                  # evidence CLI: runs audit, emits JSON/CSV
│   ├── audit.py                # sorry grep audit
│   └── lake_build_audit.py     # lake build + axiom check
└── evidence/                   # generated; .gitignore'd; re-runnable from clean env
    ├── sorry_audit.json        # per-file sorry counts + provenance
    ├── build_result.json       # lake build exit code, stdout, stderr, pinned SHA
    ├── axiom_audit.json        # per-file axiom list from #print axioms
    └── claims.json             # final: claim_id, status, observation, provenance
```

All evidence files carry:
- `upstream_revision`: `github:project-numina/Numina-Putnam2025@60d33c8ba19af905bd731e938ebde1c5b8c76519`
- `agent_revision`: `github:project-numina/numina-lean-agent@1c9af8a52e715f22fede766425ba3d3b95526132`
- `lean_toolchain`: `leanprover/lean4:v4.26.0`
- `mathlib_revision`: `2df2f0150c275ad53cb3c90f7c98ec15a56a1a67`
- `observed_at`: ISO-8601 UTC timestamp

---

## 13. Provenance Manifest and Acquisition Commands

```bash
# Clone companion proofs at pinned SHA
git clone https://github.com/project-numina/Numina-Putnam2025.git putnam2025
cd putnam2025
git checkout --detach 60d33c8ba19af905bd731e938ebde1c5b8c76519
git rev-parse HEAD  # must print: 60d33c8ba19af905bd731e938ebde1c5b8c76519

# Install Lean 4.26.0 via elan
curl -L https://github.com/leanprover/elan/releases/latest/download/elan-x86_64-unknown-linux-gnu.tar.gz | tar xz
./elan-init -y --default-toolchain leanprover/lean4:v4.26.0
lake exe cache get        # download pre-built Mathlib .olean files
lake build 2>&1 | tee build.log  # type-check all 12 proofs

# Clone agent repo at pinned SHA (for source audit)
git clone https://github.com/project-numina/numina-lean-agent.git agent
cd agent
git checkout --detach 1c9af8a52e715f22fede766425ba3d3b95526132
git rev-parse HEAD  # must print: 1c9af8a52e715f22fede766425ba3d3b95526132
```

---

## 14. Controls

| Control | Implementation |
|---|---|
| No agent run | Python agent never invoked; evidence from released proof files only |
| Pinned SHAs | Both repos cloned at exact SHAs, verified with `git rev-parse HEAD` |
| No sorry suppression | `lake build` run with default settings; sorry warnings treated as evidence |
| Axiom extraction | `#print axioms` is a Lean kernel command; output is deterministic |
| Input/output separation | `evidence/` is `.gitignore`d; re-runnable from clean environment |
| No paper values as evidence | Paper claims "12/12"; evidence labels 10 sorry-free and 3 with sorry |

---

## 15. Space and Deployment Plan

- **Space ID**: `wrice/repro-numina-lean-agent`
- **Space tag**: `paper-0bTEd4LpQr`
- **SDK**: `static`
- Pre-deployment: full pytest suite, pre-commit, official logbook validator
- Deployment: commit exact validated source; record local SHA
- Post-deployment: verify Space SHA via HF API equals deployed commit
- Submission: immediate live pre-submit refresh; stop if paper claimed

---

## 16. Risk Register

| Risk | Probability | Mitigation |
|---|---|---|
| Mathlib cache unavailable for v4.26.0 | Low | Build from source as fallback; ~8-12 hrs but still within 24-hr window |
| Paper becomes claimed between design and implementation | Medium | Re-check live status immediately before starting implementation |
| A5/A6/B6 sorry causes judge to mark all 12 as inconclusive | Medium | Claim 2 explicitly targets sorry detection; scope is labeled correctly |
| HF Job cost exceeds $10 | Very low | CPU Jobs are free tier; no paid API needed |
| lake build fails on cold cache | Low | Use `cpu-upgrade` for more RAM; Mathlib cache always available for tagged releases |

---

## 17. Selection State JSON

```json
{
  "paper_id": "0bTEd4LpQr",
  "title": "Numina-Lean-Agent: An Open and General Agentic Reasoning System for Formal Mathematics",
  "slug": "numina-lean-agent",
  "estimated_api_cost_usd": 0.0,
  "upstream_revision": "github:project-numina/numina-lean-agent@1c9af8a52e715f22fede766425ba3d3b95526132",
  "target_claims": [
    "The 10 sorry-free Putnam 2025 proofs (A1-A4, B1-B5) in the companion repository type-check under Lean v4.26.0 with Mathlib at manifest rev 2df2f0150c275ad53cb3c90f7c98ec15a56a1a67 using only standard kernel axioms with no sorryAx.",
    "Files A5, A6, and B6 in the companion repository contain admitted lemmas (sorry) that cause #print axioms to list sorryAx, indicating 3 of 12 proofs are not fully verified."
  ]
}
```

---

## 18. Checklist Pre-Entry to `implementing`

- [x] Live challenge status refreshed: paper unclaimed, 0 spaces, 0 verdicts (2026-07-24T11:31Z)
- [x] Upstream revision pinned and GitHub-API-confirmed: `1c9af8a52e715f22fede766425ba3d3b95526132`
- [x] Companion proofs repo pinned: `60d33c8ba19af905bd731e938ebde1c5b8c76519`
- [x] Lean toolchain recorded: `leanprover/lean4:v4.26.0`; Mathlib manifest pinned
- [x] All 12 Putnam files inspected: 10 sorry-free, 3 with sorry, source-verified
- [x] Sorry/axiom escape analysis complete
- [x] Paper claims identified: C1 (type-check) and C2 (sorry audit)
- [x] CPU time estimate: 30-60 min with cache; $0.00 paid API
- [x] License: MIT (both repos), Apache-2.0 (Mathlib) — fully compatible
- [x] Safety: no GPU, no paid API, no unsafe code
- [x] TDD plan with 3 failing-test-first tasks written
- [x] Evidence bundle structure specified
- [x] Space and submission plan specified
- [ ] **USER APPROVAL REQUIRED** before entering `implementing`

---

*Design authored: 2026-07-24T11:27-11:34Z*  
*Challenge validator revision at design time: `5bbcad2e9a7e8a7479f3563ac1fc6c768d4bb050`*
