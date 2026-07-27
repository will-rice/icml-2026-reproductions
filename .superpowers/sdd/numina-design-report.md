# Numina-Lean-Agent Reproduction Design Report (Final Review Corrections)

**Session**: 2026-07-24
**Worktree**: `design/numina-lean`
**Phase**: `design-pending` — independently approved; no state or HANDOFF write

## Review Record

| Event | Disposition |
|---|---|
| Original design, commit `0b5c0a6` | rejected |
| First revision, commit `01f3446` | rejected/design-pending |
| Author correction commit `3200c3e` | design-pending; independent review required |
| Independent review of `3200c3e` | **APPROVED** against orchestration state `d7fc300` |

An author cannot self-approve a design. Commit `3200c3e` recorded corrections only;
the independent decision below now approves them. Neither commit selects the paper,
transitions the loop, starts a build, or authorizes an HF Job.

## Final Corrections

1. The licensing dimension is **2**, yielding base **16**, the existing **-2**
   BrascampLieb penalty, and final **14**. Putnam has MIT terms, while the agent
   repo has no LICENSE file and BrascampLieb has no license at all.
2. The Space/evidence plan now tracks byte-stable, locally-authored normalized JSON
   only. It cannot contain BrascampLieb source, cache, binaries, raw build output,
   or raw Lean logs. Timestamps, host paths, elapsed time, and ignored evidence are
   excluded so tracked evidence is deterministic.
3. The alternative pool is closed in authoritative orchestration state `d7fc300`:
   dXPP (`2jpMiRwsrL`), TerminalTraj (`PeFSCRulgy`), and OXE-AugE (`LcswwEzzX7`)
   are persisted rejected. This worktree does not edit state or HANDOFF.
4. All 12 pinned Putnam files already contain committed `#print axioms` commands.
   The pipeline executes and parses those files directly, using the observed spelling
   `Quot.sound`, with no invented `funext` assertion. It adds only the local
   BrascampLieb query file.
5. The conservative expected-points math is explicit: Putnam
   `2×0.75 + 1×0.15 = 1.65`; BrascampLieb
   `2×0.65 + 1×0.20 = 1.50`; total **3.15**.
6. The pinned BrascampLieb commit date is **2026-04-10T15:20:33Z**.
7. `upstream_revision` now binds the agent, Putnam, and BrascampLieb SHA values in
   one immutable composite token.
8. Local CPU is the $0 default. An HF CPU Job is paid and optional; it requires a
   separate explicit approval naming hardware and maximum spend.
9. The design records the reviewer’s pinned BrascampLieb observations: Lean v4.28.0,
   `defaultTargets = ["BrascampLieb"]`, Mathlib v4.28.0, the `upperBound`
   declaration location, successful cache/build observation, and an axiom query
   without `sorryAx`. These facts do not permit retention or redistribution of its
   cache, output, or source.
10. Design/report whitespace and the current diff are checked before commit.
11. The checklist now correctly counts **4 tasks and 9 tests**.

## Source Facts Retained

| Finding | Verification / disposition |
|---|---|
| Putnam source audit | All 12 files have zero tactic-level `sorry`; earlier grep hits were comments. |
| BrascampLieb source audit | All 21 `.lean` files have zero tactic-level `sorry`. |
| Putnam axioms | All 12 pinned files contain committed `#print axioms`; review observed `Quot.sound`, and no universal `funext` allowlist is justified. |
| BrascampLieb | Pinned `413f2bfd31100187eb6c2d632c9cbf12e3115494`; commit timestamp `2026-04-10T15:20:33Z`. |
| Catalog | Paper is present with four unverified claims; this document does not change live state. |

## Commands and Boundaries

The design contains future rerun commands only. No external service was mutated in
this correction pass. Putnam axiom extraction executes each pinned committed proof
file directly; only BrascampLieb needs a locally authored query file. Raw output is
not an evidence artifact; implementation must derive and commit only the normalized
JSON summaries described in the design.

## Remaining Concerns

- The proof checks validate released artifacts, not an agent rerun; judges may award
  toy or inconclusive credit.
- Two no-file licenses make a source-free Space mandatory and lower selection value.
- Paid HF execution still requires separate explicit approval; local CPU remains the
  $0 default.

## Independent Approval

**APPROVED — 2026-07-24.** The independent reviewer checked corrected design commit
`3200c3e7ddac06633ea05dedd502c7e54adc0742` against authoritative orchestration
commit `d7fc300eff937a958eccf886ff088d2b279ddd7f`. All prior review blockers are
resolved, including persisted candidate closure, exact composite pins, licensing
score and JSON-only redistribution, Lean commands and axiom facts, deterministic
tracked evidence, local-$0 execution, expected-points arithmetic, dates, task/test
counts, and whitespace.

This approval permits the root agent to update authoritative state separately. It
does not modify state/HANDOFF, start implementation, or authorize paid execution.

## Verification

`git diff --check` and a targeted content scan are required after the patch; no
implementation test suite is run because this task changes design/report documents
only.
