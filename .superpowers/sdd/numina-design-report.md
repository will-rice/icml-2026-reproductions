# Numina-Lean-Agent Reproduction Design Report (Final Review Corrections)

**Session**: 2026-07-24
**Worktree**: `design/numina-lean`
**Phase**: `design-pending` — no state or HANDOFF write was authorized

## Review Record

| Event | Disposition |
|---|---|
| Original design, commit `0b5c0a6` | rejected |
| First revision, commit `01f3446` | rejected/design-pending |
| This correction pass | design-pending; independent different-agent review required |

An author cannot self-approve a design. This report records corrections only. A
future independent different-agent review may `APPROVE`; this author commit does
not select the paper, transition the loop, start a build, or authorize an HF Job.

## Final Corrections

1. The licensing dimension is **2**, yielding base **16**, the existing **-2**
   BrascampLieb penalty, and final **14**. Putnam has MIT terms, while the agent
   repo has no LICENSE file and BrascampLieb has no license at all.
2. The Space/evidence plan now tracks byte-stable, locally-authored normalized JSON
   only. It cannot contain BrascampLieb source, cache, binaries, raw build output,
   or raw Lean logs. Timestamps, host paths, elapsed time, and ignored evidence are
   excluded so tracked evidence is deterministic.
3. The alternative pool is recorded honestly: dXPP/TerminalTraj rejections and the
   OXE exclusion are **not yet persisted** in authoritative state. Selection cannot
   transition until the root agent serially persists them; this worktree does not
   edit state or HANDOFF.
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
- Root-serialized candidate closures and an independent different-agent `APPROVE`
  remain blockers before any state transition or paid execution.

## Verification

`git diff --check` and a targeted content scan are required after the patch; no
implementation test suite is run because this task changes design/report documents
only.
