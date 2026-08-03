# ICML 2026 Reproductions

Workspace for the [ICML 2026 Agent Repro Challenge](https://huggingface.co/spaces/ICML-2026-agent-repro/challenge)
(July 15 – August 2, 2026 AoE): a community effort to reproduce the major
claims of every ICML 2026 paper, with per-claim verdicts issued by an
automated judge against Hugging Face Space logbooks.

**Final result: 195 points, rank 31 of 372** — 92 papers submitted and judged,
with 35 claims `verified`, 5 `falsified` (correctly disproving a claim scores
the same as verifying one), and ~90 `toy`.

Root-authored content is licensed under the [MIT License](LICENSE). Bundled
NAPE components retain their own licenses; see
[`submissions/nape/LICENSE`](submissions/nape/LICENSE) and
[`submissions/nape/external/NAPE/LICENSE`](submissions/nape/external/NAPE/LICENSE).

## How this repository worked

Every paper is an independent Python project under `submissions/<paper>/` with
its own lockfile, tests, deterministic evidence generator, judge-visible
logbook (`pages/*.md`), and Hugging Face Space source. A fleet of persistent
"paper-owner" agent workers (Gemini, Claude, and GPT sessions launched by
`ops/worker_supervisor.py`) ran the reproduction lifecycle concurrently in
this single checkout, coordinated through a fenced, sharded state store:

- `skills/icml-repro-loop/`: the versioned skill the workers follow — paper
  selection, design review, guarded implementation, validation attestation,
  publication, submission, verdict import, and release, each gated by
  immutable content-addressed attestations and fencing-token leases.
- `state/repro-loop.json` + `state/repro-loop/`: the coordinator index and
  its attempt, snapshot, lease, attestation, transaction, and judgment
  shards. Live state is never git-tracked.
- `ops/worker_supervisor.py`: launches and heals the worker fleet under a
  systemd user timer.

Submission is tag-based: a Space carrying `icml2026-repro` and
`paper-<id>` tags with an unjudged revision enters the judge's queue; a new
revision of a judged Space is rejudged and its verdict replaced.

## What we learned (the endgame, condensed)

The final 30 hours nearly doubled the score (100 → 195, rank 48 → 31) without
any new reproduction science. The gains came from operational truths that are
easy to miss:

1. **The judge scores only what the Space serves.** Concrete numbers rendered
   in `pages/*.md` earned `toy`/`verified` credit; summary manifests,
   invisible evidence bundles, and Streamlit `.py` pages judged to zero.
   The publish gate now mechanically requires plural numeric markdown pages.
2. **Honesty outscores assertion.** An integrity audit found several
   submissions with hard-coded paper values presented as measurements —
   including a fabricated human-evaluation table sitting in the judge queue.
   All were rebuilt with real executed evidence; honestly-reported
   contradictions (`not_reproduced`) earned credit where fabrications earned
   nothing.
3. **Platform quotas are the real scheduler.** Space creation is a rolling
   24-hour window (20/day); concurrently *running* Spaces are capped per
   account, and auto-paused Spaces silently fail both republishing and
   judging. The endgame required treating running slots and creation slots
   as managed resources, spent in evidence-quality order.
4. **Make interfaces produce the desired behavior.** Prompt-level directives
   were unreliable across worker fleets; the durable fixes were mechanical —
   a `claim-next` that routes saturated claims onto reclaimable work instead
   of erroring, a publish gate that refuses invisible evidence, and commits
   to `main` as the only durable home for validated sources.
5. **Verify deadlines as facts.** The close was midnight Anywhere-on-Earth,
   twelve hours later than assumed; the last-verdict timestamp pattern in the
   official feed is the ground truth.

## Layout

- `submissions/<paper>/`: independent reproduction project per paper.
- `skills/icml-repro-loop/`: reproduction-loop skill (scripts, references).
- `ops/`: worker fleet supervisor.
- `state/`: coordinator state (untracked; durability via sharded store).
- `docs/`: designs, plans, and `docs/REMOTE_SETUP.md` for host setup.
- `scratch/`: workspace symlink onto a large data volume for staging clones.

Each challenge entry is deployed to a separate Hugging Face Space because
challenge metadata and judging are specific to that Space.

## NAPE Snapshot

[`submissions/nape/`](submissions/nape/) is an immutable convenience snapshot
of the canonical [NAPE reproduction repository](https://github.com/will-rice/icml-2026-repro).
It is not an independently runnable submission: its archive intentionally
contains no nested Git metadata. The canonical repository remains
authoritative. See [`submissions/nape/UPSTREAM.md`](submissions/nape/UPSTREAM.md)
for the immutable source revisions and import method.

## Verifying the workspace

```bash
uv sync --frozen
uv run pytest -q
uv run skills/icml-repro-loop/scripts/quick_validate.py skills/icml-repro-loop
uv run pre-commit run -a
```

Operational status is read from `state/repro-loop.json` and its referenced
shards through `list-attempts`, `show-attempt`, and `show-snapshot`; the
official record is the `ICML-2026-agent-repro/verdicts` dataset.
