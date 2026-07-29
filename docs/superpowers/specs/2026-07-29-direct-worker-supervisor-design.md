# Direct Paper-Owner Worker Supervisor

**Date:** 2026-07-29
**Status:** Approved for implementation planning

## Goal

Continuously maintain 10 direct Agy paper-owner workers and 5 direct Codex
paper-owner workers. Every worker runs `icml-repro-loop` as a persistent,
controller-capable paper owner. The supervisor manages host processes only; it
never claims papers or mutates reproduction coordinator state.

## Requirements

- Launch `agy` and `codex` directly, without OpenCode or another dispatch layer.
- Maintain stable identities:
  - `agy-paper-owner-01` through `agy-paper-owner-10`
  - `codex-paper-owner-01` through `codex-paper-owner-05`
- Preserve healthy existing workers during installation and reconciliation.
- Give Agy unrestricted tool approvals, internet access, workspace writes, and
  controller credential access.
- Give Codex workspace writes, internet access, and controller credential
  access.
- Never place credential values in Git, command arguments, status output, or
  logs.
- Restart exited lanes without creating duplicate processes.
- Detect Agy quota errors, try another compatible model pool, and honor the
  reported reset time when every pool is exhausted.
- Keep operational status outside the repository.

## Architecture

A user-level systemd timer invokes an idempotent Python reconciler every 30
seconds. The reconciler uses one stable tmux session per worker. Tmux supplies
the terminal required by direct Agy CLI sessions and keeps both agent types
inspectable after the launching shell exits.

Repository files:

- `ops/worker_supervisor.py`: reconciliation and status CLI.
- `ops/systemd/icml-worker-supervisor.service`: one-shot reconciliation unit.
- `ops/systemd/icml-worker-supervisor.timer`: 30-second persistent timer.
- Focused tests under `tests/`.

Runtime files live under
`~/.local/state/icml-worker-supervisor/` and include a lock, compact status
JSON, and bounded per-lane diagnostic logs. No runtime file is committed.

## Reconciliation

Each run:

1. Acquires a nonblocking host lock.
2. Enumerates the 15 desired stable worker identities.
3. Reads tmux session and foreground-command state.
4. Adopts a lane only when its expected session has a live `agy` or `codex`
   foreground process.
5. Records an idle shell, dead pane, missing session, or exited agent as
   unhealthy.
6. Applies that lane's restart and quota backoff.
7. Creates or reuses its tmux session and sends the direct launch command.
8. Writes an atomic, credential-free status snapshot.

An existing healthy process is never killed merely to change configuration or
model choice. Manual and timer reconciliation share the same lock, so they
cannot create duplicates.

## Worker Commands

Every prompt instructs the worker to use the shared `icml-repro-loop` skill
directly, keep running its persistent paper-owner loop, and names its stable
worker ID.

Agy launches with automatic permission approval in edit mode. Model
configuration is represented as compatible command profiles so options such as
reasoning effort are included only when supported. The initial model preference
order is configurable. Quota errors advance to the next profile.

Codex launches with the existing proven direct command profile: ephemeral
execution, workspace-write sandbox, explicit workspace network access, high
reasoning effort, and the repository as its working directory.

The tmux shell resolves Hugging Face and GitHub tokens at process start and
places them only in the child environment. The generated command text contains
credential lookup commands, never credential values. Environment and command
values are redacted before any diagnostic serialization.

## Failure Handling

- Ordinary unexpected exits use bounded exponential backoff with jitter.
- A parsed quota reset supplies the earliest allowed retry time for that model
  profile.
- If another profile is available, the lane tries it on the next
  reconciliation rather than waiting for the exhausted profile.
- If all Agy profiles are unavailable, the lane remains visibly degraded until
  the earliest reset.
- Repeated launch failures do not hot-loop.
- A supervisor failure affects no paper lease: stable worker identity lets the
  restarted worker inspect and continue its fenced attempt through the skill.
- The supervisor does not infer attempt ownership or repair coordinator state.

## CLI and Observability

The reconciler exposes:

- `reconcile`: enforce the configured targets.
- `status`: print live, degraded, and backed-off lanes with models and restart
  counts.
- `stop`: disable supervision and stop only supervisor-owned tmux sessions,
  requiring an explicit confirmation flag.
- `install`: install and enable the user service and timer, then reconcile.
- `--dry-run`: report actions without changing processes or systemd state.

The status JSON records timestamps, worker IDs, process health, selected model
profile, restart count, next retry time, and sanitized last error. It never
contains prompts, tokens, or environment dumps.

## Testing and Acceptance

Unit tests use fake process, tmux, clock, credential, and filesystem adapters.
They cover:

- exact 10-Agy/5-Codex desired identities;
- direct command construction;
- compatible Agy model options;
- healthy-worker adoption;
- duplicate prevention under repeated reconciliation;
- dead-pane restart;
- quota fallback and reset-time backoff;
- lock contention;
- credential and error redaction;
- atomic status output;
- dry-run behavior.

An installation smoke test must show:

- the user timer is active;
- exactly 15 desired tmux sessions exist;
- every healthy lane's foreground command is `agy` or `codex`;
- a killed disposable test lane is restored by a later timer run;
- currently healthy production workers were not interrupted.

The smoke test must not claim, release, publish, submit, or otherwise mutate a
paper attempt.
