# NAPE Import Report

## Status

Imported an immutable convenience snapshot of canonical NAPE into
`submissions/nape/` and created root operating documentation. The snapshot is
not an independently runnable submission.

## Source Revision

- Canonical repository: https://github.com/will-rice/icml-2026-repro.git
- Canonical commit: `7220279222f1abac3056da78c7b8623a2a03e12b`
- Canonical tree: `30630f17c01604fb813c0d2928602a5a5bc29ea9`
- NAPE submodule commit: `ac0d10e4dc345f982a5665a2c4bdb6b752d663f2`
- Import date: 2026-07-21

## Import Boundary

The source repository was clean before import. The snapshot was created with
`git archive` from the canonical commit. Because `external/NAPE` is a pinned
submodule, its committed tree was also imported with `git archive` at the
recorded gitlink SHA. No working-tree files or nested `.git` metadata were
copied. The only parent overlays are the archive notice prepended to
`submissions/nape/README.md` and `submissions/nape/UPSTREAM.md`; all other
snapshot files match the archived committed trees.

## Validation

- Canonical source checkout: `uv run pytest -q` reported `119 passed`.
- The archived NAPE pytest and pre-commit commands are intentionally not run:
  canonical evidence checks require Git provenance, while this archive has no
  nested Git metadata.
- Parent workspace: `uv run pytest -q` reported `260 passed`; `uv run
  pre-commit run -a` passed all hooks after this archival-model adjustment. The
  root pre-commit configuration excludes `^submissions/nape/` so parent hooks
  preserve the immutable archive while continuing to cover parent tests, skill
  files, and documentation.

## Boundary

The canonical project depends on an upstream NAPE submodule. The snapshot
includes that exact pinned submodule tree but must not be normalized or treated
as a Git checkout. Run evidence commands in the canonical repository, whose
Git metadata can satisfy the pinned-checkout checks.
