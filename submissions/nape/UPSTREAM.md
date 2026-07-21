# Upstream Provenance

## Canonical Repository

- URL: https://github.com/will-rice/icml-2026-repro.git
- Imported commit: `7220279222f1abac3056da78c7b8623a2a03e12b`
- Imported tree: `30630f17c01604fb813c0d2928602a5a5bc29ea9`
- Import date: 2026-07-21

## Pinned NAPE Submodule

The canonical repository records `external/NAPE` as a Git submodule. Its
committed revision was archived separately so the snapshot can resolve its
declared workspace dependency without nested Git metadata.

- URL: https://github.com/Tej-55/NAPE.git
- Imported commit: `ac0d10e4dc345f982a5665a2c4bdb6b752d663f2`
- Imported tree: `6f3623ec11ab58e1845df259102f8316cdfdc533`

## Import Method

Only committed trees were imported. The exact archive commands were:

```bash
git -C /Users/will/icml-2026-repro archive --format=tar --output=/tmp/nape-submission.tar HEAD
mkdir -p submissions/nape
tar -xf /tmp/nape-submission.tar -C submissions/nape
git -C /Users/will/icml-2026-repro/external/NAPE archive --format=tar --output=/tmp/nape-submodule.tar ac0d10e4dc345f982a5665a2c4bdb6b752d663f2
tar -xf /tmp/nape-submodule.tar -C submissions/nape/external/NAPE
```

Neither archive includes Git metadata. This parent copy is a snapshot for the
ICML 2026 reproduction workspace; the canonical repository above remains the
authoritative source.

## Parent Overlays

The archive payload is otherwise exact. The only parent-added overlays are:

- `README.md`: the archive notice that directs setup, submodule, and test
  commands to the canonical repository.
- `UPSTREAM.md`: this immutable provenance record.

## Evidence Validation

Evidence commands that require Git provenance must run in the canonical
repository. This archive intentionally has no nested Git metadata, so its
`external/NAPE` directory cannot satisfy the canonical evidence checks for the
pinned checkout SHA. Use the canonical repository for `uv run pytest -q` and
`uv run pre-commit run -a`.
