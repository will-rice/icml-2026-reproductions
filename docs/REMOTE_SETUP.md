# Remote Setup

Clone the parent repository on a new host, then perform the authentication and
local verification checks below. Do not place credentials, tokens, cookies, or
other secrets in repository files.

## Install Tools, Clone, And Authenticate

Install `uv` using its official installer, then confirm it is available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

Alternatively, reopen the shell or source its updated profile before running
`uv --version`.

Install or authenticate the other tools interactively as needed. Then clone the
repository and verify tool availability and authentication:

```bash
git clone https://github.com/will-rice/icml-2026-reproductions.git
cd icml-2026-reproductions
git submodule status
command -v gh
command -v hf
command -v orx
gh auth status
hf auth whoami
orx --help
```

`git submodule status` must produce no entries. Authenticate interactively if
either authentication check fails. Resume work from `state/repro-loop.json`
and its referenced attempt shards. First inspect its schema: retained schema v3
must receive only the write-free migration dry-run until an explicit,
digest-pinned migration is separately authorized; schema v6 has no single
current paper and should be inspected by listing attempts.

## Verify Required Superpowers Skills

Before starting the reproduction loop, use this diagnostic check for the three
required Superpowers skill files. It is independent of the cached plugin
version and vendor directory, but cache files alone do not prove that Codex has
activated the skills:

```bash
CODEX_HOME=${CODEX_HOME:-$HOME/.codex}
missing=0
for required_skill in brainstorming test-driven-development verification-before-completion
do
  if ! find "$CODEX_HOME/plugins/cache" -type f -path "*/superpowers/*/skills/$required_skill/SKILL.md" -print -quit | grep -q .
  then
    printf 'Missing Superpowers skill: %s\n' "$required_skill"
    missing=1
  fi
done
test "$missing" -eq 0
```

After this diagnostic passes, open a fresh Codex session and confirm that
`superpowers:brainstorming`, `superpowers:test-driven-development`, and
`superpowers:verification-before-completion` are actively listed and loadable.
If any file is absent or any skill is not active, stop before starting the loop,
install or enable the Superpowers plugin in Codex, open another fresh session,
and confirm again. There is no assumed plugin-install CLI command.

## Install The Reproduction Skill

Run these commands from the repository root on each local or remote Codex host:

```bash
CODEX_HOME=${CODEX_HOME:-$HOME/.codex}
mkdir -p "$CODEX_HOME/skills"
ln -sfn "$PWD/skills/icml-repro-loop" "$CODEX_HOME/skills/icml-repro-loop"
test -f "$CODEX_HOME/skills/icml-repro-loop/SKILL.md"
```

After the first installation, open a fresh Codex session so it discovers the
skill. The versioned source remains in this repository; update it with Git and
retain the symlink. These checks diagnose the link target but do not replace
fresh-session activation confirmation:

```bash
test -L "$CODEX_HOME/skills/icml-repro-loop"
test "$(readlink "$CODEX_HOME/skills/icml-repro-loop")" = "$PWD/skills/icml-repro-loop"
```

## Verify Persistent Paper-Owner And Subordinate Runtime Preflights

Before a directly dispatched persistent paper-owner worker begins its current
fenced iteration, verify its controller runtime preflight: controller
credentials are present and authenticated, the live network is available for
the permitted Hub/challenge operations, and the state root is writable. The
paper-owner worker may use those credentials only for the exact lifecycle of
its current fenced attempt. Do not put credentials, tokens, cookies, or their
values in Git, evidence, logs, or a subprocess environment.

An optional subordinate implementation subprocess is not the dispatched
worker. Launch it only from a paper-owner-authored contract through
`skills/icml-repro-loop/scripts/worker_guard.py`. Before the first subordinate
launch for each runtime/worktree pair, run its `preflight_runtime` probe. The
probe must execute its inside-worktree control write while denying both the
synthetic outside-worktree write and synthetic credential-file read. If it
cannot prove credential stripping and scoped write/read isolation, issue only a
read-only research contract.

The constructed subordinate environment removes `HF_TOKEN`,
`HUGGING_FACE_HUB_TOKEN`, `GH_TOKEN`, Git credential helpers, and inherited
Hugging Face caches; it sets `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` and redirects
the Hub cache to an empty ignored directory inside the assigned worktree.
Antigravity subordinate launches require `--sandbox`; Codex subordinate launches
require `-s workspace-write -C <assigned-worktree>`. Never launch a worker
with `--dangerously-skip-permissions`,
`--dangerously-bypass-approvals-and-sandbox`, danger-full-access, or
`--add-dir`.

Verify the deterministic subordinate-boundary tests on every host:

```bash
uv run pytest tests/test_repro_loop_worker_guard.py -q
```

## Verify The Points Operating Loop

Use the state CLI for every worker launch and score/capacity observation:

```bash
uv run python skills/icml-repro-loop/scripts/state.py run-worker --help
uv run python skills/icml-repro-loop/scripts/state.py candidate-census --help
uv run python skills/icml-repro-loop/scripts/state.py score-report --help
```

`run-worker` records queue/launch/exit observations around the actual guarded
child. Worker process duration comes only from complete launch/exit monotonic
counters; queue duration comes from queued/launched observations. Git
timestamps and phase timestamps are not worker runtime. A launch from
`implementing` is implementation work and one from `improving` is correction
work. Controller validation and deployment remain separately measured stages.
Incomplete intervals report `null`.

`candidate-census`, `score-report`, `show-*`, and `list-attempts` are read-only.
`refresh-live` persists snapshots, while scheduling, worker launch, leases,
design/lifecycle transitions, attestations, verdict sync, publication, release,
and repair are persistent paper-owner controller mutations. Never run the
latter commands from a subordinate implementation subprocess. Submitted and
judging attempts remain dedicated to their current paper-owner worker; exact
scored or genuine blocked release precedes the next `claim-next`. Before a
genuine blocked release, use fenced `transition-attempt` to record nonempty
`blocker` and `next_action`; only then call `release-paper --outcome blocked`.

## Direct Worker Supervisor

The direct worker supervisor launches the direct Agy and Codex CLIs for 10 Agy
and 5 Codex persistent paper-owner lanes. It does not use OpenCode, does not
mutate coordinator state, and preserves healthy lanes while reconciling missing
or unhealthy ones.

From the repository root, first inspect the non-mutating reconciliation plan,
then install the user service and verify its status:

```bash
UV_CACHE_DIR=/tmp/icml-repro-uv-cache \
  uv run python ops/worker_supervisor.py reconcile --dry-run
UV_CACHE_DIR=/tmp/icml-repro-uv-cache \
  uv run python ops/worker_supervisor.py install
UV_CACHE_DIR=/tmp/icml-repro-uv-cache \
  uv run python ops/worker_supervisor.py status
systemctl --user status icml-worker-supervisor.timer
```

The destructive boundary is separate: stopping the supervisor and its managed
lanes requires an explicit confirmation.

```bash
UV_CACHE_DIR=/tmp/icml-repro-uv-cache \
  uv run python ops/worker_supervisor.py stop --confirm
```

## Verify The Workspace

```bash
CODEX_HOME=${CODEX_HOME:-$HOME/.codex}
uv sync --frozen
uv run pytest -q
uv run "$CODEX_HOME/skills/.system/skill-creator/scripts/quick_validate.py" skills/icml-repro-loop
uv run pre-commit run -a
state_schema="$(
  uv run python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["version"])' \
    state/repro-loop.json
)"
case "$state_schema" in
  3)
    uv run python skills/icml-repro-loop/scripts/state.py \
      migrate-v6 state/repro-loop.json --dry-run
    ;;
  6)
    uv run python skills/icml-repro-loop/scripts/state.py \
      list-attempts state/repro-loop.json
    ;;
  *)
    printf 'Unsupported reproduction state schema: %s\n' "$state_schema" >&2
    exit 1
    ;;
esac
git status --short
```

The host-verification sequence never applies a migration. A separately
authorized schema-v3 migration must use both `--apply` and the exact
`--expected-source-sha256` reported by the reviewed dry-run.

`git status --short` must produce no output after fresh tests. The ignored
environment, cache, coverage, OS, and `.superpowers` paths must not dirty Git.

The archived `submissions/nape/` tree is a provenance snapshot. Do not run its
environment, tests, or hooks in place. Clone the canonical NAPE repository into
a separate sibling directory and verify the pinned revision there:

```bash
cd ..
git clone https://github.com/will-rice/icml-2026-repro.git icml-2026-repro
cd icml-2026-repro
git checkout --detach 7220279222f1abac3056da78c7b8623a2a03e12b
git submodule update --init --recursive
uv sync --frozen
uv run pytest -q
uv run pre-commit run -a
```
