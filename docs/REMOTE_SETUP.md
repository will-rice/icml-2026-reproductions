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
either authentication check fails. AgentSelect setup used to begin by reading
`docs/HANDOFF.md`; it is now judged and must not be selected, so always use
that file as the authoritative current-paper entry point.

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

## Verify The Workspace

```bash
CODEX_HOME=${CODEX_HOME:-$HOME/.codex}
uv sync --frozen
uv run pytest -q
uv run "$CODEX_HOME/skills/.system/skill-creator/scripts/quick_validate.py" skills/icml-repro-loop
uv run pre-commit run -a
git status --short
```

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
