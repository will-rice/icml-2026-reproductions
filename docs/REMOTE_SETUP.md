# Remote Setup

Clone the parent repository on a new host, then perform the authentication and
local verification checks below. Do not place credentials, tokens, cookies, or
other secrets in repository files.

## Install Tools, Clone, And Authenticate

Install `uv` using its official installer, then confirm it is available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

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

## Install The Reproduction Skill

Run these commands from the repository root on each local or remote Codex host:

```bash
CODEX_HOME=${CODEX_HOME:-$HOME/.codex}
mkdir -p "$CODEX_HOME/skills"
ln -sfn "$PWD/skills/icml-repro-loop" "$CODEX_HOME/skills/icml-repro-loop"
test -f "$CODEX_HOME/skills/icml-repro-loop/SKILL.md"
```

After the first installation, restart Codex or open a new Codex session so it
discovers the skill. The versioned source remains in this repository; update it
with Git and retain the symlink. A direct link check is sufficient when a new
session is not available:

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
```

When the canonical NAPE snapshot is present, verify that independent
submission with:

```bash
cd submissions/nape
uv sync --frozen
uv run pytest -q
uv run pre-commit run -a
```
