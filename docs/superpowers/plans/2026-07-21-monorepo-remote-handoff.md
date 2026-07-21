# Reproduction Monorepo Remote Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a parent repository containing the canonical NAPE snapshot and enough durable instructions for a fresh Codex session on a remote host to continue the AgentSelect reproduction.

**Architecture:** Keep submissions as independent projects under `submissions/`. Preserve the existing NAPE repository as the canonical source and copy its tracked revision into the parent. Store stable operating rules in `AGENTS.md`, machine setup in `docs/REMOTE_SETUP.md`, and mutable progress in `docs/HANDOFF.md`.

**Tech Stack:** Git, GitHub CLI, `uv`, Python, pytest, pre-commit, Hugging Face CLI, OpenResearch CLI

## Global Constraints

- Do not modify or replace the canonical `will-rice/icml-2026-repro` repository.
- Do not commit API keys, access tokens, cookies, credential files, or copied Git metadata.
- Every paper remains independently testable and deployable to its own Hugging Face Space.
- Do not add shared runtime code until two submissions require the same stable behavior.
- Use explicit setup and validation commands; do not add a bootstrap script.
- Run pre-commit before every implementation commit after `.pre-commit-config.yaml` exists.

---

### Task 1: Root Operating Documentation

**Files:**
- Create: `AGENTS.md`
- Create: `README.md`
- Create: `docs/REMOTE_SETUP.md`
- Create: `docs/HANDOFF.md`
- Create: `.pre-commit-config.yaml`

**Interfaces:**
- Consumes: Approved design at `docs/superpowers/specs/2026-07-21-reproduction-monorepo-design.md`
- Produces: The entry-point instructions a fresh Codex session reads before changing any submission

- [ ] **Step 1: Create the stable agent instructions**

Create `AGENTS.md` with these requirements:

```markdown
# ICML 2026 Reproduction Workspace

Read `docs/HANDOFF.md` before starting work and `docs/REMOTE_SETUP.md` before running commands on a new host.

## Objective

Build independently executable evidence for papers in the ICML 2026 Agent Repro Challenge. Recompute claims from released artifacts; never present paper-reported values as reproduced measurements.

## Layout

- `submissions/<paper>/`: independent project, tests, evidence bundle, and Space source for one paper.
- `docs/HANDOFF.md`: current mutable state and next action.
- `docs/REMOTE_SETUP.md`: host prerequisites, authentication checks, and verification commands.

## Workflow

1. Inspect the paper's live challenge status before claiming or publishing it.
2. Pin every upstream repository or dataset revision used as evidence.
3. Write a failing test before evidence-generation code.
4. Run the submission's pytest suite and `uv run pre-commit run -a`.
5. Record commands, revisions, environment, and outputs in a machine-readable evidence bundle.
6. Deploy each paper to a separate Hugging Face Space and verify the exact deployed commit.
7. Update `docs/HANDOFF.md` after every material milestone.

## Constraints

- Never commit credentials or unredacted environment dumps.
- Do not modify another submission to implement a new paper.
- Do not claim unsupported results. Mark unavailable evidence as unreplicated.
- Keep the canonical NAPE repository at `will-rice/icml-2026-repro` unchanged.
```

- [ ] **Step 2: Create remote setup instructions**

Create `docs/REMOTE_SETUP.md` with explicit commands for:

```bash
git clone https://github.com/will-rice/icml-2026-reproductions.git
cd icml-2026-reproductions
git submodule status
gh auth status
hf auth whoami
orx --help
cd submissions/nape
uv sync --frozen
uv run pytest -q
uv run pre-commit run -a
```

State that `git submodule status` must produce no entries, authentication must be performed interactively when a check fails, secrets must not be placed in repository files, and AgentSelect setup begins by reading `docs/HANDOFF.md`.

- [ ] **Step 3: Create the live handoff**

Create `docs/HANDOFF.md` recording these facts:

```markdown
# Current Handoff

## Current Objective

Build an AgentSelect reproduction for OpenReview paper `4M5Kj2UqaM` / arXiv `2603.03761`.

## Source Artifacts

- Paper: https://openreview.net/forum?id=4M5Kj2UqaM
- Official repository: https://github.com/Ancientshi/AgentSelect
- Full dataset link: https://drive.google.com/drive/folders/1wAzaUxOzPrwuF4s_iRT4NlRqV8gbLKe6?usp=sharing

## Verified Research State

- The challenge paper was unclaimed when checked on 2026-07-21.
- The repository contains Part I/II annotations, cleaned Part III artifacts, MuleRun transfer data, and deployment-run JSONL files.
- The repository has no detected license.
- The strongest targets are claims 1-3; claims 4-5 depend on whether released outputs permit independent metric recomputation.
- Do not use the inaccessible TerminalTraj repository or select WF-Bench without accounting for its missing raw fidelity series.

## Next Action

Verify that the full AgentSelect Google Drive dataset is anonymously downloadable on the remote host, record file names, sizes, and hashes, then write a paper-specific design and implementation plan before evidence code.
```

- [ ] **Step 4: Create root overview and pre-commit configuration**

Create `README.md` describing the parent repository, the canonical NAPE source, the separate-Space rule, and commands to enter and verify a submission. Create `.pre-commit-config.yaml` using `pre-commit-hooks` revision `v5.0.0` with `check-yaml`, `end-of-file-fixer`, `trailing-whitespace`, and `check-added-large-files` configured with `--maxkb=10240`.

- [ ] **Step 5: Validate documentation**

Run:

```bash
pre-commit run --all-files
rg -n 'TBD|TODO|FIXME|<your|YOUR_TOKEN|hf_' AGENTS.md README.md docs/REMOTE_SETUP.md docs/HANDOFF.md .pre-commit-config.yaml
```

Expected: all hooks pass; `rg` returns no matches.

- [ ] **Step 6: Commit**

```bash
git add AGENTS.md README.md docs/REMOTE_SETUP.md docs/HANDOFF.md .pre-commit-config.yaml
git commit -m "docs: make reproduction workspace remotely resumable"
```

### Task 2: Canonical NAPE Snapshot

**Files:**
- Create: `submissions/nape/` from tracked files in `/Users/will/icml-2026-repro`
- Create: `submissions/nape/UPSTREAM.md`

**Interfaces:**
- Consumes: Canonical NAPE Git revision `7220279` or the newer clean `main` revision observed at execution time
- Produces: An independently runnable NAPE snapshot without nested Git metadata

- [ ] **Step 1: Verify canonical source state**

Run:

```bash
git -C /Users/will/icml-2026-repro status --short
git -C /Users/will/icml-2026-repro rev-parse HEAD
git -C /Users/will/icml-2026-repro archive --format=tar --output=/tmp/nape-submission.tar HEAD
```

Expected: status is clean; the revision is recorded for `UPSTREAM.md`; the archive contains exactly the committed tree.

- [ ] **Step 2: Import tracked files**

Run:

```bash
mkdir -p submissions/nape
tar -xf /tmp/nape-submission.tar -C submissions/nape
test ! -e submissions/nape/.git
```

Expected: `submissions/nape/pyproject.toml`, `submissions/nape/tests/`, and `submissions/nape/repro_bundle/` exist; no nested `.git` exists.

- [ ] **Step 3: Record immutable provenance**

Create `submissions/nape/UPSTREAM.md` containing the canonical repository URL, imported full commit SHA, import date `2026-07-21`, and the exact `git archive` import method.

- [ ] **Step 4: Verify the imported project**

Run:

```bash
cd submissions/nape
uv sync --frozen
uv run pytest -q
uv run pre-commit run -a
```

Expected: the same test count and hook results as the canonical revision.

- [ ] **Step 5: Commit**

```bash
git add submissions/nape
git commit -m "feat: preserve canonical NAPE submission snapshot"
```

### Task 3: AgentSelect Workspace Boundary

**Files:**
- Create: `submissions/agentselect/README.md`
- Create: `submissions/agentselect/upstream/.gitkeep`
- Modify: `docs/HANDOFF.md`

**Interfaces:**
- Consumes: AgentSelect identifiers and source URLs from `docs/HANDOFF.md`
- Produces: A paper-specific project boundary without prematurely adding evidence code or copied upstream data

- [ ] **Step 1: Create submission overview**

Create `submissions/agentselect/README.md` with the paper title, OpenReview ID, arXiv ID, official repository and dataset links, and a clear `Status: artifact acquisition pending` marker. Include these five challenge claims verbatim:

1. AgentSelect reframes agent choice as narrative query-to-agent recommendation over deployable capability profiles represented by executable agent configurations.
2. AgentSelect contains 111,179 narrative queries, 107,721 deployable agents, and 251,103 positive query-agent interactions aggregated from more than 40 sources.
3. The benchmark is partitioned into LLM-only, toolkit-only, and compositional-agent parts with different interaction sparsity and long-tail reuse patterns.
4. Leaderboard results show content-aware semantic matching and tuned embedding recommenders outperform ID-centric methods in the sparse Parts II and III settings.
5. A recommender tuned on AgentSelect transfers to the unseen MuleRun agent marketplace and improves hit-rate and ranking metrics over untuned EasyRec.

State that upstream data will be pinned and either downloaded during reproduction or committed only when licensing and file size permit.

- [ ] **Step 2: Create the upstream artifact directory marker**

Create the empty marker `submissions/agentselect/upstream/.gitkeep`. Do not download or commit the upstream repository in this task.

- [ ] **Step 3: Update handoff path**

Add `submissions/agentselect/README.md` to the `Next Action` section of `docs/HANDOFF.md` as the paper-specific entry point.

- [ ] **Step 4: Validate boundaries**

Run:

```bash
test -f submissions/agentselect/README.md
test -f submissions/agentselect/upstream/.gitkeep
find submissions -name .git -print
pre-commit run --all-files
```

Expected: both files exist, `find` prints nothing, and all hooks pass.

- [ ] **Step 5: Commit**

```bash
git add submissions/agentselect docs/HANDOFF.md
git commit -m "docs: establish AgentSelect reproduction boundary"
```

### Task 4: Publish Parent Repository

**Files:**
- Modify: Git remote configuration only

**Interfaces:**
- Consumes: Completed, clean local parent repository
- Produces: Public `will-rice/icml-2026-reproductions` repository cloneable by the remote host

- [ ] **Step 1: Run final local checks**

Run:

```bash
pre-commit run --all-files
git status --short
git log --oneline --decorate -5
```

Expected: hooks pass, status is clean, and the design, plan, and three implementation commits are visible.

- [ ] **Step 2: Create and push the repository**

Run:

```bash
gh repo create will-rice/icml-2026-reproductions --public --source=. --remote=origin --push
```

Expected: GitHub reports the public repository URL and pushes `main`.

- [ ] **Step 3: Verify remote continuity**

Run:

```bash
gh repo view will-rice/icml-2026-reproductions --json url,visibility,defaultBranchRef
git ls-remote --exit-code origin refs/heads/main
```

Expected: visibility is `PUBLIC`, default branch is `main`, and `refs/heads/main` resolves to local `HEAD`.

- [ ] **Step 4: Record the remote handoff revision**

Update `docs/HANDOFF.md` with the published parent URL and the full baseline commit SHA from Step 1. Run `pre-commit run --all-files`, commit as `docs: record remote handoff revision`, and push `main`. The remote branch head will be the handoff commit; the recorded baseline identifies the complete implementation state immediately before that metadata-only commit.
