# Current Handoff

## Loop State

- `state/repro-loop.json` is authoritative and currently records
  `implementing` for `vGeNaFHdET` (EEG-FM-Bench, slug `eeg-fm-bench`) with
  `design_approved: true` under the user's 2026-07-24 standing directive that
  the loop run completely autonomously (design gates are approved on
  presentation). History holds one archived abandoned attempt (`HMu24dTKkJ` —
  see below); total API cost USD 0.00.
- Autonomous mode: do not pause for design or deployment approval. Still
  never autonomously abandon a blocked attempt; persist blocker + HANDOFF and
  return control instead.
- The state file uses schema version 3. Selection requires explicit
  `estimated_api_cost_usd`, immutable `upstream_revision`, and at least two
  unique `target_claims`. Each judging entry starts a new poll round; verdict
  history is authoritative and the final `verdict` mirrors its last record.
- The source skill is `skills/icml-repro-loop/`; install it according to
  `docs/REMOTE_SETUP.md` before using a new host.

## 2026-07-24 Live Refresh: Paper Already Scored

- The official `ICML-2026-agent-repro/verdicts` dataset (revision
  `3d3350a34469a6ee1b25a2d749f578578bf606d9`, fetched 2026-07-24) already
  contains a verdict for `HMu24dTKkJ`, judged `2026-07-22T16:03:19+00:00`
  against Space
  `wrice/repro-dimension-free-convergence-of-diffusion-models-for-approximate-gaussian-mixtures`
  at exact SHA `5af083f86c4ab0e98ee65a01e3995669f288849b`. Claim verdicts:
  three `toy` (Theorem 1 TV bound, imperfect score estimation, trace lemma)
  and two `inconclusive` (Assumption 1 closeness, prior-work contrast).
- The same dataset holds the NAPE verdict (`NvPgRwURDC`, judged
  2026-07-21T21:52:09+00:00) at Space SHA
  `6ce52a53872fbbbc73da1efe313e224a9c9c853c`.
- Both papers were therefore already scored by the official judge via the
  parallel reproduction lineage; continuing the current attempt would create
  an ineligible duplicate submission, so the loop is blocked awaiting a user
  decision.
- Before the discovery, two CLI agents completed plan Tasks 1-3 (GMM
  primitives, deterministic DDPM audits, evidence CLI) on branch
  `impl/diff-gmm-integration` (unmerged; Tasks 4-5 not started).

## Published Parent Repository

- Public repository: https://github.com/will-rice/icml-2026-reproductions
- Verified baseline revision: `02c8379d22859de0966127dfb0793b7dbc2cb7b8`.
- Clone-verified baseline at check time:
  `e1a0c4a24b92805e383d6cdbe7181db7ca6c62b3`.
- A fresh shallow clone of that baseline succeeded and contained `SKILL.md`,
  `REMOTE_SETUP.md`, and `HANDOFF.md`. Subsequent documentation commits may
  advance `main`.

Verify the current remote and local heads with:

```bash
git ls-remote origin refs/heads/main
git rev-parse HEAD
```

This document cannot contain its own commit SHA because its contents
participate in that commit's hash.

## Candidate Record

- AgentSelect, OpenReview `4M5Kj2UqaM` / arXiv `2603.03761`, is already judged
  and recorded as rejected in `state/repro-loop.json`. It must not be selected
  or resumed. Its historical artifacts were the
  [official repository](https://github.com/Ancientshi/AgentSelect) and the
  [full dataset](https://drive.google.com/drive/folders/1wAzaUxOzPrwuF4s_iRT4NlRqV8gbLKe6?usp=sharing).
- A live challenge refresh on 2026-07-22 found 2,610 reproduction Spaces.
  AdamW-style Shampoo (`gvWsViQBYB`) had two active reproductions and LoRA
  Gradient Descent (`9GRlBVAXq8`) had three; both were persisted as rejected.
- WF-Bench (`8Fhq7QpYfI`) was unclaimed but was persisted as rejected because
  its substantive fidelity/scaling reproduction requires GPU training, which
  is outside the autonomous reproduction-loop boundary.
- Submission 2493, Dimension-Free Convergence of Diffusion Models for
  Approximate Gaussian Mixtures (`HMu24dTKkJ`), was selected at immutable
  revision `arxiv:2504.05300v1` with USD 0.00 estimated API cost.

## Next Action

`state/repro-loop.json` is `implementing`. Execute
`docs/superpowers/plans/2026-07-24-eeg-fm-bench.md` autonomously with
test-driven development under `submissions/eeg-fm-bench/`: Tasks 1-2 in
parallel via codex/agy CLI agents, then Task 3 (evidence CLI), Task 4
(logbook/poster), and Task 5 (live refresh, dedicated Space, exact-SHA
verification, submission, bounded judging).

Selection comparison on 2026-07-24 (all three live and unclaimed, no verdict,
no tagged Space):

| Candidate | Base | Penalties | Final | CPU estimate | Main risk |
| --- | ---: | ---: | ---: | --- | --- |
| EEG-FM-Bench (`vGeNaFHdET`) | 18 | 0 | 18 | under 2 hours | registration-gated raw datasets; GPU performance claims out of scope |
| Graph dataset pruning (`a3GdvuPItd`) | 14 | -2 | 12 | under 30 min | no released code; synthetic theorem audits risk `toy` verdicts |
| NorMuon (`m1IRWFAMsa`) | 13 | -2 | 11 | under 2 hours | no released code; headline claims are 1.1B/5.4B GPU training |

Rejected and persisted this round: `mWxEAgz3xu` (aggregate-only data),
`MqzZ9X6m7f` and `ycj3XWCh6E` (position papers, no testable claims),
`GnqHK8Ww98` and `71030` (GPU training required).

The old `impl/*` branches/worktrees for the abandoned diffusion attempt
remain unmerged and must never be deployed or submitted.

The selected target claims are:

1. `fourteen-dataset-ten-paradigm-curation`,
2. `standardized-preprocessing-reproducibility`, and
3. `three-strategy-evaluation-harness`.

Paper-reported Figure 1 lists are context only. Computed evidence comes from
the pinned repository census, deterministic preprocessing checks, and CPU
smoke runs of the three harness strategies. GPU-only leaderboard and
representation-analysis claims remain explicitly unavailable.

Use the judging, improvement, and completion JSON examples in
`skills/icml-repro-loop/references/submission-checklist.md`; improvement and
completion verdicts must cover exactly the selected target claims.

Blocked transitions require a nonempty `blocker` and record `blocked_from`.
Resume by transitioning back to that phase. Do not archive a blocked attempt
unless the user explicitly directs `{"abandon": true}`; abandonment records the
attempt in history and accounts its actual API cost.

## Validation Commands

```bash
CODEX_HOME=${CODEX_HOME:-$HOME/.codex}
uv sync --frozen
uv run pytest -q
uv run "$CODEX_HOME/skills/.system/skill-creator/scripts/quick_validate.py" skills/icml-repro-loop
uv run pre-commit run -a
```
