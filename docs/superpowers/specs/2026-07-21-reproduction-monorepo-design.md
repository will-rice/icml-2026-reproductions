# ICML 2026 Reproduction Monorepo Design

## Objective

Create one parent repository for ICML 2026 Agent Repro Challenge submissions while keeping every paper independently reproducible and deployable. The repository must also contain enough durable context for a fresh Codex session on a remote host to continue without access to the original conversation.

## Repository Layout

```text
icml-2026-reproductions/
├── AGENTS.md
├── README.md
├── docs/
│   ├── HANDOFF.md
│   ├── REMOTE_SETUP.md
│   └── superpowers/
│       ├── plans/
│       └── specs/
└── submissions/
    ├── nape/
    └── agentselect/
```

Each directory under `submissions/` is a complete Python project with its own dependency lock, tests, evidence bundle, validation commands, and Hugging Face Space deployment. The parent does not introduce shared runtime code until at least two submissions require the same stable behavior.

## Existing NAPE Submission

The existing `will-rice/icml-2026-repro` repository remains canonical and unchanged so its OpenResearch configuration, branches, and provenance links continue to work. Its tracked contents are copied into `submissions/nape/` without its `.git` directory. The parent copy is a convenient snapshot, not a replacement for the canonical repository.

## AgentSelect Submission

`submissions/agentselect/` follows the proven NAPE package shape but contains paper-specific evidence code. It will reproduce claims from released AgentSelect artifacts rather than transcribe values from the paper.

Initial evidence priorities are:

1. Validate the executable capability-profile representation and query-to-agent schema.
2. Recompute query, agent, interaction, source, and tool counts from released data.
3. Recompute the three benchmark partitions, interaction density, sparsity, and reuse distributions.
4. Re-evaluate released baseline outputs when predictions or checkpoints make this possible.
5. Recompute MuleRun transfer metrics from released rankings or predictions when available.

Claims without independent executable evidence are marked unreplicated. The submission must distinguish author-provided inputs from values computed by our code.

## Remote Continuation

The root `AGENTS.md` records stable operating rules: repository layout, challenge objective, testing and publication workflow, artifact provenance requirements, and the rule that each paper receives a separate Space.

`docs/REMOTE_SETUP.md` contains explicit commands for cloning, installing `uv`, restoring each project environment, checking `gh`, Hugging Face, and OpenResearch authentication, and running verification. It never contains tokens or credentials.

`docs/HANDOFF.md` records mutable state: current candidate, paper and repository identifiers, completed research, known limitations, current submission status, and the exact next action. It is updated before moving hosts and after material milestones.

A fresh session starts by reading `AGENTS.md` and `docs/HANDOFF.md`, then follows `docs/REMOTE_SETUP.md`. No bootstrap script is included because explicit commands are easier to audit and troubleshoot.

## Validation

Each submission defines its own `pytest` and `pre-commit` commands. Parent-level documentation lists commands to validate one submission or iterate over all submissions, but does not hide those commands behind an orchestration script.

AgentSelect evidence tests use small committed fixtures first. Full-data tests are deterministic integration tests that verify hashes or source revisions and write machine-readable evidence artifacts. Network access is not required after documented source artifacts have been acquired.

## Publication

The parent repository is published as `will-rice/icml-2026-reproductions`. Every challenge entry is deployed to a separate Hugging Face Space because challenge metadata and judging are Space-specific. Published evidence records the parent commit, submission path, upstream artifact revision, commands, environment, and outputs.

## Error Handling

Setup commands fail with concise guidance when a required CLI, authentication context, dataset, or source revision is missing. Evidence generation fails rather than silently substituting paper-reported values or partial data. Unsupported dataset schemas raise an error naming the file and missing field.
