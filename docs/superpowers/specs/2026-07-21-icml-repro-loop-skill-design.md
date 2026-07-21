# ICML Reproduction Loop Skill Design

## Objective

Create a repository-owned Codex skill that repeatedly selects an unclaimed ICML 2026 Agent Repro Challenge paper, builds independently executable evidence, deploys and submits one Space, inspects the verdict, records what was learned, and continues to the next candidate.

## Location and Installation

The source of truth lives at `skills/icml-repro-loop/` in `will-rice/icml-2026-reproductions`. `docs/REMOTE_SETUP.md` installs it by symlinking that directory to `~/.codex/skills/icml-repro-loop`. This keeps the skill versioned and makes it discoverable by Codex on each host.

## Trigger

The skill triggers when the user asks to find, reproduce, submit, improve, or continuously process papers for the ICML 2026 Agent Repro Challenge. It does not trigger for general paper summaries or unrelated reproduction work.

## Loop

The skill executes one paper at a time:

1. Refresh the challenge paper catalog, claims, active claims, queued submissions, and verdict history.
2. Exclude papers that are claimed, queued, judging, or already judged unless the user explicitly requests an improvement attempt.
3. Inspect official paper, code, dataset, checkpoint, prediction, and result artifacts.
4. Rank candidates using the selection rubric.
5. Record the selected paper and immutable upstream revisions in `docs/HANDOFF.md`.
6. Design the paper-specific reproduction and obtain required user approval before writing implementation code.
7. Implement with tests first using the existing submission template.
8. Generate machine-readable evidence that distinguishes downloaded inputs, computed outputs, and paper-reported context.
9. Run pytest, pre-commit, evidence validation, and deployment checks.
10. Deploy a separate Hugging Face Space, submit it, and verify the exact deployed commit.
11. Poll for a verdict, record claim-level outcomes, and extract reusable selection lessons.
12. Continue to the next eligible paper.

The loop never starts a second implementation while the current paper has uncommitted work, a pending deployment mismatch, or an unresolved submission state.

## Candidate Selection

The selector prioritizes, in order:

1. Released raw data, predictions, checkpoints, or executable artifacts that directly support challenge claims.
2. Claims reproducible through deterministic statistics, metadata validation, existing predictions, or short CPU-only runs.
3. A high ratio of realistically verifiable claims to engineering time.
4. Public, revision-pinnable artifacts with a clear license.
5. Small downloads and environments that can be rebuilt without specialized hardware.

It penalizes missing artifacts, paper-table transcription, private or dead repositories, GPU training, unsafe workloads, ambiguous licenses, and claims that require subjective evaluation. A candidate must have at least two independently testable claims and a plausible path to stronger evidence than the prior submission.

## Compute and Cost Policy

Static analysis and ordinary CPU execution are preferred. GPU training is rejected by default. Codex-accessible paid APIs are allowed up to a cumulative estimated cost of USD 10 per paper. The skill records the estimate and observed usage when available.

The routine per-paper pause is design approval. The skill also pauses before exceeding USD 10, provisioning paid infrastructure, running vulnerable software outside an approved isolated environment, or performing destructive or ambiguous operations. Missing credentials also pause the loop.

## Submission Architecture

Every paper lives in `submissions/<paper-slug>/` as an independent project with its own `pyproject.toml`, lockfile, tests, evidence bundle, validation command, and Space source. The skill copies established template patterns but does not modify another submission to add a paper. Each paper receives a separate Hugging Face Space.

## Persistent State

`docs/HANDOFF.md` is the human-readable current state. `state/repro-loop.json` records candidate decisions, rejection reasons, selected upstream revisions, cost, repository and Space identifiers, deployed commit, submission status, verdict, and next action. State writes are atomic and validated against a fixed schema so a remote session can resume without conversational history.

Transient catalogs, downloaded datasets, credentials, and caches are not committed. Evidence inputs are committed only when licensing and size permit; otherwise the submission records source URLs, revisions, hashes, and acquisition commands.

## Verdict Handling

The skill does not treat successful submission as successful reproduction. It waits for the challenge verdict and records each claim as verified, partial, inconclusive, contradicted, or unavailable. One bounded improvement attempt is allowed when the verdict identifies a concrete evidence defect that can be fixed within the compute, cost, and safety policy. Otherwise the skill records the lesson and advances.

## Skill Contents

`SKILL.md` contains the concise control loop, gates, and required sub-skills. `references/selection-rubric.md` defines candidate scoring and rejection rules. `references/submission-checklist.md` defines deployment and verdict verification. `scripts/state.py` provides deterministic initialization, validation, and atomic updates for the loop state. `agents/openai.yaml` provides discoverable UI metadata.

## Validation

Skill development follows documentation TDD. A baseline agent receives a pressure scenario without the skill and is observed for common failures: selecting a high-compute paper, accepting self-reported values, skipping live claim checks, exceeding cost without approval, or stopping after deployment without inspecting the verdict. The implemented skill is then tested against the same and adversarial scenarios.

`quick_validate.py` validates skill structure and metadata. Unit tests exercise state initialization, valid transitions, duplicate-paper rejection, cost gating, and atomic writes. A dry-run scenario must produce a ranked candidate decision and stop before external mutation.

## Completion and Stop Conditions

The loop continues until no eligible candidates remain or it encounters a required approval, missing credential, external outage, unsafe execution requirement, or cost overrun. It must leave a clean repository and an explicit next action before stopping. It never fabricates a verdict or claims success while judging remains pending.
