# Candidate Selection Rubric

Run raw `refresh-live` and inspect it with `show-snapshot`, then inspect source artifacts for its current papers and extracted claims. Write an explicit assessment file against that raw snapshot revision. The 6,341-record `index.json` catalog and current challenge records are metadata only. Never infer score, feasibility, cost, target claims, or upstream pins from them.

## Eligibility

A candidate is eligible only when all of these are true:

- It has no active/history attempt, candidate lease, queued submission, tagged Space, or verdict in the snapshot and durable store. Judged and historical papers remain excluded.
- At least two distinct paper claims are independently testable. A claim is independently testable only when released artifacts or a feasible computation can verify it without treating the paper's reported value as evidence.
- It does not require GPU training. Explicitly requested GPU projects are outside this skill.
- The estimated cumulative paid-API cost for the paper is at most USD 10. More than USD 10 is ineligible, not merely a score penalty.
- The selection record includes an explicit finite `estimated_api_cost_usd`, a nonempty immutable `upstream_revision`, and at least two unique nonempty immutable `target_claims`; omitted cost is not treated as zero.
- Its execution path is not known to be unsafe. Unresolved safety ambiguity pauses selection; a workload proven safe inside an approved isolation boundary may remain eligible.
- Its paper identity, candidate status, artifact availability, and expected execution path have been checked from live or primary sources.

## Census And Assessment Input

Run `candidate-census` on the raw snapshot before research. It lists unclaimed
papers with at least two extracted claims and any revision-pinned project found
at the candidate's validated `submissions/<slug>` path in a registered
worktree. A census row has research authority only: it never supplies
feasibility, score, cost, targets, or an upstream pin.

The assessment file has top-level `challenge_revision`, `assessor`,
`assessed_at`, and `assessments`. New assessments contain exactly `paper_id`,
`score`, `target_claims`, `claim_bindings`, `upstream_revision`,
`artifact_access`, `cpu_only`, `safety_blocker`, `licensing_blocker`,
`estimated_api_cost_usd`, and `score_rate`. Historical records without
`score_rate` remain readable but cannot be newly admitted. `claim_bindings` has
one object per target claim, in target order, with exactly `target_claim`,
`challenge_claim`, and `challenge_claim_sha256`. The target must match its
corresponding `target_claims` entry; `challenge_claim` must be text from the
pinned current live extracted claims; and its SHA-256 must be the UTF-8 digest
of that exact text. Unmatched and unassessed papers remain in snapshot
provenance but are ineligible. A stale document revision aborts assessed
refresh; rerun discovery and assessment.

`score_rate` contains exactly:

```json
{
  "claim_expectations": [
    {
      "challenge_claim_sha256": "64 lowercase hex characters",
      "p_verified": 0.5,
      "p_falsified": 0.1,
      "p_toy": 0.2
    }
  ],
  "judged_before_deadline_probability": 0.8,
  "remaining_hours_p90": 2.0,
  "reusable_implementation": false,
  "direct_artifact_score": 4,
  "full_score_claim_paths": 2,
  "remaining_time_variance_hours2": 0.25,
  "primary_risk": "Artifact schema may have drifted."
}
```

There is one expectation per live claim, in live order, bound by exact claim
SHA-256. Each probability is in `[0,1]` and the three probabilities for one
claim sum to at most one. P90 hours are finite and positive. Estimate all
remaining implementation, validation, deployment, submission, and correction
work—not only the evidence command.

## Base Score

Score each dimension from 0 through 5, then sum them for a maximum base score of 25.

### Direct artifacts

| Score | Evidence available |
| --- | --- |
| 0 | No relevant artifact. |
| 1 | Paper prose, screenshots, or tables only. |
| 2 | Partial code or processed examples, but no direct claim output. |
| 3 | Public data, predictions, checkpoints, or executable code directly supports at least one claim. |
| 4 | Versioned artifacts directly support at least two claims with little reconstruction. |
| 5 | Complete, revision-pinnable raw outputs and executable evaluation artifacts support the target claims. |

### Independently testable claim count

| Score | Distinct claims with an independent test |
| --- | --- |
| 0 | None. |
| 1 | One. |
| 2 | Two. |
| 3 | Three. |
| 4 | Four. |
| 5 | Five or more. |

Count claims, not metrics or repeated dataset slices. Each counted claim needs a separate expected observation and a reproducible test.

### CPU feasibility

| Score | Expected local path |
| --- | --- |
| 0 | No credible CPU path. |
| 1 | More than 24 hours, excessive memory, or major feasibility uncertainty. |
| 2 | 8-24 CPU hours or a large, fragile environment. |
| 3 | 2-8 CPU hours with manageable downloads and dependencies. |
| 4 | At most 2 CPU hours with ordinary workstation resources. |
| 5 | Static validation or deterministic evaluation in at most 30 minutes. |

Estimate the complete evidence run, not only a smoke test.

### Provenance

| Score | Traceability |
| --- | --- |
| 0 | Origin cannot be established. |
| 1 | Unattributed copy or indirect mirror only. |
| 2 | Upstream URL exists, but version or lineage is unclear. |
| 3 | Official repository, release, or dataset is identified. |
| 4 | Exact revisions and source URLs can be pinned for core artifacts. |
| 5 | Exact revisions, hashes, acquisition commands, and lineage can be recorded for every input. |

### Licensing

| Score | Permission clarity |
| --- | --- |
| 0 | Known incompatible terms prohibit the intended use. |
| 1 | No license or materially unclear terms. |
| 2 | Some core artifacts are licensed, but important gaps remain. |
| 3 | Core artifacts have explicit compatible terms; only unused or peripheral items are unclear. |
| 4 | Every required artifact has explicit compatible use terms. |
| 5 | Use and redistribution or deployment terms are explicit, compatible, and recorded for every required artifact. |

## Penalties

Apply every relevant penalty after the base score; penalties are cumulative:

| Condition | Penalty |
| --- | ---: |
| Required artifacts are dead or private | -10 |
| Available evidence is self-report only | -5 |
| A required artifact's license is unclear | -2 |

Do not hide an eligibility failure with a high score. Record each penalty and the source observation that caused it.

## Expected Points And Selection Decision

Official claim points are exactly `verified=2`, `falsified=2`, `toy=1`, and
`inconclusive=0`. Assessments are estimates, never official verdicts:

```text
expected_points = Σ(2*p_verified + 2*p_falsified + p_toy)
priority = expected_points * judged_before_deadline_probability
           / max(remaining_hours_p90, 0.25)
```

Rank eligible candidates by descending priority. Resolve exact ties by reusable
implementation first, then higher direct-artifact score, more full-score claim
paths, lower remaining-time variance, lower paid API cost, and paper ID. The
legacy rubric score remains useful assessment context but is not the scheduler
ranking key.

Before selection, compare the top three eligible candidates side by side. If
fewer than three are eligible, document the exhausted pool and compare every
eligible candidate.

| Candidate | Expected points | Deadline probability | P90 hours | Points/hour priority | API estimate | Main risk |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A |  |  |  |  |  |  |
| B |  |  |  |  |  |  |
| C |  |  |  |  |  |  |

Persist every ineligible candidate and reason in the coordinator before
continuing. One bounded pass admits only enough candidates to restore 20
runnable paper attempts. Submitted, judging, blocked, and complete attempts do
not consume runnable implementation capacity and never stop unrelated work.

```bash
raw_id=$(uv run python skills/icml-repro-loop/scripts/state.py refresh-live state/repro-loop.json | uv run python -c 'import json,sys; print(json.load(sys.stdin)["snapshot_id"])')
uv run python skills/icml-repro-loop/scripts/state.py show-snapshot state/repro-loop.json --snapshot-id "$raw_id"
uv run python skills/icml-repro-loop/scripts/state.py candidate-census state/repro-loop.json --snapshot-id "$raw_id" --workspace-root /home/will/projects/icml-2026-reproductions
uv run python skills/icml-repro-loop/scripts/state.py refresh-live state/repro-loop.json --assessments-json state/candidate-assessments.json
uv run python skills/icml-repro-loop/scripts/state.py scheduler-pass state/repro-loop.json --snapshot-id SNAPSHOT_ID
```

Each admitted paper must include `paper_id`, `title`, `slug`,
`estimated_api_cost_usd`, `upstream_revision`, `target_claims`, and the exact
`claim_bindings` recorded against its current live claims. A candidate rejection
or exhausted pool affects admission only; existing attempts continue.
