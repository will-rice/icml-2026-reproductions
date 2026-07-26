# Leaderboard-Points Operating Design

**Date:** 2026-07-26

**Status:** User-approved operating direction; implementation pending

## Objective

Maximize `wrice`'s total judged leaderboard points before the ICML 2026 Agent
Repro Challenge deadline: Sunday, August 2, 2026 at 11:59 PM Anywhere on Earth.

The official score for one claim is:

- `2` points for `verified`;
- `2` points for `falsified`;
- `1` point for `toy`;
- `0` points for `inconclusive` or an unjudged claim.

Only the first judged logbook for one Hugging Face username and paper is
canonical. The operating system must therefore optimize expected claim points
per remaining end-to-end hour without locking a paper to avoidable
zero-point evidence.

## Current Baseline

At the design snapshot:

- `wrice` has 8 judged points from 2 papers and is approximately rank 157;
- 18 paper projects contain implementation code in the current workspace or
  its registered worktrees;
- 5 papers have schema-v6 controller attempts;
- PostTrainBench and Recurrent Samplers are judging;
- Numina and EEG are blocked after bounded judging deadlines without official
  verdicts;
- Graph Pruning is blocked by a conflicting official verdict for another
  Space;
- the current assessed candidate file covers only 6 papers and is too narrow
  to sustain a score-maximizing queue.

These values are observations, not durable score authority. Official verdict
imports continue to require a fresh immutable live snapshot.

## Scope

This design changes the reproduction loop's operating priorities, scheduling,
worker execution telemetry, and queue reporting.

It does not:

- weaken evidence standards or call paper-reported values reproduced;
- permit duplicate submissions for judged or already claimed papers;
- give workers deployment credentials or coordinator-state authority;
- add a general security platform;
- resume already judged papers;
- autonomously run GPU jobs;
- exceed USD 10 in estimated paid API cost for one paper.

## Strategy Choice

Three strategies were considered:

1. **Raw volume.** Submit the largest number of papers as quickly as possible.
   This maximizes paper count but risks canonical zero-point verdicts.
2. **Depth first.** Spend most capacity on a small number of exhaustive
   reproductions. This improves per-paper quality but cannot close the current
   leaderboard gap quickly enough.
3. **Expected points per hour.** Rank work by expected judged claim points
   divided by conservative remaining end-to-end time. This is the selected
   strategy.

The selected strategy may choose either reproduction or falsification. Both
earn full points when supported by full-scale evidence.

## Score-Rate Model

### Claim expectation

For each challenge claim `c`, an assessor records conservative probabilities:

```text
E[c] = 2 * P(verified) + 2 * P(falsified) + 1 * P(toy)
```

`P(inconclusive)` and failure consume the remaining probability and earn zero.
Probabilities are estimates used only for scheduling. They never become
official verdicts or reproduced evidence.

### Paper expectation

```text
expected_points = sum(E[c] for every challenge claim)
```

The estimate covers every live challenge claim, not only the minimum two claims
required for attempt admission. Claims without an executable evidence path
receive zero expected points.

### Remaining time

```text
remaining_hours_p90 =
    worker_queue
  + implementation
  + evidence_compute
  + controller_validation
  + correction
  + deployment
```

Judging latency is represented separately as `P(judged_before_deadline)`.
Pending judging never consumes an implementation lane.

### Priority

```text
priority =
    expected_points
  * P(judged_before_deadline)
  / max(remaining_hours_p90, 0.25)
```

Tie-breaks, in order:

1. more existing reusable implementation and evidence;
2. more direct immutable artifacts;
3. more full-score claim paths;
4. lower variance in remaining time;
5. lower estimated paid cost.

Sunk time does not raise priority. Existing work matters only when it reduces
remaining time.

## Eligibility Floor

The existing reproduction-loop eligibility rules remain mandatory. A paper is
not runnable when it:

- already has a history attempt, candidate lease, queued submission, tagged
  Space, or official verdict;
- lacks two independently testable claims;
- requires autonomous GPU training;
- exceeds the per-paper API cost cap;
- lacks immutable upstream provenance;
- has an unresolved license or execution blocker.

The priority model orders eligible papers; it does not convert ineligible work
into eligible work.

## Operating Architecture

### 1. Live candidate census

A continuous read-only census maintains a broad pool rather than a six-paper
assessment bottleneck. Candidate scouts inspect:

- the current challenge revision and anchored claim inventory;
- current tagged and queued Spaces;
- current official verdicts;
- released repositories, datasets, checkpoints, and licenses;
- existing local worktrees and implementation commits.

The controller writes assessed snapshots and admissions only after reviewing
the scouts' primary-source findings.

### 2. Harvest queue

Before starting expensive new work, the scheduler evaluates every existing
implementation worktree against live eligibility and remaining work. A
worktree is not considered complete merely because code exists. It receives a
score-rate estimate based on the work still required to produce credible
judged evidence.

Eligible high-rate worktrees enter the same queue as fresh candidates. They do
not receive a separate or automatic priority class.

### 3. Worker pool

The target is 20 runnable paper attempts, consistent with the reproduction-loop
capacity. Every available implementation slot remains occupied while eligible
work exists.

Each paper worker receives:

- one paper, one absolute worktree, and one `submissions/<paper>/` project;
- full read/write access inside that worktree;
- internet access for public paper artifacts and dependencies;
- the complete live claim inventory and immutable source pins;
- an explicit evidence goal ordered by expected points;
- no Hub, GitHub, or coordinator credentials.

Workers implement and test. They do not wait for another paper's review,
deployment, or judgment.

### 4. Thin controller lane

The controller performs only actions that need shared authority:

- live assessed snapshots and admission;
- design recording and one independent approval;
- validation;
- deployment and submission;
- bounded official-verdict polling and import;
- integration of accepted paper changes.

The controller does not perform general hardening, repeat broad review rounds,
or make presentation-only changes unless a concrete validation or scoring
failure requires them.

Root tests and pre-commit run once in authoritative validation. Workers run
paper-scoped tests and evidence generation. A correction cycle is used only
for a reproduced failure with a failing regression test.

### 5. Nonblocking judging

Submission moves a paper out of implementation capacity immediately. Judgment
watching is a lightweight controller activity with finite polling. Pending or
deadline-blocked judgment cannot prevent admission of another eligible paper.

An official verdict is imported only from a fresh immutable assessed snapshot
that matches paper, Space, and deployed SHA.

## Submission Gate

Because the first judged logbook becomes canonical, deployment alone does not
authorize submission.

A paper is submitted when:

- every included nonzero-point claim has executable evidence and provenance;
- at least two claims have a credible non-`inconclusive` path;
- evidence distinguishes full-scale, reduced-scale, and unavailable work;
- the Space exposes the machine-readable evidence and exact commands;
- authoritative validation and exact live-SHA checks pass;
- a live snapshot shows no duplicate or conflicting canonical record.

The controller may hold a deployable paper briefly when a bounded additional
experiment has higher expected points than the risk of missing judgment before
the deadline. That decision uses the same score-rate model and an explicit
deadline probability.

## Telemetry Contract

Telemetry exists to improve score-rate estimates, not as a separate platform.

The controller-owned launcher wraps the actual worker process and records:

- immutable telemetry session ID;
- attempt ID, paper ID, runtime, model, worktree, and contract digest;
- queued, launched, and exited UTC timestamps;
- monotonic launch and exit counters when available on the same host;
- exit code or terminating signal;
- paper commit before and after the session;
- bytes downloaded when the runtime exposes them;
- explicit outcome: proposal, failed, interrupted, or timed out.

Controller stages record:

- validation start/end and outcome;
- correction start/end and outcome;
- deployment start/end and outcome;
- submission observation time;
- official verdict observation time.

Derived metrics are:

- worker queue time;
- worker process elapsed time;
- implementation sessions per paper;
- evidence-compute time when separately instrumented;
- controller validation time;
- time from first launch to submission;
- first-pass validation rate;
- judged points per worker-process hour;
- judged points per end-to-end hour.

Git timestamps and state-phase transitions are never reported as worker
runtime.

Telemetry files are append-only controller records. Workers cannot edit them.
A missing exit event is reported as an open or interrupted session, never
converted into a guessed duration.

## Queue Reporting

The controller exposes a compact score dashboard with:

- current official points and rank observation;
- points pending judgment;
- runnable workers and idle capacity;
- the top candidate queue with expected points, P90 remaining hours, priority,
  and primary risk;
- papers awaiting validation or deployment;
- observed stage durations and first-pass validation rate;
- explicit blockers requiring user or challenge-operator action.

Internal candidate scores are always labeled estimates. Official points are
reported only from the verdict feed.

## Failure Handling

- **Worker exits nonzero:** retain telemetry and worktree; reassess remaining
  score rate before retrying.
- **Worker stops making progress:** interrupt the session, record it as
  interrupted, and reassess. Do not fabricate an exit duration.
- **Validation fails:** write one failing regression test and permit one
  focused correction cycle. Repeated failure blocks the attempt and refills
  capacity.
- **Artifact or license drift:** invalidate the candidate assessment and
  return to census.
- **Duplicate or verdict appears:** stop submission for that paper and refill
  the lane.
- **Judgment is delayed:** block only the judging phase; continue other papers.
- **No eligible candidates:** broaden the live census; do not fill capacity
  with known zero-point work.

Attempts are never autonomously archived. Abandonment still requires explicit
user authorization.

## Testing

Implementation must use test-driven development and include:

1. unit tests for official claim-point mapping;
2. ranking tests for expected points, deadline probability, remaining time,
   and deterministic tie-breaks;
3. tests proving ineligible candidates cannot be promoted by a high priority;
4. launcher tests proving start and exit events wrap the real child process;
5. interruption and missing-exit tests that never invent duration;
6. scheduler tests proving judging and blocked attempts release runnable
   capacity;
7. controller tests proving official and estimated scores remain distinct;
8. end-to-end fixture tests from census through ranked admission and telemetry
   reporting.

## Rollout

1. Add score-rate assessment fields and deterministic ranking without changing
   existing lifecycle authority.
2. Add controller-owned worker-process telemetry.
3. Add nonblocking capacity and queue reporting.
4. Run a fresh live census and assess the existing implementation inventory.
5. Admit the highest-rate eligible papers until runnable capacity is full.
6. Validate and submit ready work continuously.
7. Re-rank after every worker exit, validation result, verdict, or material
   deadline change.

## Success Criteria

The operating change succeeds when:

- every reported worker duration comes from actual launch/exit telemetry;
- no implementation slot is idle while an eligible candidate is available;
- pending judging never blocks admission;
- candidate ordering is reproducible from recorded inputs;
- every submitted paper passed the submission gate;
- official score and estimates are never conflated;
- measured judged points per end-to-end hour increases over the current
  baseline.

## Existing Attempt Continuation

This design does not mutate existing attempts:

- PostTrainBench attempt `cb04ab1a-a526-4137-862b-a26d68563737` remains
  `judging`;
- Recurrent Samplers attempt `534db42c-5b16-4f00-9a7d-a47056fc9dd4`
  remains `judging`;
- Numina attempt `49b12585-ca39-441d-b822-c7da77ed81e9` remains blocked from
  judging;
- EEG attempt `e20658d7-250a-5b0c-a015-be453c43e9fc` remains blocked from
  judging;
- Graph Pruning attempt `64bfe193-333b-4b37-9683-9ac25ca5ac27` remains
  blocked from deployed.

Their current owners, fences, snapshots, and attestations remain unchanged.
The next action for pending judgments is a fresh live observation followed by
official import only when an exact verdict exists. Graph Pruning remains
blocked until the challenge operator resolves its conflicting canonical
record.
