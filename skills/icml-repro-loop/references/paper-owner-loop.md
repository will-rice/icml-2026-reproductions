# Persistent Paper-Owner Event Loop

One persistent paper-owner worker owns one fenced attempt per iteration. After every mutation, reread its attempt shard and live lease. Never infer owner, fence, phase, Space SHA, or verdict. A submitted or judging attempt remains the worker's current paper; the worker may not claim another paper until the iteration is released.

Endgame saturation: while publish-ready lanes (`validated` plus
`blocked_from: validated`) number at least the daily Space-creation quota
(20), do not select new papers — publish the backlog as quota allows and
run improvement cycles on judged zero-score attempts instead. Improvement
evidence must be computed, deterministic, and rendered into the served
`pages/*.md`; honest `unreplicated` beats asserted success.

Before selecting new work, inspect every active released blocked attempt
against the fresh assessed immutable snapshot. If its recorded blocker is
resolved or its `next_action` is actionable, explicitly reclaim the
highest-priority eligible attempt with `claim-next --reclaim-attempt-id` and a
fresh fencing token. Otherwise leave unresolved blockers reclaimable and
select new work. This routing is mandatory: ordinary `claim-next` must not
auto-reclaim unresolved blocked attempts.

| Event | Required reaction | Terminal behavior |
| --- | --- | --- |
| `worker-exited` | `validate-or-correct`: inspect proposal commit/diff and run fresh controller validation | continue same attempt |
| `worker-noop` | `repair-permissions-and-relaunch`: fix scoped runtime permission or change runtime/model | continue same attempt |
| `validation-rejected` | `correct-and-relaunch`: give exact scientific/integrity findings to the same attempt | continue same attempt |
| `validated` | `commit-then-publish`: commit the exact attested source to the shared repository on `main` (`submissions/<slug>/` with its `.gitignore` negation block), then publish only that attested source | continue same attempt |
| `deployment-invalid` | `repair-publication`: correct SHA/tags/runtime before attestation | continue same attempt |
| `submitted` | `remain-dedicated` | watch; do not select another paper |
| `pending` | `keep-watching`: verify exact healthy visibility; do not alter evidence solely for queue age | continue same attempt |
| `inconclusive` | `improve-redeploy-resubmit`: correct the cited evidence deficiency and watch the new SHA | continue same attempt |
| `judging` | `remain-dedicated` | watch; do not select another paper |
| `scored` | `release-scored-and-repeat` | release after exact `sync-verdict`, then `claim-next` |
| `genuine-external-blocker` | `notify-release-and-repeat` | persist, release reclaimably, then `claim-next` |

submitted/judging are dedicated states and do not release. Keep bounded verdict watching on the current attempt and do not select another paper.

After `sync-verdict` imports the exact official claim statuses,
`release-scored-and-repeat` uses `release-paper --outcome scored` and starts the
next iteration with `claim-next`. A genuine external blocker first requires a
fenced `transition-attempt` to `blocked` with nonempty `blocker` and
`next_action`; only then does `notify-release-and-repeat` call
`release-paper --outcome blocked`, notify the root coordinator, and use
`claim-next`. Release never abandons the blocked attempt: later reclamation by
the same or another worker preserves its attempt ID and history, uses a fresh
fencing token, and uses one fresh assessed immutable snapshot for selection or
reclamation.

## Validation rejection

Passing subordinate implementation subprocess tests are not controller evidence. Reject hard-coded outcomes, paper values in measurement fields, missing/tamperable provenance, stale root pages, nondeterministic bundles, incorrect algorithms, authority claims, cross-paper edits, or a dirty source tree. `validation-rejected` is an event, not a phase: before an official verdict the attempt remains `implementing`. Write exact defects into the correction contract, reclaim an expired lease with its predecessor token, and call normal fenced `run-worker`. There is no `--work-kind` flag; telemetry derives `implementation` or `correction` from the attempt phase.

## No-score diagnosis

Distinguish queue state from evidence failure:

- no live submission: repair publication or submission observation;
- exact healthy submission pending: keep watching within the deadline;
- official correctable inconclusive/rejected claim: call
  `sync-verdict --improvement-reason REASON` to preserve the exact official
  verdict and transition to `improving`, then correct its stated evidence
  deficiency with `run-worker`; the improving phase derives correction
  telemetry;
- official scored verdict: import it exactly, even when lower than expected.

Never resubmit unchanged evidence merely to refresh queue position.
