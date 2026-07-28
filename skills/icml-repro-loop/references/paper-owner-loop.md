# Paper-Owner Event Loop

One paper-owner controller owns exactly one fenced attempt. After every
mutation, reread its attempt shard and live lease. Never infer owner, fence,
phase, Space SHA, or verdict.

| Event | Required reaction | Terminal? |
| --- | --- | --- |
| `worker-exited` | `validate-or-correct`: inspect commit/diff and run fresh controller validation | no |
| `worker-noop` | `repair-permissions-and-relaunch`: fix scoped runtime permission or change runtime/model | no |
| `validation-rejected` | `correct-and-relaunch`: give exact scientific/integrity findings to the same attempt | no |
| `validated` | `publish`: publish only the exact attested source | no |
| `deployment-invalid` | `repair-publication`: correct SHA/tags/runtime before attestation | no |
| `submitted` | `watch`: start bounded official observation and release implementation capacity | no |
| `pending` | `keep-watching`: verify exact healthy visibility; do not alter evidence solely for queue age | no |
| `inconclusive` | `improve-redeploy-resubmit`: correct the cited evidence deficiency and watch the new SHA | no |
| `judging` | `release-implementation-capacity`: notify the competition coordinator to refill independently | no |
| `scored` | `sync-verdict`: import exact official claim statuses and notify coordinator | yes |
| `deadline` | `persist-blocker`: record exact phase, observation, next action, and unperformed writes | yes |
| `unresolvable-blocker` | `persist-blocker`: retain attempt; never auto-abandon | yes |

## Validation rejection

Passing worker tests are not controller evidence. Reject hard-coded outcomes,
paper values in measurement fields, missing/tamperable provenance, stale root
pages, nondeterministic bundles, incorrect algorithms, authority claims,
cross-paper edits, or a dirty source tree. `validation-rejected` is an event,
not a phase: before an official verdict the attempt remains `implementing`.
Write exact defects into the correction contract, reclaim an expired lease
with its predecessor token, and call normal fenced `run-worker`. There is no
`--work-kind` flag; telemetry derives `implementation` or `correction` from
the attempt phase.

## No-score diagnosis

Distinguish queue state from evidence failure:

- no live submission: repair publication or submission observation;
- exact healthy submission pending: keep watching within the deadline;
- official correctable inconclusive/rejected claim: call
  `sync-verdict --improvement-reason REASON` to preserve the exact official
  verdict and enters `improving`, then correct its stated evidence deficiency
  with `run-worker`; the improving phase derives correction telemetry;
- official scored verdict: import it exactly, even when lower than expected.

Never resubmit unchanged evidence merely to refresh queue position.

## Capacity notification

The paper owner does not select a second paper. On `judging`, `scored`, or
`blocked`, emit an event containing attempt ID, paper ID, phase, owner, fence,
snapshot, Space SHA, next action, blocker, and whether implementation capacity
is free. The competition coordinator dispatches another top-level paper owner.
