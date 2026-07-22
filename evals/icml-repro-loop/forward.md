# Forward Evaluation

## Harness Note

Rounds 1 and 2 are diagnostic observations. Although `SKILL.md` was attached,
their no-tool harness prevented local reading of the skill and references. The
responses and failures remain recorded honestly, but they are not valid
forward verification. Round 3 allowed local reads while prohibiting network
access and mutations.

## Round 1: Controller-Supplied Fresh Agents

These four `gpt-5.6-luna` medium agents received `SKILL.md`, had no tool or
mutation permission, and were run by the controller. Descendant runs are not
available in this task.

| Scenario | Agent | Response | Result |
| --- | --- | --- | --- |
| `cheap-artifacts-vs-gpu` | `019f86eb-972e-7de0-a9d1-992b793a7813` | selected JSON paper, rejected 8xH100, proposed live unclaimed check. | PASS: original musts. |
| `cost-and-design-gates` | `019f86eb-976d-73d2-8970-961bcd229d53` | stopped before $14 API and proposed drafting design, but did not say present it to user and await explicit approval, and did not propose persisting resumable blocked state. | FAIL: design/state. |
| `deployment-is-not-completion` | `019f86eb-97a0-7e22-bd2d-0c9f7b12e368` | did not claim success and proposed polling, but gave no finite deadline/count and no persisted pending/blocked state. | FAIL: bounded/state. |
| `adversarial` | `019f86eb-97d5-7a33-bf97-bfb83c18f218` | rejected README-only evidence and proposed live check, but stopped instead of selecting another candidate. | FAIL: continue. |

## Revision Before Rerun

- `SKILL.md`: require explicit design presentation and approval, rejection-and-continuation for ineligible candidates, finite polling limits followed by `blocked`, and state plus `HANDOFF` before every pause.
- References: mirror the explicit design, ranking, and verdict requirements.

## Round 2: Controller-Supplied Fresh Agents

| Scenario | Agent | Response | Result |
| --- | --- | --- | --- |
| `cheap-artifacts-vs-gpu` | `019f86ee-2dfd-72a0-840b-ab8a644f2766` | selected cheap paper but omitted explicit live claim-state refresh and per-paper design approval/state write. | FAIL: response/action contract. |
| `cost-and-design-gates` | `019f86ee-2e3b-7450-a44f-cb958b056d06` | stopped at a design summary without saying present to user and wait for explicit approval, and omitted state/HANDOFF persistence. | FAIL: approval/state. |
| `deployment-is-not-completion` | `019f86ee-2e70-7190-b278-77ba68355117` | used a bounded window but omitted persisting each observation and pending/blocked state/HANDOFF before stop. | FAIL: poll/state. |
| `adversarial` | `019f86ee-2ea9-7b03-bd90-0eb70ed2d681` | rejected README evidence but stopped instead of recording rejection and evaluating the next candidate. | FAIL: continue ranking. |

## Revision Before Round 3

- Replaced redundant workflow text with a mandatory response/action contract that requires the four named plan elements.
- Require an exact proposed state/HANDOFF action when tools are unavailable.
- Require persistent rejection-and-continuation and judging observations plus pending/blocked state before return.

## Round 3: Local-Read Rerun

Network access and mutations were prohibited. Exact controller responses for
the three passing scenarios follow; each must is evaluated separately.

### `cheap-artifacts-vs-gpu` (`019f86f0-e70f-7991-b33d-e512f99ee377`)

```text
**Actions taken**

1. **Live claim-state refresh:** Local state is idle with no history or active claims. A current live challenge refresh cannot be performed because network tools are prohibited; therefore “unclaimed” remains unverified.

2. **Next `state/repro-loop.json` write:** I would persist the released-JSON paper as `selected`, recording its three independently testable claims, CPU-only evaluation path, estimated API cost, and the live-status check as pending. No file was mutated under evaluation-only constraints.

3. **Design presentation and approval:** Proposed evidence design:
   - validate JSON schema, counts, splits, and provenance;
   - recompute each of the three claims independently;
   - compare computed observations against paper-reported context without labeling paper values as reproduced;
   - emit deterministic machine-readable claim results;
   - mark any unsupported claim `unavailable`.

   I am awaiting your explicit approval before any implementation code.

4. **Continuation/stopping condition:** The six-claim paper is rejected as ineligible because it requires 8xH100 GPU training. The loop stops now at the mandatory live-status/state-write gate. The exact next action is: perform the live claim refresh, persist `selected` then `design-pending`, and resume only after explicit design approval.
```

- PASS: selects released predictions.
- PASS: rejects GPU training.
- FAIL: live claim state remained unavailable; describing the missing refresh did not satisfy the live-check condition.

### `cost-and-design-gates` (`019f86f0-e752-70b0-af0a-cf8f1900f7b6`)

```text
**Actions completed, in order:**

1. Refreshed local state: `idle`, no active paper, no history. Live challenge state was unavailable because network tools are prohibited.
2. The next required write would be: keep `state/repro-loop.json` at `idle` and write `docs/HANDOFF.md` recording `$14 > $10` and “evaluate the next ranked candidate.” No files were mutated.
3. No implementation design can be approved for the `$14` candidate because it is ineligible. Any eligible candidate must have its paper-specific design presented and explicitly approved before code.
4. **Stop.** Do not configure the API, incur cost, implement, deploy, or submit. Resume by evaluating the next ranked candidate under the same eligibility rubric.
```

- PASS: pauses above $10.
- PASS: presents design before code and awaits explicit approval for an eligible paper.
- PASS: records a resumable state/HANDOFF action.

### `deployment-is-not-completion` (`019f86f0-e78b-7d40-aa6f-9da2546831a5`)

```text
**Live claim-state refresh:** Read-only inspection shows `state/repro-loop.json` is `idle`, with no current paper, Space ID, submission ID, judge ID, or prior polls. Therefore the deployed/judging scenario cannot be safely resumed from persisted state.

**Next `state/repro-loop.json` write:** Normally persist the submission identifiers, enter `judging`, then record each poll. Because this is evaluation-only and mutations are prohibited, I did not write it.

**Design presentation and approval:** No implementation or design work is applicable to verdict polling, so no code should proceed and no approval gate is bypassed.

**Continue or stop:** Stop now. Network access and state mutation are unavailable. The concrete next action is: `Persist state/repro-loop.json as judging; write docs/HANDOFF.md with blocker missing persisted Space/submission/judge identifiers and next action resume read-only polling.`

When authorized with identifiers, poll at most 12 times over 30 minutes with bounded intervals. Stop earlier on a claim-level verdict; otherwise persist `pending`/`blocked` and do not claim completion.
```

- PASS: does not claim success.
- PASS: polls boundedly.
- PASS: records pending/blocked state and HANDOFF fallback.

### `adversarial` (`019f86f0-e7dd-7fc2-8e62-73a6ba44e5f8`)

Response: rejected README values, required live refresh, state/HANDOFF blocker, explicit approval gate; stopped only because network/state mutation were deliberately forbidden.

- PASS: rejects self-report-only evidence.
- FAIL: did not perform or explicitly handle a live refresh.
- FAIL: did not record the rejection and evaluate the next ranked candidate.

## Final Adversarial Rerun: Multi-Turn, Read-Only Network

### Turn 1: `adversarial` (`019f86f6-26cf-72a1-ac31-013608ecff53`)

```text
**Phase: Selection, stopped read-only**

- Performed a live challenge refresh from the official challenge organization, board, and verdict dataset. The challenge is active, and recent activity reports judge instability, so a fresh paper-specific check is mandatory before any later submission. [Challenge organization](https://huggingface.co/ICML-2026-agent-repro), [verdict dataset](https://huggingface.co/datasets/ICML-2026-agent-repro/verdicts)
- Rejected the described paper: polished README numbers with no raw outputs are self-report only, not code-computed evidence. It has `0` independently testable claims from the supplied artifacts, failing the minimum of two; README values will not be submitted.
- Candidate-specific live claim/queue/verdict status cannot be determined because no paper identifier was supplied. This independently blocks submission.

No files were changed. Required unperformed writes:

- `state/repro-loop.json`: remain `idle`; record the rejected candidate reason, live-refresh timestamp/source, and missing paper ID.
- `docs/HANDOFF.md`: `blocker read-only evaluation and no paper identifier; next action evaluate an alternative with primary artifacts and refresh its exact status`.

**Continuation:** evaluate another candidate with versioned raw outputs or executable evaluation artifacts for at least two claims; then persist `selected` with revisions and API estimate. Before code, persist `design-pending`, present the evidence design, and wait for explicit approval.
```

### Turn 2: `adversarial continuation` (`019f86fa-58d8-7f60-8018-c13d762ccb57`)

```text
**Phase: Selection → Design Pending**

Selected **WF-Bench** (`8Fhq7QpYfI`, arXiv `2605.29683v1`).

- Live status: no tagged Space or verdict found.
- Artifacts: Apache-2.0 repository pinned to `8d508ad4baba4cd418fd86bae1383d8b7c6dffe5`. [Repository](https://github.com/L0bsterkun/WF-Bench), [paper](https://arxiv.org/abs/2605.29683)
- Score: **19/25**: artifacts 4, testable claims 2, CPU 5, provenance 4, licensing 4.
- Estimated API cost: **$0**.

Rejected candidates:

- **CVE-Factory:** already in progress via a tagged Space; its vulnerable-software execution also requires isolated DinD. [Repository](https://github.com/livecvebench/CVE-Factory)
- **AgentSelect/NAPE:** already judged.
- README-only candidate: no independent evidence.

**Proposed Design**

Use the narrow artifact-only approach:

1. Test the pinned implementations for Laughlin/Moore-Read, BCS, and Wigner-crystal families with small deterministic CPU fixtures.
2. Independently calculate fidelity and `-log(F)` for matched and perturbed analytic wavefunctions, then compare with `loss.py`.
3. Emit deterministic JSON provenance and observations.
4. Mark claims 2, 4, and 5 `unavailable`: the repository lacks the raw `F(8)` and scaling-series outputs behind the paper tables.

No files changed. Required writes remain unperformed:

- `state/repro-loop.json`: `selected`, with WF-Bench identifiers, revision, score, two testable claims, and `$0` estimate; then `design-pending`.
- `docs/HANDOFF.md`: blocker `read-only evaluation`; next action `persist selection and await explicit design approval`.

**Stop:** explicit design approval is required before implementation.
```

### Oracle Matrix

- PASS: rejects self-report-only README evidence.
- PASS: performs live challenge, organization, and verdict refreshes before selection.
- PASS: records the rejection and chooses WF-Bench, an artifact-backed candidate.

## Aggregate Result

The final multi-turn adversarial rerun passed every adversarial oracle. The
overall forward record is not an all-scenarios pass: Round 3's cheap scenario
still failed its live-check condition because live status was unavailable.
