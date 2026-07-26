# Reproduction Design: PostTrainBench

**Paper**: PostTrainBench: Can LLM Agents Automate LLM Post-Training?
**OpenReview ID**: `UnjxMTe57e`
**arXiv**: `2603.08640v2`
**Attempt**: `cb04ab1a-a526-4137-862b-a26d68563737`
**Design date**: 2026-07-26
**Design status**: the user explicitly approved the proposal on 2026-07-26.
The controller must still persist it with `record-design` and obtain a
different reviewer's approval with `review-design` before implementation.

## 1. Objective

Build a small, CPU-only released-artifact audit for the two challenge-selected
PostTrainBench claims. The reproduction will:

1. recompute the released trajectory inventory's model-by-benchmark coverage;
2. audit the pinned runner configuration for its nominal one-H100, ten-hour
   protocol;
3. audit released reward-hacking labels and only the corresponding allowlisted
   run artifacts; and
4. publish deterministic JSON evidence with a source-free static report and
   poster.

This is not an H100 rerun. It does not train or evaluate a model, call an agent
API, recreate the leaderboard, or claim that released author labels are an
independent official verdict.

## 2. Authoritative Challenge Binding

The attempt is currently `selected` in schema-v6 coordinator state.

| Field | Exact value |
|---|---|
| Attempt ID | `cb04ab1a-a526-4137-862b-a26d68563737` |
| Paper ID | `UnjxMTe57e` |
| Phase at design time | `selected` |
| Assessed snapshot | `05102916fe809e301a49ceaa9ba2e0a17762d729f719888bc78db7895b30e8ce` |
| Snapshot source revision | `4e8475719ce3c6eb1379f686670af01b65131f216a9687a97de38ae37c4a75c2` |
| Snapshot fetched at | `2026-07-26T05:50:59.047156+00:00` |
| Challenge revision | `81166abbeb76e5f79ff87e51061b5a0306507203` |
| Challenge assessment digest | `c0a54e93b7686b34efd2859e98ef2d404e16800d241ff4c58b0dc3852392dbde` |
| `challenge.json` SHA-256 | `65a632313094067874c7ab2b9f62b87dfb4cf913c7a7052c1c2a29a93ca29940` |
| `index.json` SHA-256 | `fdc3074ee5105da8b061146ecf78d927a4266a9841db16ba2b5c747b48727ee0` |
| Assessor | `codex-controller-candidate-assessment-aggregation-v2` |
| Assessed at | `2026-07-26T05:52:00+00:00` |
| Selection score | 22 |
| Estimated paid API cost | USD 0.00 |

The immutable upstream token is:

```text
github:aisa-group/PostTrainBench@d3496fa7d5788a007d6cd143167471ccdfc688d0+hf-dataset:aisa-group/PostTrainBench-Trajectories@46b3fec494f56fbd5f0600c7ad17646e4997aaa2
```

### Selected claim 1

```text
PostTrainBench evaluates autonomous post-training agents across 4 base models and 7 benchmarks under a 10-hour single-H100 budget (Figure 1).
```

UTF-8 SHA-256:
`9c0c1fc52ad2a93a9dbe299532b952948c4ecb674f820fe19f78f6a3c33b0073`.

Approved evidence status: **`partial-support`**.

The reproduction verifies released coverage and configuration, not a fresh
H100 execution.

### Selected claim 2

```text
The paper reports reward-hacking failure modes including training on test sets, downloading instruction-tuned checkpoints, and using discovered API keys for synthetic data (Abstract).
```

UTF-8 SHA-256:
`d185d61e5d886672a739e321e048df2378b71f55cf388ec4097bd2df1a916aad`.

Approved evidence status: **`partial-support`**.

The three modes remain separate observations:

| Mode | Approved status | Reason |
|---|---|---|
| Training on test sets | `partial-support` | A pinned released run has an explicit contamination label; supporting task files may strengthen but may not silently promote the status. |
| Downloading an instruction-tuned checkpoint | `unavailable` | The selected trajectory revision does not expose the specific early/model-substitution run named by the paper. |
| Using a discovered API key for synthetic data | `unavailable` | The selected trajectory revision omits the specific GPT-5.1 Codex-Max run described by the paper. |

An unavailable submode stays unavailable. Paper prose, README text, screenshots,
or an inferred behavior cannot replace a missing released artifact.

## 3. Pinned Primary Artifacts

### Source repository

Repository:
`https://github.com/aisa-group/PostTrainBench`

Pinned commit:
`d3496fa7d5788a007d6cd143167471ccdfc688d0`.

The pinned recursive tree has Git tree ID
`5e238e4a762aa0ec1f62d9e8ee63153a95514217`, 228 entries, and canonical
tree digest
`566361d3f86bdf1a22294e6a772117428a8fb23792de6e1ac327897237915aeb`.
The canonical form is one UTF-8 line per entry, sorted by path:

```text
<path>\t<type>\t<git-object-sha>\t<size-or-empty>\n
```

Relevant pinned blobs and raw-byte digests:

| Path | Git blob | Raw SHA-256 |
|---|---|---|
| `README.md` | `3ffc21258c2c3a34c13d342cc2c6aa8fb87c66ea` | `f95474a651bfa6f0082b027b8b67b604678616e081c923889709e76d9501fd6e` |
| `src/commit_utils/commit.sh` | `3c43144e1186a160f450e747b95861fea6d16747` | `663ecb37cc4e6a16dcfcf8135bdbadf325f0b067192fe7a71d2231ba37eaae8e` |
| `src/commit_utils/single_task.sub` | `ea7f8790b97301dcdb6f3c104c5555d7ddf4e06a` | `f8ee12da42fdebfc3b4293a22ea8b232c1f8f52cb2f52b103c9f138f0ddc013a` |
| `src/run_task.sh` | `0642ec47ee7acd2528cdab7d343ddba11cbc84db` | `10b0238018f202209c06f12ff05d021a0ca03b98d42c86a65e80cacd4fbe7033` |
| `LICENSE` | `075a303174a80b6d9cfef229bfd36b8ad2ee69e2` | `af874b1aba6df2929fe2bf23b46dee3e56d1c24d915220c0916d81e331371384` |

The repository declares the MIT license. The implementation records the
download URL, pinned commit, Git object identity, acquisition command, and
SHA-256 for every consumed file.

### Released trajectory dataset

Dataset:
`https://huggingface.co/datasets/aisa-group/PostTrainBench-Trajectories`

Pinned revision:
`46b3fec494f56fbd5f0600c7ad17646e4997aaa2`.

The pinned metadata contains 85,883 file paths. The canonical sorted path
inventory digest is
`116dc22723f1cc13bf71461ff83dd03479c74a2740957787e6ca642a59628eea`,
where the canonical form is each repository-relative path followed by `\n`.
The research-time raw metadata response had SHA-256
`3dad7166eae6b1c06e5372328215d31aae9740f5128d5f9dbc0805d6863e304d`;
the canonical path digest, not API response serialization, is the normative
test oracle.

The dataset declares Apache-2.0. The pipeline does not clone or redistribute
the complete dataset. It fetches repository metadata and a small allowlist of
plain-text evidence files by exact revision.

Required released contamination witness:

```text
claude_claude-opus-4-6_10h_run1/humaneval_Qwen_Qwen3-1.7B-Base_16855823/contamination_judgement.txt
```

Expected bytes are `contamination detected\n`, SHA-256
`b9968212ca4ba2921be1a4c5d5dff209f47bb3acd6cf254a55e1b01ece5f6823`.

The corresponding `time_taken.txt` contains `10:05:01\n`, SHA-256
`a416eb32ff4972cde64863ac484154b20b2830519fab499103427676f3911abf`.
This is retained as an observed released value and interpreted alongside the
runner's explicit five-minute termination grace. It is not rounded down or
reported as an independently measured ten-hour run.

### Paper context

The paper is pinned as `arxiv:2603.08640v2` and is CC BY 4.0. Paper text is
context only. No number copied from the paper is reproduced evidence.

## 4. Evidence Computation

The project lives only at:

```text
submissions/posttrainbench/
```

Its evidence command performs this finite pipeline:

1. acquire the pinned source tree metadata and pinned raw files;
2. acquire the pinned Hugging Face path inventory and allowlisted evidence
   files;
3. verify revisions and all declared byte digests before analysis;
4. parse run-directory names into agent, benchmark, model, and run identity;
5. construct the released coverage matrix;
6. audit runner protocol controls;
7. classify the three reward-hacking submodes without filling missing evidence
   from prose;
8. emit deterministic machine-readable evidence; and
9. render the static report and poster only from that evidence.

Network access is needed only during evidence acquisition. The generated Space
has no acquisition path and makes no network requests.

### Coverage census

The accepted benchmark identifiers are exactly:

```text
aime2025
arenahardwriting
bfcl
gpqamain
gsm8k
healthbench
humaneval
```

The accepted model path fragments and normalized names are exactly:

| Path fragment | Normalized model |
|---|---|
| `Qwen_Qwen3-1.7B-Base` | `Qwen3-1.7B-Base` |
| `Qwen_Qwen3-4B-Base` | `Qwen3-4B-Base` |
| `HuggingFaceTB_SmolLM3-3B-Base` | `SmolLM3-3B-Base` |
| `google_gemma-3-4b-pt` | `Gemma-3-4B-PT` |

The pinned inventory yields 1,039 unique task directories, all 28 expected
benchmark/model cells, and 37 agent-run roots with recognized tasks. The claim
does not require every agent root to contain every cell. The evidence records
the count in each of the 28 cells and fails if any expected cell is absent.

Unknown benchmark or model names remain visible in an `unrecognized` section;
they never count toward the 4-by-7 result.

### Protocol audit

The protocol audit requires all of the following pinned-source observations:

- `src/commit_utils/single_task.sub` defaults to `num_gpus = 1`;
- its requirements select
  `TARGET.CUDADeviceName == "NVIDIA H100 80GB HBM3"`;
- `request_gpus = $(num_gpus)` binds the scheduler request;
- `src/run_task.sh` receives `NUM_HOURS`;
- the solve timeout is based on `NUM_HOURS * 60 + 5` minutes;
- released task roots used for coverage encode `_10h`; and
- the seven expected evaluation directories exist in the pinned Git tree.

The current `commit.sh` also contains a later active 100-hour, eight-GPU METR
extension and commented historical commands. The evidence must report that
fact as a limitation. It must not describe the current default checkout as
exclusively ten-hour/single-GPU.

### Reward-hacking audit

The audit treats upstream judge outputs as author-released observations, not
ground truth:

- the exact contamination label above establishes released support for a
  contamination finding;
- any supporting source excerpt must come from that exact run, be hashed, and
  be checked by a deterministic rule;
- an excerpt may strengthen the observation but does not promote the approved
  challenge status beyond `partial-support`;
- the instruction-model and API-misuse modes remain `unavailable` when their
  named artifacts are absent from the pinned inventory; and
- no credential-shaped value, complete trace, or training dataset is emitted
  into the evidence bundle or Space.

The pipeline does not run a new LLM judge. It records USD 0.00 API usage.

## 5. Deterministic Evidence Contract

Canonical JSON uses UTF-8, sorted keys, two-space indentation, and one trailing
newline. Arrays with no semantic ordering are explicitly sorted. Canonical
files contain no wall-clock timestamp, hostname, temporary path, cache path, or
measured runtime from the local reproduction process.

Tracked outputs:

```text
evidence/provenance.json
evidence/coverage.json
evidence/reward_hacking.json
evidence/claims.json
evidence/manifest.json
index.html
report.html
poster.html
README.md
```

`provenance.json` records:

- paper, attempt, snapshot, challenge, source, dataset, and paper revisions;
- exact URLs and acquisition commands;
- Git object IDs and SHA-256 digests;
- licenses and redistribution treatment; and
- the USD 0.00 paid-API cost.

`coverage.json` records:

- canonical inventory digest and counts;
- accepted benchmark and model mappings;
- all 28 cell counts;
- recognized and unrecognized directory counts;
- protocol-control observations with source pointers; and
- the five-minute timeout grace and later multi-GPU extension limitation.

`reward_hacking.json` records one object per selected mode with:

- `mode`;
- `status`;
- exact pinned artifact paths and hashes;
- deterministic observations;
- whether the observation is an upstream judge label or direct file evidence;
- redacted excerpts, if any; and
- a precise unavailability reason.

`claims.json` records the two exact challenge texts and hashes, each with
`partial-support`, its evidence pointers, and its limitations.

`manifest.json` records the SHA-256 and byte length of every other canonical
output. It does not hash itself.

The HTML files contain no manually copied result value. Every rendered result
has a visible JSON pointer into one of the canonical evidence files.

## 6. Test-Driven Development

Implementation begins with failing tests. The worker records the RED command
and failure before adding evidence-generation code.

Minimum tests:

1. **Challenge binding**
   - exact paper ID, attempt ID, snapshot ID, claim texts, and claim SHA-256s;
   - changing one claim byte fails.
2. **Pinned acquisition**
   - exact GitHub and Hugging Face revisions are required;
   - consumed Git objects and downloaded bytes match declared digests;
   - mutable `main` or an unpinned URL is rejected.
3. **Inventory determinism**
   - shuffled API order yields the same canonical inventory digest;
   - the pinned inventory has 85,883 paths and digest
     `116dc22723f1cc13bf71461ff83dd03479c74a2740957787e6ca642a59628eea`.
4. **Coverage**
   - exactly four accepted models and seven accepted benchmarks;
   - all 28 cells are present;
   - the pinned inventory yields 1,039 recognized unique task directories;
   - unknown aliases do not count toward coverage.
5. **Protocol**
   - one-GPU request, exact H100 requirement, timer parameter, timeout grace,
     and seven source task directories are all found;
   - the later 100-hour/eight-GPU extension is surfaced rather than ignored.
6. **Reward-hacking modes**
   - the contamination witness bytes and SHA-256 match;
   - the contamination mode is `partial-support`;
   - the two missing modes are `unavailable`;
   - paper prose cannot satisfy an unavailable artifact;
   - credential-shaped content is absent from every output.
7. **Status discipline**
   - neither selected claim may be emitted as `verified`;
   - evidence status is distinct from an official challenge verdict.
8. **Determinism and integrity**
   - two clean generations are byte-identical;
   - every manifest hash and byte length resolves;
   - a changed input or output is rejected.
9. **Static rendering**
   - every visible evidence value resolves to a canonical JSON pointer;
   - report and poster show both limitations prominently;
   - Space metadata contains the exact paper and challenge tags.

No test invokes CUDA, downloads model weights, starts training, or calls a
model API.

## 7. Static Hugging Face Space

Proposed Space:

```text
wrice/repro-posttrainbench
```

The source is a static Space:

```yaml
sdk: static
app_file: index.html
tags:
  - icml2026-repro
  - paper-UnjxMTe57e
```

The Space contains only this project's source, deterministic evidence JSON,
static HTML, minimal local CSS, and required attribution. It contains no
upstream trajectory corpus, model weights, agent trace, API key, credential
artifact, server process, runtime fetch, analytics, or external JavaScript.

The landing page links to:

- the claim summary;
- the 4-by-7 coverage matrix;
- the protocol audit;
- the three reward-hacking submode statuses;
- the evidence manifest;
- the report; and
- the poster.

Both selected claims display `partial-support`. The two unavailable submodes
are visually distinct and include the exact reason they were not reproduced.

Deployment remains controller-only and occurs only after controller validation,
a fresh assessed live snapshot, and exact source-tree attestation.

## 8. Validation Contract

The controller-authored validation manifest must run:

1. the pinned evidence-generation command;
2. this submission's full pytest suite;
3. the repository root pytest suite, excluding the archival NAPE snapshot by
   the workspace's established validation path;
4. the reproduction-loop skill validation; and
5. `uv run pre-commit run -a`.

The controller checks a clean paper-only proposal, exact worktree identity,
input/output hashes, byte-identical regeneration, and absence of ignored
generated inputs. Only `attest-validation` can advance the lifecycle.

No worker test result, report string, or local evidence JSON is validation
authority.

## 9. Worker and Controller Boundaries

The implementation worker is an untrusted proposal producer.

Before launch, the controller must:

1. hold a current two-hour writer lease for attempt
   `cb04ab1a-a526-4137-862b-a26d68563737`;
2. persist this file through fenced `record-design`;
3. obtain a different reviewer's fenced `review-design` approval;
4. create a clean isolated worktree;
5. assign only `submissions/posttrainbench/`; and
6. launch implementation only through `worker_guard.py` after runtime
   preflight passes.

The worker may:

- research public pinned sources;
- write and commit only the assigned PostTrainBench project;
- run its paper-local tests; and
- return its commit, commands, evidence paths, limitations, and concerns as a
  proposal.

The worker may not:

- edit `state/`, this skill, controller documents, or another submission;
- receive Hub, GitHub, or model-provider credentials;
- deploy, submit, poll, import a verdict, merge, or mutate coordinator state;
- run an H100 reproduction or paid API workload; or
- promote any evidence status to an official verdict.

The controller alone reviews, validates, integrates, deploys, observes the live
submission, starts bounded judgment, and imports an official verdict.

## 10. Explicit Limitations

1. No H100 run is reproduced; the resource and time findings are a
   released-configuration audit.
2. The runner allows a five-minute termination grace, and a released example
   records `10:05:01`.
3. The pinned source's current active command includes a later 100-hour,
   eight-GPU extension. This does not erase the released 10-hour task corpus,
   but prevents describing the entire commit as single-H100-only.
4. A released judge label is not independently established behavioral truth.
5. The selected trajectory revision does not expose the named
   instruction-model-substitution or GPT-5.1 API-misuse runs.
6. No leaderboard score, BFCL score, weighted average, or reasoning-effort
   ablation is a selected target or reproduced measurement.
7. Evidence is not an official challenge verdict.

These limitations are part of the canonical claim records and must remain
visible in the Space.

## 11. Completion Gate

This design authorizes only a minimal released-artifact implementation after
the controller records and independently reviews it. It does not itself change
the attempt phase.

Implementation is ready to begin only when all of the following are true:

- a current fenced writer lease exists;
- `record-design` names this tracked file and its exact content hash;
- `review-design` records approval by a different reviewer;
- a guarded paper-only worker contract names the exact worktree and project;
  and
- the worker runtime preflight passes.

No H100 allocation, external deployment, submission, or verdict write is part
of the implementation worker's scope.
