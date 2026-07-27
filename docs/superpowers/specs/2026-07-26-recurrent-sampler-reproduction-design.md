# Reproduction Design: Efficient Parallel Samplers for Recurrent-Depth Models (Approved)

**Paper**: Efficient Parallel Samplers for Recurrent-Depth Models and Their
Connection to Diffusion Language Models

**OpenReview / paper ID**: `h7WBYYJF1Q`

**arXiv**: `2510.14961v1`

**Attempt ID**: `534db42c-5b16-4f00-9a7d-a47056fc9dd4`

**Design date**: 2026-07-26

**Approval**: The user explicitly approved the minimal proposal on 2026-07-26.

**Phase gate**: The attempt remains `selected`. This document records the
approved design but does not mutate coordinator state or authorize a worker by
itself.

---

## 1. Scope

This reproduction addresses exactly the two challenge claims already bound to
the attempt. It produces deterministic CPU evidence from the released paper
source and released sampler implementation.

The reproduction does not:

- load or run the 3.5B Huginn-0125 checkpoint;
- use a GPU or reproduce A100 timing;
- reproduce the paper's benchmark accuracy or approximately 5x throughput
  result;
- infer an official challenge verdict;
- expand to the paper's three other live claims; or
- build a general source-verification or sandboxing framework.

The result is deliberately narrow:

1. a released-code and schedule-level mechanism reproduction for the sampler;
2. an exact theorem-number and assumption audit for the expressiveness claim;
3. a static evidence Space presenting those results and their limitations.

---

## 2. Live Attempt Binding

The admitted immutable snapshot is
`44cf759e3c503f46d2e8bcdea78a2824a588a0f70bf5aacd07357ac980ca84d6`,
fetched at `2026-07-26T05:52:33.573231+00:00`.

At that snapshot:

- the paper was an eligible, unverified candidate;
- no queued submission matched `h7WBYYJF1Q`;
- no tagged Space matched `h7WBYYJF1Q`; and
- no official verdict matched `h7WBYYJF1Q`.

The attempt was created under owner
`scheduler-00ca43f3-45b7-4006-88a4-bbd3d40111fa`, fencing token `1`.
That short scheduler lease expired. The controller reclaimed the attempt as
`controller-recurrent-design-20260726`, fencing token `2`, at
`2026-07-26T09:11:41.700319+00:00`. Before implementation, the controller
must still record and independently review this design through the schema-v6
lifecycle. This design does not perform either action.

---

## 3. Exact Target Claims and Approved Evidence Status

### Claim 1: sampler wavefront mechanism

Exact challenge text:

> The sampler decodes new tokens every forward pass while refining latent states for those tokens in parallel through recurrent depth (Section 3.1).

Challenge text SHA-256:
`d0da87ee16f7485d3dff369e7465f66299c55ac003a54e1cf8c00b3a0ad8b265`

Approved evidence status: **`partial`**.

The reproduction will verify the released implementation's observable control
flow and independently recompute its wavefront schedule:

- `generate(...)` dispatches diffusion arguments to
  `generate_diffusion_style(...)`;
- each outer sampler iteration applies `inner_recurrence` updates to the
  complete active latent-state tensor;
- logits are decoded for the active wavefront;
- the default `headway=1` appends one new candidate position per outer sampler
  iteration;
- a converged prefix may be frozen by the latent-difference criterion; and
- `max_wavefront` bounds the active state width.

This is partial support, not a full model reproduction. In particular, the
released implementation decodes after the inner-recurrence loop, not after
each individual `iterate_one_step` call. The evidence must describe this
precisely and must not silently reinterpret "every forward pass" as every
inner recurrent-block update. It also does not show that the 3.5B model
produces correct tokens or the reported speedup.

The paper's Section 3.1 describes the recurrent-depth architecture. The
sampler pseudocode itself appears in Section 3.3 and Appendix A.1. The report
must preserve this citation distinction.

### Claim 2: expressiveness theorem

Exact challenge text:

> The paper proves the sampler is strictly more expressive than baseline autoregressive generation under the same time budget on modern hardware (Theorem 4.2).

Challenge text SHA-256:
`2e15221c8b5516b0ab705e29a3d7c5d924ed5f0187c970a0caf60a1402757804`

Approved evidence status: **`unavailable`**.

The arXiv v1 source does not support the challenge citation as written:

- Theorem 4.2 is the informal **prefilling** result. It compares recurrent
  depth scaling with two token-replication width-scaling constructions.
- The same-runtime **decoding** statement is Theorem 4.4. It states equal
  depth and strictly greater width for diffusion-forcing decoding, conditional
  on `r > 1`, KV-cache sharing, and wavefront size `W <= L_*`.
- The expressiveness interpretation follows in Remark 4.5 and depends on the
  paper's width/parallelism model and hardware I/O assumptions.

The released v1 source does not contain a complete independently checkable
proof of the informal decoding theorem. A CPU source audit can identify the
statement, numbering, definitions, and assumptions, but cannot establish the
hardware-dependent strict-expressiveness claim. The reproduction therefore
records this claim as unavailable. It must not label the claim `verified`,
`falsified`, `toy`, or `inconclusive`; those are official-verdict concepts
owned by the challenge judge.

---

## 4. Immutable Inputs and Digests

### Paper

| Input | Pin / digest |
|---|---|
| arXiv version | `2510.14961v1` |
| PDF SHA-256 | `74e7985abe41ee2a75914a65e3778a15353fb0c0964d6ea34e7bfeb1f18312c8` |
| arXiv source archive SHA-256 | `60a795d123a2d2d642971834b6e0cba6dda80b5dfcd539f78d01639582d9c41d` |
| `arxiv_submission.tex` SHA-256 | `cdc058830d1e51f631e4fb8d1f2de0b79de91670fd4111646fe624f8c258d3b8` |
| License | CC BY 4.0 |

The project may retain the exact `arxiv_submission.tex` bytes as a vendored
audit input because the paper is CC BY 4.0. It must include author/title,
source URL, version, license URL, and an explicit statement that the vendored
file is unchanged.

### Released implementation

| Input | Pin / digest |
|---|---|
| Repository | `seal-rg/recurrent-pretraining` |
| Commit | `1ea7220ec7eb42d13e89db0663df254d0bcdc28e` |
| Commit tree | `407d9ff62687510b413fd0b54afa9f873371f7e1` |
| Sampler path | `recpre/raven_modeling_minimal.py` |
| Sampler Git blob | `0e83a0766644df9113a8923f43350c6a1b5a182c` |
| Sampler file SHA-256 | `18fcacd53fb5696a76c0d3bda44480f2f3900aa9659c137a08962c593a9a9e42` |
| Last sampler-changing commit | `59e0b69b2d96a59cbbe79c9d5034d89ecb5ab6f6` |
| License path | `LICENSE` |
| License Git blob | `d8ec087bc3eb28ba9883a9251e2f63a630878e76` |
| License file SHA-256 | `bc6c264d8ba4450599cf95c4699c6b82142f32ca1ecd91011c17b50a5a36a2f5` |
| License | Apache-2.0 |

The immutable upstream revision token is:

```text
arxiv:2510.14961v1+
pdf-sha256:74e7985abe41ee2a75914a65e3778a15353fb0c0964d6ea34e7bfeb1f18312c8+
source-sha256:60a795d123a2d2d642971834b6e0cba6dda80b5dfcd539f78d01639582d9c41d+
github:seal-rg/recurrent-pretraining@1ea7220ec7eb42d13e89db0663df254d0bcdc28e+
git-blob:recpre/raven_modeling_minimal.py@0e83a0766644df9113a8923f43350c6a1b5a182c
```

The implementation should vendor only the exact paper TeX, sampler file, and
upstream license needed by the offline audit. It must not vendor model weights,
datasets, the entire repository, or benchmark outputs.

---

## 5. Project Shape

The controller-assigned project path is:

```text
submissions/efficient-parallel-samplers-for-recurrent-depth-models/
```

Minimum contents:

```text
pyproject.toml
uv.lock
README.md
src/recurrent_sampler_repro/
  __init__.py
  evidence.py
tests/
  test_claim_bindings.py
  test_provenance.py
  test_source_audit.py
  test_wavefront_schedule.py
  test_theorem_audit.py
  test_determinism.py
  test_space_assets.py
vendor/
  arxiv/arxiv_submission.tex
  arxiv/ATTRIBUTION.md
  recurrent-pretraining/recpre/raven_modeling_minimal.py
  recurrent-pretraining/LICENSE
evidence/
space/
```

One small Python module is sufficient. Do not split the implementation into a
general parsing framework or introduce dependencies that are not needed for
the deterministic audit. Python's standard library is sufficient for source
hashing, AST inspection, schedule simulation, JSON generation, and static HTML
rendering. `pytest` is the only required development dependency.

---

## 6. Evidence Computation

### 6.1 Provenance verification

Before computing evidence, the CLI must:

1. hash every vendored input and compare it with Section 4;
2. compute the sampler file's Git blob identifier from its bytes and compare it
   with `0e83a0766644df9113a8923f43350c6a1b5a182c`;
3. verify the two exact challenge strings and their SHA-256 values;
4. verify the paper and repository attribution records; and
5. fail closed on any mismatch.

No network access is required for evidence generation after the controller has
acquired and verified the pinned bytes.

### 6.2 Released-code audit

Use Python AST inspection on the exact vendored sampler file. The audit should
record source spans and normalized findings for only the control-flow
properties needed by Claim 1:

- the diffusion dispatcher;
- the `generate_diffusion_style` defaults for `headway`,
  `inner_recurrence`, `freeze_strategy`, and `max_wavefront`;
- the active-state `inner_recurrence` loop;
- latent update, logits projection, sampling, and new-state append order;
- latent-difference freezing; and
- maximum-wavefront truncation.

AST evidence authenticates the implementation structure. It is not a model
execution and must be labeled accordingly.

### 6.3 Independent schedule mechanism

Implement a small integer-only reference schedule corresponding to the
released control flow. Its canonical run uses:

```text
outer_steps = 8
inner_recurrence = 4
headway = 1
max_wavefront = 8
```

For every outer step, record:

- active position identifiers before refinement;
- each position's cumulative recurrence count;
- decoded position identifiers;
- appended position identifiers;
- frozen position identifiers, if a supplied deterministic freeze fixture
  permits them; and
- active width after enforcing `max_wavefront`.

The canonical schedule must demonstrate that previously active positions gain
four recurrence updates while one new candidate position is appended per
outer iteration. The simulator does not generate text, execute a transformer,
measure wall-clock performance, or estimate accuracy.

Include two negative controls:

- `headway=0`, which must fail the one-new-position invariant; and
- a sequential autoregressive fixture with active width one, which must not
  exhibit a multi-position wavefront.

### 6.4 Theorem audit

Parse the vendored TeX sufficiently to identify theorem environments in
document order and record:

- Theorem 4.2 title, scope, and referenced definitions;
- Theorem 4.4 title, statement, and explicit assumptions;
- Remark 4.5's hardware/I/O interpretation; and
- whether a complete proof environment for the decoding theorem is present.

The generated record must state that the challenge's theorem number is
inconsistent with arXiv v1 and that the decoding theorem is not independently
reproduced.

---

## 7. Test-Driven Implementation

Every implementation step begins with a failing test.

1. **Claim binding tests**
   - assert both exact challenge strings and SHA-256 values;
   - reject punctuation, whitespace, or status changes.
2. **Provenance tests**
   - assert every full digest and Git blob;
   - mutate one byte in temporary fixtures and require failure.
3. **Source-audit tests**
   - assert the dispatcher and required sampler operations are found in the
     pinned AST;
   - remove or reorder a required operation in a temporary fixture and require
     failure.
4. **Schedule tests**
   - assert one appended candidate per canonical outer step;
   - assert all prior active positions gain `inner_recurrence` updates;
   - assert the wavefront bound;
   - assert both negative controls fail the relevant invariant.
5. **Theorem-audit tests**
   - assert Theorem 4.2 is the prefilling result;
   - assert Theorem 4.4 is the conditional decoding result;
   - assert Claim 2 remains `unavailable`.
6. **Determinism tests**
   - generate the complete evidence and Space trees twice;
   - compare file lists and SHA-256 values byte for byte.
7. **Static Space tests**
   - assert required metadata, tags, claim text, statuses, pins, download
     links, and limitation labels;
   - reject any unsupported 3.5B, speedup, accuracy, or official-verdict
     assertion.

Tests must exercise temporary copies and must not alter vendored inputs.

---

## 8. Deterministic Outputs

The canonical CLI writes:

```text
evidence/manifest.json
evidence/claim-1-wavefront.json
evidence/claim-2-theorem-audit.json
evidence/results.json
evidence/REPORT.md
space/README.md
space/index.html
space/poster.html
space/evidence/manifest.json
space/evidence/claim-1-wavefront.json
space/evidence/claim-2-theorem-audit.json
space/evidence/results.json
space/REPORT.md
```

Output rules:

- JSON uses sorted keys, two-space indentation, UTF-8, and one trailing
  newline.
- Generated files contain no wall-clock generation timestamp, hostname,
  absolute path, random identifier, or environment dump.
- The evidence manifest records input revisions, full hashes, command name,
  Python version requirement, output hashes, and the two approved statuses.
- The report and HTML are rendered solely from the authenticated evidence
  JSON.
- Running the generator twice from the same source tree must produce
  byte-identical outputs.

`claim-1-wavefront.json` must clearly distinguish `source_audit` from
`schedule_mechanism`. `claim-2-theorem-audit.json` must clearly distinguish
`statement_found` from `proof_reproduced`, with the latter set to `false`.

---

## 9. Static Space Contract

The deployable directory is `space/`; it is a static Space with no Python
runtime and no network calls.

`space/README.md` frontmatter must include:

```yaml
title: Recurrent-Depth Parallel Sampler Reproduction
sdk: static
app_file: index.html
tags:
  - icml2026-repro
  - paper-h7WBYYJF1Q
```

The static page contains:

- paper and attempt identity;
- the exact two challenge claims and their hashes;
- a Claim 1 card labeled `partial`;
- a Claim 2 card labeled `unavailable`;
- an interactive-free schedule table generated from the canonical run;
- the Theorem 4.2 / 4.4 distinction;
- complete upstream pins and licenses;
- download links for every evidence JSON file; and
- a prominent limitations section.

The page must not claim a 3.5B model run, GPU timing, benchmark reproduction,
approximately 5x speedup, official submission, or official verdict.

The controller alone publishes the static directory and verifies the exact
Hub commit, required tags, and healthy runtime through
`publish-deployment`.

---

## 10. Validation Commands

The paper project declares exact commands suitable for a controller-authored
validation manifest:

```bash
uv run --project submissions/efficient-parallel-samplers-for-recurrent-depth-models \
  python -m recurrent_sampler_repro.evidence \
  --project-root submissions/efficient-parallel-samplers-for-recurrent-depth-models

uv run --project submissions/efficient-parallel-samplers-for-recurrent-depth-models \
  python -m pytest \
  submissions/efficient-parallel-samplers-for-recurrent-depth-models/tests
```

The controller must also run the repository's required root pytest command,
skill validation, and:

```bash
uv run pre-commit run -a
```

The implementation proposal is acceptable only if the project tests pass, two
evidence regenerations are byte-identical, all intended files are tracked, and
the project contains no credential, cache, model weight, external symlink, or
unrelated submission change.

---

## 11. Limitations and Blockers

1. **No model execution**: the evidence does not run Huginn-0125 or establish
   generated-text quality.
2. **No hardware reproduction**: the modern-GPU runtime model and reported
   speedups are not tested.
3. **Conditional theorem**: Theorem 4.4 depends on `r > 1`, KV-cache sharing,
   `W <= L_*`, and hardware/I/O assumptions.
4. **Citation mismatch**: the challenge cites Theorem 4.2 for a decoding claim,
   while arXiv v1 places the relevant decoding statement in Theorem 4.4.
5. **Incomplete proof evidence**: the released source is insufficient for an
   independent proof reproduction of the decoding theorem.
6. **Mechanism-only schedule**: the integer schedule demonstrates wavefront
   bookkeeping, not semantic token correctness or practical throughput.

These limitations are expected outputs, not reasons to broaden the
implementation.

---

## 12. Worker and Controller Boundaries

The implementation worker is an untrusted proposal producer. After
`worker_guard.py` constructs an approved launch and its runtime preflight
passes, the worker may write only the controller-assigned worktree and:

```text
submissions/efficient-parallel-samplers-for-recurrent-depth-models/
```

The worker must not:

- edit `state/`, `docs/`, this skill, HANDOFF, another submission, or Git
  integration metadata;
- receive Hub or GitHub credentials;
- deploy, submit, poll, import a verdict, merge, or claim an external phase;
- download or run the 3.5B checkpoint; or
- change the approved claim scope or statuses.

The worker returns a proposal containing its commit, commands, evidence paths,
and concerns.

The controller alone:

- acquires or renews fenced attempt authority;
- records this design and obtains review by a different reviewer;
- creates the guarded worker contract;
- verifies every proposal diff;
- runs `attest-validation`;
- performs a fresh live eligibility check;
- runs `publish-deployment`, `attest-submission`, `watch-attempt`,
  `record-poll`, and `sync-verdict` when their respective conditions are met;
  and
- integrates the approved branch.

No external phase may be claimed without its dedicated immutable controller
attestation.

---

## 13. Approval Record and Next Action

The user approved the minimal proposal with:

- Claim 1 status `partial`;
- Claim 2 status `unavailable`;
- source/schedule/theorem evidence only;
- deterministic static Space output; and
- no 3.5B or GPU run.

The next controller action is to record this exact design for attempt
`534db42c-5b16-4f00-9a7d-a47056fc9dd4` under its current fenced authority and
request an independent review from a different identity. Only an approved
`review-design` transition may advance the attempt to guarded implementation.
