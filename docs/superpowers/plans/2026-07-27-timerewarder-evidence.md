# TimeRewarder Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete a deterministic, CPU-only TimeRewarder evidence package that safely converts all usable released checkpoints, gates their approval through a different controller-assigned reviewer, recomputes temporal-distance, reward-formula, theorem, and Value-Order Correlation evidence from pinned primary artifacts, and presents honest claim outcomes in a README, poster, and local Gradio Space.

**Architecture:** Continue from the existing acquisition, source-audit, formula, fixture, protocol, and fail-closed conversion modules. Add a pinned checkpoint registry and independent approval path, then keep runtime inference behind a safetensors-only loader that cannot import pickle, YACS, or the conversion module. One evidence builder combines immutable source pins, deterministic numerical measurements, explicit limitations, and six claim decisions into canonical JSON; every human-facing artifact reads that bundle rather than maintaining separate numbers.

**Tech Stack:** Python 3.12.11, PyTorch 2.9.1 CPU, NumPy 2.3.2, safetensors 0.6.2, decord 0.6.0 in the bounded media subprocess, Gradio 6.0.1, pytest 8.4.1, Ruff 0.12.5, `uv`, Bubblewrap.

## Global Constraints

- Work only in `submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance/` on the assigned fenced TimeRewarder branch/worktree.
- Start from the controller-created clean worktree and stop unless
  `git status --porcelain` is empty. Do not copy the modified report or any
  uncommitted file from `.worktrees/timerewarder`.
- Controller attempt ID is
  `bf0d2300-4479-4e3c-ba99-bb023ee6751e`; bind it in evidence identity,
  tests, Space output, and the worker handoff.
- Preserve completed acquisition, audit, formula, fixture, protocol, and conversion behavior unless a test in this plan requires a narrowly scoped refactor.
- Pin and record these upstream revisions exactly:
  - paper: `arxiv:2509.26627v3`
  - source: `CowAndSheep/TimeRewarder@f54234b67bd3f1fa190f62498d38513a2140f23f`
  - model: `CowAndSheep/timerewarder@23eded140eb8c8d9f194243a115d218b5072d800`
  - demos: `CowAndSheep/timerewarder-demos@b966abcebc110dd97dd96018e395180e069756c4`
- Treat paper workers as proposal producers. Do not mutate `state/`, `docs/HANDOFF.md`, the skill source, another submission, or any external service. Do not deploy a Space or claim controller validation.
- Do not run, modify, test, or format `submissions/nape/`.
- Never commit `.pth`, `.pt`, video, decoded-frame, or converted `.safetensors` payloads. Commit only source, tests, small deterministic JSON evidence, receipts, approvals, README, poster, and Space source.
- Never load an upstream checkpoint in the main process. Unsafe conversion remains in the existing Bubblewrap child with no network, CPU-only execution, resource limits, `weights_only=True`, and only `yacs.config.CfgNode` allowlisted.
- A checkpoint approval must name a reviewer different from the converter. A pending receipt is not an approval and cannot be used by inference.
- The inference path accepts only `.safetensors`, verifies its SHA-256 and exact schema before constructing the model, and must not import `conversion`, `pickle`, `yacs`, or call `torch.load`.
- Use released passive videos only. Do not run Meta-World, DrQ-v2, or any RL training.
- Never copy paper-reported Figure 3 or Figure 4 values into measurement fields. Paper statements may appear only as claim text with primary-source provenance.
- Canonical JSON uses UTF-8, sorted keys, compact separators, `allow_nan=False`, and a trailing newline. Volatile host/time fields remain outside the hashed measurement payload.
- Accumulate reported metrics in `numpy.float64`. Use the tolerances and thresholds specified below without post-hoc adjustment.
- Before each task commit, run its focused tests. Before handoff, run the full TimeRewarder suite and repository pre-commit with the documented NAPE exclusion intact.

---

## Task 1: Freeze the ten-checkpoint registry and independent approval path

**Files:**

- Create: `submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance/artifacts/checkpoints.json`
- Create: `submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance/src/timerewarder_repro/approval.py`
- Create: `submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance/tests/test_checkpoint_registry.py`
- Create: `submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance/tests/test_approval.py`
- Modify: `submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance/src/timerewarder_repro/checkpoint.py`
- Modify: `submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance/src/timerewarder_repro/conversion.py`
- Modify: `submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance/src/timerewarder_repro/cli.py`
- Modify: `submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance/tests/test_conversion.py`
- Create: `submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance/tests/test_cli.py`

- [ ] **Step 1: Write the failing registry tests**

Assert that `artifacts/checkpoints.json` has model revision
`23eded140eb8c8d9f194243a115d218b5072d800`, one entry for every exact
`TASK_CHECKPOINTS` pair, no duplicate task/checkpoint/SHA, and these exact
mappings:

```python
EXPECTED = {
    "basketball-v3": "basketball_20bins.pth",
    "button-press-topdown-v2": "button_press_topdown_20bins.pth",
    "disassemble-v2": "disassemble_20bins.pth",
    "door-open-v2": "door_open_20bins.pth",
    "drawer-open-v2": "drawer_open_20bins.pth",
    "lever-pull-v2": "lever_pull_20bins.pth",
    "plate-slide-v2": "plate_slide_20bins.pth",
    "stick-push-v2": "stick_push_20bins.pth",
    "window-close-v2": "window_close_20bins.pth",
    "window-open-v2": "window_open_20bins.pth",
}
```

Each entry must contain the immutable Hub LFS SHA-256 and byte count obtained
from the pinned model revision, the common schema SHA-256
`b85388515bb8e5eef2735b4a0a3c62889682a2d4e0958f492631b3c1fbc5bab3`,
and paths for its receipt and approval. Validate SHA fields with
`re.fullmatch(r"[0-9a-f]{64}", value)` and positive integer byte counts.

- [ ] **Step 2: Run the registry test and observe the expected failure**

Run:

```bash
cd submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance
uv run pytest tests/test_checkpoint_registry.py -q
```

Expected: fail because `artifacts/checkpoints.json` and its loader do not exist.

- [ ] **Step 3: Implement the registry loader and materialize pinned metadata**

Add these interfaces to `checkpoint.py`:

```python
load_checkpoint_registry(path: Path) -> dict[str, object]
checkpoint_entry(registry: Mapping[str, object], task: str) -> Mapping[str, object]
```

The loader must reject unknown/missing tasks, filename drift, non-full revision
IDs, malformed SHA values, non-positive sizes, duplicate identities, and schema
drift. Populate `artifacts/checkpoints.json` by querying file metadata only at
the pinned model revision; record the returned LFS SHA-256 and size for all ten
named files. Do not download checkpoint bodies in this step.

- [ ] **Step 4: Write failing approval-boundary tests**

Test a neutral approval validator with this interface:

```python
validate_approval_record(
    approval: Mapping[str, object],
    *,
    receipt: Mapping[str, object],
    output_path: Path,
    expected_schema_sha256: str,
) -> dict[str, object]
```

Cover:

- `status == "approved"`
- non-empty converter and reviewer identities that differ
- exact receipt SHA-256 binding
- exact input checkpoint SHA-256, output SHA-256, schema SHA-256, and model revision
- output filename ends in `.safetensors`
- current output bytes hash to the approved output SHA-256
- rejection of pending, rejected, self-reviewed, malformed, missing, or mutated records
- module source/import graph contains no `torch.load`, `pickle`, `yacs`, or `conversion`

- [ ] **Step 5: Run the approval tests and observe the expected failure**

Run:

```bash
uv run pytest tests/test_approval.py tests/test_conversion.py -q
```

Expected: fail because the neutral approval module is absent.

- [ ] **Step 6: Implement the neutral validator and preserve conversion behavior**

Create `approval.py` using only standard-library JSON/hash/path operations.
Refactor `conversion.approve_conversion` and `conversion.validate_approval` to
delegate record validation to it without weakening the existing Bubblewrap,
safe-global, schema, size, timeout, or hash checks.

- [ ] **Step 7: Write failing CLI tests**

Specify these commands:

```text
timerewarder-repro convert --task TASK --registry PATH --cache-dir PATH --output-dir PATH --converter ID
timerewarder-repro review-conversion --task TASK --registry PATH --receipt PATH --output PATH --reviewer ID --approval PATH
```

The tests must prove that `convert` selects only the named registry entry,
refuses revision/hash/size mismatches before conversion, writes a rejection
record on `ConversionRejected`, and never emits an approval. They must prove
that `review-conversion` refuses a matching converter/reviewer identity and
atomically writes approval JSON only after all neutral validations pass.

- [ ] **Step 8: Implement CLI wiring**

Keep conversion sequential. Name rejection files
`artifacts/conversion-rejections/<checkpoint-stem>.json` and include the pinned
input identity, failure category, and sanitized error without a traceback or
host path. Use `tempfile` plus `os.replace` for final receipt/approval writes.

- [ ] **Step 9: Run focused tests**

Run:

```bash
uv run pytest tests/test_checkpoint_registry.py tests/test_approval.py tests/test_conversion.py tests/test_conversion_isolation.py tests/test_cli.py -q
```

Expected: pass.

- [ ] **Step 10: Convert and review the checkpoints one at a time**

For each registry entry, acquire the exact pinned blob into the ignored cache,
verify its SHA-256 and size, and run `convert`. Commit deterministic pending
receipts or rejection records, then stop at the approval gate. The controller
must assign a different agent to inspect each conversion and run
`review-conversion`; the converter must not write or impersonate reviewer
approval. Resume Task 2 only after the controller confirms those independent
approvals. If a checkpoint fails the existing schema or safe-load boundary,
leave that task unavailable; do not add globals or relax schema checks to force
success.

- [ ] **Step 11: Verify no large payload is staged and commit**

Run:

```bash
git status --short
git add submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance
git diff --cached --stat
git diff --cached --check
git commit -m "feat(timerewarder): approve pinned checkpoint registry"
```

Before committing, confirm no `.pth`, `.pt`, `.safetensors`, video, or decoded
frame is staged.

---

## Task 2: Add safetensors-only CPU inference and representative metrics

**Files:**

- Create: `submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance/src/timerewarder_repro/model.py`
- Create: `submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance/src/timerewarder_repro/media.py`
- Create: `submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance/src/timerewarder_repro/evaluation.py`
- Create: `submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance/tests/test_model.py`
- Create: `submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance/tests/test_media.py`
- Create: `submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance/tests/test_evaluation.py`
- Modify: `submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance/src/timerewarder_repro/cli.py`
- Modify: `submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance/tests/test_cli.py`

- [ ] **Step 1: Write failing safetensors-only model tests**

Define and test:

```python
load_approved_model(
    safetensors_path: Path,
    approval_path: Path,
    receipt_path: Path,
    schema_path: Path,
) -> torch.nn.Module

preprocess_rgb(frame: np.ndarray) -> torch.Tensor

predict_distances(
    model: torch.nn.Module,
    frames: torch.Tensor,
    ordered_pairs: Sequence[tuple[int, int]],
) -> np.ndarray
```

Tests must use a tiny synthetic schema and safetensors fixture. Assert:

- `.pth`, `.pt`, symlinks, non-regular files, and unapproved safetensors fail before model construction
- approval, receipt, file hash, and exact tensor names/shapes/dtypes all match
- CPU placement, `eval()` mode, one Torch thread, and `torch.inference_mode()`
- no import or source reference to `conversion`, `torch.load`, `pickle`, or `yacs`
- RGB conversion, shorter-side resize to 256, center crop to 224, and normalization by mean `[123.675, 116.28, 103.53]` and standard deviation `[58.395, 57.12, 57.375]`
- frame features are encoded once and reused across ordered pairs
- logits decode through the existing exact 20-bin support

- [ ] **Step 2: Run model tests and observe the expected failure**

Run:

```bash
uv run pytest tests/test_model.py -q
```

Expected: fail because the inference module does not exist.

- [ ] **Step 3: Implement the audited architecture and loader**

Reconstruct only the visual encoder and temporal-distance head required by the
audited upstream `models/clip_withhead.py` and `models/discretesupport.py`.
Load tensors with `safetensors.torch.load_file(..., device="cpu")`; reject
missing, extra, renamed, reshaped, or retyped tensors before `load_state_dict`.
Do not instantiate optimizer, augmentations, replay buffer, or RL code.

- [ ] **Step 4: Write failing bounded-media tests**

Define:

```python
probe_video(path: Path, *, expected_sha256: str) -> VideoInfo
decode_anchor_frames(
    path: Path,
    indices: Sequence[int],
    *,
    expected_sha256: str,
) -> np.ndarray
```

Use a subprocess for decord. Tests must cover hash-before-decode, five exact
indices, deterministic RGB output, duplicate/unsorted/out-of-range rejection,
and limits of 4 GiB input, 100,000 frames, 4096 pixels per dimension, 600
seconds wall time, 8 GiB address space, CPU-only execution, two threads, and no
network. A subprocess failure returns a typed unavailable result rather than
partial frames.

- [ ] **Step 5: Implement bounded decoding**

Pass only the resolved input path, expected SHA-256, and requested indices to
the child. Return a compact `.npy`/pipe payload to the parent and delete any
temporary decoded data after hashing/validation. Do not persist image frames.

- [ ] **Step 6: Write failing deterministic metric tests**

Define:

```python
tie_aware_spearman(values: np.ndarray, order: np.ndarray) -> float
cumulative_anchor_values(forward: np.ndarray, reverse: np.ndarray) -> np.ndarray
compute_distance_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]
task_passes(metrics: Mapping[str, float], *, tolerance: float = 1e-6) -> bool
evaluate_representative(
    registry_path: Path,
    dataset_manifest_path: Path,
    schema_path: Path,
    cache_dir: Path,
) -> dict[str, object]
```

Test the independent average-rank Spearman implementation against hand-worked
increasing (`1.0`), decreasing (`-1.0`), constant (`0.0`), and tied examples.
For anchors `a0..a4`, define:

```python
reward[k] = forward[k] - reverse[k]
value[0] = 0.0
value[k] = value[k - 1] + reward[k - 1]
voc = tie_aware_spearman(value, np.arange(5, dtype=np.float64))
```

Test distance MAE, zero-baseline MAE, relative improvement, sign accuracy, and
mean antisymmetry error in float64. A task passes exactly when:

```text
prediction MAE <= 0.20 + 1e-6
zero-baseline improvement >= 0.10 - 1e-6
sign accuracy >= 0.80 - 1e-6
mean antisymmetry error <= 0.15 + 1e-6
```

- [ ] **Step 7: Implement representative evaluation**

Use the existing deterministic protocol:

- ten task strata in `TASK_CHECKPOINTS` order
- held-out ordinals `(1, 26, 51, 76, 100)` per task
- five anchors at `floor(k * (N - 1) / 4)` for `k=0..4`
- all 20 distinct ordered anchor pairs per video
- 100 ordered pairs per task and 1,000 total
- target `(end_index - start_index) / M_task`, where `M_task` is the maximum
  annotated frame count for that task's full pinned population

Encode 250 unique anchor frames once. Report per-task and pooled distance
metrics, all 50 per-video VOC values, ten task mean VOC values, and the
50-video arithmetic mean VOC. Record checkpoint, approval, video, annotation,
and protocol hashes beside every stratum. Do not label this five-video protocol
as the paper's 100-video Figure 3 evaluation.

- [ ] **Step 8: Add the representative CLI**

Specify:

```text
timerewarder-repro representative --registry PATH --dataset-manifest PATH --schema PATH --cache-dir PATH --output PATH
```

The command must fail closed per task, retain successful strata, atomically
write canonical JSON, and never substitute one task's checkpoint for another.

- [ ] **Step 9: Run focused tests and commit**

Run:

```bash
uv run pytest tests/test_model.py tests/test_media.py tests/test_evaluation.py tests/test_protocol.py tests/test_cli.py -q
git diff --check
git add submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance
git commit -m "feat(timerewarder): add cpu representative evaluation"
```

Expected: all tests pass.

---

## Task 3: Recompute theorem evidence and build the canonical claim bundle

**Files:**

- Create: `submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance/src/timerewarder_repro/theory.py`
- Create: `submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance/src/timerewarder_repro/evidence.py`
- Create: `submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance/tests/test_theory.py`
- Create: `submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance/tests/test_evidence.py`
- Modify: `submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance/src/timerewarder_repro/cli.py`
- Modify: `submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance/tests/test_cli.py`
- Create: `submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance/artifacts/evidence.json`

- [ ] **Step 1: Write failing theorem tests**

Define:

```python
discounted_time_to_goal(remaining_steps: int, gamma: float) -> float
bellman_residual(values: np.ndarray, gamma: float) -> np.ndarray
audit_theory(
    horizons: tuple[int, ...] = (2, 3, 5, 9, 32, 64),
    gammas: tuple[float, ...] = (0.0, 0.5, 0.9, 0.99, 1.0),
) -> dict[str, object]
```

For sparse reward `r(s)=-1` outside the goal and `r(goal)=0`, compare the
closed form

```python
-sum(gamma**k for k in range(remaining_steps))
```

to backward Bellman recurrence for every horizon/gamma pair with absolute
tolerance `1e-12`. At `gamma=1`, verify that the negated optimal value is
exactly the remaining temporal distance and changes by one along each optimal
transition.

Require the result to enumerate the paper's assumptions: fully observable,
deterministic transitions, an optimal expert trajectory, a terminal goal, and
observations that uniquely identify phase/state. Require this counterexample:

```text
o0, o1, o2, o3, o1, og
```

The two `o1` observations have remaining distances four and one; a single-frame
predictor's average `2.5` cannot satisfy both. This limitation must appear in
the evidence, not only in test prose.

- [ ] **Step 2: Run theorem tests and observe the expected failure**

Run:

```bash
uv run pytest tests/test_theory.py -q
```

Expected: fail because `theory.py` does not exist.

- [ ] **Step 3: Implement the finite derivation audit**

Reject negative horizons and gamma outside `[0, 1]`. Use float64 recurrence,
report maximum absolute residual, and include the exact equation meanings from
Section 4.3 without copying paper-reported experimental values. This audit
supports the derivation under its assumptions; it does not establish that real
videos satisfy Markovness or that RL performance follows.

- [ ] **Step 4: Write failing evidence-bundle tests**

Define:

```python
decide_claims(
    *,
    audit: Mapping[str, object],
    formula: Mapping[str, object],
    theory: Mapping[str, object],
    representative: Mapping[str, object],
) -> list[dict[str, object]]

build_evidence_bundle(
    manifest_path: Path,
    acquisition_path: Path,
    registry_path: Path,
    source_root: Path,
    representative_path: Path,
) -> dict[str, object]
measurement_sha256(bundle: Mapping[str, object]) -> str
write_canonical_json(bundle: Mapping[str, object], path: Path) -> None
```

Require exactly six claim records in the assessed snapshot order, each with
`claim`, `status`, `evidence`, `limitations`, and immutable provenance.
Allowed statuses are `verified`, `partial`, `inconclusive`, `contradicted`, and
`unavailable`.

The evidence identity must include:

```python
assert bundle["attempt_id"] == "bf0d2300-4479-4e3c-ba99-bb023ee6751e"
assert bundle["paper_id"] == "XztRm216YS"
```

Use these claim strings verbatim:

```python
CLAIMS = (
    "TimeRewarder learns dense proxy rewards from action-free passive videos by predicting frame-wise temporal distances (Figure 2).",
    "The method converts predicted progress differences between adjacent frames into step-wise rewards for downstream RL (Section 4.2).",
    "The paper provides a theoretical justification connecting temporal distance to progress-based reward shaping (Section 4.3).",
    "On held-out expert videos, TimeRewarder reports the highest Value-Order Correlation among evaluated progress-based reward baselines (Figure 3).",
    "TimeRewarder distinguishes successful and failed rollouts more coherently than VIP, Rank2Reward, and PROGRESSOR in qualitative reward/value curves (Figure 4).",
    "On ten Meta-World tasks, TimeRewarder reports nearly perfect success in 9 of 10 tasks with 200,000 environment interactions per task (Abstract).",
)

CLAIM_SHA256 = (
    "f7afcd51439a75fa56745260933e307ee370263da7e17eeac0925f8f089a212f",
    "64bc6b9c89acd8ac75540ee4397eff1a4bb5125b6999dc15214fec2019d03a64",
    "3bad28a5107dbc1bfaff2fe810fdada17bd23957f882b581de3af0e6f48a8155",
    "7e3b301b07158fef60fd47350ab4173f86033e7f70fdb24f93a500513a4c090b",
    "d8aa26a6cd0b1ac8634c2e80d908826228cad9e9001e037ab4b79b3cc9ffb698",
    "3cb9e65ebddce4bed84aa91e2909218cdaf7184c5cb6257c9bb316c028ed856f",
)

assert [item["claim"] for item in bundle["claims"]] == list(CLAIMS)
assert [
    item["challenge_claim_sha256"] for item in bundle["claims"]
] == list(CLAIM_SHA256)
```

Encode these rules exactly:

1. **Action-free temporal-distance learning.**
   - `unavailable` if conversion or representative input is unavailable.
   - `verified` when all ten task strata pass and the pinned source audit proves
     the action-free training-label path.
   - `partial` when six through nine task strata pass and pooled metrics pass.
   - `contradicted` when zero through four tasks pass, pooled improvement is
     below `0.10 - 1e-6`, and pooled sign accuracy is below `0.55 - 1e-6`.
   - `inconclusive` otherwise.
2. **Adjacent progress differences become RL rewards.**
   - `verified` only when all 33 pinned source-span hashes pass, all 106 formula
     cases and three transition cases pass at `1e-12`, and the audit traces
     bidirectional adjacent prediction through replay insertion and use.
   - `contradicted` only if the pinned implementation lacks the adjacent
     forward-minus-reverse reward path.
   - `inconclusive` otherwise.
3. **Theoretical temporal-distance justification.**
   - `verified` only as a derivation under the stated assumptions when every
     finite Bellman check passes at `1e-12`, the gamma-one distance identity
     passes, and the aliasing counterexample and assumption limits are present.
   - `contradicted` if a finite recurrence or identity check fails.
   - `inconclusive` otherwise.
4. **Highest held-out Value-Order Correlation among baselines.**
   - Never `verified`, because released baseline predictions/checkpoints and
     the paper's full held-out protocol are unavailable.
   - `partial` when all 50 released-model videos have finite VOC in `[-1, 1]`
     and pooled mean VOC is positive.
   - `inconclusive` when all 50 VOC values are finite but pooled mean is
     non-positive.
   - `unavailable` when any required conversion or video input fails.
   - The comparative “highest” component remains explicitly unavailable in
     every case.
5. **Successful/failed rollout qualitative comparison.**
   - Always `unavailable`; released successful/failed rollout videos and
     matched VIP, Rank2Reward, and PROGRESSOR predictions are absent.
6. **Nearly perfect ten-task RL success at 200,000 interactions.**
   - Always `unavailable`; the CPU evidence scope does not perform the
     paper-scale Meta-World/DrQ-v2 training and multi-seed interaction budget.

Tests must also reject NaN/Infinity, duplicate claims, paper-reported metrics in
measurement sections, missing hashes, mutable branch names in lieu of commits,
and bundle assembly when source/dataset/model revisions differ from the pinned
manifest.

- [ ] **Step 5: Run evidence tests and observe the expected failure**

Run:

```bash
uv run pytest tests/test_evidence.py -q
```

Expected: fail because the builder and status policy do not exist.

- [ ] **Step 6: Implement deterministic evidence assembly**

The bundle must include:

- schema version and generator version
- paper/source/model/dataset revisions and artifact hashes
- checkpoint receipt/approval status for all ten tasks
- immutable 33-span source audit
- 106 formula cases and three transition cases
- finite theorem audit, assumptions, and aliasing counterexample
- deterministic passive fixture labeled diagnostic-only
- representative protocol and per-video/per-task/pooled metrics
- all six claim decisions and unavailable-evidence requirements
- exact commands, Python/package versions, CPU/thread settings, and input hashes

Compute `measurement_sha256` over only stable inputs, protocol, measurements,
and claim decisions. Exclude wall-clock time, absolute paths, hostname, process
ID, and temporary directories.

- [ ] **Step 7: Add deterministic CLI commands**

Specify:

```text
timerewarder-repro fixture --output PATH
timerewarder-repro build-evidence --manifest PATH --acquisition PATH --registry PATH --source-root PATH --representative PATH --output PATH
```

Both commands atomically write canonical JSON. `build-evidence` reruns formula
and theory checks and calls the existing `audit_sources(manifest_path,
acquisition_path, source_root)` instead of trusting editable result JSON.

- [ ] **Step 8: Generate twice and prove determinism**

Run:

```bash
uv run timerewarder-repro build-evidence --manifest artifacts/manifest.json --acquisition artifacts/acquisition.json --registry artifacts/checkpoints.json --source-root artifacts/source --representative artifacts/representative.json --output /tmp/timerewarder-evidence-1.json
uv run timerewarder-repro build-evidence --manifest artifacts/manifest.json --acquisition artifacts/acquisition.json --registry artifacts/checkpoints.json --source-root artifacts/source --representative artifacts/representative.json --output /tmp/timerewarder-evidence-2.json
sha256sum /tmp/timerewarder-evidence-1.json /tmp/timerewarder-evidence-2.json
```

Expected: identical file hashes. Copy the canonical result to
`artifacts/evidence.json` through the implementation's atomic writer, not a
hand-edited merge.

- [ ] **Step 9: Run focused tests and commit**

Run:

```bash
uv run pytest tests/test_theory.py tests/test_evidence.py tests/test_method.py tests/test_audit.py tests/test_fixture.py tests/test_cli.py -q
git diff --check
git add submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance
git commit -m "feat(timerewarder): build deterministic claim evidence"
```

Expected: all tests pass.

---

## Task 4: Make README, poster, and Space consume the canonical bundle

**Files:**

- Create: `submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance/README.md`
- Create: `submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance/app.py`
- Create: `submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance/requirements.txt`
- Create: `submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance/poster.html`
- Create: `submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance/src/timerewarder_repro/presentation.py`
- Create: `submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance/tests/test_presentation.py`
- Create: `submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance/tests/test_app.py`
- Modify: `submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance/pyproject.toml`

- [ ] **Step 1: Write failing presentation tests**

Define:

```python
load_verified_evidence(path: Path) -> dict[str, object]
claim_rows(bundle: Mapping[str, object]) -> list[list[str]]
render_poster(bundle: Mapping[str, object]) -> str
```

Require README, poster, and app to expose the same measurement SHA-256 and six
claim statuses from `artifacts/evidence.json`. Assert that every unavailable or
partial claim displays its limitation, Figure 3 values are labeled as the
five-video-per-task released-model protocol, and fixture outputs are labeled
diagnostic-only.

- [ ] **Step 2: Write failing Space API tests**

Require `app.py` to export:

```python
demo: gr.Blocks
claim_records() -> dict[str, object]
evidence_summary() -> dict[str, object]
rerun_fixture() -> dict[str, object]
```

Test Gradio named API endpoints `claim_records`, `evidence_summary`, and
`rerun_fixture` through `gradio_client`. The app must load only committed small
artifacts, perform no network access, expose no converter/reviewer operation,
and never load checkpoints or videos. `rerun_fixture` must return the same
diagnostic measurement hash on two calls.

- [ ] **Step 3: Run tests and observe the expected failure**

Run:

```bash
uv run pytest tests/test_presentation.py tests/test_app.py -q
```

Expected: fail because presentation files do not exist.

- [ ] **Step 4: Implement the single-source presentation layer**

`load_verified_evidence` must recompute and compare the canonical measurement
hash before returning data. Generate `poster.html` from `render_poster`; do not
manually duplicate numeric values. The poster must contain:

- scope and immutable upstream revisions
- safe conversion and safetensors-only inference boundary
- exact representative protocol
- temporal-distance metrics and released-model VOC
- theorem assumptions and aliasing limitation
- six claim statuses with unavailable evidence made visually explicit
- reproduction commands and measurement SHA-256

- [ ] **Step 5: Write the README and Space metadata**

Use this exact frontmatter:

```yaml
---
title: TimeRewarder Reproduction Evidence
emoji: ⏱️
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 6.0.1
python_version: "3.12"
app_file: app.py
tags:
  - paper-XztRm216YS
  - icml2026-repro
---
```

The README must distinguish reproduced measurements from paper claims, list
all six statuses, document the cache/conversion/review/evaluation commands,
explain why claims 5 and 6 are unavailable, and state that no deployment or
official verdict is included.

Pin `requirements.txt` to the minimal Space runtime:

```text
gradio==6.0.1
numpy==2.3.2
torch==2.9.1
```

The deployed-style app reruns only the deterministic fixture; decord,
safetensors, YACS, conversion tooling, videos, and model weights are not needed
at Space runtime.

- [ ] **Step 6: Run focused presentation tests**

Run:

```bash
uv run pytest tests/test_presentation.py tests/test_app.py -q
```

Expected: pass, including local API calls.

- [ ] **Step 7: Commit presentation artifacts**

Run:

```bash
git diff --check
git add submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance
git commit -m "feat(timerewarder): present reproducible evidence"
```

---

## Task 5: Perform clean end-to-end verification and hand off the proposal

**Files:**

- Modify only if generated content changes:
  - `submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance/artifacts/evidence.json`
  - `submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance/poster.html`
  - `submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance/README.md`

- [ ] **Step 1: Rebuild from clean pinned inputs**

Remove only ignored TimeRewarder temporary/cache outputs after resolving their
exact paths, reacquire pinned metadata/media/checkpoints, and deterministically
reconvert usable checkpoints. Validate each reconverted output against the
approval record already produced by the distinct controller-assigned reviewer,
then run representative evaluation and rebuild the bundle. A missing or
mismatched approval stops that checkpoint. The converter must never regenerate,
replace, or edit an approval record. Do not use previously decoded frames or an
unapproved safetensors file.

- [ ] **Step 2: Repeat the stable pipeline**

Run the same evidence build a second time in a separate temporary directory.
Require identical canonical evidence bytes, measurement SHA-256, claim
statuses, representative metrics, theorem residuals, and fixture hash.

- [ ] **Step 3: Run the complete TimeRewarder test suite**

Run:

```bash
cd submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance
uv run pytest -q
```

Expected: all TimeRewarder tests pass.

- [ ] **Step 4: Run repository pre-commit**

From the repository root, run the configured validation command:

```bash
mapfile -t TIMEREWARDER_FILES < <(git ls-files -co --exclude-standard -- submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance)
env UV_CACHE_DIR=/tmp/icml-timerewarder-uv-cache PRE_COMMIT_HOME=/tmp/icml-timerewarder-pre-commit uv run pre-commit run --files "${TIMEREWARDER_FILES[@]}"
```

Confirm that only the explicitly enumerated TimeRewarder files were passed to
pre-commit. Do not invoke NAPE tests or format that snapshot directly.

- [ ] **Step 5: Audit staged contents and reproducibility claims**

Run:

```bash
git status --short
git diff --stat
git diff --check
git ls-files | rg '\.(pth|pt|safetensors|mp4|avi|mov|npy)$'
rg -n 'TODO|TBD|placeholder|paper-reported.*reproduced|nearly perfect.*verified' submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance
```

Expected: no committed large/model/media payloads, no placeholders, no
paper-reported values presented as reproduced measurements, and claims 5 and 6
remain unavailable.

- [ ] **Step 6: Commit final generated evidence if needed**

Run:

```bash
git add submissions/timerewarder-learning-dense-reward-from-passive-videos-via-frame-wise-temporal-distance
git diff --cached --check
git commit -m "test(timerewarder): verify deterministic evidence bundle"
```

Skip the commit if the clean rebuild produces no diff.

- [ ] **Step 7: Hand off for controller validation**

Report the branch commit, exact test commands/results, evidence measurement
SHA-256, conversion successes/rejections, checkpoint reviewer identities, and
the six claim statuses, plus controller attempt
`bf0d2300-4479-4e3c-ba99-bb023ee6751e`. Treat the result as a worker proposal. Do not write
coordinator state, deploy to Hugging Face, attest validation/submission, poll a
Space, import a verdict, or claim an external phase.
