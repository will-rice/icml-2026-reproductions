# AGoQ Pinned Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace AGoQ's hard-coded and synthetic outputs with deterministic,
source-pinned evidence for exact activation-memory algebra, pipeline allocation
arithmetic, and released-source tracing, while marking unavailable distributed
training tables honestly.

**Architecture:** Checked-in paper transcription and a manifest of selected files
from the exact official code revision form the immutable inputs. Small
standard-library audit modules verify every input before recomputing rational
memory and pipeline results or tracing specific source semantics. One canonical
evidence builder feeds tests, the committed JSON bundle, documentation, poster,
and read-only Gradio Space; none of those presentation surfaces calculate or
invent results independently.

**Tech Stack:** Python 3.12, Python standard library (`ast`, `dataclasses`,
`decimal`, `fractions`, `hashlib`, `json`, `pathlib`), Gradio 4.44.1, pytest,
canonical JSON, Git object IDs.

## Global Constraints

- Work only in
  `submissions/agoq-activation-and-gradient-quantization-for-memory-efficient-distributed-training-of-llms/`.
  This plan itself is already approved and must not be rewritten during
  implementation.
- Start from the controller-created clean worktree and stop unless
  `git status --porcelain` is empty. Do not copy uncommitted files from
  `.worktrees/agoq-memory-accounting`.
- Controller attempt ID is
  `2fc3b006-3307-4fc3-8df6-c000379298c4`; bind it in evidence identity,
  tests, and the worker handoff.
- Use paper `arXiv:2605.00539v2`, whose acquired PDF is 3,196,252 bytes with
  SHA-256
  `6a5095edf64e730a824fc076a0cbf3d97922b370dc827f173e872e17eb95e0d7`.
- Use only official repository `https://github.com/Eutenacity/AGoQ.git` at
  commit `006fa0f6318228d1fcd6727f0578c0e548e5cbff`.
- Treat paper transcription as `paper_context`, never as a reproduced
  measurement. Reproduced observations are only arithmetic derived from that
  transcription or source facts verified against pinned file bytes.
- Do not run training or claim the paper's Tables 2 or 3. Their hardware
  requirements (64 GPUs and 16 NVIDIA Blackwell GPUs respectively) are
  unavailable in this CPU-bounded reproduction.
- Claims 1–4 remain `partial`; claims 5–6 are `unavailable`. Do not promote a
  status merely because an arithmetic identity or source location is verified.
- Record the paper's printed pipeline equation and its reported discrete
  allocation separately. For four stages, preserve stored-batch counts in
  paper device order `(11, 9, 7, 5)`, exact raw allocations
  `(4, 44/9, 44/7, 44/5)`, and paper-reported integer allocation `(4, 5, 6, 8)`.
  Report storage products `(44, 45, 42, 40)` and the one-unit overshoot at the
  second device; do not claim exact peak equality or invent an undocumented
  rounding rule.
- A source trace may establish that quantize/dequantize calls are adjacent to
  matrix operations. It must not claim a fused single-kernel implementation
  unless such an implementation is present in the pinned source. Record the
  absent fused-kernel body as a limitation.
- Canonical evidence must exclude wall-clock timestamps, host details, random
  values, environment dumps, current Git state, absolute paths, and network
  observations.
- Use failing tests before each implementation change. Keep all tests CPU-only,
  offline, and deterministic.
- Preserve `demo.launch(server_name="0.0.0.0", server_port=7860)`.
- Space README metadata must include tags `paper-ymHDVBwmta` and
  `icml2026-repro`.
- Do not mutate coordinator state, `docs/HANDOFF.md`, the skill source, another
  submission, or NAPE. Do not access Hub credentials, deploy, submit, poll,
  import verdicts, or make controller attestations.
- Every commit below is a worker proposal. Controller validation and publication
  remain separate operations after the implementation branch is handed back.

---

### Task 1: Replace implicit constants with verified immutable inputs

**Files:**

- Create:
  `submissions/agoq-activation-and-gradient-quantization-for-memory-efficient-distributed-training-of-llms/evidence/inputs/paper_transcription.json`
- Create:
  `submissions/agoq-activation-and-gradient-quantization-for-memory-efficient-distributed-training-of-llms/evidence/inputs/upstream_manifest.json`
- Create:
  `submissions/agoq-activation-and-gradient-quantization-for-memory-efficient-distributed-training-of-llms/evidence/inputs/upstream/`
- Create:
  `submissions/agoq-activation-and-gradient-quantization-for-memory-efficient-distributed-training-of-llms/scripts/acquire_upstream.py`
- Create:
  `submissions/agoq-activation-and-gradient-quantization-for-memory-efficient-distributed-training-of-llms/src/agoq_repro/provenance.py`
- Create:
  `submissions/agoq-activation-and-gradient-quantization-for-memory-efficient-distributed-training-of-llms/tests/test_provenance.py`

**Pinned upstream files:**

| Relative path | Bytes | Git blob | SHA-256 |
|---|---:|---|---|
| `LICENSE` | 14,567 | `80e0a79551d81847f5d0e9e858415bf77547c31d` | `41e59bea3bfd8fd8f236fbda2f86266e9041d6eaa6ef3d7522bd2460f951e093` |
| `README.md` | 3,462 | `ae0df36d8d0b69cdd014c322613d02c832660431` | `817622fa7ffb6daabb1a1ab9969f3f11df5ead9e0406ddd6a40045633d734b4a` |
| `megatron/core/tensor_parallel/layers.py` | 52,929 | `2824feba8425ec726ccb98d8e2fd47d89584994c` | `d9ad74e0e5c137203f55836a75292c007fdc5b94457a36f0588935ffc22cc4c6` |
| `megatron/core/quantizer/activation_quantization.py` | 52,966 | `3e331afc6f4dd3952b9b1cf40f08f6cce0a52ba1` | `84aecf446545d0df7fd64a7209189c5104caa4a5deb2ce08a5288e7085c85ee7` |
| `megatron/core/distributed/distributed_data_parallel.py` | 13,706 | `43061bf6eb05c2dcf72fb56b945fc650e2909cfe` | `9a382929c024030f5d2a8057355904dd1555c6e0e34ac56fbe2ebbe758a59546` |
| `megatron/core/distributed/param_and_grad_buffer.py` | 43,538 | `c0ab2dca64671a28cb10bb79f638a69c0f63501e` | `b5cc3378b84794f2de204ff1cfe3299075d880b66248cbb2de4fa43faffd218e` |
| `changes_te/linear.py` | 44,429 | `925b84050b65361e593e50c44734f181c30c6d0f` | `cb8e3625defb185d779ec4a26a9f54f69478dc218f39222a0f9aad8e75368d9a` |
| `changes_te/layernorm_linear.py` | 55,176 | `b7da3015ad930eca694254070cab9da5b12f30aa` | `a48fe77bd2c7bd77c8140ae490226815052c5141c6fd7c169d14ac7516e88896` |
| `changes_te/layernorm_mlp.py` | 72,525 | `a32d9707e00d3e4666a8771000f6c119a9825501` | `733a9113799e59d1cb96f2e8d96f9030bbec70178492747c009a5c3086085675` |
| `megatron/core/pipeline_parallel/schedules.py` | 58,679 | `12484ff09549aa1d8a4ab19e9e5f59f6c6fa8dfd` | `062f50f6bbb541bb90cdff1c76a4509e75cbe6096f80b3d32fe7ef81090347ba` |

**Interfaces:**

```python
class IntegrityError(ValueError):
    """Pinned input bytes or metadata failed verification."""

@dataclass(frozen=True)
class VerifiedFile:
    path: str
    git_blob: str
    sha256: str
    size_bytes: int

canonical_sha256(path: Path) -> str
load_verified_transcription(project_root: Path) -> dict[str, object]
load_verified_sources(project_root: Path) -> Sequence[VerifiedFile]
```

- [ ] **Step 1: Add exact paper transcription and upstream manifest**

Transcribe Table 1 in units of `U = B*S*H*2 bytes`:

```json
{
  "schema_version": 1,
  "paper": {
    "arxiv_id": "2605.00539v2",
    "pdf_sha256": "6a5095edf64e730a824fc076a0cbf3d97922b370dc827f173e872e17eb95e0d7",
    "pdf_size_bytes": 3196252,
    "license": "CC BY 4.0"
  },
  "table_1_units": {
    "bf16": {"qkv": "1", "attention": "5", "linear_1": "1", "rmsnorm": "4", "ffn_1": "1", "activation": "12", "ffn_2": "4"},
    "coat": {"qkv": "1", "attention": "5", "linear_1": "1", "rmsnorm": "1", "ffn_1": "1/2", "activation": "6", "ffn_2": "2"},
    "agoq": {"qkv": "0", "attention": "5", "linear_1": "1/4", "rmsnorm": "1/2", "ffn_1": "0", "activation": "2", "ffn_2": "0"}
  },
  "pipeline": {
    "printed_equation": "N_i = n + 2*i - 1, 1 <= i <= n",
    "stored_batches_device_order_n4": [11, 9, 7, 5],
    "minimum_bits": "4",
    "reported_bits_device_order_n4": [4, 5, 6, 8]
  },
  "training_tables": {
    "table_2_required_hardware": "64 GPUs",
    "table_3_required_hardware": "16 NVIDIA Blackwell GPUs"
  }
}
```

The upstream manifest contains the repository URL, exact commit, license path,
and all ten rows in the table above. Add the corresponding exact bytes under
`evidence/inputs/upstream/`, preserving relative paths.

- [ ] **Step 2: Write failing provenance tests**

```python
def test_all_pinned_sources_verify(project_root):
    files = load_verified_sources(project_root)
    assert len(files) == 10
    assert {item.path for item in files} >= {
        "LICENSE",
        "megatron/core/tensor_parallel/layers.py",
        "megatron/core/distributed/param_and_grad_buffer.py",
    }


def test_modified_source_is_rejected(project_root, tmp_path):
    copied = copytree(project_root, tmp_path / "project")
    target = copied / "evidence/inputs/upstream/changes_te/linear.py"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(IntegrityError, match="SHA-256"):
        load_verified_sources(copied)


def test_transcription_is_bound_to_paper_hash(project_root):
    data = load_verified_transcription(project_root)
    assert data["paper"]["arxiv_id"] == "2605.00539v2"
    assert data["paper"]["pdf_sha256"] == (
        "6a5095edf64e730a824fc076a0cbf3d97922b370dc827f173e872e17eb95e0d7"
    )
```

- [ ] **Step 3: Run the provenance tests and record RED**

```bash
cd submissions/agoq-activation-and-gradient-quantization-for-memory-efficient-distributed-training-of-llms
uv run pytest -q tests/test_provenance.py
```

Expected: collection fails because `agoq_repro.provenance` does not exist.

- [ ] **Step 4: Implement fail-closed verification**

`load_verified_sources` must reject a wrong repository URL, wrong 40-character
commit, duplicate path, unsafe absolute or `..` path, absent file, SHA mismatch,
and Git blob mismatch. Compute a Git blob locally as:

```python
def git_blob_id(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()
```

Parse JSON with duplicate-key rejection and validate the transcription's exact
schema before returning it. `scripts/acquire_upstream.py` accepts
`--repository-url`, `--revision`, `--manifest`, and `--output`; it uses
`git init`, `git fetch --depth=1 <url> <revision>`, and
`git show FETCH_HEAD:<path>` in a temporary directory, verifies every byte, and
atomically replaces only the selected output files. The checked-in evidence
path remains usable without network access.

- [ ] **Step 5: Run GREEN and commit**

```bash
uv run pytest -q tests/test_provenance.py
git diff --check
git add evidence/inputs scripts/acquire_upstream.py src/agoq_repro/provenance.py tests/test_provenance.py
git commit -m "evidence: pin AGoQ paper and source inputs"
```

Expected: all provenance tests pass.

---

### Task 2: Recompute Table 1 and model memory with exact rational algebra

**Files:**

- Replace:
  `submissions/agoq-activation-and-gradient-quantization-for-memory-efficient-distributed-training-of-llms/src/agoq_repro/memory_accounting.py`
- Replace:
  `submissions/agoq-activation-and-gradient-quantization-for-memory-efficient-distributed-training-of-llms/tests/test_memory_accounting.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class LayerMemoryAudit:
    method: str
    components_u: dict[str, Fraction]
    total_u: Fraction

@dataclass(frozen=True)
class ModelProjection:
    batch: int
    sequence: int
    hidden: int
    layers: int
    bytes_per_u: int
    totals_bytes: dict[str, Fraction]

audit_table_1(transcription: Mapping[str, object]) -> Sequence[LayerMemoryAudit]
project_model(audits: Iterable[LayerMemoryAudit], batch: int, sequence: int, hidden: int, layers: int) -> ModelProjection
fraction_text(value: Fraction) -> str
```

- [ ] **Step 1: Write exact failing arithmetic tests**

```python
def test_table_1_totals_are_recomputed(project_root):
    source = load_verified_transcription(project_root)
    audits = {row.method: row for row in audit_table_1(source)}
    assert audits["bf16"].total_u == Fraction(28)
    assert audits["coat"].total_u == Fraction(33, 2)
    assert audits["agoq"].total_u == Fraction(31, 4)
    assert audits["agoq"].components_u["linear_1"] == Fraction(1, 4)


def test_model_projection_uses_u_definition(project_root):
    audits = audit_table_1(load_verified_transcription(project_root))
    result = project_model(audits, batch=2, sequence=4096, hidden=8192, layers=32)
    assert result.bytes_per_u == 134_217_728
    assert result.totals_bytes["bf16"] == 120_259_084_288
    assert result.totals_bytes["coat"] == 70_866_960_384
    assert result.totals_bytes["agoq"] == 33_285_996_544


@pytest.mark.parametrize("field", ["batch", "sequence", "hidden", "layers"])
def test_projection_rejects_nonpositive_dimensions(project_root, field):
    kwargs = {"batch": 1, "sequence": 1, "hidden": 1, "layers": 1}
    kwargs[field] = 0
    with pytest.raises(ValueError, match=field):
        project_model(
            audit_table_1(load_verified_transcription(project_root)), **kwargs
        )
```

- [ ] **Step 2: Run RED**

```bash
uv run pytest -q tests/test_memory_accounting.py
```

Expected: failures because the existing module does not accept verified
transcription or preserve `Fraction` values.

- [ ] **Step 3: Implement strict rational calculation**

Parse only integer strings or `numerator/denominator` strings with positive
denominators. Require exactly the seven component keys shown in the
transcription. Compute totals using `sum(component_values, Fraction())`; never store expected
totals in implementation code. Define `bytes_per_u = batch * sequence * hidden
* 2`, then multiply each per-layer total by `bytes_per_u * layers`.
`fraction_text` returns an integer string when the denominator is one and
`"numerator/denominator"` otherwise.

- [ ] **Step 4: Run GREEN and commit**

```bash
uv run pytest -q tests/test_memory_accounting.py tests/test_provenance.py
git diff --check
git add src/agoq_repro/memory_accounting.py tests/test_memory_accounting.py
git commit -m "evidence: recompute AGoQ memory algebra"
```

Expected: both test files pass.

---

### Task 3: Replace the pipeline heuristic with an equation audit

**Files:**

- Replace:
  `submissions/agoq-activation-and-gradient-quantization-for-memory-efficient-distributed-training-of-llms/src/agoq_repro/pipeline_allocator.py`
- Replace:
  `submissions/agoq-activation-and-gradient-quantization-for-memory-efficient-distributed-training-of-llms/tests/test_pipeline_allocator.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class PipelineStageAudit:
    device_index: int
    stored_batches: int
    raw_bits: Fraction
    reported_bits: int | None
    reported_storage_units: int | None

@dataclass(frozen=True)
class PipelineAudit:
    stage_count: int
    equation_order_counts: Sequence[int]
    device_order_counts: Sequence[int]
    stages: Sequence[PipelineStageAudit]
    target_storage_units: int
    maximum_reported_storage_units: int | None
    maximum_reported_overshoot_units: int | None
    reported_rounding_rule_available: bool

audit_pipeline(transcription: Mapping[str, object], stage_count: int) -> PipelineAudit
```

- [ ] **Step 1: Write failing equation and discrepancy tests**

```python
def test_four_stage_equation_and_reported_allocation(project_root):
    result = audit_pipeline(load_verified_transcription(project_root), 4)
    assert result.equation_order_counts == (5, 7, 9, 11)
    assert result.device_order_counts == (11, 9, 7, 5)
    assert tuple(stage.raw_bits for stage in result.stages) == (
        Fraction(4), Fraction(44, 9), Fraction(44, 7), Fraction(44, 5)
    )
    assert tuple(stage.reported_bits for stage in result.stages) == (4, 5, 6, 8)
    assert tuple(stage.reported_storage_units for stage in result.stages) == (
        44, 45, 42, 40
    )
    assert result.target_storage_units == 44
    assert result.maximum_reported_storage_units == 45
    assert result.maximum_reported_overshoot_units == 1
    assert result.reported_rounding_rule_available is False


def test_non_four_stage_case_has_no_invented_integer_policy(project_root):
    result = audit_pipeline(load_verified_transcription(project_root), 3)
    assert result.equation_order_counts == (4, 6, 8)
    assert result.device_order_counts == (8, 6, 4)
    assert all(stage.reported_bits is None for stage in result.stages)
    assert result.maximum_reported_storage_units is None


@pytest.mark.parametrize("stage_count", [0, -1, True, 2.5])
def test_invalid_stage_count_is_rejected(project_root, stage_count):
    with pytest.raises((TypeError, ValueError), match="stage_count"):
        audit_pipeline(load_verified_transcription(project_root), stage_count)
```

- [ ] **Step 2: Run RED**

```bash
uv run pytest -q tests/test_pipeline_allocator.py
```

Expected: failures because the existing heuristic exposes unsupported
average-scaling behavior instead of the paper equation.

- [ ] **Step 3: Implement exact equation audit**

For `i = 1..n`, compute printed equation counts `n + 2*i - 1`; reverse that
tuple for paper device order. Let the first device's stored-batch count be
`N_1`, minimum bits be `m`, and raw bits at a device be
`m * N_1 / N_i`. Only when `stage_count == 4` may the implementation attach the
transcribed reported tuple. It must calculate reported storage products and
overshoot, never infer `round`, `ceil`, or `floor` as the paper's policy.

- [ ] **Step 4: Run GREEN and commit**

```bash
uv run pytest -q tests/test_pipeline_allocator.py tests/test_memory_accounting.py
git diff --check
git add src/agoq_repro/pipeline_allocator.py tests/test_pipeline_allocator.py
git commit -m "evidence: audit AGoQ pipeline allocation exactly"
```

Expected: both test files pass.

---

### Task 4: Trace released activation and gradient source without overstating fusion

**Files:**

- Create:
  `submissions/agoq-activation-and-gradient-quantization-for-memory-efficient-distributed-training-of-llms/src/agoq_repro/source_audit.py`
- Create:
  `submissions/agoq-activation-and-gradient-quantization-for-memory-efficient-distributed-training-of-llms/tests/test_source_audit.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class SourceObservation:
    observation_id: str
    disposition: Literal["verified", "partial", "absent"]
    files: Sequence[str]
    symbol_names: Sequence[str]
    detail: str

audit_released_source(project_root: Path) -> Sequence[SourceObservation]
```

- [ ] **Step 1: Write failing semantic source tests**

```python
def test_activation_quantization_trace_is_source_bound(project_root):
    rows = {r.observation_id: r for r in audit_released_source(project_root)}
    row = rows["activation_quantization_integration"]
    assert row.disposition == "verified"
    assert row.files == (
        "megatron/core/tensor_parallel/layers.py",
        "megatron/core/quantizer/activation_quantization.py",
    )
    assert {"op_quantize", "op_dequantize"} <= set(row.symbol_names)


def test_gradient_collective_trace_is_source_bound(project_root):
    rows = {r.observation_id: r for r in audit_released_source(project_root)}
    assert rows["local_gradient_accumulation"].disposition == "verified"
    assert rows["all_to_all_reduce_all_gather_path"].disposition == "verified"
    assert rows["all_to_all_reduce_all_gather_path"].files == (
        "megatron/core/distributed/param_and_grad_buffer.py",
    )


def test_fused_kernel_body_is_not_claimed(project_root):
    rows = {r.observation_id: r for r in audit_released_source(project_root)}
    row = rows["single_gpu_fused_kernel_body"]
    assert row.disposition == "absent"
    assert row.files == (
        "changes_te/linear.py",
        "changes_te/layernorm_linear.py",
        "changes_te/layernorm_mlp.py",
    )
    assert "call sites" in row.detail
    assert "kernel body" in row.detail
```

- [ ] **Step 2: Run RED**

```bash
uv run pytest -q tests/test_source_audit.py
```

Expected: collection fails because `agoq_repro.source_audit` does not exist.

- [ ] **Step 3: Implement AST-backed source tracing**

First call `load_verified_sources`; no trace is valid before all pinned bytes
verify. Parse each Python file with `ast.parse`. Collect imports, function and
class names, and qualified call names. Require these source relationships:

- `layers.py` integrates activation quantization helpers from
  `activation_quantization.py`;
- the three `changes_te` files import `gact.ops` names `op_quantize` and/or
  `op_dequantize`;
- `distributed_data_parallel.py` contains the local gradient accumulation path;
- `param_and_grad_buffer.py` contains the All-to-All, local reduction, and
  AllGather quantized-gradient path;
- `schedules.py` supplies pipeline schedule context but does not establish the
  paper's discrete bit-allocation rule.

Use exact AST facts, not line numbers, because line numbers are presentation
metadata. Emit fixed observation IDs in sorted order. The fused-kernel
observation remains `absent`: the selected release exposes call sites but no
implementation body proving a single fused GPU kernel.

- [ ] **Step 4: Add mutation tests**

Copy the verified project fixture, replace the `op_quantize` identifier in one
vendored file, update neither hash nor manifest, and assert `IntegrityError`.
Then make an internally verified test fixture with the required call removed
and assert the semantic audit fails with a message naming the missing
relationship. This distinguishes byte integrity from semantic completeness.

- [ ] **Step 5: Run GREEN and commit**

```bash
uv run pytest -q tests/test_source_audit.py tests/test_provenance.py
git diff --check
git add src/agoq_repro/source_audit.py tests/test_source_audit.py
git commit -m "evidence: trace AGoQ released source semantics"
```

Expected: both test files pass.

---

### Task 5: Build one canonical six-claim evidence bundle

**Files:**

- Create:
  `submissions/agoq-activation-and-gradient-quantization-for-memory-efficient-distributed-training-of-llms/src/agoq_repro/evidence.py`
- Replace:
  `submissions/agoq-activation-and-gradient-quantization-for-memory-efficient-distributed-training-of-llms/generate_evidence.py`
- Replace:
  `submissions/agoq-activation-and-gradient-quantization-for-memory-efficient-distributed-training-of-llms/evidence.json`
- Create:
  `submissions/agoq-activation-and-gradient-quantization-for-memory-efficient-distributed-training-of-llms/tests/test_evidence.py`
- Delete:
  `submissions/agoq-activation-and-gradient-quantization-for-memory-efficient-distributed-training-of-llms/src/agoq_repro/quantization_sim.py`
- Delete:
  `submissions/agoq-activation-and-gradient-quantization-for-memory-efficient-distributed-training-of-llms/tests/test_quantization_sim.py`

**Claim IDs and immutable hashes:**

```text
claim-1 0b198b87a5abf16409a547a6f5277a41a62eac4a791b71cada94b054c65a1a13
claim-2 89292ed940125355f402bc04bc847acbed65f01bd0718124cceb88416ec24228
claim-3 a5a088563e0ab1a912f212da4246d90e8df679e6312e494ec486f0c38953b5bf
claim-4 88c789000f385b4692435064cb66b427ecbfd05b92c0632adde7681cb7b69eaa
claim-5 a513e6751344f810d77db2b7cd9a2fac9cf9ceab94f2a583a0247f917e64145d
claim-6 7391424029d3da524d5b5dfe17c88119ee6b0b7d1808ec6d0bc80366630efd1a
```

**Exact ordered live claim texts:**

```python
LIVE_CLAIMS = (
    "AGoQ combines layer-aware activation quantization and precision-preserved gradient quantization within Megatron-LM-style distributed training (Section 3).",
    "Layer-aware activation quantization reduces cached activation memory for a transformer layer from 28U in BF16 and 16.5U in COAT to 7.75U (Table 1).",
    "Dynamic Bit-width Compensation for Pipeline Parallelism assigns higher activation bit-widths to underutilized pipeline stages while maintaining near-4-bit activation storage (Section 4.2).",
    "Kernel fusion combines quantization/dequantization with adjacent GEMM operations to reduce activation-quantization overhead (Figure 4).",
    "On LLaMA2-13B sequence lengths from 32K to 80K, AGoQ reports faster training than Megatron-LM and ZeRO-1 while avoiding activation recomputation in the listed settings (Table 2).",
    "AGoQ reports lower memory than COAT and comparable or faster training time on OLMo-1B at 24K and 32K sequence lengths (Table 3).",
)
```

**Interfaces:**

```python
build_evidence(project_root: Path) -> dict[str, object]
canonical_json_bytes(evidence: Mapping[str, object]) -> bytes
write_evidence(project_root: Path, output: Path) -> None
```

- [ ] **Step 1: Write failing bundle tests**

```python
def test_bundle_has_all_live_claims_in_order(project_root):
    evidence = build_evidence(project_root)
    assert evidence["schema_version"] == 3
    assert evidence["identity"]["attempt_id"] == (
        "2fc3b006-3307-4fc3-8df6-c000379298c4"
    )
    assert [c["claim_id"] for c in evidence["claims"]] == [
        "claim-1", "claim-2", "claim-3",
        "claim-4", "claim-5", "claim-6",
    ]
    assert [c["claim"] for c in evidence["claims"]] == list(LIVE_CLAIMS)
    assert [c["challenge_claim_sha256"] for c in evidence["claims"]] == [
        "0b198b87a5abf16409a547a6f5277a41a62eac4a791b71cada94b054c65a1a13",
        "89292ed940125355f402bc04bc847acbed65f01bd0718124cceb88416ec24228",
        "a5a088563e0ab1a912f212da4246d90e8df679e6312e494ec486f0c38953b5bf",
        "88c789000f385b4692435064cb66b427ecbfd05b92c0632adde7681cb7b69eaa",
        "a513e6751344f810d77db2b7cd9a2fac9cf9ceab94f2a583a0247f917e64145d",
        "7391424029d3da524d5b5dfe17c88119ee6b0b7d1808ec6d0bc80366630efd1a",
    ]
    assert [c["status"] for c in evidence["claims"]] == [
        "partial", "partial", "partial", "partial",
        "unavailable", "unavailable",
    ]


def test_bundle_separates_context_observations_and_limitations(project_root):
    evidence = build_evidence(project_root)
    assert evidence["paper_context"]["table_1"]["agoq_total_u"] == "31/4"
    assert evidence["reproduced_observations"]["table_1"]["agoq_total_u"] == "31/4"
    assert evidence["reproduced_observations"]["pipeline"]["maximum_reported_overshoot_units"] == 1
    assert evidence["limitations"]["single_gpu_fused_kernel_body"] == (
        "Call sites are present, but a fused GPU kernel implementation body "
        "is not present in the pinned selected source."
    )
    assert "64 GPUs" in evidence["limitations"]["claim-5"]
    assert "16 NVIDIA Blackwell GPUs" in evidence["limitations"]["claim-6"]


def test_bundle_is_deterministic_and_contains_no_synthetic_fields(project_root):
    first = canonical_json_bytes(build_evidence(project_root))
    second = canonical_json_bytes(build_evidence(project_root))
    assert first == second
    text = first.decode()
    for forbidden in (
        "generated_at", "timestamp", "hostname", "git_head",
        "random_seed", "quantization_error", "proxy_score",
    ):
        assert forbidden not in text
```

- [ ] **Step 2: Run RED**

```bash
uv run pytest -q tests/test_evidence.py
```

Expected: collection fails because `agoq_repro.evidence` does not exist.

- [ ] **Step 3: Implement canonical evidence composition**

`build_evidence` must call all three verified audits. Its top-level keys are:
`schema_version`, `identity`, `upstream`, `paper_context`,
`reproduced_observations`, `claims`, and `limitations`. Use the exact paper and
repository pins from Task 1 and set `identity.attempt_id` exactly to
`2fc3b006-3307-4fc3-8df6-c000379298c4`. Store all fractions as canonical
strings and source files as repository-relative paths.

Map evidence conservatively:

- claim 1: activation-quantization and precision-preserved-gradient source
  traces, including local accumulation and All-to-All/reduce/AllGather paths;
- claim 2: Table 1 exact rational activation-memory arithmetic;
- claim 3: exact four-stage pipeline equation audit, including the discrete
  allocation discrepancy;
- claim 4: adjacent quantize/dequantize and GEMM call sites, explicit absent
  fused-kernel body, and no measured overhead-reduction claim;
- claim 5: unavailable Table 2 distributed-training measurements;
- claim 6: unavailable Table 3 Blackwell measurements.

Serialize with:

```python
def canonical_json_bytes(evidence):
    return (
        json.dumps(evidence, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
```

`write_evidence` writes to a sibling temporary file, `fsync`s it, and uses
`Path.replace` for atomic publication. The CLI accepts only `--output`, resolves
the project directory from `generate_evidence.py`, and prints the output's
SHA-256. It performs no network or Git calls.

- [ ] **Step 4: Generate twice and prove byte identity**

```bash
uv run python generate_evidence.py --output evidence.json
cp evidence.json /tmp/agoq-evidence-first.json
uv run python generate_evidence.py --output evidence.json
cmp /tmp/agoq-evidence-first.json evidence.json
sha256sum evidence.json
```

Expected: `cmp` exits zero. Record the resulting SHA-256 in the implementation
handoff; do not hard-code it back into the bundle.

- [ ] **Step 5: Remove synthetic simulation and run GREEN**

Remove every import and reference to `quantization_sim`, NumPy random sampling,
quantization-error proxies, and generated timestamps.

```bash
uv run pytest -q tests/test_evidence.py tests/test_memory_accounting.py \
  tests/test_pipeline_allocator.py tests/test_source_audit.py tests/test_provenance.py
git diff --check
git add evidence.json generate_evidence.py src/agoq_repro tests
git commit -m "evidence: emit deterministic AGoQ claim bundle"
```

Expected: all evidence tests pass and the deleted simulator stays absent.

---

### Task 6: Make README, poster, and Space read the same evidence

**Files:**

- Replace:
  `submissions/agoq-activation-and-gradient-quantization-for-memory-efficient-distributed-training-of-llms/app.py`
- Replace:
  `submissions/agoq-activation-and-gradient-quantization-for-memory-efficient-distributed-training-of-llms/README.md`
- Replace:
  `submissions/agoq-activation-and-gradient-quantization-for-memory-efficient-distributed-training-of-llms/DESIGN.md`
- Create:
  `submissions/agoq-activation-and-gradient-quantization-for-memory-efficient-distributed-training-of-llms/POSTER.md`
- Create:
  `submissions/agoq-activation-and-gradient-quantization-for-memory-efficient-distributed-training-of-llms/tests/test_app.py`
- Modify:
  `submissions/agoq-activation-and-gradient-quantization-for-memory-efficient-distributed-training-of-llms/pyproject.toml`
- Modify:
  `submissions/agoq-activation-and-gradient-quantization-for-memory-efficient-distributed-training-of-llms/uv.lock`
- Delete generated tracked directories:
  `submissions/agoq-activation-and-gradient-quantization-for-memory-efficient-distributed-training-of-llms/agoq_repro.egg-info/`
  and
  `submissions/agoq-activation-and-gradient-quantization-for-memory-efficient-distributed-training-of-llms/src/agoq_repro.egg-info/`

**Interfaces:**

```python
load_committed_evidence(path: Path = PROJECT_ROOT / "evidence.json") -> dict[str, object]
evidence_summary() -> tuple[list[list[str]], list[list[str]], str]
create_demo() -> gr.Blocks
```

- [ ] **Step 1: Write failing presentation-contract tests**

```python
def test_space_metadata_and_launch_contract():
    readme = (PROJECT_ROOT / "README.md").read_text()
    app = (PROJECT_ROOT / "app.py").read_text()
    assert "paper-ymHDVBwmta" in readme
    assert "icml2026-repro" in readme
    assert 'server_name="0.0.0.0"' in app
    assert "server_port=7860" in app


def test_space_uses_committed_evidence_without_random_simulation():
    source = (PROJECT_ROOT / "app.py").read_text()
    assert "evidence.json" in source
    assert "quantization_sim" not in source
    assert "numpy" not in source
    claims, memory, limitation = evidence_summary()
    assert [row[0] for row in claims] == [
        "claim-1", "claim-2", "claim-3",
        "claim-4", "claim-5", "claim-6",
    ]
    assert any(row[:2] == ["agoq", "31/4"] for row in memory)
    assert "one-unit" in limitation


def test_docs_do_not_present_training_tables_as_reproduced():
    for name in ("README.md", "POSTER.md"):
        text = (PROJECT_ROOT / name).read_text()
        assert "unavailable" in text.lower()
        assert "64 GPUs" in text
        assert "16 NVIDIA Blackwell GPUs" in text
        assert "reproduced throughput" not in text.lower()
```

- [ ] **Step 2: Run RED**

```bash
uv run pytest -q tests/test_app.py
```

Expected: failures for missing `POSTER.md`, stale tags, and simulator UI.

- [ ] **Step 3: Replace the app with a read-only evidence explorer**

At import, verify `evidence.json` is canonical by rebuilding it and comparing
bytes; fail with a visible integrity message on mismatch. `evidence_summary`
returns claim ID/status/basis rows, memory method/total rows, and a pipeline
limitation string. `create_demo` has four tabs:

1. **Claim Status** — all six hashes, statuses, evidence basis, and limitations;
2. **Exact Memory Algebra** — seven components and recomputed totals for BF16,
   COAT, and AGoQ;
3. **Pipeline Audit** — equation order, device order, raw fractions, reported
   integers, storage products, and one-unit discrepancy;
4. **Pinned Source Trace** — repository commit, file SHA-256 values, verified
   semantic observations, and absent fused-kernel body.

Inputs for optional model projection must be positive integers and call
`project_model`; no result may be stored as claim evidence. Remove every random
simulation, chart of synthetic error, and heuristic allocation control.

- [ ] **Step 4: Rewrite README, approved-design record, and poster**

README frontmatter uses:

```yaml
---
title: AGoQ Evidence Audit
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
tags:
  - paper-ymHDVBwmta
  - icml2026-repro
---
```

README and POSTER contain the exact upstream pins, Table 1 component arithmetic,
pipeline discrepancy, source-trace limits, six claim dispositions, offline
generation/test commands, and a clear distinction between `paper_context` and
`reproduced_observations`. They state that Tables 2 and 3 are unavailable and
quote no throughput, accuracy, or convergence number as reproduced.

Update DESIGN to document the implemented immutable-input → verified-audit →
canonical-evidence → presentation flow, without changing the approved scope.

- [ ] **Step 5: Remove obsolete dependencies and generated metadata**

Remove NumPy and any simulator-only dependency. Keep Gradio exactly `4.44.1`
and place pytest in the development dependency group. Regenerate the lock:

```bash
uv lock
```

Delete both tracked `*.egg-info` directories. Ensure the repository ignore rules
cover `*.egg-info/` before committing; if the root rule already does, do not add
a duplicate project rule.

- [ ] **Step 6: Run GREEN and commit**

```bash
uv run pytest -q tests/test_app.py tests/test_evidence.py
uv run python -c "import app; assert app.create_demo() is not None"
git diff --check
git add app.py README.md DESIGN.md POSTER.md pyproject.toml uv.lock tests/test_app.py
git add -u agoq_repro.egg-info src/agoq_repro.egg-info
git commit -m "docs: present AGoQ evidence without synthetic claims"
```

Expected: presentation tests pass and the app constructs without launching a
server.

---

### Task 7: Final deterministic validation and worker handoff

**Files:**

- Verify all changed files in the AGoQ submission.
- Do not create a controller attestation or edit coordinator files.

- [ ] **Step 1: Run the complete submission suite twice**

```bash
cd submissions/agoq-activation-and-gradient-quantization-for-memory-efficient-distributed-training-of-llms
uv run pytest -q
uv run python generate_evidence.py --output evidence.json
cp evidence.json /tmp/agoq-evidence-validation.json
uv run pytest -q
uv run python generate_evidence.py --output evidence.json
cmp /tmp/agoq-evidence-validation.json evidence.json
```

Expected: both suites pass and `cmp` exits zero.

- [ ] **Step 2: Run repository-wide formatting and policy checks**

From the workspace root:

```bash
mapfile -t AGOQ_FILES < <(git ls-files -co --exclude-standard -- submissions/agoq-activation-and-gradient-quantization-for-memory-efficient-distributed-training-of-llms)
env UV_CACHE_DIR=/tmp/icml-agoq-uv-cache \
  PRE_COMMIT_HOME=/tmp/icml-agoq-pre-commit \
  uv run pre-commit run --files "${AGOQ_FILES[@]}"
```

Expected: all hooks pass with only the explicitly enumerated AGoQ paths.

- [ ] **Step 3: Audit forbidden content and exact pins**

```bash
rg -n 'generated_at|timestamp|hostname|git_head|random_seed|quantization_error|proxy_score' \
  submissions/agoq-activation-and-gradient-quantization-for-memory-efficient-distributed-training-of-llms
rg -n '006fa0f6318228d1fcd6727f0578c0e548e5cbff|2605\.00539v2|paper-ymHDVBwmta|icml2026-repro' \
  submissions/agoq-activation-and-gradient-quantization-for-memory-efficient-distributed-training-of-llms
git diff --check
```

Expected: the first search has no matches outside negative test assertions; the
second finds all four pins in the intended evidence and presentation files.

- [ ] **Step 4: Inspect only the proposal diff and commit final generated bytes**

```bash
git status --short
git diff --stat
git diff -- evidence.json README.md POSTER.md app.py
git add evidence.json
git commit -m "test: validate deterministic AGoQ evidence"
```

Expected: no coordinator state, `docs/HANDOFF.md`, skill source, other
submission, or credential file appears in the proposal. If `evidence.json` is
already unchanged, omit the empty commit.

- [ ] **Step 5: Prepare the controller handoff**

Report:

- proposal branch and final commit;
- controller attempt `2fc3b006-3307-4fc3-8df6-c000379298c4`;
- exact paper, repository, and ten file hashes;
- generated `evidence.json` SHA-256;
- full pytest and pre-commit commands with exit status;
- claim statuses `partial, partial, partial, partial, unavailable, unavailable`;
- the pipeline one-unit discrepancy and absent fused-kernel-body limitation;
- confirmation that no training, Hub mutation, deployment, submission,
  coordinator-state mutation, or controller attestation was performed.

The controller must independently inspect the immutable live snapshot, validate
the worker proposal, generate its attestation, and perform any authorized
publication.
