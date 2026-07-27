# mHC Evidence Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce deterministic CPU evidence for mHC's pinned Sinkhorn/manifold invariants, dimensional ablations, and explicitly toy-only propagation behavior, while marking training, systems, and 27B claims unavailable.

**Architecture:** Keep the existing small PyTorch package and separate three responsibilities: numerical projection diagnostics, dimension/variant sweeps, and paired toy residual propagation. A pure evidence builder will bind all five live challenge claims to exact text and hashes, while the README, poster, and Gradio Space render only generated evidence and its limitations.

**Tech Stack:** Python 3.12, PyTorch, NumPy, pytest, Gradio 6.20.0, JSON, CSV, static HTML

## Global Constraints

- Work only in `submissions/mhc-manifold-constrained-hyper-connections/` inside the assigned mHC worktree.
- Start from the controller-created clean worktree and stop unless `git status --porcelain` is empty. Do not reuse or copy any uncommitted file from `.worktrees/mhc-manifold-connections`, which contains dirty mHC and unrelated submission changes.
- Controller attempt ID is `3d164e18-39ef-416e-b986-96b5a5d4e12d`; bind it in evidence identity, tests, and the worker handoff.
- Pin upstream context to `arxiv:2512.24880v2+github:tokenbender/mHC-manifold-constrained-hyper-connections@ad20d0d8db4d6fc7e8d9b148281167141da20d47`.
- Execute only deterministic CPU calculations with paid API cost USD 0.00.
- Treat seeded random matrices and synthetic tensors as toy evidence, never as trained-checkpoint, loss-gap, systems-overhead, or 27B benchmark measurements.
- Claims 1–3 must have status `partial`; Claims 2–3 use `toy` only as `evidence_kind`. Claims 4–5 must have status `unavailable`.
- Bind all five claims to their exact live challenge text and SHA-256 listed in Task 4.
- Generated JSON must be strict RFC-compatible JSON: no `NaN`, `Infinity`, or `-Infinity`.
- Do not modify `state/`, `docs/HANDOFF.md`, the reproduction-loop skill, another submission, or any Hub resource.
- Do not deploy, submit, poll, import a verdict, or claim controller validation.

---

## File Structure

- `mhc_repro/sinkhorn.py`: stable Sinkhorn projection and reusable invariant diagnostics.
- `mhc_repro/ablation.py`: deterministic sweep across stream counts, hidden dimensions, seeds, and the eight mapping variants.
- `mhc_repro/propagation.py`: paired toy residual-matrix composition only; no training or checkpoint language.
- `mhc_repro/cli.py`: pure evidence construction plus deterministic JSON/CSV serialization.
- `tests/test_sinkhorn.py`: projection input validation and manifold invariant tests.
- `tests/test_ablation.py`: complete dimensional-grid and determinism tests.
- `tests/test_propagation.py`: paired toy propagation and honest-scope tests.
- `tests/test_cli.py`: exact claim binding, statuses, provenance, finite JSON, and byte reproducibility.
- `app.py`: read-only Gradio presentation of committed generated evidence.
- `tests/test_app.py`: Space metadata and evidence-summary contract.
- `poster.html`: static poster that labels computed, toy, and unavailable evidence distinctly.
- `tests/test_poster.py`: poster fidelity and prohibited-claim regression checks.
- `README.md`: commands, exact scope, provenance, and limitations.
- `pyproject.toml`, `uv.lock`: pin Gradio and retain a frozen executable environment.
- `evidence.json`, `summary.csv`: regenerated canonical evidence artifacts.

### Task 1: Harden Sinkhorn Projection and Manifold Diagnostics

**Files:**
- Modify: `submissions/mhc-manifold-constrained-hyper-connections/mhc_repro/sinkhorn.py`
- Modify: `submissions/mhc-manifold-constrained-hyper-connections/tests/test_sinkhorn.py`

**Interfaces:**
- Consumes: square `torch.Tensor` logits with shape `(..., K, K)`.
- Produces: `sinkhorn_knopp_projection(logits, n_iters=100, eps=1e-12) -> torch.Tensor`.
- Produces: `projection_diagnostics(matrix, atol=1e-6) -> dict[str, float | bool]` with keys `nonnegative`, `max_row_error`, `max_column_error`, `spectral_norm`, and `is_doubly_stochastic`.

- [ ] **Step 1: Add failing validation and invariant tests**

```python
import pytest
import torch

from mhc_repro.sinkhorn import (
    projection_diagnostics,
    sinkhorn_knopp_projection,
)


@pytest.mark.parametrize("stream_count", [2, 4, 8])
def test_projection_diagnostics_hold_across_stream_counts(stream_count):
    generator = torch.Generator().manual_seed(42 + stream_count)
    logits = torch.randn(
        3, stream_count, stream_count, dtype=torch.float64, generator=generator
    )
    projected = sinkhorn_knopp_projection(logits)
    diagnostics = projection_diagnostics(projected)

    assert diagnostics["nonnegative"] is True
    assert diagnostics["max_row_error"] <= 1e-6
    assert diagnostics["max_column_error"] <= 1e-6
    assert diagnostics["spectral_norm"] <= 1.0 + 1e-6
    assert diagnostics["is_doubly_stochastic"] is True


@pytest.mark.parametrize(
    ("logits", "n_iters", "message"),
    [
        (torch.randn(2, 3), 100, "square"),
        (torch.randn(2, 2), 0, "positive"),
    ],
)
def test_projection_rejects_invalid_inputs(logits, n_iters, message):
    with pytest.raises(ValueError, match=message):
        sinkhorn_knopp_projection(logits, n_iters=n_iters)
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
uv run python -m pytest tests/test_sinkhorn.py -q
```

Expected: FAIL because `projection_diagnostics` does not exist and invalid inputs are accepted.

- [ ] **Step 3: Implement stable projection and diagnostics**

```python
def sinkhorn_knopp_projection(
    logits: torch.Tensor,
    n_iters: int = 100,
    eps: float = 1e-12,
) -> torch.Tensor:
    if logits.ndim < 2 or logits.shape[-1] != logits.shape[-2]:
        raise ValueError("logits must contain square matrices")
    if n_iters < 1:
        raise ValueError("n_iters must be positive")
    if eps <= 0:
        raise ValueError("eps must be positive")

    matrix = torch.exp(logits - logits.amax(dim=(-2, -1), keepdim=True))
    for _ in range(n_iters):
        matrix = matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(eps)
        matrix = matrix / matrix.sum(dim=-2, keepdim=True).clamp_min(eps)
    return matrix


def projection_diagnostics(
    matrix: torch.Tensor,
    atol: float = 1e-6,
) -> dict[str, float | bool]:
    row_error = (matrix.sum(dim=-1) - 1.0).abs().max()
    column_error = (matrix.sum(dim=-2) - 1.0).abs().max()
    spectral_norm = torch.linalg.matrix_norm(matrix, ord=2).max()
    nonnegative = bool(torch.all(matrix >= 0).item())
    return {
        "nonnegative": nonnegative,
        "max_row_error": float(row_error.item()),
        "max_column_error": float(column_error.item()),
        "spectral_norm": float(spectral_norm.item()),
        "is_doubly_stochastic": (
            nonnegative
            and row_error.item() <= atol
            and column_error.item() <= atol
        ),
    }
```

Keep `is_doubly_stochastic()` as a compatibility wrapper returning
`bool(projection_diagnostics(matrix, atol)["is_doubly_stochastic"])`.

- [ ] **Step 4: Run focused tests**

Run:

```bash
uv run python -m pytest tests/test_sinkhorn.py tests/test_layers.py -q
```

Expected: all projection and layer tests PASS.

- [ ] **Step 5: Commit the independently reviewable invariant change**

```bash
git add submissions/mhc-manifold-constrained-hyper-connections/mhc_repro/sinkhorn.py submissions/mhc-manifold-constrained-hyper-connections/tests/test_sinkhorn.py
git commit -m "test(mhc): harden Sinkhorn manifold invariants"
```

### Task 2: Add Deterministic Dimensional Ablations

**Files:**
- Modify: `submissions/mhc-manifold-constrained-hyper-connections/mhc_repro/ablation.py`
- Modify: `submissions/mhc-manifold-constrained-hyper-connections/tests/test_ablation.py`

**Interfaces:**
- Consumes: Task 1's `projection_diagnostics`.
- Produces: `run_dimensional_ablations(stream_counts=(2, 4, 8), hidden_dims=(8, 16, 32), seeds=(17, 42, 123), n_samples=8, n_sinkhorn_iters=100) -> list[dict[str, object]]`.
- Each result contains `seed`, `stream_count`, `hidden_dim`, `variant_name`, `pre_mode`, `post_mode`, `res_mode`, `expected_shape`, `observed_shape`, `output_shape_valid`, and `residual_projection`.

- [ ] **Step 1: Add a failing complete-grid test**

```python
from mhc_repro.ablation import run_dimensional_ablations


def test_dimensional_ablations_cover_complete_grid_deterministically():
    kwargs = {
        "stream_counts": (2, 4),
        "hidden_dims": (8, 16),
        "seeds": (17, 42),
        "n_samples": 3,
        "n_sinkhorn_iters": 100,
    }
    first = run_dimensional_ablations(**kwargs)
    second = run_dimensional_ablations(**kwargs)

    assert first == second
    assert len(first) == 2 * 2 * 2 * 8
    assert all(row["output_shape_valid"] for row in first)
    assert {
        (row["stream_count"], row["hidden_dim"], row["seed"])
        for row in first
    } == {
        (stream_count, hidden_dim, seed)
        for stream_count in (2, 4)
        for hidden_dim in (8, 16)
        for seed in (17, 42)
    }
```

- [ ] **Step 2: Run the ablation tests and verify failure**

Run:

```bash
uv run python -m pytest tests/test_ablation.py -q
```

Expected: FAIL because `run_dimensional_ablations` is undefined.

- [ ] **Step 3: Implement the explicit dimensional sweep**

Define one module-level immutable `VARIANT_CONFIGS` tuple containing the
existing eight named `(pre_mode, post_mode, res_mode)` configurations. Make
`run_component_ablations()` delegate to the new sweep for one dimension/seed,
then implement:

```python
def run_dimensional_ablations(
    stream_counts: tuple[int, ...] = (2, 4, 8),
    hidden_dims: tuple[int, ...] = (8, 16, 32),
    seeds: tuple[int, ...] = (17, 42, 123),
    n_samples: int = 8,
    n_sinkhorn_iters: int = 100,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        for stream_count in stream_counts:
            for hidden_dim in hidden_dims:
                generator = torch.Generator().manual_seed(seed)
                x = torch.randn(
                    n_samples,
                    stream_count,
                    hidden_dim,
                    generator=generator,
                )
                torch.manual_seed(seed)
                for name, pre_mode, post_mode, res_mode in VARIANT_CONFIGS:
                    layer = ManifoldHyperConnectionLayer(
                        K=stream_count,
                        d_model=hidden_dim,
                        n_sinkhorn_iters=n_sinkhorn_iters,
                        pre_mode=pre_mode,
                        post_mode=post_mode,
                        res_mode=res_mode,
                    )
                    output = layer(x)
                    expected_shape = [n_samples, stream_count, hidden_dim]
                    diagnostics = projection_diagnostics(
                        layer.get_effective_residual_matrix()
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "stream_count": stream_count,
                            "hidden_dim": hidden_dim,
                            "variant_name": name,
                            "pre_mode": pre_mode,
                            "post_mode": post_mode,
                            "res_mode": res_mode,
                            "expected_shape": expected_shape,
                            "observed_shape": list(output.shape),
                            "output_shape_valid": list(output.shape) == expected_shape,
                            "residual_projection": diagnostics,
                        }
                    )
    return rows
```

Do not report these rows as Table 1 loss ablations. They establish only shape
preservation and, for manifold residual variants, projection invariants.

- [ ] **Step 4: Run ablation and layer tests**

Run:

```bash
uv run python -m pytest tests/test_ablation.py tests/test_layers.py -q
```

Expected: all tests PASS and the default sweep produces `3 * 3 * 3 * 8 = 216`
records.

- [ ] **Step 5: Commit the dimensional evidence unit**

```bash
git add submissions/mhc-manifold-constrained-hyper-connections/mhc_repro/ablation.py submissions/mhc-manifold-constrained-hyper-connections/tests/test_ablation.py
git commit -m "feat(mhc): add dimensional mapping ablations"
```

### Task 3: Replace Training-Like Evidence With Paired Toy Propagation

**Files:**
- Modify: `submissions/mhc-manifold-constrained-hyper-connections/mhc_repro/propagation.py`
- Modify: `submissions/mhc-manifold-constrained-hyper-connections/tests/test_propagation.py`

**Interfaces:**
- Produces: `evaluate_toy_propagation(depths=(10, 50, 100), stream_counts=(2, 4, 8), seeds=(17, 42, 123), n_sinkhorn_iters=100) -> list[dict[str, float | int | str]]`.
- Each record contains paired raw/projected results from the same seeded logits: `seed`, `stream_count`, `depth`, `evidence_kind`, `unconstrained_forward_amax`, `unconstrained_backward_amax`, `projected_forward_amax`, and `projected_backward_amax`.
- `evidence_kind` is always `toy_random_matrix_propagation`.

- [ ] **Step 1: Add failing paired-propagation scope tests**

```python
import math

import pytest

from mhc_repro.propagation import evaluate_toy_propagation


def test_toy_propagation_is_paired_bounded_and_explicitly_toy():
    rows = evaluate_toy_propagation(
        depths=(10, 50),
        stream_counts=(2, 4),
        seeds=(17, 42),
        n_sinkhorn_iters=100,
    )

    assert len(rows) == 2 * 2 * 2
    assert all(row["evidence_kind"] == "toy_random_matrix_propagation" for row in rows)
    assert all(math.isfinite(value) for row in rows for value in (
        row["projected_forward_amax"],
        row["projected_backward_amax"],
    ))
    for row in rows:
        assert row["projected_forward_amax"] == pytest.approx(1.0, abs=1e-5)
        assert row["projected_backward_amax"] == pytest.approx(1.0, abs=1e-5)


def test_toy_propagation_rejects_nonpositive_depths():
    with pytest.raises(ValueError, match="positive"):
        evaluate_toy_propagation(depths=(0,))
```

- [ ] **Step 2: Run propagation tests and verify failure**

Run:

```bash
uv run python -m pytest tests/test_propagation.py -q
```

Expected: FAIL because `evaluate_toy_propagation` is undefined.

- [ ] **Step 3: Implement paired matrix composition**

```python
def evaluate_toy_propagation(
    depths: tuple[int, ...] = (10, 50, 100),
    stream_counts: tuple[int, ...] = (2, 4, 8),
    seeds: tuple[int, ...] = (17, 42, 123),
    n_sinkhorn_iters: int = 100,
) -> list[dict[str, float | int | str]]:
    requested_depths = tuple(sorted(set(depths)))
    if not requested_depths or requested_depths[0] < 1:
        raise ValueError("depths must contain positive integers")

    rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        for stream_count in stream_counts:
            generator = torch.Generator().manual_seed(seed)
            raw_composite = torch.eye(stream_count, dtype=torch.float64)
            projected_composite = torch.eye(stream_count, dtype=torch.float64)
            for depth in range(1, requested_depths[-1] + 1):
                logits = torch.randn(
                    stream_count,
                    stream_count,
                    dtype=torch.float64,
                    generator=generator,
                )
                raw_composite = logits @ raw_composite
                projected_composite = (
                    sinkhorn_knopp_projection(
                        logits,
                        n_iters=n_sinkhorn_iters,
                    )
                    @ projected_composite
                )
                if depth in requested_depths:
                    raw = amax_gain_magnitudes(raw_composite)
                    projected = amax_gain_magnitudes(projected_composite)
                    rows.append(
                        {
                            "seed": seed,
                            "stream_count": stream_count,
                            "depth": depth,
                            "evidence_kind": "toy_random_matrix_propagation",
                            "unconstrained_forward_amax": raw["forward_amax"],
                            "unconstrained_backward_amax": raw["backward_amax"],
                            "projected_forward_amax": projected["forward_amax"],
                            "projected_backward_amax": projected["backward_amax"],
                        }
                    )
    return rows
```

Retain `evaluate_signal_propagation()` only for backward compatibility, but
remove it from evidence generation. It must not supply training-loss,
gradient-norm, or full-model conclusions.

- [ ] **Step 4: Run propagation tests**

Run:

```bash
uv run python -m pytest tests/test_propagation.py -q
```

Expected: all tests PASS.

- [ ] **Step 5: Commit the toy-only propagation unit**

```bash
git add submissions/mhc-manifold-constrained-hyper-connections/mhc_repro/propagation.py submissions/mhc-manifold-constrained-hyper-connections/tests/test_propagation.py
git commit -m "fix(mhc): scope propagation evidence to paired toy matrices"
```

### Task 4: Build Strict Five-Claim Evidence and Regenerate Artifacts

**Files:**
- Modify: `submissions/mhc-manifold-constrained-hyper-connections/mhc_repro/cli.py`
- Modify: `submissions/mhc-manifold-constrained-hyper-connections/tests/test_cli.py`
- Modify: `submissions/mhc-manifold-constrained-hyper-connections/evidence.json`
- Modify: `submissions/mhc-manifold-constrained-hyper-connections/summary.csv`

**Interfaces:**
- Consumes: `projection_diagnostics`, `run_dimensional_ablations`, and `evaluate_toy_propagation`.
- Produces: `build_evidence(n_sinkhorn_iters=100) -> dict[str, object]`.
- Produces: `write_evidence(bundle, output_json, output_csv) -> None`.
- CLI retains `--output-json`, `--output-csv`, and `--n-iters`; remove `--depth` because depths are the fixed audited tuple `(10, 50, 100)`.

- [ ] **Step 1: Add failing exact-schema and reproducibility tests**

```python
import json

from mhc_repro.cli import build_evidence, write_evidence


EXPECTED_BINDINGS = {
    "claim-1": "bdf296450b900b06ca2efbc1ffe702d9547371e3d847e6a31c71d977e0bfa052",
    "claim-2": "fa35812d9e1626bcfe1702f946f3926128f6012071af6ef378e0241e30881823",
    "claim-3": "15537486e4923b51864ce7b52999581519cde779ca9e03eda2ad93117abc9735",
    "claim-4": "2fd1e3570d1437de16597b0b942dc8d2f4a0045e84fd066bf01da44a86c86959",
    "claim-5": "f67e1f1f781f58d9e6c928002254a5dda7d94078db32408893971d6985094a02",
}


def test_evidence_binds_all_live_claims_with_honest_statuses():
    bundle = build_evidence()
    claims = {claim["claim_id"]: claim for claim in bundle["claims"]}

    assert bundle["attempt_id"] == "3d164e18-39ef-416e-b986-96b5a5d4e12d"
    assert {claim_id: claim["status"] for claim_id, claim in claims.items()} == {
        "claim-1": "partial",
        "claim-2": "partial",
        "claim-3": "partial",
        "claim-4": "unavailable",
        "claim-5": "unavailable",
    }
    assert {
        claim_id: claim["challenge_claim_sha256"]
        for claim_id, claim in claims.items()
    } == EXPECTED_BINDINGS
    assert claims["claim-3"]["evidence_kind"] == "toy_random_matrix_propagation"
    assert "27B" in claims["claim-5"]["limitation"]


def test_serialized_evidence_is_strict_and_byte_reproducible(tmp_path):
    bundle = build_evidence()
    first_json = tmp_path / "first.json"
    first_csv = tmp_path / "first.csv"
    second_json = tmp_path / "second.json"
    second_csv = tmp_path / "second.csv"

    write_evidence(bundle, first_json, first_csv)
    write_evidence(build_evidence(), second_json, second_csv)

    assert first_json.read_bytes() == second_json.read_bytes()
    assert first_csv.read_bytes() == second_csv.read_bytes()
    json.loads(
        first_json.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
```

- [ ] **Step 2: Run CLI tests and verify failure**

Run:

```bash
uv run python -m pytest tests/test_cli.py -q
```

Expected: FAIL because the builder/writer interfaces and five-claim schema do
not exist.

- [ ] **Step 3: Define exact immutable claim bindings**

Add this immutable constant:

```python
LIVE_CLAIMS = (
    (
        "claim-1",
        "mHC constrains hyper-connection residual mappings by projecting them onto a manifold to preserve stability relative to unconstrained HC (Figure 1, Section 4.1).",
        "bdf296450b900b06ca2efbc1ffe702d9547371e3d847e6a31c71d977e0bfa052",
    ),
    (
        "claim-2",
        "HC component ablations test the roles of pre, post, and residual mappings while maintaining dimensional consistency with fixed mappings (Table 1).",
        "fa35812d9e1626bcfe1702f946f3926128f6012071af6ef378e0241e30881823",
    ),
    (
        "claim-3",
        "Training and propagation analyses show unconstrained HC has larger loss gaps, gradient norms, and residual propagation instability than mHC (Figure 2, Figure 3, Figure 5, Figure 7).",
        "15537486e4923b51864ce7b52999581519cde779ca9e03eda2ad93117abc9735",
    ),
    (
        "claim-4",
        "The paper introduces kernel fusion, recomputing, and communication-overlap infrastructure to reduce mHC system overhead (Section 4.3, Table 2, Table 3, Figure 4).",
        "2fd1e3570d1437de16597b0b942dc8d2f4a0045e84fd066bf01da44a86c86959",
    ),
    (
        "claim-5",
        "At 27B scale, mHC outperforms the baseline and surpasses HC on most zero-shot and few-shot downstream benchmarks (Table 4).",
        "f67e1f1f781f58d9e6c928002254a5dda7d94078db32408893971d6985094a02",
    ),
)
```

- [ ] **Step 4: Implement pure construction and strict serialization**

`build_evidence()` must:

- compute the Task 1 diagnostic for seeded `K=4` logits;
- include all 216 Task 2 dimensional-ablation rows;
- include all 27 Task 3 propagation rows;
- set statuses exactly to `partial`, `partial`, `partial`, `unavailable`,
  `unavailable`;
- set top-level `attempt_id` exactly to
  `3d164e18-39ef-416e-b986-96b5a5d4e12d`;
- state that Claim 2 tests dimensional consistency only;
- state that Claim 3 tests the residual-propagation mechanism only and does not
  reproduce loss gaps or trained gradient norms;
- state that no kernel/system benchmark was run for Claim 4;
- state that no 27B training or downstream benchmark was run for Claim 5;
- record Python, PyTorch, CPU, seed tuple, dimensions, command, pinned source
  revision, and USD 0.00 API cost;
- set `all_claims_verified` to `false`.

Implement serialization as:

```python
def write_evidence(bundle, output_json, output_csv):
    output_json.write_text(
        json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["claim_id", "status", "evidence_kind", "observation", "limitation"]
        )
        for claim in bundle["claims"]:
            writer.writerow(
                [
                    claim["claim_id"],
                    claim["status"],
                    claim["evidence_kind"],
                    claim["observation"],
                    claim["limitation"],
                ]
            )
```

Use `Path` for both output arguments and recursively reject non-finite floats
before serialization so no raw propagation overflow can enter the bundle.

- [ ] **Step 5: Run tests, regenerate twice, and compare bytes**

Run:

```bash
uv run python -m pytest tests/test_cli.py -q
uv run python -m mhc_repro.cli --output-json evidence.json --output-csv summary.csv --n-iters 100
cp evidence.json /tmp/mhc-evidence-first.json
cp summary.csv /tmp/mhc-summary-first.csv
uv run python -m mhc_repro.cli --output-json evidence.json --output-csv summary.csv --n-iters 100
cmp evidence.json /tmp/mhc-evidence-first.json
cmp summary.csv /tmp/mhc-summary-first.csv
```

Expected: tests PASS; both `cmp` commands exit 0.

- [ ] **Step 6: Commit code and generated evidence together**

```bash
git add submissions/mhc-manifold-constrained-hyper-connections/mhc_repro/cli.py submissions/mhc-manifold-constrained-hyper-connections/tests/test_cli.py submissions/mhc-manifold-constrained-hyper-connections/evidence.json submissions/mhc-manifold-constrained-hyper-connections/summary.csv
git commit -m "fix(mhc): emit honest five-claim evidence"
```

### Task 5: Align README, Poster, and Gradio Space With Generated Evidence

**Files:**
- Modify: `submissions/mhc-manifold-constrained-hyper-connections/README.md`
- Create: `submissions/mhc-manifold-constrained-hyper-connections/poster.html`
- Create: `submissions/mhc-manifold-constrained-hyper-connections/app.py`
- Create: `submissions/mhc-manifold-constrained-hyper-connections/tests/test_app.py`
- Create: `submissions/mhc-manifold-constrained-hyper-connections/tests/test_poster.py`
- Modify: `submissions/mhc-manifold-constrained-hyper-connections/pyproject.toml`
- Modify: `submissions/mhc-manifold-constrained-hyper-connections/uv.lock`

**Interfaces:**
- Produces: `app.load_evidence() -> dict[str, object]`.
- Produces: `app.evidence_summary() -> list[list[str]]`, one row per exact live claim with columns `Claim`, `Status`, `Evidence kind`, `Observation`, and `Limitation`.
- Gradio renders committed `evidence.json`; it does not recompute, mutate, deploy, or call external services.
- `poster.html` fetches `./evidence.json` and renders the same five statuses.

- [ ] **Step 1: Add failing Space and poster contract tests**

```python
from pathlib import Path

import app


def test_space_summary_exposes_all_five_honest_statuses():
    rows = app.evidence_summary()
    assert len(rows) == 5
    assert [row[1] for row in rows] == [
        "partial",
        "partial",
        "partial",
        "unavailable",
        "unavailable",
    ]
    assert "27B" in rows[-1][4]


def test_space_readme_has_exact_discovery_tags():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "paper-mDhyxu8WRb" in readme
    assert "icml2026-repro" in readme
    assert "app_file: app.py" in readme
```

```python
from pathlib import Path


def test_poster_labels_toy_and_unavailable_evidence():
    poster = Path("poster.html").read_text(encoding="utf-8")
    assert 'fetch("./evidence.json")' in poster
    assert "Toy CPU mechanism audit" in poster
    assert "No 27B training or downstream evaluation was run." in poster
    assert "reproduces Table 4" not in poster
    assert "verified all claims" not in poster.lower()
```

- [ ] **Step 2: Run presentation tests and verify failure**

Run:

```bash
uv run python -m pytest tests/test_app.py tests/test_poster.py -q
```

Expected: FAIL because `app.py` and `poster.html` do not exist.

- [ ] **Step 3: Implement the read-only Space**

Pin `gradio==6.20.0` in project dependencies and regenerate `uv.lock`. Implement:

```python
from pathlib import Path
import json

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent


def load_evidence() -> dict[str, object]:
    return json.loads((PROJECT_ROOT / "evidence.json").read_text(encoding="utf-8"))


def evidence_summary() -> list[list[str]]:
    return [
        [
            claim["claim_id"],
            claim["status"],
            claim["evidence_kind"],
            claim["observation"],
            claim["limitation"],
        ]
        for claim in load_evidence()["claims"]
    ]


with gr.Blocks(title="mHC CPU Evidence") as demo:
    gr.Markdown(
        "# mHC: Manifold-Constrained Hyper-Connections\n"
        "Pinned CPU evidence. Toy results are not full-training measurements."
    )
    gr.Dataframe(
        value=evidence_summary,
        headers=["Claim", "Status", "Evidence kind", "Observation", "Limitation"],
        interactive=False,
    )


if __name__ == "__main__":
    demo.launch()
```

- [ ] **Step 4: Add exact Space metadata and rewrite evidence documentation**

Start `README.md` with:

```yaml
---
title: mHC CPU Evidence
emoji: 🧭
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
python_version: "3.12"
tags:
  - paper-mDhyxu8WRb
  - icml2026-repro
---
```

Then document:

- the exact pinned revision;
- the frozen execution command;
- Claim 1 as a partial projection-invariant audit that does not verify the full
  stability claim;
- Claims 2 and 3 as partial results with `toy` evidence kind;
- Claims 4 and 5 as unavailable;
- the dimension grid, seeds, depths, and tolerances;
- the distinction between `evidence.json` computed values and paper-reported
  context;
- USD 0.00 API cost and CPU-only execution.

Create `poster.html` with a visible scope banner, five claim cards populated
from `evidence.json`, tables for dimensional and propagation summaries, and
the exact sentence `No 27B training or downstream evaluation was run.` The
poster must not contain manually copied measurements absent from
`evidence.json`.

- [ ] **Step 5: Run presentation and complete paper tests**

Run:

```bash
uv lock
uv sync --frozen
uv run python -m pytest tests -q
uv run python -c 'import app; assert len(app.evidence_summary()) == 5'
```

Expected: all paper tests PASS; the import check exits 0 without launching a
server.

- [ ] **Step 6: Commit the presentation surface**

```bash
git add submissions/mhc-manifold-constrained-hyper-connections/README.md submissions/mhc-manifold-constrained-hyper-connections/poster.html submissions/mhc-manifold-constrained-hyper-connections/app.py submissions/mhc-manifold-constrained-hyper-connections/tests/test_app.py submissions/mhc-manifold-constrained-hyper-connections/tests/test_poster.py submissions/mhc-manifold-constrained-hyper-connections/pyproject.toml submissions/mhc-manifold-constrained-hyper-connections/uv.lock
git commit -m "feat(mhc): add evidence-bound Space and poster"
```

### Task 6: Final Reproduction and Scope Verification

**Files:**
- Verify only: `submissions/mhc-manifold-constrained-hyper-connections/`

**Interfaces:**
- Consumes all prior task outputs.
- Produces one clean, independently executable proposal commit range for controller review; it does not produce lifecycle authority.

- [ ] **Step 1: Run the exact paper suite**

```bash
cd submissions/mhc-manifold-constrained-hyper-connections
uv sync --frozen
uv run python -m pytest tests -q
```

Expected: all tests PASS.

- [ ] **Step 2: Regenerate canonical evidence and prove byte stability**

```bash
uv run python -m mhc_repro.cli --output-json evidence.json --output-csv summary.csv --n-iters 100
sha256sum evidence.json summary.csv
cp evidence.json /tmp/mhc-final-evidence.json
cp summary.csv /tmp/mhc-final-summary.csv
uv run python -m mhc_repro.cli --output-json evidence.json --output-csv summary.csv --n-iters 100
cmp evidence.json /tmp/mhc-final-evidence.json
cmp summary.csv /tmp/mhc-final-summary.csv
```

Expected: both `cmp` commands exit 0; record both SHA-256 values in the worker
handoff.

- [ ] **Step 3: Scan for prohibited overclaims and invalid JSON**

```bash
uv run python -c 'import json; json.load(open("evidence.json", encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))'
rg -n -i 'all claims verified|reproduced (the )?27b|reproduces table 4|training loss reproduced|systems overhead reproduced' README.md poster.html app.py evidence.json summary.csv
```

Expected: JSON parsing exits 0; `rg` returns no matches.

- [ ] **Step 4: Run repository hygiene checks without touching NAPE**

```bash
cd ../..
git diff --check
env UV_CACHE_DIR=/tmp/icml-mhc-uv-cache PRE_COMMIT_HOME=/tmp/icml-mhc-pre-commit uv run pre-commit run --files submissions/mhc-manifold-constrained-hyper-connections
git status --short
```

Expected: `git diff --check` and scoped pre-commit PASS. Status contains only
the intended mHC proposal changes; any unrelated or pre-existing dirt stops
execution and must be reported. No `state/`, `docs/HANDOFF.md`, skill, NAPE, or
other submission change was introduced by this plan.

- [ ] **Step 5: Commit any deterministic formatting-only corrections**

If scoped pre-commit changed mHC files, inspect the diff and commit only those
mechanical changes:

```bash
git add submissions/mhc-manifold-constrained-hyper-connections
git commit -m "style(mhc): apply scoped verification fixes"
```

If scoped pre-commit made no changes, do not create an empty commit.

- [ ] **Step 6: Prepare the worker proposal handoff**

Report:

- paper `mDhyxu8WRb`;
- assigned mHC worktree and final commit range;
- exact test, generation, comparison, and pre-commit commands;
- `evidence.json` and `summary.csv` SHA-256 values;
- controller attempt `3d164e18-39ef-416e-b986-96b5a5d4e12d`;
- Claims 1–3 `partial`, with Claims 2–3 using `toy` evidence kind, and Claims
  4–5 `unavailable`;
- no full-training, systems, or 27B measurements;
- no controller validation, deployment, submission, Hub, state, or verdict
  writes.
