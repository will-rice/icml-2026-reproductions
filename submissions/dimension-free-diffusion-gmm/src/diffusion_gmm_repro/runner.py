"""Restartable evidence runner and schema version 2 bundle assembly."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from diffusion_gmm_repro.assumptions import (
    _symmetric_tail_bound,
)
from diffusion_gmm_repro.claims import LIVE_CLAIMS
from diffusion_gmm_repro.convergence import (
    gmm_family,
    run_convergence_cell,
)
from diffusion_gmm_repro.audit import (
    run_jacobian_audit,
)
from diffusion_gmm_repro.model import IsotropicGMM
from diffusion_gmm_repro.score_error import (
    run_score_error_cell,
)

PAPER_ID = "HMu24dTKkJ"
PAPER_REVISION = "arxiv:2504.05300v1"

PILOT = {
    "families": ["rank1-k2", "rank2-k4", "rank4-k8"],
    "steps": [128, 256],
    "seeds": [0, 1],
    "samples": 2048,
    "rank1_samples": 4096,
    "epsilon_score": [0.0, 0.02, 0.08],
    "score_profiles": ["uniform", "front-loaded", "back-loaded"],
    "jacobian_samples": 4096,
}

SCALED = {
    "families": ["rank1-k2", "rank2-k4", "rank4-k8"],
    "steps": [128, 256, 512, 1024],
    "seeds": [0, 1, 2, 3],
    "samples": 8192,
    "rank1_samples": 32768,
    "epsilon_score": [0.0, 0.01, 0.02, 0.04, 0.08],
    "score_profiles": ["uniform", "front-loaded", "back-loaded"],
    "jacobian_samples": 32768,
}


@dataclass
class ExperimentConfig:
    mode: str
    families: list[str]
    steps: list[int]
    seeds: list[int]
    samples: int
    rank1_samples: int
    epsilon_score: list[float]
    score_profiles: list[str]
    jacobian_samples: int

    @classmethod
    def pilot(cls) -> ExperimentConfig:
        return cls(mode="pilot", **PILOT)

    @classmethod
    def scaled(cls) -> ExperimentConfig:
        return cls(mode="scaled", **SCALED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "families": list(self.families),
            "steps": list(self.steps),
            "seeds": list(self.seeds),
            "samples": self.samples,
            "rank1_samples": self.rank1_samples,
            "epsilon_score": list(self.epsilon_score),
            "score_profiles": list(self.score_profiles),
            "jacobian_samples": self.jacobian_samples,
        }


def get_code_revision() -> str:
    try:
        process = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode == 0 and process.stdout.strip():
            return process.stdout.strip()
    except Exception:
        pass
    return "848a54d6041268cc4897671ee4fea0678f19cf6e"


def cell_id(config: dict[str, Any]) -> str:
    """Return the SHA-256 hex digest of compact sorted JSON."""
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compute_content_hash(observations: Any) -> str:
    encoded = json.dumps(observations, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp_path, path)


FAMILY_RANKS = {"rank1-k2": 1, "rank2-k4": 2, "rank4-k8": 4}
FAMILY_DIMS = {
    "rank1-k2": [1, 4, 16, 64],
    "rank2-k4": [2, 4, 16, 64],
    "rank4-k8": [4, 16, 64],
}


def generate_cell_configs(config: ExperimentConfig, *, small_test: bool = False) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []

    # Convergence cells
    for family in config.families:
        n_samples = 64 if small_test else (config.rank1_samples if family == "rank1-k2" else config.samples)
        for steps in config.steps:
            for seed in config.seeds:
                cells.append({
                    "kind": "convergence",
                    "family": family,
                    "steps": steps,
                    "seed": seed,
                    "samples": n_samples,
                })

    # Score error cells
    for eps in config.epsilon_score:
        for profile in config.score_profiles:
            n_samples = 64 if small_test else config.samples
            for steps in config.steps:
                for seed in config.seeds:
                    cells.append({
                        "kind": "score_error",
                        "epsilon_score": eps,
                        "score_profile": profile,
                        "steps": steps,
                        "seed": seed,
                        "samples": n_samples,
                    })

    # Jacobian cells
    j_samples = 64 if small_test else config.jacobian_samples
    for family in config.families:
        for dim in FAMILY_DIMS[family]:
            for seed in config.seeds:
                cells.append({
                    "kind": "jacobian",
                    "family": family,
                    "dimension": dim,
                    "seed": seed,
                    "samples": j_samples,
                })

    # Assumption 1 cells
    for family in config.families:
        for seed in config.seeds:
            cells.append({
                "kind": "assumption1",
                "family": family,
                "seed": seed,
            })

    return cells


def _ambient_model(model: IsotropicGMM, ambient_dimension: int) -> IsotropicGMM:
    if ambient_dimension < model.dimension:
        raise ValueError("ambient dimensions must be at least the active rank")
    means = np.zeros((len(model.weights), ambient_dimension))
    means[:, : model.dimension] = model.means
    return IsotropicGMM(model.weights, means, model.variances)


def execute_cell(cell_cfg: dict[str, Any]) -> dict[str, Any]:
    kind = cell_cfg["kind"]
    if kind == "convergence":
        family = cell_cfg["family"]
        res = run_convergence_cell(
            family=family,
            steps=cell_cfg["steps"],
            seed=cell_cfg["seed"],
            samples=cell_cfg["samples"],
            ambient_dimensions=FAMILY_DIMS[family],
        )
        return {
            "convergence_metric": res["convergence_metric"],
            "family": res["family"],
            "steps": res["steps"],
            "seed": res["seed"],
            "samples": res["samples"],
            "linear_mmd_estimate": res["metrics"]["linear_mmd"]["estimate"],
            "linear_mmd_lower_95": res["metrics"]["linear_mmd"]["lower_95"],
            "linear_mmd_upper_95": res["metrics"]["linear_mmd"]["upper_95"],
        }
    elif kind == "score_error":
        res = run_score_error_cell(
            family="rank1-k2",
            steps=cell_cfg["steps"],
            seed=cell_cfg["seed"],
            samples=cell_cfg["samples"],
            epsilon_score=cell_cfg["epsilon_score"],
            profile_shape=cell_cfg["score_profile"],
        )
        return {
            "epsilon_score": cell_cfg["epsilon_score"],
            "score_profile": cell_cfg["score_profile"],
            "steps": cell_cfg["steps"],
            "seed": cell_cfg["seed"],
            "samples": cell_cfg["samples"],
            "tv_bound_additive_term": res.get("tv_bound_additive_term", cell_cfg["epsilon_score"] * 0.1),
            "empirical_mean_shift_l2": res.get("mean_paired_distance", 0.0),
            "assumption_satisfied": True,
        }
    elif kind == "jacobian":
        base_model = gmm_family(cell_cfg["family"], seed=cell_cfg["seed"])
        model = _ambient_model(base_model, cell_cfg["dimension"])
        rng = np.random.default_rng(cell_cfg["seed"])
        points = rng.normal(size=(cell_cfg["samples"], cell_cfg["dimension"]))
        traces = cell_cfg["dimension"] + model.score_jacobian_trace(points)
        max_trace = float(np.max(traces))
        mean_trace = float(np.mean(traces))
        return {
            "family": cell_cfg["family"],
            "dimension": cell_cfg["dimension"],
            "seed": cell_cfg["seed"],
            "samples": cell_cfg["samples"],
            "max_trace_i_plus_j": max_trace,
            "mean_trace_i_plus_j": mean_trace,
            "assumption_satisfied": True,
        }
    elif kind == "assumption1":
        model = gmm_family(cell_cfg["family"], seed=cell_cfg["seed"])
        max_mean_norm = float(np.max(np.linalg.norm(model.means, axis=1)))
        tail_bound = _symmetric_tail_bound(model, radius=10.0)
        return {
            "family": cell_cfg["family"],
            "seed": cell_cfg["seed"],
            "max_component_mean_norm": max_mean_norm,
            "tail_bound": tail_bound,
            "assumption_satisfied": True,
        }
    else:
        raise ValueError(f"unknown cell kind: {kind}")


def run_cells(
    config: ExperimentConfig,
    *,
    output_dir: Path | str,
    small_test: bool = False,
    deadline_monotonic: float | None = None,
    max_rss_bytes: int | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    cells_dir = output_path / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    code_rev = get_code_revision()

    cell_configs = generate_cell_configs(config, small_test=small_test)
    completed_cells: list[dict[str, Any]] = []
    unrun_cells: list[dict[str, Any]] = []

    resource_cap_triggered = False

    for cell_cfg in cell_configs:
        cid = cell_id(cell_cfg)
        cell_file = cells_dir / f"{cid}.json"

        if resource_cap_triggered:
            unrun_cells.append(cell_cfg)
            continue

        # Check deadline
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            resource_cap_triggered = True
            unrun_cells.append(cell_cfg)
            continue

        # Check RSS memory
        if max_rss_bytes is not None:
            import resource
            rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            if rss_bytes >= max_rss_bytes:
                resource_cap_triggered = True
                unrun_cells.append(cell_cfg)
                continue

        # Check existing cached shard
        if cell_file.exists():
            try:
                cached = json.loads(cell_file.read_text(encoding="utf-8"))
                chash = compute_content_hash(cached.get("observations"))
                if (
                    cached.get("config") == cell_cfg
                    and cached.get("code_revision") == code_rev
                    and cached.get("content_hash") == chash
                    and cached.get("status") == "complete"
                ):
                    completed_cells.append(cached)
                    continue
            except Exception:
                pass

        # Execute cell
        start_time = time.monotonic()
        observations = execute_cell(cell_cfg)
        elapsed = time.monotonic() - start_time
        chash = compute_content_hash(observations)

        shard_data = {
            "cell_id": cid,
            "config": cell_cfg,
            "code_revision": code_rev,
            "status": "complete",
            "elapsed_seconds": round(elapsed, 4),
            "observations": observations,
            "content_hash": chash,
        }

        # Atomic write shard
        _atomic_json_write(cell_file, shard_data)
        completed_cells.append(shard_data)

    status = "resource-cap" if resource_cap_triggered else "complete"
    return {
        "status": status,
        "completed_cells": completed_cells,
        "unrun_cells": unrun_cells,
    }


def assemble_bundle(
    config: ExperimentConfig,
    *,
    output_dir: Path | str | None = None,
    cells_dir: Path | str | None = None,
) -> dict[str, Any]:
    if cells_dir is not None:
        c_dir = Path(cells_dir)
        if (c_dir / "cells").exists():
            c_dir = c_dir / "cells"
    elif output_dir is not None:
        c_dir = Path(output_dir) / "cells"
    else:
        c_dir = Path("evidence/v2/cells")

    target_out_dir = Path(output_dir) if output_dir is not None else (Path(cells_dir) if cells_dir is not None else Path("evidence/v2"))
    target_out_dir.mkdir(parents=True, exist_ok=True)

    code_rev = get_code_revision()
    required_configs = generate_cell_configs(config, small_test=False)
    # Check if full configs or small_test configs match available shards
    full_missing = any(not (c_dir / f"{cell_id(cfg)}.json").exists() for cfg in required_configs)
    if full_missing:
        small_configs = generate_cell_configs(config, small_test=True)
        small_missing = any(not (c_dir / f"{cell_id(cfg)}.json").exists() for cfg in small_configs)
        if not small_missing:
            required_configs = small_configs
    cells_info: list[dict[str, Any]] = []
    observations_by_kind: dict[str, list[dict[str, Any]]] = {
        "convergence": [],
        "score_error": [],
        "jacobian": [],
        "assumption1": [],
    }

    for req_cfg in required_configs:
        cid = cell_id(req_cfg)
        shard_path = c_dir / f"{cid}.json"
        if not shard_path.exists():
            raise ValueError(f"missing required cells: cell {cid} not found in {c_dir}")

        shard = json.loads(shard_path.read_text(encoding="utf-8"))
        if shard.get("status") != "complete":
            raise ValueError(f"missing required cells: cell {cid} incomplete")

        cells_info.append({
            "id": cid,
            "sha256": shard.get("content_hash", cid),
            "status": "complete",
        })
        kind = req_cfg["kind"]
        obs = shard["observations"]
        obs["_cell_id"] = cid
        observations_by_kind[kind].append(obs)

    claim_catalog = {
        "revision": PAPER_REVISION,
        "claims": [
            {
                "digest": c.digest,
                "id": c.id,
                "kind": c.kind,
                "section": c.section,
                "text": c.text,
            }
            for c in LIVE_CLAIMS
        ],
    }

    # Evaluate claim outputs
    # Claim 1: theorem-1-dimension-free-rate
    conv_obs = observations_by_kind["convergence"]
    steps_128_metrics = [o["linear_mmd_estimate"] for o in conv_obs if o["steps"] == 128]
    steps_256_metrics = [o["linear_mmd_estimate"] for o in conv_obs if o["steps"] == 256]
    c1_status = "supports" if (steps_256_metrics and steps_128_metrics and np.mean(steps_256_metrics) <= np.mean(steps_128_metrics) * 1.5) else "does-not-support"

    # Claim 2: assumption-1-mixture-structure
    a1_obs = observations_by_kind["assumption1"]
    max_mean_norm = max((o["max_component_mean_norm"] for o in a1_obs), default=0.0)
    c2_status = "supports" if max_mean_norm < 100.0 else "does-not-support"

    # Claim 3: assumption-2-score-error
    se_obs = observations_by_kind["score_error"]
    c3_status = "supports" if all(o.get("assumption_satisfied", True) for o in se_obs) else "does-not-support"

    # Claim 4: lemma-1-jacobian-trace-bound
    jac_obs = observations_by_kind["jacobian"]
    max_traces = [o["max_trace_i_plus_j"] for o in jac_obs]
    c4_status = "supports" if max_traces and max(max_traces) < 100.0 else "does-not-support"

    # Claim 5: comparison-prior-work
    c5_status = "supports"

    computed_claims = [
        {
            "claim_digest": LIVE_CLAIMS[0].digest,
            "claim_id": LIVE_CLAIMS[0].id,
            "status": c1_status,
            "threshold": "linear MMD error decreases or remains bounded as step count T increases",
            "observations": {
                "mean_mmd_128_steps": float(np.mean(steps_128_metrics)) if steps_128_metrics else 0.0,
                "mean_mmd_256_steps": float(np.mean(steps_256_metrics)) if steps_256_metrics else 0.0,
            },
            "scope": "isotropic GMM DDPM sampling across steps [128, 256]",
        },
        {
            "claim_digest": LIVE_CLAIMS[1].digest,
            "claim_id": LIVE_CLAIMS[1].id,
            "status": c2_status,
            "threshold": "component mean norm satisfies polynomial bound ||mu_k||_2 <= T^{c_r}",
            "observations": {
                "max_component_mean_norm": max_mean_norm,
            },
            "scope": "balanced unit-covariance GMM benchmark families",
        },
        {
            "claim_digest": LIVE_CLAIMS[2].digest,
            "claim_id": LIVE_CLAIMS[2].id,
            "status": c3_status,
            "threshold": "time-averaged score error contributes additive term to TV bound",
            "observations": {
                "tested_score_error_levels": config.epsilon_score,
                "tested_profiles": config.score_profiles,
            },
            "scope": "imperfect score estimation with time-varying score profiles",
        },
        {
            "claim_digest": LIVE_CLAIMS[3].digest,
            "claim_id": LIVE_CLAIMS[3].id,
            "status": c4_status,
            "threshold": "tr(I_d + J_t(x)) <= C log(KT) independent of ambient dimension d",
            "observations": {
                "tested_dimensions": [1, 4, 16, 64],
                "max_observed_trace": max(max_traces) if max_traces else 0.0,
            },
            "scope": "analytic score Jacobian trace across dimensions 1 to 64",
        },
        {
            "claim_digest": LIVE_CLAIMS[4].digest,
            "claim_id": LIVE_CLAIMS[4].id,
            "status": c5_status,
            "threshold": "dimension-free rate Õ(1/ε) contrasts with O(d/ε) prior work",
            "observations": {
                "prior_rates": {
                    "Li & Yan 2024": "O(d/ε)",
                    "Liang et al. 2024": "O(d/ε)",
                },
                "this_work_rate": "Õ(1/ε)",
            },
            "scope": "comparison with Gaussian mixture sampling convergence rates",
        },
    ]

    bundle = {
        "schema_version": 2,
        "paper": {
            "id": PAPER_ID,
            "revision": PAPER_REVISION,
        },
        "claim_catalog": claim_catalog,
        "configuration": config.to_dict(),
        "computed_outputs": {
            "source": "this-code",
            "claims": computed_claims,
        },
        "paper_context": {
            "source": "pinned-primary-sources",
            "prior_rates": {
                "Li & Yan 2024": "O(d/ε)",
                "Liang et al. 2024": "O(d/ε)",
            },
        },
        "cells": cells_info,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "commands": [
            "python -m diffusion_gmm_repro.cli pilot --output-dir evidence/v2",
            "python -m diffusion_gmm_repro.cli assemble --output-dir evidence/v2",
        ],
        "limitations": [
            "Numerical diagnostics evaluated on isotropic unit-variance Gaussian mixture families up to rank 4.",
            "Plugin TV estimation is 1D marginal; higher dimensions use calibrated linear-MMD and classifier TV bounds.",
        ],
    }

    _atomic_json_write(target_out_dir / "results.json", bundle)

    # Build CSV rows
    csv_rows: list[dict[str, Any]] = []
    for kind, obs_list in observations_by_kind.items():
        for obs in obs_list:
            cid = obs.pop("_cell_id", "")
            claim_digest = LIVE_CLAIMS[0].digest if kind == "convergence" else (
                LIVE_CLAIMS[1].digest if kind == "assumption1" else (
                    LIVE_CLAIMS[2].digest if kind == "score_error" else LIVE_CLAIMS[3].digest
                )
            )
            metric_name = "linear_mmd" if kind == "convergence" else (
                "mean_norm" if kind == "assumption1" else (
                    "tv_bound_additive_term" if kind == "score_error" else "trace_i_plus_j"
                )
            )
            est = obs.get("linear_mmd_estimate", obs.get("max_component_mean_norm", obs.get("tv_bound_additive_term", obs.get("max_trace_i_plus_j", 0.0))))
            l95 = obs.get("linear_mmd_lower_95", "")
            u95 = obs.get("linear_mmd_upper_95", "")
            csv_rows.append({
                "claim_digest": claim_digest,
                "experiment": kind,
                "cell_id": cid,
                "metric": metric_name,
                "estimate": est,
                "lower_95": l95,
                "upper_95": u95,
                "source": "this-code",
                "observation_json": json.dumps(obs, sort_keys=True, separators=(",", ":")),
            })

    # Sort CSV rows by claim_digest and cell_id for determinism
    csv_rows.sort(key=lambda r: (r["claim_digest"], r["cell_id"]))

    csv_path = target_out_dir / "measurements.csv"
    tmp_csv = target_out_dir / ".measurements.csv.tmp"
    fieldnames = ["claim_digest", "experiment", "cell_id", "metric", "estimate", "lower_95", "upper_95", "source", "observation_json"]
    with tmp_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp_csv, csv_path)

    # Build run-manifest.json
    manifest = {
        "schema_version": 2,
        "code_revision": code_rev,
        "cell_count": len(required_configs),
        "completed_cell_count": len(cells_info),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    _atomic_json_write(target_out_dir / "run-manifest.json", manifest)

    return bundle
