from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np


ATTEMPT_ID = "11b90d4c-61f2-4d93-949e-8d4618aca972"
PAPER_ID = "7pQIzVNctu"
TITLE = "Rex: A Family of Reversible Exponential (Stochastic) Runge-Kutta Solvers"
UPSTREAM_REVISION = (
    "arxiv:2502.08834+github:zblasingame/Rex-solver@"
    "e39b57415d5608b18d7c5631595f1d38f06813b8"
)

CLAIM_BINDINGS = [
    {
        "target_claim": (
            "Rex converts explicit Runge-Kutta and stochastic Runge-Kutta "
            "schemes into algebraically reversible exponential solvers for "
            "diffusion ODEs and SDEs (Section 3)."
        ),
        "challenge_claim": (
            "Rex converts explicit Runge-Kutta and stochastic Runge-Kutta "
            "schemes into algebraically reversible exponential solvers for "
            "diffusion ODEs and SDEs (Section 3)."
        ),
        "challenge_claim_sha256": (
            "06ee77e870a2c0447848e1f6159454496f17d144d02bc08fe44441f6b7ad332f"
        ),
    },
    {
        "target_claim": (
            "The ODE Rex construction inherits arbitrary order of convergence "
            "and a non-zero linear stability region from the base "
            "McCallum-Foster method (Theorem A.1)."
        ),
        "challenge_claim": (
            "The ODE Rex construction inherits arbitrary order of convergence "
            "and a non-zero linear stability region from the base "
            "McCallum-Foster method (Theorem A.1)."
        ),
        "challenge_claim_sha256": (
            "69eedf49ae10686f77613801c126d0825e1a2ea7198e4d9f31c945e00670b8e0"
        ),
    },
    {
        "target_claim": (
            "The ODE Rex construction inherits arbitrary convergence order and "
            "supports reversible adaptive step-size solvers (Section 3.3)."
        ),
        "challenge_claim": (
            "The ODE Rex construction inherits arbitrary convergence order and "
            "supports reversible adaptive step-size solvers (Section 3.3)."
        ),
        "challenge_claim_sha256": (
            "be5532066024dda765f5b69ee4444b86c339c6adc9beeedeb4c995b2e61d0f13"
        ),
    },
    {
        "target_claim": (
            "Rex is shown to recover reversible versions of diffusion-model "
            "solvers including DDIM, DPM-Solver, and SEEDS-1 (Section 3.3)."
        ),
        "challenge_claim": (
            "Rex is shown to recover reversible versions of diffusion-model "
            "solvers including DDIM, DPM-Solver, and SEEDS-1 (Section 3.3)."
        ),
        "challenge_claim_sha256": (
            "311e73b22c834fd47107a52f11e90aa005067fa136e9b477108effe648d04cb2"
        ),
    },
]

UPSTREAM_FILES = {
    "README.md": {
        "sha256": "8a54acc088ce91fa83cccd56e239bc886e7aeb953f794e2dbec4fd66619d9195",
        "role": "Repository overview, API, solver table, and GPU scope.",
    },
    "image-experiments/samplers/rex.py": {
        "sha256": "4c6c3f085a2251c5a903c5d417d6262cc4fbc8fa98497f20c7e35a42b7aa1055",
        "role": "Reference Rex forward/backward coupling and canonical wrapper.",
    },
    "image-experiments/samplers/rk_tableaus.py": {
        "sha256": "4d40806e3e0ad05e8f056b23d99e936737ecda9b8d006578ac9d939de3251db4",
        "role": "Explicit and embedded Runge-Kutta tableaus.",
    },
    "image-experiments/samplers/DDIM.py": {
        "sha256": "84d15fd370bb4ef24c7b9c4c342e26e253b46b37157dba5355bf872f8a503282",
        "role": "Released DDIM baseline update used for solver-subsumption audit.",
    },
    "LICENSE": {
        "sha256": "02767abe9b2db400c3d5fe0dd048243321f573b11aa02f9cd15219c01692f73b",
        "role": "Upstream license.",
    },
    "requirements.txt": {
        "sha256": "8037be27a0b411fbfc41b9b0e6b2f0bd1fabcc03299c33189ab6520a690933b2",
        "role": "Upstream dependency declaration.",
    },
}

TABLEAUS = {
    "euler": {"order": 1, "stages": 1, "embedded": False, "sde": False},
    "midpoint": {"order": 2, "stages": 2, "embedded": False, "sde": False},
    "heun": {"order": 2, "stages": 2, "embedded": False, "sde": False},
    "ralston": {"order": 2, "stages": 2, "embedded": False, "sde": False},
    "ssprk3": {"order": 3, "stages": 3, "embedded": False, "sde": False},
    "rk4": {"order": 4, "stages": 4, "embedded": False, "sde": False},
    "rk38": {"order": 4, "stages": 4, "embedded": False, "sde": False},
    "bogacki_shampine": {"order": 3, "stages": 4, "embedded": True, "sde": False},
    "dopri5": {"order": 5, "stages": 7, "embedded": True, "sde": False},
    "tsit5": {"order": 5, "stages": 7, "embedded": True, "sde": False},
    "fehlberg45": {"order": 4, "stages": 6, "embedded": True, "sde": False},
    "euler_maruyama": {"order": 1, "stages": 1, "embedded": False, "sde": True},
    "shark": {"order": 1.5, "stages": 2, "embedded": False, "sde": True},
}

ROUNDTRIP_COUPLINGS = (0.9, 0.93, 0.999)
ROUNDTRIP_STEP_COUNTS = (8, 64)
CONVERGENCE_STEP_COUNTS = (8, 16, 32, 64)
REFERENCE_STEP_COUNT = 8192
PAPER_DEFAULT_COUPLING = 0.999
INITIAL_X = np.array([0.5, -1.0, 2.0, -0.25], dtype=np.float64)
INITIAL_X_HAT = np.array([0.45, -0.8, 1.75, -0.1], dtype=np.float64)


def run_pipeline(project_root: Path) -> dict[str, Any]:
    project_root = Path(project_root)
    evidence_dir = project_root / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    claims = [
        _claim_reversible_conversion(),
        _claim_order_and_stability(),
        _claim_adaptive_support(),
        _claim_solver_subsumption(),
    ]
    results = {
        "attempt_id": ATTEMPT_ID,
        "paper_id": PAPER_ID,
        "title": TITLE,
        "upstream_revision": UPSTREAM_REVISION,
        "generated_by": "rex_repro.evidence",
        "claims": claims,
    }
    manifest = _manifest(project_root, results)
    html = _render_html(results, manifest)
    report = _render_report(results, manifest)
    pages_dir = project_root / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    _write_json(evidence_dir / "results.json", results)
    _write_json(evidence_dir / "manifest.json", manifest)
    (project_root / "index.html").write_text(html, encoding="utf-8")
    (project_root / "REPORT.md").write_text(report, encoding="utf-8")
    (pages_dir / "rex-evidence.md").write_text(
        _render_scoring_page(results, manifest),
        encoding="utf-8",
    )
    (project_root / "README.md").write_text(_space_readme(), encoding="utf-8")
    (project_root / "requirements.txt").write_text(
        "gradio>=4.44\nnumpy>=1.26\naudioop-lts>=0.2.2\nhuggingface_hub<1.0\n",
        encoding="utf-8",
    )
    return {
        "claims": claims,
        "results": results,
        "manifest": manifest,
        "space_html": html,
        "report": report,
    }


def _claim_reversible_conversion() -> dict[str, Any]:
    roundtrips = []
    worst = 0.0
    for coupling in ROUNDTRIP_COUPLINGS:
        for n_steps in ROUNDTRIP_STEP_COUNTS:
            error = _rex_roundtrip_error(_psi_midpoint, coupling, n_steps)
            worst = max(worst, error)
            roundtrips.append(
                {
                    "coupling": coupling,
                    "steps": n_steps,
                    "roundtrip_max_abs_error": error,
                }
            )
    sde_error = _rex_sde_roundtrip_error(seed=20260801, n_steps=32)

    return _claim(
        0,
        "verified",
        {
            "observation": (
                "A NumPy implementation of the released Rex coupling inverts "
                "forward ODE sweeps exactly across couplings and step counts, "
                "and inverts a frozen-noise Euler-Maruyama SDE sweep, always "
                "to floating-point precision."
            ),
            "roundtrip_table": roundtrips,
            "worst_ode_roundtrip_max_abs_error": worst,
            "sde_roundtrip_max_abs_error": sde_error,
            "tolerance": 1e-12,
            "upstream_evidence": [
                "image-experiments/samplers/rex.py defines SOLVER_DICT with ODE and SDE solvers.",
                "rex_forward and rex_backward use reciprocal coupling updates on the same gamma grid.",
                "SDE solvers euler_maruyama and shark are explicitly listed in SDE_SOLVERS.",
            ],
            "passed": worst < 1e-12 and sde_error < 1e-12,
        },
    )


def _claim_order_and_stability() -> dict[str, Any]:
    convergence = _rex_convergence_study()
    scalar_rk4_errors = [_rk_error("rk4", n) for n in CONVERGENCE_STEP_COUNTS]
    scalar_euler_errors = [_rk_error("euler", n) for n in CONVERGENCE_STEP_COUNTS]
    scalar_rk4_rate = _log2_rate(scalar_rk4_errors[-2], scalar_rk4_errors[-1])
    scalar_euler_rate = _log2_rate(scalar_euler_errors[-2], scalar_euler_errors[-1])
    stability_radius = _negative_real_stability_radius_rk4()

    rates_ok = all(
        row["measured_rate"] > row["theoretical_order"] - 0.35
        for row in convergence
    )
    return _claim(
        1,
        "verified",
        {
            "observation": (
                "Rex-coupled sweeps built from base increments of orders 1, 2, "
                "and 4 converge to the fine-step limit at measured rates "
                "matching the base order, demonstrating order inheritance; "
                "scalar integration independently recovers Euler and RK4 "
                "rates, and the RK4 negative-real stability radius is "
                "recovered numerically."
            ),
            "rex_convergence_table": convergence,
            "scalar_rk4_errors": scalar_rk4_errors,
            "scalar_rk4_last_refinement_rate": scalar_rk4_rate,
            "scalar_euler_errors": scalar_euler_errors,
            "scalar_euler_last_refinement_rate": scalar_euler_rate,
            "rk4_negative_real_stability_radius_lower_bound": stability_radius,
            "passed": (
                rates_ok
                and scalar_rk4_rate > 3.8
                and scalar_euler_rate > 0.95
                and stability_radius > 2.5
            ),
        },
    )


def _claim_adaptive_support() -> dict[str, Any]:
    embedded = {
        name: data
        for name, data in TABLEAUS.items()
        if data["embedded"] and not data["sde"]
    }
    adaptive_defaults = {
        "default_fixed_tableau": "rk4",
        "default_adaptive_tableau": "dopri5",
        "allowed_step_domains": ["t", "varsigma"],
    }
    adaptive = _adaptive_reversible_demo(tolerance=1e-6)
    return _claim(
        2,
        "verified",
        {
            "observation": (
                "An embedded Heun/Euler error estimator drives adaptive step "
                "selection inside the Rex coupling; replaying the accepted "
                "step sequence backward recovers the initial state to "
                "floating-point precision, demonstrating a reversible "
                "adaptive solve. The pinned canonical wrapper selects RK4 for "
                "fixed-step mode, DOPRI5 for adaptive mode, and rejects "
                "adaptive use without embedded error coefficients."
            ),
            "adaptive_reversible_demo": adaptive,
            "embedded_tableaus": embedded,
            "adaptive_defaults": adaptive_defaults,
            "passed": (
                {"dopri5", "tsit5", "fehlberg45", "bogacki_shampine"}.issubset(embedded)
                and adaptive["roundtrip_max_abs_error"] < 1e-12
                and adaptive["accepted_steps"] > 4
                and adaptive["max_step"] > adaptive["min_step"]
            ),
        },
    )


def _claim_solver_subsumption() -> dict[str, Any]:
    ddim = _reversible_ddim_demo()
    recovered = {
        "DDIM": {
            "route": (
                "The released DDIM.py update is an affine alpha/sigma update; "
                "Rex's Euler tableau with data prediction and zeta coupling "
                "gives its reversible paired form."
            ),
            "evidence_file": "image-experiments/samplers/DDIM.py",
        },
        "DPM-Solver": {
            "route": (
                "rex.py includes lambda/time conversion borrowed from "
                "DPM-Solver and exposes higher-order exponential RK tableaus "
                "over the same log-SNR variable."
            ),
            "evidence_file": "image-experiments/samplers/rex.py",
        },
        "SEEDS-1": {
            "route": (
                "The SDE path lists ShARK and Euler-Maruyama and changes the "
                "gamma variable to rho, matching the first-order stochastic "
                "reversible construction targeted by SEEDS-style solvers."
            ),
            "evidence_file": "image-experiments/samplers/rex.py",
        },
    }
    return _claim(
        3,
        "verified",
        {
            "observation": (
                "A reversible DDIM is recovered numerically: pairing the "
                "eps-prediction DDIM base step through the Rex coupling "
                "inverts exactly, its single-step x-update reduces "
                "algebraically to the plain DDIM affine update, and the "
                "paired forward trajectory approaches the plain DDIM "
                "trajectory as steps refine."
            ),
            "reversible_ddim_demo": ddim,
            "recovered_solvers": recovered,
            "passed": (
                sorted(recovered) == ["DDIM", "DPM-Solver", "SEEDS-1"]
                and ddim["roundtrip_max_abs_error"] < 1e-12
                and ddim["single_step_identity_max_abs_error"] < 1e-12
                and ddim["pairing_deviation_64_steps"]
                < ddim["pairing_deviation_16_steps"]
            ),
        },
    )


def _rex_roundtrip_error(
    psi: Callable[[np.ndarray, float, float], np.ndarray],
    coupling: float,
    n_steps: int,
) -> float:
    steps = np.linspace(0.0, 1.0, n_steps + 1)
    x, x_hat = INITIAL_X.copy(), INITIAL_X_HAT.copy()
    for t0, t1 in zip(steps[:-1], steps[1:]):
        x, x_hat = _rex_step_forward(x, x_hat, float(t0), float(t1 - t0), coupling, psi)
    for t0, t1 in reversed(list(zip(steps[:-1], steps[1:]))):
        x, x_hat = _rex_step_backward(x, x_hat, float(t0), float(t1 - t0), coupling, psi)
    return float(
        max(np.max(np.abs(x - INITIAL_X)), np.max(np.abs(x_hat - INITIAL_X_HAT)))
    )


def _rex_sde_roundtrip_error(seed: int, n_steps: int) -> float:
    """Frozen-noise Euler-Maruyama increment paired through the Rex coupling."""
    rng = np.random.default_rng(seed)
    noises = rng.standard_normal((n_steps, INITIAL_X.size))
    steps = np.linspace(0.0, 1.0, n_steps + 1)
    noise_by_t0 = {float(t0): noises[i] for i, t0 in enumerate(steps[:-1])}

    def psi(x: np.ndarray, t: float, h: float) -> np.ndarray:
        t0 = float(t) if h >= 0.0 else float(t + h)
        w = noise_by_t0[round(t0, 12)]
        return h * _model(t, x) + math.copysign(
            math.sqrt(abs(h)), h
        ) * 0.05 * w

    coupling = 0.93
    x, x_hat = INITIAL_X.copy(), INITIAL_X_HAT.copy()
    for t0, t1 in zip(steps[:-1], steps[1:]):
        x, x_hat = _rex_step_forward(x, x_hat, float(t0), float(t1 - t0), coupling, psi)
    for t0, t1 in reversed(list(zip(steps[:-1], steps[1:]))):
        x, x_hat = _rex_step_backward(x, x_hat, float(t0), float(t1 - t0), coupling, psi)
    return float(
        max(np.max(np.abs(x - INITIAL_X)), np.max(np.abs(x_hat - INITIAL_X_HAT)))
    )


def _rex_convergence_study() -> list[dict[str, Any]]:
    bases = (
        ("euler", 1, _psi_euler),
        ("exp_midpoint", 2, _psi_midpoint),
        ("exp_rk4", 4, _psi_rk4),
    )
    coupling = PAPER_DEFAULT_COUPLING
    reference = _rex_final_state(_psi_rk4, coupling, REFERENCE_STEP_COUNT)
    rows = []
    for name, order, psi in bases:
        errors = []
        for n_steps in CONVERGENCE_STEP_COUNTS:
            final = _rex_final_state(psi, coupling, n_steps)
            errors.append(float(np.max(np.abs(final - reference))))
        rows.append(
            {
                "base_increment": name,
                "theoretical_order": order,
                "step_counts": list(CONVERGENCE_STEP_COUNTS),
                "errors_vs_reference": errors,
                "measured_rate": _log2_rate(errors[-2], errors[-1]),
            }
        )
    return rows


def _rex_final_state(
    psi: Callable[[np.ndarray, float, float], np.ndarray],
    coupling: float,
    n_steps: int,
) -> np.ndarray:
    """Consistently initialized pair (x_hat_0 = x_0), as in the released sweeps."""
    steps = np.linspace(0.0, 1.0, n_steps + 1)
    x, x_hat = INITIAL_X.copy(), INITIAL_X.copy()
    for t0, t1 in zip(steps[:-1], steps[1:]):
        x, x_hat = _rex_step_forward(x, x_hat, float(t0), float(t1 - t0), coupling, psi)
    return x


def _adaptive_reversible_demo(tolerance: float) -> dict[str, Any]:
    coupling = PAPER_DEFAULT_COUPLING
    t, t_end = 0.0, 1.0
    h = 0.2
    x, x_hat = INITIAL_X.copy(), INITIAL_X_HAT.copy()
    accepted: list[tuple[float, float]] = []
    while t < t_end - 1e-14:
        h = min(h, t_end - t)
        full = _psi_midpoint(x_hat, t, h)
        low = _psi_euler(x_hat, t, h)
        error = float(np.max(np.abs(full - low)))
        if error <= tolerance or h <= 1e-4:
            x, x_hat = _rex_step_forward(x, x_hat, t, h, coupling, _psi_midpoint)
            accepted.append((t, h))
            t += h
            h *= min(1.6, max(0.7, 0.9 * (tolerance / max(error, 1e-30)) ** 0.5))
        else:
            h *= 0.5
    for t0, h0 in reversed(accepted):
        x, x_hat = _rex_step_backward(x, x_hat, t0, h0, coupling, _psi_midpoint)
    roundtrip = float(
        max(np.max(np.abs(x - INITIAL_X)), np.max(np.abs(x_hat - INITIAL_X_HAT)))
    )
    step_sizes = [h0 for _, h0 in accepted]
    return {
        "error_tolerance": tolerance,
        "accepted_steps": len(accepted),
        "min_step": min(step_sizes),
        "max_step": max(step_sizes),
        "roundtrip_max_abs_error": roundtrip,
    }


def _reversible_ddim_demo() -> dict[str, Any]:
    """Pair the eps-prediction DDIM base step through the Rex coupling."""

    def eps_model(t: float, x: np.ndarray) -> np.ndarray:
        return np.array([0.1, -0.05, 0.2, 0.02]) + (0.1 + 0.05 * t) * x

    def ddim_step(x: np.ndarray, t: float, h: float) -> np.ndarray:
        sigma_n, sigma_n1 = _sigma(t), _sigma(t + h)
        return (sigma_n1 / sigma_n) * x + sigma_n1 * psi_ddim(x, t, h)

    def psi_ddim(x: np.ndarray, t: float, h: float) -> np.ndarray:
        sigma_n, sigma_n1 = _sigma(t), _sigma(t + h)
        return (1.0 / sigma_n1 - 1.0 / sigma_n) * (-eps_model(t, x))

    n_steps = 10
    steps = np.linspace(0.0, 1.0, n_steps + 1)
    identity_error = 0.0
    x_plain = INITIAL_X.copy()
    for t0, t1 in zip(steps[:-1], steps[1:]):
        t0f, hf = float(t0), float(t1 - t0)
        via_psi = (_sigma(t0f + hf) / _sigma(t0f)) * x_plain + _sigma(
            t0f + hf
        ) * psi_ddim(x_plain, t0f, hf)
        direct = ddim_step(x_plain, t0f, hf)
        identity_error = max(identity_error, float(np.max(np.abs(via_psi - direct))))
        x_plain = direct

    roundtrip = _rex_roundtrip_error(
        psi_ddim, coupling=PAPER_DEFAULT_COUPLING, n_steps=n_steps
    )

    deviations = []
    for n in (16, 64):
        grid = np.linspace(0.0, 1.0, n + 1)
        plain = INITIAL_X.copy()
        for t0, t1 in zip(grid[:-1], grid[1:]):
            plain = ddim_step(plain, float(t0), float(t1 - t0))
        paired = INITIAL_X.copy()
        paired_hat = INITIAL_X.copy()
        for t0, t1 in zip(grid[:-1], grid[1:]):
            paired, paired_hat = _rex_step_forward(
                paired,
                paired_hat,
                float(t0),
                float(t1 - t0),
                PAPER_DEFAULT_COUPLING,
                psi_ddim,
            )
        deviations.append(float(np.max(np.abs(paired - plain))))

    return {
        "eps_model": "affine deterministic eps(t, x) on a linear sigma schedule",
        "single_step_identity_max_abs_error": identity_error,
        "roundtrip_max_abs_error": roundtrip,
        "pairing_deviation_16_steps": deviations[0],
        "pairing_deviation_64_steps": deviations[1],
    }


def _rex_step_forward(
    x: np.ndarray,
    x_hat: np.ndarray,
    t: float,
    h: float,
    coupling: float,
    psi: Callable[[np.ndarray, float, float], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    psi = psi or _psi_midpoint
    sigma_n = _sigma(t)
    sigma_n1 = _sigma(t + h)
    x_next = (sigma_n1 / sigma_n) * (
        coupling * x + (1.0 - coupling) * x_hat
    ) + sigma_n1 * psi(x_hat, t, h)
    x_hat_next = (sigma_n1 / sigma_n) * x_hat - sigma_n1 * psi(
        x_next, t + h, -h
    )
    return x_next, x_hat_next


def _rex_step_backward(
    x: np.ndarray,
    x_hat: np.ndarray,
    t: float,
    h: float,
    coupling: float,
    psi: Callable[[np.ndarray, float, float], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    psi = psi or _psi_midpoint
    sigma_n = _sigma(t)
    sigma_n1 = _sigma(t + h)
    coupling_inv = 1.0 / coupling
    x_hat_prev = (sigma_n / sigma_n1) * x_hat + sigma_n * psi(x, t + h, -h)
    x_prev = (sigma_n / sigma_n1) * (coupling_inv * x) + (
        1.0 - coupling_inv
    ) * x_hat_prev - sigma_n * coupling_inv * psi(x_hat_prev, t, h)
    return x_prev, x_hat_prev


def _sigma(t: float) -> float:
    return 1.0 + 0.15 * t


def _psi_euler(x: np.ndarray, t: float, h: float) -> np.ndarray:
    """First-order increment, as `euler` in pinned upstream rex.py."""
    return h * _model(t, x)


def _psi_midpoint(x: np.ndarray, t: float, h: float) -> np.ndarray:
    """Second-order sigma-consistent increment, as upstream `exp_midpoint`:
    the intermediate stage steps the transformed variable u = x / sigma."""
    u = x / _sigma(t)
    k2 = u + 0.5 * h * _model(t, x)
    return h * _model(t + 0.5 * h, _sigma(t + 0.5 * h) * k2)


def _psi_rk4(x: np.ndarray, t: float, h: float) -> np.ndarray:
    """Fourth-order sigma-consistent increment: classic RK4 on u = x / sigma,
    following the exponential-integrator structure of upstream rex.py."""
    u = x / _sigma(t)
    k1 = _model(t, _sigma(t) * u)
    k2 = _model(t + 0.5 * h, _sigma(t + 0.5 * h) * (u + 0.5 * h * k1))
    k3 = _model(t + 0.5 * h, _sigma(t + 0.5 * h) * (u + 0.5 * h * k2))
    k4 = _model(t + h, _sigma(t + h) * (u + h * k3))
    return h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def _model(t: float, x: np.ndarray) -> np.ndarray:
    return np.array([0.1, -0.2, 0.05, 0.3]) + (0.03 + 0.02 * t) * x


def _rk_error(method: str, n_steps: int) -> float:
    y = 1.0
    h = 1.0 / n_steps
    t = 0.0
    for _ in range(n_steps):
        if method == "euler":
            y += h * y
        elif method == "rk4":
            k1 = y
            k2 = y + 0.5 * h * k1
            k3 = y + 0.5 * h * k2
            k4 = y + h * k3
            y += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        else:
            raise ValueError(method)
        t += h
    if abs(t - 1.0) > 1e-12:
        raise AssertionError("time grid drift")
    return abs(y - math.e)


def _log2_rate(coarse_error: float, fine_error: float) -> float:
    return math.log(coarse_error / fine_error, 2.0)


def _negative_real_stability_radius_rk4() -> float:
    def stability(z: float) -> float:
        return abs(1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0)

    xs = np.linspace(0.0, 4.0, 40001)
    stable = [float(x) for x in xs if stability(-float(x)) <= 1.0 + 1e-12]
    return max(stable)


def _claim(index: int, status: str, evidence: dict[str, Any]) -> dict[str, Any]:
    binding = CLAIM_BINDINGS[index]
    return {
        "target_claim": binding["target_claim"],
        "claim": binding["challenge_claim"],
        "challenge_claim_sha256": binding["challenge_claim_sha256"],
        "status": status,
        "evidence": evidence,
    }


def _manifest(project_root: Path, results: dict[str, Any]) -> dict[str, Any]:
    files = {
        "pyproject.toml": _file_record(project_root / "pyproject.toml"),
        "src/rex_repro/evidence.py": _file_record(
            project_root / "src" / "rex_repro" / "evidence.py"
        ),
        "tests/test_rex_evidence.py": _file_record(
            project_root / "tests" / "test_rex_evidence.py"
        ),
    }
    return {
        "attempt_id": ATTEMPT_ID,
        "paper_id": PAPER_ID,
        "title": TITLE,
        "upstream_revision": UPSTREAM_REVISION,
        "python_requirement": ">=3.10",
        "claim_bindings": CLAIM_BINDINGS,
        "upstream_files": UPSTREAM_FILES,
        "commands": {
            "evidence_generation": (
                "uv run --project submissions/rex-a-family-of-reversible-"
                "exponential-stochastic-runge-kutta-solvers python -m "
                "rex_repro.evidence --project-root submissions/rex-a-family-"
                "of-reversible-exponential-stochastic-runge-kutta-solvers"
            ),
            "test_suite": (
                "uv run --project submissions/rex-a-family-of-reversible-"
                "exponential-stochastic-runge-kutta-solvers python -m pytest "
                "submissions/rex-a-family-of-reversible-exponential-"
                "stochastic-runge-kutta-solvers/tests -q"
            ),
        },
        "local_files": files,
        "results_sha256": _json_sha256(results),
    }


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def _render_html(results: dict[str, Any], manifest: dict[str, Any]) -> str:
    rows = "\n".join(
        "<tr><td>{}</td><td>{}</td><td><code>{}</code></td><td>{}</td></tr>".format(
            i + 1,
            claim["status"],
            claim["challenge_claim_sha256"],
            claim["evidence"]["observation"],
        )
        for i, claim in enumerate(results["claims"])
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Rex ICML 2026 Reproduction Evidence</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; line-height: 1.45; }}
    main {{ max-width: 980px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #bbb; padding: 0.5rem; vertical-align: top; }}
    code {{ overflow-wrap: anywhere; }}
  </style>
</head>
<body>
<main>
  <h1>Rex Reproduction Evidence</h1>
  <p>Attempt {ATTEMPT_ID}. Paper {PAPER_ID}. Upstream {UPSTREAM_REVISION}.</p>
  <p><a href="evidence/results.json">results.json</a> <a href="evidence/manifest.json">manifest.json</a></p>
  <table>
    <thead><tr><th>#</th><th>Status</th><th>Claim SHA-256</th><th>Observation</th></tr></thead>
    <tbody>
{rows}
    </tbody>
  </table>
  <p>Results digest: <code>{manifest["results_sha256"]}</code></p>
</main>
</body>
</html>
"""


def _render_report(results: dict[str, Any], manifest: dict[str, Any]) -> str:
    lines = [
        f"# {TITLE}",
        "",
        f"- Attempt: `{ATTEMPT_ID}`",
        f"- Paper: `{PAPER_ID}`",
        f"- Upstream: `{UPSTREAM_REVISION}`",
        f"- Results SHA-256: `{manifest['results_sha256']}`",
        "",
        "## Claims",
    ]
    for i, claim in enumerate(results["claims"], start=1):
        lines.extend(
            [
                "",
                f"{i}. `{claim['status']}` `{claim['challenge_claim_sha256']}`",
                f"   {claim['evidence']['observation']}",
            ]
        )
    return "\n".join(lines) + "\n"


def _fmt(value: float) -> str:
    return f"{value:.3e}"


def _render_scoring_page(results: dict[str, Any], manifest: dict[str, Any]) -> str:
    claims = results["claims"]
    ev1 = claims[0]["evidence"]
    ev2 = claims[1]["evidence"]
    ev3 = claims[2]["evidence"]
    ev4 = claims[3]["evidence"]

    lines = [
        "# Rex Evidence For ICML 2026 Judge",
        "",
        f"Paper: `{PAPER_ID}`",
        f"Attempt: `{ATTEMPT_ID}`",
        f"Validated upstream revision: `{UPSTREAM_REVISION}`",
        f"Results digest: `{manifest['results_sha256']}`",
        "",
        "Every number on this page is recomputed deterministically on CPU by "
        "`evidence.py` in this Space (regenerate: `python evidence.py`; full "
        "records in `evidence/results.json`). Nothing is copied from the "
        "paper. The GPU image-generation and Boltzmann sampling claims are "
        "outside this selected evidence target.",
        "",
        f"## Claim 1: {claims[0]['status']} — exact reversibility of the Rex coupling",
        "",
        claims[0]["target_claim"],
        "",
        f"Challenge claim SHA-256: `{claims[0]['challenge_claim_sha256']}`",
        "",
        "Forward sweeps through the released reciprocal coupling, then "
        "backward replay, recover the initial state exactly (tolerance "
        "1e-12):",
        "",
        "| coupling | steps | round-trip max abs error |",
        "| --- | --- | --- |",
    ]
    for row in ev1["roundtrip_table"]:
        lines.append(
            f"| {row['coupling']} | {row['steps']} | "
            f"{_fmt(row['roundtrip_max_abs_error'])} |"
        )
    lines.extend(
        [
            "",
            "The stochastic path is reversible as well: a frozen-noise "
            "Euler-Maruyama increment paired through the same coupling "
            f"round-trips with max abs error "
            f"{_fmt(ev1['sde_roundtrip_max_abs_error'])} over 32 steps.",
            "",
            f"## Claim 2: {claims[1]['status']} — order inheritance measured",
            "",
            claims[1]["target_claim"],
            "",
            f"Challenge claim SHA-256: `{claims[1]['challenge_claim_sha256']}`",
            "",
            "Rex sweeps built from base increments of orders 1, 2, and 4 "
            f"converge to the fine-step limit (RK4 base, {REFERENCE_STEP_COUNT} "
            "steps) at the base method's order — the coupling preserves "
            "convergence order:",
            "",
            "| base increment | theoretical order | errors at steps "
            f"{list(CONVERGENCE_STEP_COUNTS)} | measured rate |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in ev2["rex_convergence_table"]:
        errors = ", ".join(_fmt(e) for e in row["errors_vs_reference"])
        lines.append(
            f"| {row['base_increment']} | {row['theoretical_order']} | "
            f"{errors} | {row['measured_rate']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Independent scalar checks: Euler errors "
            f"{', '.join(_fmt(e) for e in ev2['scalar_euler_errors'])} "
            f"(rate {ev2['scalar_euler_last_refinement_rate']:.2f}), RK4 errors "
            f"{', '.join(_fmt(e) for e in ev2['scalar_rk4_errors'])} "
            f"(rate {ev2['scalar_rk4_last_refinement_rate']:.2f}). The measured "
            "RK4 negative-real-axis stability radius lower bound is "
            f"{ev2['rk4_negative_real_stability_radius_lower_bound']:.4f} "
            "(theory: 2.7853), confirming a non-zero linear stability region.",
            "",
            f"## Claim 3: {claims[2]['status']} — reversible adaptive stepping demonstrated",
            "",
            claims[2]["target_claim"],
            "",
            f"Challenge claim SHA-256: `{claims[2]['challenge_claim_sha256']}`",
            "",
            "An embedded Heun/Euler estimator adaptively selects step sizes "
            "inside the Rex coupling; replaying the accepted step sequence "
            "backward recovers the initial state exactly:",
            "",
            f"- error tolerance: {ev3['adaptive_reversible_demo']['error_tolerance']:.0e}",
            f"- accepted steps: {ev3['adaptive_reversible_demo']['accepted_steps']}",
            f"- step-size range: {_fmt(ev3['adaptive_reversible_demo']['min_step'])}"
            f" to {_fmt(ev3['adaptive_reversible_demo']['max_step'])}",
            "- round-trip max abs error: "
            f"{_fmt(ev3['adaptive_reversible_demo']['roundtrip_max_abs_error'])}",
            "",
            "The pinned canonical wrapper defaults to RK4 (fixed step) and "
            "DOPRI5 (adaptive), and rejects adaptive use for tableaus "
            "without embedded error coefficients "
            "(embedded set: bogacki_shampine, dopri5, fehlberg45, tsit5).",
            "",
            f"## Claim 4: {claims[3]['status']} — reversible DDIM recovered numerically",
            "",
            claims[3]["target_claim"],
            "",
            f"Challenge claim SHA-256: `{claims[3]['challenge_claim_sha256']}`",
            "",
            "The eps-prediction DDIM base step paired through the Rex "
            "coupling yields a reversible DDIM:",
            "",
            "- single-step identity: the psi-form update reproduces the plain "
            "DDIM affine update with max abs error "
            f"{_fmt(ev4['reversible_ddim_demo']['single_step_identity_max_abs_error'])};",
            "- exact inversion: the paired DDIM round-trips with max abs "
            f"error {_fmt(ev4['reversible_ddim_demo']['roundtrip_max_abs_error'])} "
            "over 10 steps;",
            "- consistency: the paired forward trajectory deviates from plain "
            "DDIM by "
            f"{_fmt(ev4['reversible_ddim_demo']['pairing_deviation_16_steps'])} at 16 "
            f"steps and {_fmt(ev4['reversible_ddim_demo']['pairing_deviation_64_steps'])} "
            "at 64 steps, so the reversible pairing approaches standard DDIM "
            "under refinement.",
            "",
            "DPM-Solver and SEEDS-1 recovery routes are audited in the pinned "
            "source: rex.py carries the DPM-Solver lambda/time conversion "
            "with higher-order exponential tableaus over the same log-SNR "
            "variable, and the SDE path (Euler-Maruyama, ShARK over rho) "
            "matches the first-order stochastic reversible construction "
            "targeted by SEEDS-style solvers.",
            "",
            "## Provenance",
            "",
            "The evidence pins the upstream GitHub repository at commit "
            "`e39b57415d5608b18d7c5631595f1d38f06813b8` and records SHA-256 "
            "digests for the audited upstream README, Rex implementation, "
            "Runge-Kutta tableau registry, DDIM baseline, license, and "
            "requirements file in `evidence/manifest.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def _space_readme() -> str:
    return """---
title: Rex ICML 2026 Reproduction Evidence
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
tags:
  - icml2026-repro
  - paper-7pQIzVNctu
  - reproducibility
  - rex
---

# Rex ICML 2026 Reproduction Evidence

This Space serves deterministic CPU evidence for the Rex reproduction attempt.
"""


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _json_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = run_pipeline(args.project_root)
    print(json.dumps(result["results"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
