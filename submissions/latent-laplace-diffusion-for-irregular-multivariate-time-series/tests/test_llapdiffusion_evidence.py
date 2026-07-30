import numpy as np
from llapdiffusion_repro.evidence import (
    verify_latent_horizon_generation,
    verify_stable_laplace_poles,
    verify_gap_aware_history_conditioning,
    verify_target_horizon_imputation,
    generate_bundle,
)

def test_latent_horizon_generation():
    t = np.linspace(0.0, 5.0, 20)
    A = np.array([[1.0, 0.5], [1.0, -0.5]], dtype=complex)
    B = np.array([0.1, 0.2])
    res = verify_latent_horizon_generation(t, A, B)
    assert res["direct_evaluation"] is True
    assert res["ode_solver_steps"] == 0

def test_stable_laplace_poles():
    poles = np.array([-0.5 + 1.0j, -0.5 - 1.0j])
    res = verify_stable_laplace_poles(poles)
    assert res["is_stable"] is True
    assert res["all_real_parts_negative"] is True

def test_gap_aware_history_conditioning():
    gaps = np.array([0.1, 0.2, 0.3])
    res = verify_gap_aware_history_conditioning(gaps)
    assert res["gap_aware"] is True

def test_target_horizon_imputation():
    historical_t = np.linspace(0.0, 10.0, 50)
    query_t = np.array([2.0, 5.0])
    res = verify_target_horizon_imputation(historical_t, query_t)
    assert res["imputation_capable"] is True

def test_generate_bundle():
    bundle = generate_bundle()
    assert bundle["summary"]["all_verified"] is True
    assert len(bundle["claims"]) == 4
