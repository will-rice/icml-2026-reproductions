import pytest
import numpy as np

def test_demix_weight_normalization():
    from demix.merging import normalize_weights

    ratios = {"general": 0.4, "code": 0.3, "math": 0.3}
    weights = normalize_weights(ratios)
    assert pytest.approx(sum(weights.values())) == 1.0
    assert pytest.approx(weights["general"]) == 0.4
    assert pytest.approx(weights["code"]) == 0.3
    assert pytest.approx(weights["math"]) == 0.3

def test_demix_linear_merge():
    from demix.merging import merge_parameters

    comp_a = {"weight1": np.array([1.0, 2.0]), "weight2": np.array([3.0, 4.0])}
    comp_b = {"weight1": np.array([3.0, 6.0]), "weight2": np.array([5.0, 8.0])}

    merged = merge_parameters({"a": comp_a, "b": comp_b}, {"a": 0.5, "b": 0.5})
    np.testing.assert_allclose(merged["weight1"], np.array([2.0, 4.0]))
    np.testing.assert_allclose(merged["weight2"], np.array([4.0, 6.0]))

def test_demix_spearman_evaluation():
    from demix.eval import eval_correlations

    pred_data = {
        "mix_0": {"general_avg": 0.65, "code_avg": 0.40, "math_avg": 0.30},
        "mix_1": {"general_avg": 0.70, "code_avg": 0.50, "math_avg": 0.45},
        "mix_2": {"general_avg": 0.75, "code_avg": 0.60, "math_avg": 0.55},
        "mix_3": {"general_avg": 0.80, "code_avg": 0.70, "math_avg": 0.65},
    }
    gt_data = {
        "mix_0": {"general_avg": 0.60, "code_avg": 0.38, "math_avg": 0.28},
        "mix_1": {"general_avg": 0.68, "code_avg": 0.48, "math_avg": 0.42},
        "mix_2": {"general_avg": 0.74, "code_avg": 0.58, "math_avg": 0.52},
        "mix_3": {"general_avg": 0.79, "code_avg": 0.68, "math_avg": 0.62},
    }

    rho_domain, top25_domain, maintain_domain = eval_correlations(pred_data, gt_data)
    assert rho_domain["avg"] > 0.80
    assert "general_avg" in rho_domain
    assert "code_avg" in rho_domain
    assert "math_avg" in rho_domain

def test_demix_pipeline_run():
    from demix.pipeline import run_demix_reproduction

    bundle = run_demix_reproduction()
    assert bundle["paper_id"] == "uyRIOjFgOn"
    assert bundle["reproduction_status"] == "verified"
    assert bundle["macro_spearman"] >= 0.80
    assert len(bundle["target_claims"]) == 3
