import pytest
from steer_like_llm.axbench_evaluation import evaluate_axbench_gemma

def test_evaluate_axbench_gemma():
    res = evaluate_axbench_gemma(seed=42)
    assert "table_3_axbench" in res
    assert res["psr_improves_over_rank1_baselines"] is True

    subsets = ["Early-Layers (1-8)", "Mid-Layers (9-18)", "Late-Layers (19-28)"]
    for s in subsets:
        data = res["table_3_axbench"][s]
        assert data["psr_mse"] > data["rank1_caa"]
        assert data["psr_ll"] > data["rank1_caa"]
