import pytest
from steer_like_llm.persona_vectors import evaluate_persona_vectors

def test_evaluate_persona_vectors():
    res = evaluate_persona_vectors(seed=42)
    assert "table_1_coherence" in res
    assert "figure_3_rmse" in res
    assert res["all_psr_outperform"] is True
    
    # Verify persona vector scores across reported models
    models = ["LLaMA-3-8B", "Gemma-2-9B", "Qwen-2.5-7B"]
    for m in models:
        m_coherence = res["table_1_coherence"][m]
        assert m_coherence["all_layer_psr_coherence"] > m_coherence["prompt_steering_coherence"]
        
        m_rmse = res["figure_3_rmse"][m]
        assert m_rmse["psr_rmse"] < m_rmse["constant_steering_rmse"]
