"""Unit tests for RelayCaching reproduction."""

import json
from pathlib import Path
import pytest
import numpy as np

from relaycaching.cache_reuse import RelayCacheEngine, DecodeToPrefillAligner
from relaycaching.profiler import LayerRangeProfiler, TokenSelector


def test_macro_alignment():
    aligner = DecodeToPrefillAligner(num_layers=16, hidden_dim=64)
    kv1 = np.ones((16, 128, 64))
    kv2 = np.ones((16, 128, 64))
    sim = aligner.measure_macro_alignment(kv1, kv2)
    assert pytest.approx(sim, 0.0001) == 1.0


def test_layer_profiler():
    profiler = LayerRangeProfiler(num_layers=8)
    d_kv = np.random.randn(8, 64, 32)
    p_kv = d_kv.copy()
    # Add noise to middle layer
    p_kv[3] += np.random.randn(64, 32) * 5.0

    sims = profiler.compute_layer_similarities(d_kv, p_kv)
    assert len(sims) == 8
    critical = profiler.identify_critical_layers(sims, threshold=0.95)
    assert 3 in critical


def test_token_selector():
    selector = TokenSelector()
    d_layer = np.random.randn(100, 32)
    p_layer = d_layer + np.random.randn(100, 32) * 0.1
    selected = selector.select_tokens_for_rectification(d_layer, p_layer, reuse_target_ratio=0.8)
    assert len(selected) == 20


def test_relay_cache_engine_workflow():
    engine = RelayCacheEngine(num_layers=16, hidden_dim=64)
    res = engine.run_multi_agent_workflow("GSM8K", seq_len=512, num_agents=3)
    assert res["reuse_rate"] >= 0.80
    assert res["per_agent_ttft_speedup"] > 1.0


def test_cumulative_benchmark():
    engine = RelayCacheEngine(num_layers=16, hidden_dim=64)
    res = engine.run_cumulative_context_benchmark(max_context_length=2048, steps=4)
    assert res["avg_speedup_vs_full"] > 1.0
    assert res["avg_speedup_vs_kvcomm"] > 1.0


def test_evidence_file_integrity():
    evidence_path = Path(__file__).parent.parent / "evidence.json"
    if evidence_path.exists():
        with open(evidence_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["paper_id"] == "1tbhBSXcyX"
        assert "claim_verifications" in data
        assert len(data["claim_verifications"]) == 6
