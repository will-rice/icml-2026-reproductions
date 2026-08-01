"""
Unit tests for TTT Streaming Memory Layer and Memory Compression Metrics.
"""

import sys
sys.dont_write_bytecode = True

from video_salmonn_s.ttt_memory import TTTStreamingMemoryLayer, compute_memory_token_reduction

def test_ttt_streaming_memory_forward():
    hidden_dim = 16
    memory_dim = 8
    layer = TTTStreamingMemoryLayer(hidden_dim=hidden_dim, memory_dim=memory_dim)

    sequence = [[0.1 * (i + j) for j in range(hidden_dim)] for i in range(10)]
    out, loss = layer.forward(sequence)

    assert len(out) == 10
    assert len(out[0]) == memory_dim
    assert isinstance(loss, float)
    assert loss >= 0.0

def test_ttt_parameter_freezing_stage2():
    hidden_dim = 16
    memory_dim = 8
    layer = TTTStreamingMemoryLayer(hidden_dim=hidden_dim, memory_dim=memory_dim)

    layer.set_freeze_ttt(True)
    assert layer.ttt_frozen is True

    layer.set_freeze_ttt(False)
    assert layer.ttt_frozen is False

def test_memory_token_reduction_ratio():
    res = compute_memory_token_reduction(seq_len=10000, memory_dim=64, similarity_merge_ratio=0.5)

    assert res["sequence_length"] == 10000
    assert res["ttt_tokens"] == 64
    assert res["similarity_merge_tokens"] == 5000
    assert res["ratio_ttt_to_similarity"] < 0.25
    assert res["achieves_under_25_percent"] is True
