"""
Unit tests for TTT Streaming Memory Layer and Memory Compression Metrics.
"""

import pytest
import torch
from video_salmonn_s.ttt_memory import TTTStreamingMemoryLayer, compute_memory_token_reduction

def test_ttt_streaming_memory_forward():
    hidden_dim = 64
    memory_dim = 32
    layer = TTTStreamingMemoryLayer(hidden_dim=hidden_dim, memory_dim=memory_dim)
    
    x = torch.randn(2, 10, hidden_dim)
    out, loss = layer(x)
    
    assert out.shape == (2, 10, memory_dim)
    assert isinstance(loss.item(), float)
    assert loss.item() >= 0.0

def test_ttt_parameter_freezing_stage2():
    hidden_dim = 64
    memory_dim = 32
    layer = TTTStreamingMemoryLayer(hidden_dim=hidden_dim, memory_dim=memory_dim)
    
    layer.set_freeze_ttt(True)
    for p in layer.parameters():
        assert not p.requires_grad

    layer.set_freeze_ttt(False)
    for p in layer.parameters():
        assert p.requires_grad

def test_memory_token_reduction_ratio():
    # Long video stream sequence of 10,000 frames
    res = compute_memory_token_reduction(seq_len=10000, memory_dim=64, similarity_merge_ratio=0.5)
    
    assert res["sequence_length"] == 10000
    assert res["ttt_tokens"] == 64
    assert res["similarity_merge_tokens"] == 5000
    assert res["ratio_ttt_to_similarity"] < 0.25
    assert res["achieves_under_25_percent"] is True
