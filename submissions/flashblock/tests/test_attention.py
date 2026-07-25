import pytest
import torch
from flashblock_repro.attention import (
    scaled_dot_product_attention,
    log_space_attention_composition,
    BlockCausalAttentionCache,
    FlashBlockAttention,
)

def test_log_space_attention_composition_exactness():
    """Verify log-space composition equals single-pass full attention within numerical precision."""
    torch.manual_seed(42)
    batch_size = 2
    num_heads = 4
    d_k = 64
    seq_len_out = 128
    seq_len_in = 16
    
    Q = torch.randn(batch_size, num_heads, seq_len_in, d_k)
    K_out = torch.randn(batch_size, num_heads, seq_len_out, d_k)
    V_out = torch.randn(batch_size, num_heads, seq_len_out, d_k)
    K_in = torch.randn(batch_size, num_heads, seq_len_in, d_k)
    V_in = torch.randn(batch_size, num_heads, seq_len_in, d_k)
    
    # 1. Full single-pass attention over K_full = [K_out, K_in], V_full = [V_out, V_in]
    K_full = torch.cat([K_out, K_in], dim=2)
    V_full = torch.cat([V_out, V_in], dim=2)
    
    A_full_expected, L_full_expected = scaled_dot_product_attention(Q, K_full, V_full)
    
    # 2. Decomposed attention: block-external and block-internal
    A_out, L_out = scaled_dot_product_attention(Q, K_out, V_out)
    A_in, L_in = scaled_dot_product_attention(Q, K_in, V_in)
    
    # 3. Log-space composition
    A_composed, L_composed = log_space_attention_composition(A_out, L_out, A_in, L_in)
    
    # Assert numerical fidelity
    assert torch.allclose(L_composed, L_full_expected, atol=1e-5, rtol=1e-5)
    assert torch.allclose(A_composed, A_full_expected, atol=1e-5, rtol=1e-5)

def test_block_causal_attention_cache_lifecycle():
    """Test caching, retrieval, and threshold-based updating in BlockCausalAttentionCache."""
    cache = BlockCausalAttentionCache(update_threshold=2)
    
    batch_size, num_heads, block_size, d_k = 2, 4, 8, 32
    layer_idx = 0
    
    A_out = torch.randn(batch_size, num_heads, block_size, d_k)
    L_out = torch.randn(batch_size, num_heads, block_size, 1)
    
    # First step: store cache
    cache.update_cache(layer_idx=layer_idx, A_out=A_out, L_out=L_out)
    assert cache.has_cache(layer_idx)
    
    ret_A, ret_L = cache.get_cache(layer_idx)
    assert torch.equal(ret_A, A_out)
    assert torch.equal(ret_L, L_out)
    
    # Test threshold checking: num_updated < threshold (e.g. 1 < 2) -> reuse
    assert cache.should_reuse_cache(layer_idx=layer_idx, num_updated_tokens=1) is True
    # num_updated >= threshold (e.g. 2 >= 2) -> recompute
    assert cache.should_reuse_cache(layer_idx=layer_idx, num_updated_tokens=2) is False

def test_flashblock_attention_layer_forward():
    """Test FlashBlockAttention module forward pass with and without caching."""
    torch.manual_seed(123)
    embed_dim = 64
    num_heads = 4
    layer = FlashBlockAttention(embed_dim=embed_dim, num_heads=num_heads, update_threshold=2)
    cache = BlockCausalAttentionCache(update_threshold=2)
    
    batch_size = 2
    seq_len_out = 64
    block_size = 8
    
    x_out = torch.randn(batch_size, seq_len_out, embed_dim)
    x_in = torch.randn(batch_size, block_size, embed_dim)
    
    # Step 1: full compute & populate cache (num_updated = 8 >= 2)
    out_step1 = layer(x_in=x_in, x_out=x_out, layer_idx=0, cache=cache, num_updated_tokens=8)
    assert out_step1.shape == (batch_size, block_size, embed_dim)
    assert cache.has_cache(0)
    
    # Step 2: cached reuse (num_updated = 1 < 2)
    out_step2 = layer(x_in=x_in, x_out=x_out, layer_idx=0, cache=cache, num_updated_tokens=1)
    assert out_step2.shape == (batch_size, block_size, embed_dim)
