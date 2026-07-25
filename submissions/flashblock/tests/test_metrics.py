import pytest
import torch
from flashblock_repro.metrics import compute_speedup_and_flops, compute_composition_error

def test_compute_speedup_and_flops():
    """Test computation of attention FLOPs and speedups for dense vs cached FlashBlock."""
    batch_size = 2
    num_heads = 4
    d_k = 64
    context_len = 1024
    block_size = 8
    num_steps = 4
    update_threshold = 2
    
    metrics = compute_speedup_and_flops(
        batch_size=batch_size,
        num_heads=num_heads,
        d_k=d_k,
        context_len=context_len,
        block_size=block_size,
        num_steps=num_steps,
        update_threshold=update_threshold,
    )
    
    assert "dense_flops" in metrics
    assert "flashblock_flops" in metrics
    assert "theoretical_speedup" in metrics
    assert metrics["flashblock_flops"] < metrics["dense_flops"]
    assert metrics["theoretical_speedup"] > 1.20

def test_compute_composition_error():
    """Test computation of L1 and L_infinity error between composed and full attention."""
    torch.manual_seed(42)
    shape = (2, 4, 8, 32)
    A_full = torch.randn(*shape)
    A_composed = A_full + 1e-6 * torch.randn(*shape)
    
    err = compute_composition_error(A_full, A_composed)
    assert "l1_error" in err
    assert "linf_error" in err
    assert err["linf_error"] < 1e-4
