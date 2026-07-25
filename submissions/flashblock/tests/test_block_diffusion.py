import pytest
import torch
from flashblock_repro.block_diffusion import BlockDiffusionModel, BlockDiffusionGenerator
from flashblock_repro.metrics import compute_cross_step_stability

def test_block_diffusion_generator_run():
    """Test multi-step block diffusion generation pipeline."""
    torch.manual_seed(42)
    vocab_size = 100
    embed_dim = 64
    num_heads = 4
    num_layers = 2
    block_size = 4
    num_blocks = 3
    num_steps_per_block = 3

    model = BlockDiffusionModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
    )
    generator = BlockDiffusionGenerator(model=model, block_size=block_size, update_threshold=2)

    result = generator.generate(
        num_blocks=num_blocks,
        num_steps_per_block=num_steps_per_block,
        use_flashblock=True
    )

    assert "tokens" in result
    assert "stability_metrics" in result
    assert "speedup_metrics" in result
    assert len(result["tokens"]) == block_size * num_blocks

def test_cross_step_stability_analysis():
    """Verify stability discrepancy between block-external and block-internal attention."""
    torch.manual_seed(99)
    batch_size, num_heads, block_size, d_k = 2, 4, 8, 32

    # Simulate step s and step s+1
    # External attention: highly similar (stable)
    A_out_s = torch.randn(batch_size, num_heads, block_size, d_k)
    A_out_s1 = A_out_s + 0.01 * torch.randn_like(A_out_s)

    # Internal attention: significantly changing
    A_in_s = torch.randn(batch_size, num_heads, block_size, d_k)
    A_in_s1 = torch.randn(batch_size, num_heads, block_size, d_k)

    metrics = compute_cross_step_stability(
        A_out_s=A_out_s, A_out_s1=A_out_s1,
        A_in_s=A_in_s, A_in_s1=A_in_s1
    )

    assert metrics["external_cosine_similarity"] > 0.95
    assert metrics["external_l1_distance"] < 0.05
    assert metrics["internal_cosine_similarity"] < 0.80
    assert metrics["external_cosine_similarity"] > metrics["internal_cosine_similarity"]
