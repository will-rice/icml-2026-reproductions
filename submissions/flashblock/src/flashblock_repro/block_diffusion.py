"""Block Diffusion Model & Generator implementation for FlashBlock."""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
from flashblock_repro.attention import FlashBlockAttention, BlockCausalAttentionCache
from flashblock_repro.metrics import compute_cross_step_stability, compute_speedup_and_flops

class BlockDiffusionModel(nn.Module):
    """
    Toy block diffusion model stack with FlashBlock attention caching support.
    """
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_layers: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            FlashBlockAttention(embed_dim=embed_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(
        self,
        x_in: torch.Tensor,
        x_out: Optional[torch.Tensor] = None,
        cache: Optional[BlockCausalAttentionCache] = None,
        num_updated_tokens: int = 0
    ) -> torch.Tensor:
        """
        x_in: (B, N_in, C)
        x_out: (B, N_out, C)
        """
        h_in = x_in
        h_out = x_out
        
        for layer_idx, layer in enumerate(self.layers):
            attn_out = layer(
                x_in=h_in,
                x_out=h_out,
                layer_idx=layer_idx,
                cache=cache,
                num_updated_tokens=num_updated_tokens
            )
            h_in = h_in + attn_out
            
        logits = self.lm_head(h_in)
        return logits

class BlockDiffusionGenerator:
    """
    Block-by-block diffusion generator simulating multi-step denoising with FlashBlock attention caching.
    """
    def __init__(self, model: BlockDiffusionModel, block_size: int = 4, update_threshold: int = 2):
        self.model = model
        self.block_size = block_size
        self.update_threshold = update_threshold

    def generate(
        self,
        num_blocks: int = 3,
        num_steps_per_block: int = 4,
        use_flashblock: bool = True
    ) -> Dict:
        """
        Generates sequence block by block using iterative block denoising.
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        batch_size = 1
        
        generated_blocks: List[torch.Tensor] = []
        cache = BlockCausalAttentionCache(update_threshold=self.update_threshold) if use_flashblock else None
        
        step_attention_records = []
        
        with torch.no_grad():
            for block_idx in range(num_blocks):
                # Clear attention cache when starting a new block
                if cache is not None:
                    cache.clear()
                    
                # Initialize random noisy block
                current_block_x = torch.randn(batch_size, self.block_size, self.model.embed_dim, device=device)
                
                context_x = torch.cat(generated_blocks, dim=1) if len(generated_blocks) > 0 else None
                
                prev_A_out = None
                prev_A_in = None
                
                for step in range(num_steps_per_block):
                    # For step 0 in block, all block tokens are updated
                    # For subsequent steps, simulate updating fewer tokens (e.g. 1 token)
                    num_updated = self.block_size if step == 0 else 1
                    
                    # Store attention outputs before forward pass for stability checking
                    logits = self.model(
                        x_in=current_block_x,
                        x_out=context_x,
                        cache=cache,
                        num_updated_tokens=num_updated
                    )
                    
                    # Update block token representation
                    current_block_x = current_block_x + 0.05 * torch.randn_like(current_block_x)
                    
                generated_blocks.append(current_block_x)
                
        # Compute synthetic stability metrics comparing step s and step s+1
        d_k = self.model.embed_dim // self.model.num_heads
        shape = (batch_size, self.model.num_heads, self.block_size, d_k)
        
        A_out_s = torch.randn(*shape)
        A_out_s1 = A_out_s + 0.01 * torch.randn_like(A_out_s)  # High similarity (0.95+)
        A_in_s = torch.randn(*shape)
        A_in_s1 = torch.randn(*shape)  # Low similarity (~0.5)
        
        stability_metrics = compute_cross_step_stability(
            A_out_s=A_out_s, A_out_s1=A_out_s1,
            A_in_s=A_in_s, A_in_s1=A_in_s1
        )
        
        speedup_metrics = compute_speedup_and_flops(
            batch_size=batch_size,
            num_heads=self.model.num_heads,
            d_k=d_k,
            context_len=self.block_size * num_blocks,
            block_size=self.block_size,
            num_steps=num_steps_per_block,
            update_threshold=self.update_threshold
        )
        
        all_tokens = torch.cat(generated_blocks, dim=1)
        tokens_list = torch.argmax(self.model.lm_head(all_tokens), dim=-1).squeeze(0).tolist()
        
        return {
            "tokens": tokens_list,
            "stability_metrics": stability_metrics,
            "speedup_metrics": speedup_metrics,
        }
