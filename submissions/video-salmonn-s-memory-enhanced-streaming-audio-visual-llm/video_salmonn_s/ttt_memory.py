"""
Test-Time Training (TTT) Streaming Memory Layer Implementation for video-SALMONN S.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class TTTStreamingMemoryLayer(nn.Module):
    """
    TTT Streaming Memory Layer that uses fast-weight updates
    as streaming memory for long-sequence audio-visual inputs.
    """
    def __init__(self, hidden_dim: int, memory_dim: int, learning_rate: float = 0.01):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.learning_rate = learning_rate
        
        # Base projection weights (frozen in Stage 2 during scale-up)
        self.W_key = nn.Linear(hidden_dim, memory_dim, bias=False)
        self.W_val = nn.Linear(hidden_dim, memory_dim, bias=False)
        
        # Fast weights matrix W_fast: shape (memory_dim, memory_dim)
        self.register_buffer("W_fast", torch.eye(memory_dim))
        
        # Parameter freeze flag for Stage 2 training
        self.ttt_frozen = False

    def set_freeze_ttt(self, freeze: bool):
        """Freeze or unfreeze TTT base parameters (Stage 2 training control)."""
        self.ttt_frozen = freeze
        for p in self.parameters():
            p.requires_grad = not freeze

    def reset_fast_weights(self):
        """Reset fast-weight streaming memory state."""
        self.W_fast = torch.eye(self.memory_dim, device=self.W_key.weight.device)

    def fast_weight_update(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Perform fast-weight update step:
        W_fast_new = W_fast - lr * grad_loss(W_fast * k, v)
        """
        # Reconstruction / long-span prediction loss gradient update
        pred = torch.matmul(k, self.W_fast.t()) # (batch, memory_dim)
        err = pred - v                          # (batch, memory_dim)
        grad = torch.matmul(err.t(), k) / k.size(0) # (memory_dim, memory_dim)
        
        with torch.no_grad():
            self.W_fast = self.W_fast - self.learning_rate * grad
        return self.W_fast

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass over input sequence x of shape (batch, seq_len, hidden_dim).
        Returns projected memory states and prediction loss.
        """
        keys = self.W_key(x) # (batch, seq_len, memory_dim)
        vals = self.W_val(x) # (batch, seq_len, memory_dim)
        
        # Fast weight update across time steps
        batch_size, seq_len, _ = keys.shape
        loss = 0.0
        outputs = []
        for t in range(seq_len):
            kt = keys[:, t, :]
            vt = vals[:, t, :]
            
            # Predict using current fast weights
            pred = torch.matmul(kt, self.W_fast.t())
            step_loss = F.mse_loss(pred, vt)
            loss += step_loss
            
            # Update fast weights
            self.fast_weight_update(kt, vt)
            outputs.append(pred)
            
        out_tensor = torch.stack(outputs, dim=1)
        return out_tensor, loss / seq_len


def compute_memory_token_reduction(
    seq_len: int,
    memory_dim: int,
    similarity_merge_ratio: float = 0.5
) -> Dict[str, Any]:
    """
    Compute memory token footprint comparison between TTT streaming memory
    and similarity-based token merging over a long video stream.
    """
    # TTT maintains a fixed memory representation of size memory_dim * memory_dim tokens equivalent
    ttt_token_footprint = memory_dim # fixed memory size independent of seq_len
    
    # Similarity merging retains a fraction of sequence length (e.g. 50% or linear scaling with seq_len)
    similarity_token_footprint = int(seq_len * similarity_merge_ratio)
    
    ratio = ttt_token_footprint / max(1, similarity_token_footprint)
    
    return {
        "sequence_length": seq_len,
        "ttt_tokens": ttt_token_footprint,
        "similarity_merge_tokens": similarity_token_footprint,
        "ratio_ttt_to_similarity": ratio,
        "achieves_under_25_percent": ratio < 0.25
    }
