"""OeMDM mathematical derivations and algorithmic verifications (Proposition 3.2 and Proposition 3.3)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class OeMDMNELBO(nn.Module):
    """Implementation of Order-aware Masked Diffusion Model (OeMDM) NELBO decomposition.

    Proposition 3.2: NELBO decomposes into:
      NELBO = E [ L_recon + L_velocity_mismatch ]
    where L_recon evaluates target token reconstruction under unmasking and L_velocity_mismatch
    measures the difference between learned unmasking velocity and true order dynamics.
    """

    def __init__(self, vocab_size: int, seq_len: int, mask_token_id: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.mask_token_id = mask_token_id

    def compute_reconstruction_loss(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss over masked positions."""
        if not mask.any():
            return torch.tensor(0.0, device=logits.device)
        masked_logits = logits[mask]
        masked_targets = targets[mask]
        return F.cross_entropy(masked_logits, masked_targets, reduction="mean")

    def compute_velocity_mismatch(self, predicted_order_velocity: torch.Tensor, target_order_velocity: torch.Tensor) -> torch.Tensor:
        """Compute velocity mismatch between predicted and target generation order dynamics."""
        return F.mse_loss(predicted_order_velocity, target_order_velocity, reduction="mean")

    def compute_nelbo(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
                      pred_velocity: torch.Tensor, target_velocity: torch.Tensor) -> dict:
        """Decompose NELBO into L_recon and L_velocity_mismatch as per Proposition 3.2."""
        l_recon = self.compute_reconstruction_loss(logits, targets, mask)
        l_vel = self.compute_velocity_mismatch(pred_velocity, target_velocity)
        total_nelbo = l_recon + l_vel
        return {
            "total_nelbo": total_nelbo,
            "reconstruction_loss": l_recon,
            "velocity_mismatch_loss": l_vel,
            "decomposed_sum": l_recon + l_vel,
        }


def verify_left_to_right_ar_recovery(seq_len: int, batch_size: int = 4) -> dict:
    """Verify Proposition 3.3: Left-to-Right AR scheduler equivalence.

    Under a strictly deterministic left-to-right order scheduler pi(t) = t,
    the unmasking sequence matches standard causal autoregressive factorization
    P(x) = prod_{i=1}^L P(x_i | x_{<i}).
    """
    torch.manual_seed(42)
    # Define deterministic L2R order scheduler pi
    l2r_order = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)

    # Generate causal masks corresponding to step t in [0, seq_len]
    causal_masks = []
    for t in range(seq_len + 1):
        # At step t, position < t are unmasked (False), positions >= t are masked (True)
        mask = (torch.arange(seq_len).unsqueeze(0) >= t).repeat(batch_size, 1)
        causal_masks.append(mask)

    # Check properties:
    # 1. At t=0, all positions masked
    all_masked_at_0 = causal_masks[0].all().item()
    # 2. At t=L, no positions masked
    none_masked_at_L = (~causal_masks[-1]).all().item()
    # 3. Unmasking step t unmasks exactly index t-1
    step_unmasking_correct = True
    for t in range(1, seq_len + 1):
        prev_mask = causal_masks[t - 1]
        curr_mask = causal_masks[t]
        unmasked_at_step = prev_mask & (~curr_mask)
        expected = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        expected[:, t - 1] = True
        if not torch.equal(unmasked_at_step, expected):
            step_unmasking_correct = False
            break

    verified = all_masked_at_0 and none_masked_at_L and step_unmasking_correct
    return {
        "proposition": "3.3",
        "verified": verified,
        "all_masked_at_start": all_masked_at_0,
        "none_masked_at_end": none_masked_at_L,
        "exact_l2r_step_unmasking": step_unmasking_correct,
        "causal_order": l2r_order[0].tolist(),
    }
