"""LoMDM single-objective joint training architecture (Section 4.1)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableOrderScheduler(nn.Module):
    """Learnable order scheduler that outputs unmasking probabilities across sequence steps."""

    def __init__(self, hidden_dim: int, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.order_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute unmasking order logits per token position [B, L]."""
        order_logits = self.order_net(hidden_states).squeeze(-1)
        return order_logits


class LoMDMBackbone(nn.Module):
    """LoMDM model combining diffusion backbone and learnable order scheduler.

    Section 4.1: LoMDM jointly optimizes token prediction (masked diffusion) and
    learnable order scheduling with a single unified objective.
    """

    def __init__(self, vocab_size: int, seq_len: int, hidden_dim: int = 64, mask_token_id: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.mask_token_id = mask_token_id

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)

        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.token_head = nn.Linear(hidden_dim, vocab_size)
        self.order_scheduler = LearnableOrderScheduler(hidden_dim, seq_len)

    def forward(self, x_t: torch.Tensor, mask: torch.Tensor) -> dict:
        """Forward pass outputting token prediction logits and order logits."""
        # x_t: [B, L], mask: [B, L] bool
        h = self.token_embedding(x_t) + self.pos_embedding
        h = self.encoder(h)

        token_logits = self.token_head(h)
        order_logits = self.order_scheduler(h)

        return {
            "token_logits": token_logits,
            "order_logits": order_logits,
            "hidden_states": h,
        }

    def compute_joint_loss(self, outputs: dict, targets: torch.Tensor, mask: torch.Tensor, target_order: torch.Tensor) -> dict:
        """Single unified objective for joint training of diffusion backbone and order scheduler."""
        token_logits = outputs["token_logits"]
        order_logits = outputs["order_logits"]

        # Token reconstruction loss on masked positions
        if mask.any():
            token_loss = F.cross_entropy(token_logits[mask], targets[mask])
        else:
            token_loss = torch.tensor(0.0, device=token_logits.device)

        # Order scheduling loss (matching target unmasking order profile)
        order_loss = F.mse_loss(F.softmax(order_logits, dim=-1), F.softmax(target_order, dim=-1))

        # Unified single objective (Section 4.1)
        joint_loss = token_loss + order_loss

        return {
            "joint_loss": joint_loss,
            "token_loss": token_loss,
            "order_loss": order_loss,
        }


def verify_lomdm_joint_training(vocab_size: int = 100, seq_len: int = 16, hidden_dim: int = 32) -> dict:
    """Verify single-objective joint optimization of LoMDM backbone and order scheduler."""
    torch.manual_seed(42)
    model = LoMDMBackbone(vocab_size, seq_len, hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Synthetic batch
    B = 4
    targets = torch.randint(1, vocab_size, (B, seq_len))
    mask = torch.rand(B, seq_len) > 0.5
    x_t = targets.clone()
    x_t[mask] = model.mask_token_id
    target_order = torch.randn(B, seq_len)

    initial_loss_info = model.compute_joint_loss(model(x_t, mask), targets, mask, target_order)
    initial_joint_loss = initial_loss_info["joint_loss"].item()

    # Perform 5 optimization steps
    losses = []
    for step in range(5):
        optimizer.zero_grad()
        outputs = model(x_t, mask)
        loss_dict = model.compute_joint_loss(outputs, targets, mask, target_order)
        loss_dict["joint_loss"].backward()
        optimizer.step()
        losses.append(loss_dict["joint_loss"].item())

    final_joint_loss = losses[-1]
    loss_decreased = final_joint_loss < initial_joint_loss

    return {
        "verified": loss_decreased,
        "initial_joint_loss": initial_joint_loss,
        "final_joint_loss": final_joint_loss,
        "loss_history": losses,
        "single_objective_unified": True,
    }
