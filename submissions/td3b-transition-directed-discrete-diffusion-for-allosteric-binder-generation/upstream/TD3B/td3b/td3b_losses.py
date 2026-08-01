"""
TD3B Loss Functions
Implements contrastive loss for separating agonist/antagonist embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ContrastiveLoss(nn.Module):
    """
    Margin-based contrastive loss for separating agonist and antagonist embeddings.

    For a pair of sequences (y_i, y_j):
        - If both are agonists OR both are antagonists (similar): minimize distance
        - If one is agonist and one is antagonist (dissimilar): maximize distance

    Loss formula:
        L_ctr = (1 - y_ij) * 0.5 * d²
              + y_ij * 0.5 * max(0, margin - d)²

    where:
        - d = ||emb_i - emb_j||_2 (Euclidean distance)
        - y_ij = 0 if similar, 1 if dissimilar
        - margin = minimum distance between dissimilar pairs
    """

    def __init__(self, margin: float = 1.0, distance_metric: str = 'euclidean', adaptive_margin: bool = False):
        """
        Args:
            margin: Minimum distance between dissimilar pairs (base margin)
            distance_metric: 'euclidean' or 'cosine'
            adaptive_margin: If True, adjust margin based on actual dissimilar distances
        """
        super().__init__()
        self.base_margin = margin
        self.distance_metric = distance_metric
        self.adaptive_margin = adaptive_margin

    def compute_distance(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise distance between embeddings.

        Args:
            emb1: (batch_size, embedding_dim)
            emb2: (batch_size, embedding_dim)
        Returns:
            distances: (batch_size,)
        """
        if self.distance_metric == 'euclidean':
            # L2 distance
            distances = torch.norm(emb1 - emb2, p=2, dim=-1)  # (B,)
        elif self.distance_metric == 'cosine':
            # Cosine distance = 1 - cosine_similarity
            cos_sim = F.cosine_similarity(emb1, emb2, dim=-1)  # (B,)
            distances = 1.0 - cos_sim
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        return distances

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        confidences: Optional[torch.Tensor] = None,
        debug: bool = False
    ) -> torch.Tensor:
        """
        Compute contrastive loss for a batch.

        Args:
            embeddings: (batch_size, embedding_dim) sequence embeddings
            labels: (batch_size,) directional labels in {-1, +1}
                +1 = agonist, -1 = antagonist
            confidences: (batch_size,) oracle confidence scores; pairs with product <= 0 are masked out
            debug: If True, print detailed debugging information
        Returns:
            loss: scalar contrastive loss
        """
        batch_size = embeddings.size(0)
        if batch_size < 2:
            if debug:
                print(f"[ContrastiveLoss DEBUG] Batch size {batch_size} < 2, returning 0 loss")
            return torch.tensor(0.0, device=embeddings.device)

        if confidences is not None:
            if not torch.is_tensor(confidences):
                confidences = torch.as_tensor(confidences, device=embeddings.device)
            else:
                confidences = confidences.to(embeddings.device)
            confidences = confidences.view(-1)
            if confidences.numel() != batch_size:
                raise ValueError(
                    f"Confidences size {confidences.numel()} does not match batch size {batch_size}"
                )

        # Compute pairwise distances (all pairs)
        if self.distance_metric == 'euclidean':
            distances = torch.cdist(embeddings, embeddings, p=2)  # (B, B)
        elif self.distance_metric == 'cosine':
            emb_norm = F.normalize(embeddings, p=2, dim=-1)
            distances = 1.0 - torch.matmul(emb_norm, emb_norm.T)  # (B, B)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        # Compute pairwise similarity labels
        # y_ij = 0 if same class (both agonist or both antagonist)
        # y_ij = 1 if different class
        labels = labels.view(-1)
        labels_expanded = labels.unsqueeze(1)  # (B, 1)
        label_product = labels_expanded * labels_expanded.T  # (B, B)
        # label_product > 0 means same class (both +1 or both -1)
        # label_product < 0 means different class
        dissimilar_mask = (label_product < 0)  # (B, B) bool

        # Exclude diagonal
        eye_mask = torch.eye(batch_size, device=embeddings.device, dtype=torch.bool)
        pos_mask = (~dissimilar_mask) & ~eye_mask
        neg_mask = dissimilar_mask & ~eye_mask

        # Apply confidence mask: remove pairs with confidence product <= 0
        conf_mask = None
        if confidences is not None:
            conf_product = confidences.unsqueeze(0) * confidences.unsqueeze(1)
            conf_mask = conf_product > 0
            pos_mask = pos_mask & conf_mask
            neg_mask = neg_mask & conf_mask

        # Adaptive margin: set margin based on actual dissimilar distances
        if self.adaptive_margin and neg_mask.any():
            # Get all dissimilar distances
            dissimilar_distances = distances[neg_mask]
            # Set margin to 150% of mean dissimilar distance
            # This ensures there's always room for optimization
            adaptive_margin = 1.5 * dissimilar_distances.mean().item()
            # Use max of base_margin and adaptive_margin
            margin = max(self.base_margin, adaptive_margin)
        else:
            margin = self.base_margin

        pos_count = pos_mask.sum()
        neg_count = neg_mask.sum()
        total_pairs = pos_count + neg_count
        if total_pairs.item() == 0:
            if debug:
                print("[ContrastiveLoss DEBUG] No valid pairs after filtering, returning 0 loss")
            return torch.tensor(0.0, device=embeddings.device)

        # Contrastive loss
        # For similar pairs: minimize squared distance
        # For dissimilar pairs: squared hinge loss with margin
        pos_loss = distances[pos_mask].pow(2).sum() / (pos_count + 1e-8)
        neg_loss = torch.clamp(margin - distances[neg_mask], min=0.0).pow(2).sum() / (neg_count + 1e-8)
        loss = pos_loss + neg_loss

        if debug:
            print(f"\n[ContrastiveLoss DEBUG]")
            print(f"  Batch size: {batch_size}")
            print(f"  Labels: {labels.cpu().tolist()}")
            print(f"  Unique labels: {torch.unique(labels).cpu().tolist()}")
            print(f"  Embedding shape: {embeddings.shape}")
            print(f"  Embedding norm (mean): {embeddings.norm(dim=-1).mean().item():.4f}")
            print(f"  Embedding norm (std): {embeddings.norm(dim=-1).std().item():.4f}")
            valid_mask = pos_mask | neg_mask
            if valid_mask.any():
                valid_dists = distances[valid_mask]
                print(f"  Distance stats (valid pairs): mean={valid_dists.mean().item():.4f} "
                      f"min={valid_dists.min().item():.4f} max={valid_dists.max().item():.4f}")
            if self.adaptive_margin and neg_mask.any():
                print(f"  Margin: {margin:.4f} (adaptive, base={self.base_margin})")
            else:
                print(f"  Margin: {margin:.4f} (fixed)")
            print(f"  Num similar pairs: {pos_count.item():.0f}")
            print(f"  Num dissimilar pairs: {neg_count.item():.0f}")
            if conf_mask is not None:
                print(f"  Confidence-passing pairs: {conf_mask.sum().item():.0f}")
            print(f"  Similar loss (mean): {pos_loss.item():.4f}")
            print(f"  Dissimilar loss (mean): {neg_loss.item():.4f}")
            print(f"  Total loss: {loss.item():.4f}")

            # Show which dissimilar pairs have margin violations
            margin_violations = (distances < margin) & neg_mask
            if margin_violations.sum() > 0:
                print(f"  Margin violations: {margin_violations.sum().item():.0f} dissimilar pairs have distance < margin")
            else:
                print(f"  Margin violations: 0 (all dissimilar pairs are already separated)")

        return loss


class InfoNCELoss(nn.Module):
    """
    Alternative: InfoNCE contrastive loss (used in SimCLR, CLIP).
    Treats agonists as positive class, antagonists as negative class.

    For each agonist, pull it close to other agonists and push away from antagonists.
    For each antagonist, pull it close to other antagonists and push away from agonists.
    """

    def __init__(self, temperature: float = 0.1):
        """
        Args:
            temperature: Temperature parameter for softmax
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        confidences: Optional[torch.Tensor] = None,
        debug: bool = False
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            embeddings: (batch_size, embedding_dim)
            labels: (batch_size,) in {-1, +1}
            confidences: (batch_size,) oracle confidence scores; pairs with product <= 0 are masked out
            debug: Unused (kept for API compatibility)
        Returns:
            loss: scalar
        """
        batch_size = embeddings.size(0)
        if confidences is not None:
            if not torch.is_tensor(confidences):
                confidences = torch.as_tensor(confidences, device=embeddings.device)
            else:
                confidences = confidences.to(embeddings.device)
            confidences = confidences.view(-1)
            if confidences.numel() != batch_size:
                raise ValueError(
                    f"Confidences size {confidences.numel()} does not match batch size {batch_size}"
                )
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device)

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=-1)  # (B, D)

        # Compute similarity matrix
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature  # (B, B)

        # Create positive/negative masks
        labels_expanded = labels.unsqueeze(1)  # (B, 1)
        label_product = labels_expanded * labels_expanded.T  # (B, B)
        positive_mask = (label_product > 0)  # Same class
        negative_mask = (label_product < 0)  # Different class

        # Remove self-similarity
        positive_mask.fill_diagonal_(0)

        if confidences is not None:
            conf_product = confidences.unsqueeze(0) * confidences.unsqueeze(1)
            conf_mask = conf_product > 0
            positive_mask = positive_mask & conf_mask
            negative_mask = negative_mask & conf_mask

        # For each sample, compute InfoNCE loss
        # log( exp(sim_pos) / (exp(sim_pos) + sum(exp(sim_neg))) )
        losses = []
        for i in range(batch_size):
            # Positive samples
            pos_sims = similarity[i][positive_mask[i]]  # (num_pos,)
            # Negative samples
            neg_sims = similarity[i][negative_mask[i]]  # (num_neg,)

            # Check if there are positive samples
            if pos_sims.numel() == 0:
                continue

            # LogSumExp for numerical stability
            pos_exp = torch.exp(pos_sims)  # (num_pos,)
            neg_exp = torch.exp(neg_sims)  # (num_neg,)

            if neg_exp.numel() == 0:
                continue

            # Average over positive samples
            denominator = pos_exp.sum() + neg_exp.sum()
            loss_i = -torch.log(pos_exp.sum() / (denominator + 1e-8))
            losses.append(loss_i)

        if len(losses) == 0:
            return torch.tensor(0.0, device=embeddings.device)

        return torch.stack(losses).mean()


class TD3BTotalLoss:
    """
    Combined TD3B loss: L_total = L_WDCE + λ * L_ctr + β * L_KL

    Components:
        - L_WDCE: Weighted Denoising Cross-Entropy (from TR2-D2)
        - L_ctr: Contrastive loss for agonist/antagonist separation
        - L_KL: KL divergence regularization between policy and reference model
    """

    def __init__(
        self,
        contrastive_weight: float = 0.1,
        contrastive_margin: float = 1.0,
        contrastive_type: str = 'margin',  # 'margin' or 'infonce'
        kl_beta: float = 0.1,  # β coefficient for KL divergence
        reference_model: Optional[nn.Module] = None,
        adaptive_margin: bool = True  # Enable adaptive margin by default
    ):
        """
        Args:
            contrastive_weight: λ coefficient for contrastive loss
            contrastive_margin: Margin for margin-based contrastive loss (base margin if adaptive)
            contrastive_type: Type of contrastive loss ('margin' or 'infonce')
            kl_beta: β coefficient for KL divergence regularization
            reference_model: Frozen reference model for KL divergence (deepcopy of pretrained)
            adaptive_margin: If True, automatically adjust margin based on dissimilar distances
        """
        self.contrastive_weight = contrastive_weight
        self.kl_beta = kl_beta
        self.reference_model = reference_model

        # Freeze reference model if provided
        if self.reference_model is not None:
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad = False

            # Verify all parameters are frozen
            assert all(not p.requires_grad for p in self.reference_model.parameters()), \
                "ERROR: Reference model has parameters with requires_grad=True!"

        if contrastive_type == 'margin':
            self.contrastive_loss = ContrastiveLoss(
                margin=contrastive_margin,
                distance_metric='euclidean',
                adaptive_margin=adaptive_margin
            )
        elif contrastive_type == 'infonce':
            self.contrastive_loss = InfoNCELoss(temperature=0.1)
        else:
            raise ValueError(f"Unknown contrastive type: {contrastive_type}")

    def compute_kl_categorical(
        self,
        log_p: torch.Tensor,
        log_ref_p: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence between categorical distributions.

        KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
                   = Σ P(x) * (log P(x) - log Q(x))

        Args:
            log_p: (B, L, Vocab) log-probabilities from policy model
            log_ref_p: (B, L, Vocab) log-probabilities from reference model
        Returns:
            kl: (B, L) KL divergence per position
        """
        # Convert log-probs to probabilities
        p = torch.exp(log_p)  # (B, L, Vocab)

        # KL divergence element-wise
        kl_elementwise = p * (log_p - log_ref_p)  # (B, L, Vocab)

        # Handle numerical issues: 0 * log(0) should be 0
        # Replace NaNs or Infs that occur at -inf locations with 0
        kl_elementwise = torch.where(
            torch.isfinite(kl_elementwise),
            kl_elementwise,
            torch.zeros_like(kl_elementwise)
        )

        # Sum over vocabulary dimension
        kl = kl_elementwise.sum(dim=-1)  # (B, L)

        return kl

    def compute_kl_loss(
        self,
        policy_model: nn.Module,
        sequences: torch.Tensor,
        attn_mask: torch.Tensor,
        sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence loss between policy model and reference model.

        Args:
            policy_model: Current policy model (being trained)
            sequences: (B, L) input sequences
            attn_mask: (B, L) attention mask
            sigma: (B,) noise schedule
        Returns:
            kl_loss: Scalar KL divergence loss
        """
        if self.reference_model is None:
            return torch.tensor(0.0, device=sequences.device)

        # Ensure reference model is in eval mode
        assert not self.reference_model.training, \
            "ERROR: Reference model is in training mode! It should always be in eval mode."

        # Forward through policy model (already computed in WDCE, but need logits)
        policy_logits = policy_model(sequences, attn_mask=attn_mask, sigma=sigma)  # (B, L, Vocab)

        # Forward through reference model (frozen, no gradients)
        with torch.no_grad():
            ref_logits = self.reference_model(sequences, attn_mask=attn_mask, sigma=sigma)  # (B, L, Vocab)

        # Convert to log-probabilities
        log_p = F.log_softmax(policy_logits, dim=-1)  # (B, L, Vocab)
        log_ref_p = F.log_softmax(ref_logits, dim=-1)  # (B, L, Vocab)

        # Compute KL divergence
        kl_per_position = self.compute_kl_categorical(log_p, log_ref_p)  # (B, L)

        # Mask out padding positions
        kl_masked = kl_per_position * attn_mask.float()  # (B, L)

        # Average over all non-padding positions
        num_valid = attn_mask.float().sum()
        kl_loss = kl_masked.sum() / (num_valid + 1e-8)

        return kl_loss

    def compute_loss(
        self,
        wdce_loss: torch.Tensor,
        embeddings: torch.Tensor,
        directional_labels: torch.Tensor,
        confidences: Optional[torch.Tensor] = None,
        kl_loss: Optional[torch.Tensor] = None,
        debug: bool = False
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute total TD3B loss.

        Args:
            wdce_loss: Precomputed WDCE loss (scalar)
            embeddings: (batch_size, embedding_dim) sequence embeddings from MDLM
            directional_labels: (batch_size,) labels in {-1, +1}
            confidences: (batch_size,) oracle confidence scores; pairs with product <= 0 are masked out
            kl_loss: Precomputed KL divergence loss (optional)
            debug: If True, enable debugging output in contrastive loss
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        # Contrastive loss (pass debug flag)
        contrastive_loss = self.contrastive_loss(
            embeddings,
            directional_labels,
            confidences=confidences,
            debug=debug
        )

        # KL divergence loss
        if kl_loss is None:
            kl_loss = torch.tensor(0.0, device=embeddings.device)

        # Total loss: L_total = L_WDCE + λ * L_ctr + β * L_KL
        total_loss = wdce_loss + self.contrastive_weight * contrastive_loss + self.kl_beta * kl_loss

        loss_dict = {
            'total_loss': total_loss.item(),
            'wdce_loss': wdce_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'kl_loss': kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
        }

        return total_loss, loss_dict


def extract_embeddings_from_mdlm(
    model,
    sequences: torch.Tensor,
    pool_method: str = 'mean'
) -> torch.Tensor:
    """
    Extract sequence-level embeddings from MDLM backbone.

    Args:
        model: MDLM model with backbone (Roformer)
        sequences: (batch_size, seq_len) token sequences
        pool_method: 'mean', 'max', or 'cls'
    Returns:
        embeddings: (batch_size, hidden_dim)
    """
    # Create attention mask (1 for real tokens, 0 for padding)
    attn_mask = (sequences != 0).long()  # (B, L)

    # Forward through Roformer backbone to get hidden states
    # IMPORTANT: DO NOT use torch.no_grad() here - we need gradients for backprop!
    # Access the underlying RoFormerForMaskedLM model and request hidden states
    outputs = model.backbone.model(
        input_ids=sequences,
        attention_mask=attn_mask,
        output_hidden_states=True,
        return_dict=True
    )

    # Extract last hidden state from outputs
    # outputs.hidden_states is a tuple of (embedding_output, layer1, layer2, ..., layerN)
    # We want the last layer's hidden states
    hidden_states = outputs.hidden_states[-1]  # (B, L, D)

    # Pool to get sequence-level embeddings
    if pool_method == 'mean':
        # Mean pooling (ignore padding)
        mask = attn_mask.float().unsqueeze(-1)  # (B, L, 1)
        pooled = (hidden_states * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # (B, D)
    elif pool_method == 'max':
        # Max pooling
        pooled = hidden_states.max(dim=1)[0]  # (B, D)
    elif pool_method == 'cls':
        # Use first token (CLS-style)
        pooled = hidden_states[:, 0, :]  # (B, D)
    else:
        raise ValueError(f"Unknown pool method: {pool_method}")

    return pooled
