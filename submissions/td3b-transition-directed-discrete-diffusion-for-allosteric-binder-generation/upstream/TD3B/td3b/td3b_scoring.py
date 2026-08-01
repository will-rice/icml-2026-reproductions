"""
TD3B Scoring Functions
Implements gated allosteric reward combining affinity prediction and directional oracle.
"""

import os
import torch
import numpy as np
from typing import List, Tuple, Optional
from .direction_oracle import DirectionalOracle
from scoring.functions.binding import BindingAffinity


class TD3BRewardFunction:
    """
    Implements the TD3B gated total reward with sigmoid temperature scaling:
        S_total(y; d*, x) = g_ψ(y, x) · σ(d* · (f_φ(y, x) -0.5) / τ)

    where:
        - g_ψ(y, x): affinity predictor (BindingAffinity)
        - σ: sigmoid function σ(z) = 1 / (1 + exp(-z))
        - d* ∈ {+1, -1}: target direction (agonist/antagonist)
        - f_φ(y, x): directional oracle (DirectionalOracle)
          * Directional oracle outputs p(agonist) in [0, 1]
        - τ: temperature parameter (lower = sharper distribution)
        - y: peptide sequence
        - x: target protein sequence

    Note: The placeholder oracle outputs 0.5, which makes (f_φ - 0.5) = 0, resulting in
    neutral gating during initial training before a real oracle is trained.

    Benefits of sigmoid formulation:
        1. Output always in [0, 1] → bounded gated rewards
        2. Temperature τ controls sharpness of selection
        3. Differentiable gating for smooth optimization
        4. Sharper discrimination between aligned and misaligned directions

    OLD FORMULA (replaced):
        S_total(y; d*, x) = g_ψ(y, x) · (d* · f_φ(y, x))
    """

    def __init__(
        self,
        affinity_predictor: BindingAffinity,
        directional_oracle: DirectionalOracle,
        target_direction: float,  # +1 for agonist, -1 for antagonist
        target_protein_tokens: torch.Tensor,
        peptide_tokenizer,
        device: torch.device,
        min_affinity_threshold: float = 0.0,  # Minimum g_ψ for allosteric control
        use_confidence_weighting: bool = True,
        temperature: float = 0.1  # Temperature for sigmoid sharpening
    ):
        """
        Args:
            affinity_predictor: Pretrained g_ψ model (BindingAffinity)
            directional_oracle: Pretrained f_φ model (DirectionalOracle)
            target_direction: d* in {+1, -1} for agonist/antagonist
            target_protein_tokens: Tokenized target protein sequence
            peptide_tokenizer: Tokenizer for converting SMILES to tokens
            device: Computation device
            min_affinity_threshold: Only apply directional control if g_ψ > threshold
            use_confidence_weighting: Whether to use κ(y) for importance weights
            temperature: Temperature τ for sigmoid sharpening (lower = sharper)
                        Default 0.1 makes distribution sharper than standard sigmoid
        """
        self.g_psi = affinity_predictor  # Affinity predictor
        self.f_phi = directional_oracle  # Directional oracle
        self.target_direction = target_direction  # d* ∈ {+1, -1}
        self.protein_tokens = target_protein_tokens
        self.peptide_tokenizer = peptide_tokenizer
        self.device = device
        self.min_affinity_threshold = min_affinity_threshold
        self.use_confidence_weighting = use_confidence_weighting
        self.temperature = temperature  # τ for sigmoid temperature

    def compute_affinity(self, peptide_seqs: List[str]) -> np.ndarray:
        """
        Compute binding affinity g_ψ(y, x).

        Args:
            peptide_seqs: List of peptide SMILES strings
        Returns:
            affinities: (N,) array of affinity scores
        """
        affinities = self.g_psi(peptide_seqs)  # Returns list of scores
        return np.array(affinities)

    def compute_direction(self, peptide_seqs: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute directional bias f_φ(y, x) and confidence κ(y).

        Args:
            peptide_seqs: List of peptide SMILES strings
        Returns:
            directions: (N,) tensor of directional biases
                - DirectionalOracle: p(agonist) in [0, 1]
            confidences: (N,) tensor of confidence scores in [0, 1]
        """
        # Tokenize peptides in a single batch for speed
        peptide_tokens = None
        peptide_token_dict = None
        try:
            peptide_token_dict = self.peptide_tokenizer(
                peptide_seqs,
                return_tensors='pt',
                padding=True
            )
            peptide_token_dict = {k: v.to(self.device) for k, v in peptide_token_dict.items()}
            peptide_tokens = peptide_token_dict.get('input_ids')
        except Exception:
            peptide_tokens_list = []
            for seq in peptide_seqs:
                tokens = self.peptide_tokenizer(seq, return_tensors='pt', padding=True)
                peptide_tokens_list.append(tokens['input_ids'].to(self.device))

            # Batch tokenization (simple stacking, assumes same length after padding)
            try:
                peptide_tokens = torch.cat(peptide_tokens_list, dim=0)  # (N, L)
            except Exception:
                # Fallback: pad to max length
                max_len = max(t.size(1) for t in peptide_tokens_list)
                peptide_tokens = torch.zeros(len(peptide_tokens_list), max_len, dtype=torch.long, device=self.device)
                for i, tokens in enumerate(peptide_tokens_list):
                    peptide_tokens[i, :tokens.size(1)] = tokens[0]

        # Expand protein tokens to batch size
        protein_tokens = self.protein_tokens.expand(len(peptide_seqs), -1)  # (N, L_prot)

        # Compute direction and confidence
        with torch.no_grad():
            if peptide_token_dict is not None and hasattr(self.f_phi, "_normalize_token_dict"):
                directions, confidences = self.f_phi.predict_with_confidence(
                    peptide_token_dict, protein_tokens
                )
            else:
                directions, confidences = self.f_phi.predict_with_confidence(
                    peptide_tokens, protein_tokens
                )

        return directions, confidences

    def compute_gated_reward(
        self,
        peptide_seqs: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute gated total reward with sigmoid temperature scaling.

        NEW FORMULA:
            S_total = g_ψ · σ(d* · (f_φ-0.5) / τ)

        Where:
            - g_ψ: affinity score
            - σ: sigmoid function
            - d*: target direction (+1 or -1)
            - f_φ: directional oracle prediction (in [-1, +1])
            - τ: temperature (lower = sharper distribution)

        OLD FORMULA (replaced):
            S_total = g_ψ · (d* · f_φ)

        Args:
            peptide_seqs: List of peptide SMILES strings
        Returns:
            total_rewards: (N,) array of gated total rewards
            affinities: (N,) array of affinity scores g_ψ
            confidences: (N,) array of confidence scores κ
            directions: (N,) array of directional predictions f_φ
        """
        # Compute affinity g_ψ(y, x)
        affinities = self.compute_affinity(peptide_seqs)  # (N,)

        # Compute directional bias f_φ(y, x) and confidence κ(y)
        directions, confidences = self.compute_direction(peptide_seqs)  # (N,), (N,)
        directions = directions.cpu().numpy()
        confidences = confidences.cpu().numpy()

        # NEW: Sigmoid-based gated reward with temperature scaling
        # S_total = g_ψ · σ(d* · (f_φ-0.5) / τ), use 0.5 as the threshold to make it balanced/symmetric.
        directional_score = self.target_direction * (directions - 0.5)  # (N,) in [-1, +1]

        # Apply temperature scaling (lower τ → sharper sigmoid)
        scaled_score = directional_score / self.temperature  # (N,)

        # Apply sigmoid to get value in [0, 1]
        # σ(x) = 1 / (1 + exp(-x))
        sigmoid_weight = 1.0 / (1.0 + np.exp(-scaled_score))  # (N,) in [0, 1]

        # Gate affinity with sigmoid weight
        gated_rewards = affinities * sigmoid_weight  # (N,)

        # Optional: only apply directional control if affinity is high enough
        # This implements the "allosteric control only for binders" principle
        low_affinity_mask = affinities < self.min_affinity_threshold
        gated_rewards[low_affinity_mask] = affinities[low_affinity_mask] * 0.1  # Downweight

        return gated_rewards, affinities, confidences, directions

    def __call__(
        self,
        input_seqs: List[str]
    ) -> Tuple[np.ndarray, dict]:
        """
        Main interface for reward computation.

        Args:
            input_seqs: List of peptide SMILES strings
        Returns:
            rewards: (N,) array of total rewards
            info: dict with 'affinities', 'confidences', 'directions', 'score_vectors'
        """
        total_rewards, affinities, confidences, directions = self.compute_gated_reward(input_seqs)

        info = {
            'affinities': affinities,
            'confidences': confidences,
            'directions': directions,  # Add direction predictions
            'score_vectors': np.stack([affinities, total_rewards], axis=1)  # (N, 2)
        }

        return total_rewards, info


class TD3BConfidenceWeighting:
    """
    Implements confidence-weighted importance sampling for TD3B.

    The importance weights w(y) are modulated by confidence κ(y):
        w(y) = κ(y) · exp(S_total(y) / α)

    This distinguishes between:
        - Full agonists/antagonists: high κ (|f_φ| ≈ 1)
        - Partial agonists/antagonists: medium κ (|f_φ| ≈ 0.5)
        - Non-selective binders: low κ (|f_φ| ≈ 0)
    """

    def __init__(
        self,
        alpha: float = 0.1,  # Temperature for reward scaling
        min_confidence: float = 0.1  # Minimum confidence to avoid zero weights
    ):
        """
        Args:
            alpha: Temperature parameter for reward scaling
            min_confidence: Minimum confidence threshold
        """
        self.alpha = alpha
        self.min_confidence = min_confidence

    def compute_importance_weights(
        self,
        rewards: np.ndarray,
        confidences: np.ndarray
    ) -> np.ndarray:
        """
        Compute confidence-weighted importance weights.

        Args:
            rewards: (N,) array of total rewards S_total
            confidences: (N,) array of confidence scores κ ∈ [0, 1]
        Returns:
            weights: (N,) array of importance weights
        """
        # Clip confidences to avoid zero weights
        confidences = np.maximum(confidences, self.min_confidence)

        # Compute importance weights: w(y) = κ(y) · exp(S_total / α)
        log_weights = rewards / self.alpha  # (N,)
        weights = confidences * np.exp(log_weights)  # (N,)

        return weights

    def compute_log_importance_weights(
        self,
        rewards: np.ndarray,
        confidences: np.ndarray
    ) -> np.ndarray:
        """
        Compute log importance weights for numerical stability.

        Args:
            rewards: (N,) array of total rewards
            confidences: (N,) array of confidence scores
        Returns:
            log_weights: (N,) array of log importance weights
        """
        # Clip confidences
        confidences = np.maximum(confidences, self.min_confidence)

        # log w(y) = log κ(y) + S_total / α
        log_weights = np.log(confidences) + (rewards / self.alpha)  # (N,)

        return log_weights


# Factory function for creating TD3B reward function
def create_td3b_reward_function(
    affinity_predictor: BindingAffinity,
    target_protein_seq: str,
    target_direction: str,  # 'agonist' or 'antagonist'
    peptide_tokenizer,
    device: torch.device,
    directional_oracle: Optional[DirectionalOracle] = None,
    directional_oracle_checkpoint: Optional[str] = None,
    base_path: Optional[str] = None,
    direction_oracle_tr2d2_checkpoint: Optional[str] = None,
    direction_oracle_tokenizer_vocab: Optional[str] = None,
    direction_oracle_tokenizer_splits: Optional[str] = None,
    direction_oracle_esm_name: str = "facebook/esm2_t33_650M_UR50D",
    direction_oracle_esm_cache_dir: Optional[str] = None,
    direction_oracle_esm_local_files_only: bool = False,
    direction_oracle_max_ligand_length: int = 768,
    direction_oracle_max_protein_length: int = 1024,
    direction_oracle_d_model: int = 256,
    direction_oracle_n_heads: int = 4,
    direction_oracle_n_self_attn_layers: int = 1,
    direction_oracle_n_bmca_layers: int = 2,
    direction_oracle_dropout: float = 0.3,
    **kwargs
) -> TD3BRewardFunction:
    """
    Factory function to create TD3B reward function.

    Args:
        affinity_predictor: Pretrained binding affinity model
        directional_oracle: Preloaded DirectionalOracle instance (optional)
        directional_oracle_checkpoint: Path to Directional oracle checkpoint (optional if instance provided)
        base_path: Base path for default oracle assets
        direction_oracle_tr2d2_checkpoint: TR2-D2 checkpoint for ligand encoder
        direction_oracle_tokenizer_vocab: SMILES tokenizer vocab path
        direction_oracle_tokenizer_splits: SMILES tokenizer splits path
        target_protein_seq: Target protein amino acid sequence
        target_direction: 'agonist' (+1) or 'antagonist' (-1)
        peptide_tokenizer: Tokenizer for peptides
        device: Computation device
        **kwargs: Additional arguments for TD3BRewardFunction

    Returns:
        reward_function: TD3BRewardFunction instance
    """
    if directional_oracle is None:
        if base_path is None:
            base_path = "To Be Added"
        tr2d2_root = os.path.join(base_path, "tr2d2-pep")
        if directional_oracle_checkpoint is None:
            directional_oracle_checkpoint = os.path.join(
                tr2d2_root, "direction_oracle.pt"
            )
        if direction_oracle_tr2d2_checkpoint is None:
            direction_oracle_tr2d2_checkpoint = os.path.join(
                tr2d2_root, "pretrained", "peptune-pretrained.ckpt"
            )
        if direction_oracle_tokenizer_vocab is None:
            direction_oracle_tokenizer_vocab = os.path.join(
                tr2d2_root, "tokenizer", "new_vocab.txt"
            )
        if direction_oracle_tokenizer_splits is None:
            direction_oracle_tokenizer_splits = os.path.join(
                tr2d2_root, "tokenizer", "new_splits.txt"
            )

        directional_oracle = DirectionalOracle(
            model_ckpt=directional_oracle_checkpoint,
            tr2d2_checkpoint=direction_oracle_tr2d2_checkpoint,
            tokenizer_vocab=direction_oracle_tokenizer_vocab,
            tokenizer_splits=direction_oracle_tokenizer_splits,
            esm_name=direction_oracle_esm_name,
            d_model=direction_oracle_d_model,
            n_heads=direction_oracle_n_heads,
            n_self_attn_layers=direction_oracle_n_self_attn_layers,
            n_bmca_layers=direction_oracle_n_bmca_layers,
            dropout=direction_oracle_dropout,
            max_ligand_length=direction_oracle_max_ligand_length,
            max_protein_length=direction_oracle_max_protein_length,
            device=device,
            esm_cache_dir=direction_oracle_esm_cache_dir,
            esm_local_files_only=direction_oracle_esm_local_files_only,
        )

    directional_oracle.eval()

    protein_tokens = directional_oracle.encode_protein(target_protein_seq)

    # Convert direction string to numerical value
    direction_map = {'agonist': +1.0, 'antagonist': -1.0}
    d_star = direction_map.get(target_direction.lower(), +1.0)

    # Create reward function
    reward_function = TD3BRewardFunction(
        affinity_predictor=affinity_predictor,
        directional_oracle=directional_oracle,
        target_direction=d_star,
        target_protein_tokens=protein_tokens,
        peptide_tokenizer=peptide_tokenizer,
        device=device,
        **kwargs
    )

    return reward_function
