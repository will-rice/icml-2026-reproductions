"""
TD3B-specific MCTS modifications.
Extends the base MCTS to support directional rewards and confidence weighting.
"""

import numpy as np
import torch
from mcts.peptide_mcts import MCTS as BaseMCTS
from .td3b_scoring import TD3BRewardFunction, TD3BConfidenceWeighting


class TD3B_MCTS(BaseMCTS):
    """
    TD3B version of MCTS that:
    1. Uses gated directional rewards instead of multi-objective scalarization
    2. Stores directional labels and confidence scores in the buffer
    3. Applies confidence-weighted importance sampling
    """

    def __init__(
        self,
        args,
        diffusion_model,
        td3b_reward_function: TD3BRewardFunction,
        confidence_weighting: TD3BConfidenceWeighting,
        mask_index: int,
        buffer_size: int = 100,
        noise=None,
        tokenizer=None,
        enforce_validity=None,
    ):
        """
        Args:
            args: Configuration arguments
            diffusion_model: MDLM model for sampling
            td3b_reward_function: TD3BRewardFunction instance
            confidence_weighting: TD3BConfidenceWeighting instance
            mask_index: Token ID for masked positions
            buffer_size: Maximum buffer size
            noise: Noise schedule
            tokenizer: Peptide tokenizer
            enforce_validity: Validity gate toggle for MCTS expansion. None
                (default) consults args.enforce_validity, falling back to True
                (unchanged behavior). False disables the is_peptide gate so
                invalid peptides are kept/scored on pure affinity x direction.
        """
        # Initialize base MCTS (will set self.rewardFunc later)
        # Note: base MCTS expects 'policy_model' not 'diffusion_model'
        # Create a minimal config object for base MCTS
        class MinimalConfig:
            def __init__(self):
                self.noise = type('obj', (object,), {
                    'type': 'loglinear',
                    'sigma_min': 1e-4,
                    'sigma_max': 20
                })()
        config = MinimalConfig()

        super().__init__(
            args=args,
            config=config,
            policy_model=diffusion_model,
            pretrained=diffusion_model,  # Use same model
            score_func_names=['affinity', 'gated_reward', 'placeholder1', 'placeholder2', 'placeholder3'],  # 5 objectives
            enforce_validity=enforce_validity,
        )

        # Set TD3B-specific attributes
        self.td3b_reward_func = td3b_reward_function
        self.confidence_weighting = confidence_weighting
        self.mask_index = mask_index
        self.buffer_size = buffer_size
        self.noise = noise
        self.tokenizer = tokenizer if tokenizer is not None else diffusion_model.tokenizer

        # Override num_obj to ensure it's 5 (matching our padded rewards)
        self.num_obj = 5

        # Override rewardFunc for compatibility
        self.rewardFunc = self._td3b_reward_wrapper

    def _td3b_reward_wrapper(self, input_seqs):
        """
        Wrapper to make TD3BRewardFunction compatible with existing MCTS interface.
        Returns (N, 5) array to match base MCTS expectations.
        The 5 columns are: [affinity, gated_reward, 0, 0, 0] (padding last 3)
        """
        import numpy as np
        total_rewards, info = self.td3b_reward_func(input_seqs)
        # info contains: 'affinities', 'confidences', 'score_vectors'

        # Store confidences for later use (attach to self for access in updateBuffer)
        self._last_confidences = info['confidences']

        # Pad score_vectors from (N, 2) to (N, 5) to match base MCTS
        # Original columns: [affinity, gated_reward]
        # Padded to: [affinity, gated_reward, 0, 0, 0]
        score_vectors = info['score_vectors']  # (N, 2)
        padded = np.zeros((score_vectors.shape[0], 5))
        padded[:, :2] = score_vectors  # Copy affinity and gated_reward

        return padded

    def updateBuffer(self, x_final, log_rnd, score_vectors, childSequences):
        """
        TD3B version: stores directional labels and confidence scores.

        Args:
            x_final: (B, L) final sequence tokens
            log_rnd: (B,) log importance weights (trajectory-level)
            score_vectors: (B, K) score arrays
            childSequences: List of B SMILES strings
        Returns:
            traj_log_rnds: (B,) updated log importance weights
            scalar_rewards: (B,) scalar rewards
        """
        B = x_final.shape[0]
        traj_log_rnds, scalar_rewards = [], []

        # Get confidences from last reward computation
        confidences = getattr(self, '_last_confidences', np.ones(B))

        for i in range(B):
            sv = np.asarray(score_vectors[i], dtype=float)  # [affinity, gated_reward]
            confidence = confidences[i]

            # For TD3B, the "scalar reward" is the gated reward (second element)
            scalar_reward = float(sv[1])  # gated_reward = g_ψ · (d* · sigmoid(f_φ-0.5)/α)

            # Compute confidence-weighted importance weight
            # w(y) = κ(y) · exp(S_total / α)
            # In log space: log w(y) = log κ(y) + S_total / α
            log_confidence = np.log(np.maximum(confidence, self.confidence_weighting.min_confidence))
            traj_log_rnd = log_rnd[i] + (scalar_reward / self.args.alpha) + log_confidence

            # Infer directional label from oracle (sign of gated reward)
            # If gated_reward > 0, peptide is predicted as target direction
            # This is approximate; in practice you might want to query f_φ directly
            directional_label = np.sign(scalar_reward) if scalar_reward != 0 else 0.0

            item = {
                "x_final": x_final[i].clone(),
                "log_rnd": traj_log_rnd.clone() if isinstance(traj_log_rnd, torch.Tensor) else torch.tensor(traj_log_rnd),
                "final_reward": scalar_reward,
                "score_vector": sv.copy(),
                "seq": childSequences[i],
                # TD3B-specific additions
                "directional_label": directional_label,
                "confidence": confidence,
            }

            # Pareto dominance filtering (same as base class)
            from mcts.peptide_mcts import dominated_by, dominates

            if any(dominated_by(sv, bi["score_vector"]) for bi in self.buffer):
                self._debug_buffer_decision(sv, "rejected_dominated")
                continue

            # Remove dominated items
            keep = []
            for bi in self.buffer:
                if not dominates(sv, bi["score_vector"]):
                    keep.append(bi)
            self.buffer = keep

            # Insert with capacity constraint
            if len(self.buffer) < self.buffer_size:
                self.buffer.append(item)
            else:
                # Replace worst item
                worst_i = int(np.argmin([np.sum(bi["score_vector"]) for bi in self.buffer]))
                self.buffer[worst_i] = item

            self._debug_buffer_decision(sv, "inserted", {"new_len": len(self.buffer)})

            traj_log_rnds.append(traj_log_rnd)
            scalar_rewards.append(scalar_reward)

        traj_log_rnds = torch.stack([torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in traj_log_rnds], dim=0) if traj_log_rnds else torch.empty(0)
        scalar_rewards = np.asarray(scalar_rewards, dtype=float)
        return traj_log_rnds, scalar_rewards

    def forward(self, resetTree=False):
        """
        TD3B version of forward that returns 7 values.

        Returns:
            x_final: (N, L) sequence tokens
            log_rnd: (N,) log importance weights
            final_rewards: (N,) scalar rewards
            score_vectors: (N, K) score arrays
            sequences: List of N SMILES strings
            directional_labels: (N,) directional labels
            confidences: (N,) confidence scores
        """
        self.reset(resetTree)

        while (self.iter_num < self.num_iter):
            self.iter_num += 1

            # traverse the tree form the root node until a leaf node
            with self.timer.section("select"):
                leafNode, _ = self.select(self.rootNode)

            # expand leaf node into num_children partially unmasked sequences at the next timestep
            with self.timer.section("expand"):
                self.expand(leafNode)

        final_x, log_rnd, final_rewards, score_vectors, sequences, directional_labels, confidences = self.consolidateBuffer()

        rows = self.timer.summary()
        print("\n=== Timing summary (by total time) ===")
        for name, cnt, total, mean, p50, p95 in rows:
            print(f"{name:30s}  n={cnt:5d}  total={total:8.3f}s  mean={mean*1e3:7.2f}ms  "
                f"p50={p50*1e3:7.2f}ms  p95={p95*1e3:7.2f}ms")

        return final_x, log_rnd, final_rewards, score_vectors, sequences, directional_labels, confidences

    def consolidateBuffer(self):
        """
        TD3B version: includes directional labels and confidences.

        Returns:
            x_final: (N, L) sequence tokens
            log_rnd: (N,) log importance weights
            final_rewards: (N,) scalar rewards
            score_vectors: (N, K) score arrays
            sequences: List of N SMILES strings
            directional_labels: (N,) directional labels
            confidences: (N,) confidence scores
        """
        # Handle empty buffer case - return empty tensors/arrays
        if len(self.buffer) == 0:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("MCTS buffer is empty - no valid sequences found. Returning empty results.")

            # Return empty tensors/arrays with correct shapes
            # Use policy_model (set by base MCTS class) to get device
            device = self.policy_model.device if hasattr(self.policy_model, 'device') else 'cpu'
            return (
                torch.empty(0, 0, dtype=torch.long, device=device),  # x_final: (0, 0)
                torch.empty(0, dtype=torch.float32, device=device),  # log_rnd: (0,)
                np.empty(0, dtype=np.float32),  # final_rewards: (0,)
                np.empty((0, 0), dtype=np.float32),  # score_vectors: (0, 0)
                [],  # sequences: empty list
                np.empty(0, dtype=np.float32),  # directional_labels: (0,)
                np.empty(0, dtype=np.float32)   # confidences: (0,)
            )

        x_final = []
        log_rnd = []
        final_rewards = []
        score_vectors = []
        sequences = []
        directional_labels = []
        confidences = []

        for item in self.buffer:
            x_final.append(item["x_final"])
            log_rnd.append(item["log_rnd"])
            final_rewards.append(item["final_reward"])
            score_vectors.append(item["score_vector"])
            sequences.append(item["seq"])
            directional_labels.append(item.get("directional_label", 0.0))
            confidences.append(item.get("confidence", 1.0))

        x_final = torch.stack(x_final, dim=0)  # (N, L)
        log_rnd = torch.stack(log_rnd, dim=0).to(dtype=torch.float32)  # (N,)
        final_rewards = np.stack(final_rewards, axis=0).astype(np.float32)
        score_vectors = np.stack(score_vectors, axis=0).astype(np.float32)
        directional_labels = np.array(directional_labels, dtype=np.float32)
        confidences = np.array(confidences, dtype=np.float32)

        return x_final, log_rnd, final_rewards, score_vectors, sequences, directional_labels, confidences


def create_td3b_mcts(
    args,
    diffusion_model,
    td3b_reward_function: TD3BRewardFunction,
    alpha: float = 0.1,
    **kwargs
) -> TD3B_MCTS:
    """
    Factory function to create TD3B MCTS instance.

    Args:
        args: Configuration arguments
        diffusion_model: MDLM model
        td3b_reward_function: TD3BRewardFunction instance
        alpha: Temperature for importance weighting
        **kwargs: Additional MCTS arguments (forwarded verbatim to TD3B_MCTS).
            This includes ``enforce_validity`` (validity gate toggle); when not
            passed here it defaults to None inside TD3B_MCTS and is resolved from
            ``args.enforce_validity`` (i.e. the --validity_reward flag).

    Returns:
        mcts: TD3B_MCTS instance
    """
    # Create confidence weighting module
    confidence_weighting = TD3BConfidenceWeighting(
        alpha=alpha,
        min_confidence=0.1
    )

    # Create TD3B MCTS
    mcts = TD3B_MCTS(
        args=args,
        diffusion_model=diffusion_model,
        td3b_reward_function=td3b_reward_function,
        confidence_weighting=confidence_weighting,
        **kwargs
    )

    return mcts
