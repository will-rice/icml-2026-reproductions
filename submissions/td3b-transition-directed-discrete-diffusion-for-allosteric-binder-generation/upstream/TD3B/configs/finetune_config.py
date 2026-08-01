"""
Shared Configuration Classes for TD3B Finetuning

This module contains all configuration dataclasses used by both:
- finetune_v1.py (single-target training)
- finetune_multi_target.py (multi-target training)

Extracted to avoid code duplication and ensure consistency.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RoFormerConfig:
    """Configuration for RoFormer model architecture."""
    hidden_size: int
    n_layers: int
    n_heads: int
    max_position_embeddings: int = 1035  # Must match pretrained model


@dataclass(frozen=True)
class NoiseConfig:
    """Configuration for noise scheduling."""
    type: str = 'loglinear'
    sigma_min: float = 1e-4
    sigma_max: float = 20.0


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for training parameters."""
    sampling_eps: float


@dataclass(frozen=True)
class SamplingConfig:
    """Configuration for sampling parameters."""
    steps: int
    sampling_eps: float
    predictor: str = 'ddpm_cache'


@dataclass(frozen=True)
class EvalConfig:
    """Configuration for evaluation parameters."""
    gen_ppl_eval_model_name_or_path: str = 'gpt2-large'


@dataclass(frozen=True)
class OptimConfig:
    """Configuration for optimizer parameters."""
    lr: float


@dataclass(frozen=True)
class MCTSConfig:
    """Configuration for MCTS parameters."""
    sampling: int = 0  # 0 for Gumbel sampling


class DiffusionConfig:
    """
    Complete configuration for Diffusion model.

    This class encapsulates all nested configuration objects required
    by the Diffusion model, providing a clean interface and type safety.
    """

    def __init__(
        self,
        roformer: RoFormerConfig,
        noise: NoiseConfig,
        training: TrainingConfig,
        sampling: SamplingConfig,
        eval_cfg: EvalConfig,
        optim: OptimConfig,
        mcts: MCTSConfig
    ):
        # Create anonymous objects for backward compatibility
        self.roformer = type('RoFormerObj', (), {
            'hidden_size': roformer.hidden_size,
            'n_layers': roformer.n_layers,
            'n_heads': roformer.n_heads,
            'max_position_embeddings': roformer.max_position_embeddings
        })()

        self.noise = type('NoiseObj', (), {
            'type': noise.type,
            'sigma_min': noise.sigma_min,
            'sigma_max': noise.sigma_max
        })()

        self.training = type('TrainingObj', (), {
            'sampling_eps': training.sampling_eps
        })()

        self.sampling = type('SamplingObj', (), {
            'steps': sampling.steps,
            'sampling_eps': sampling.sampling_eps,
            'predictor': sampling.predictor
        })()

        self.eval = type('EvalObj', (), {
            'gen_ppl_eval_model_name_or_path': eval_cfg.gen_ppl_eval_model_name_or_path
        })()

        self.optim = type('OptimObj', (), {
            'lr': optim.lr
        })()

        self.mcts = type('MCTSObj', (), {
            'sampling': mcts.sampling
        })()

        # Fixed parameters
        self.backbone = 'roformer'
        self.parameterization = 'subs'
        self.time_conditioning = False
        self.T = 0
