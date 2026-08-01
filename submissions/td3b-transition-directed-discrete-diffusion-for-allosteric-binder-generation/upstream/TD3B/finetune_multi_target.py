"""
Multi-Target TD3B Fine-Tuning Script

Trains TD3B on multiple protein targets with random sampling strategy.
Uses the GPCR directional oracle for direction-aware gating.

Architecture: Transition-Directed Discrete Diffusion for Binders (TD3B)
Training: Random K-target sampling + MCTS-guided trajectory optimization + contrastive learning

Key Features:
- Random K targets sampled per MCTS round
- Small-batch training to prevent OOM
- Periodic validation on held-out targets
- Checkpoint saving with validation metrics
"""

import os
import sys
import argparse
import logging
import warnings
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.diffusion import Diffusion
from tokenizer.my_tokenizers import SMILES_SPE_Tokenizer
from utils.app import PeptideAnalyzer
from scoring.functions.binding import MultiTargetBindingAffinity, TargetSpecificBindingAffinity
from td3b.data_utils import peptide_seq_to_smiles, smiles_token_length

# TD3B imports
from td3b.td3b_losses import TD3BTotalLoss
from td3b.td3b_finetune import (
    extract_embeddings_from_mdlm,
    add_td3b_sampling_to_model
)
from td3b.direction_oracle import DirectionalOracle

# Import shared configuration classes
from configs.finetune_config import (
    RoFormerConfig,
    NoiseConfig,
    TrainingConfig,
    SamplingConfig,
    EvalConfig,
    OptimConfig,
    MCTSConfig,
    DiffusionConfig
)

# Import shared utilities
from training.finetune_utils import (
    load_tokenizer,
    initialize_device,
    create_output_directory,
    save_model,
    setup_wandb,
    cleanup_wandb,
    create_mcts_instance,
    create_reward_function,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Constants
SEPARATOR_LINE = "=" * 80
eps = 1e-5

class TargetDataset:
    """Dataset handler for multi-target training."""

    def __init__(self, csv_path: str, tokenizer: Optional[SMILES_SPE_Tokenizer] = None):
        """
        Load target dataset from CSV.

        Args:
            csv_path: Path to CSV file with columns:
                - Target_Sequence: Protein target sequence
                - Ligand_Sequence: Binder sequence (for length reference)
                - label: 'agonist' or 'antagonist'
            tokenizer: Tokenizer used to compute SMILES token length
        """
        self.df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(self.df)} samples from {csv_path}")
        self.tokenizer = tokenizer

        # Group by target
        self.targets = {}
        for target_seq in self.df['Target_Sequence'].unique():
            target_df = self.df[self.df['Target_Sequence'] == target_seq]

            # Get binder lengths for each direction
            agonist_binders = target_df[target_df['label'] == 'agonist']['Ligand_Sequence'].tolist()
            antagonist_binders = target_df[target_df['label'] == 'antagonist']['Ligand_Sequence'].tolist()

            # Store actual sequence lengths
            agonist_lengths = [self._binder_length(seq) for seq in agonist_binders] if agonist_binders else []
            antagonist_lengths = [self._binder_length(seq) for seq in antagonist_binders] if antagonist_binders else []

            # Use most common length for each direction, or average if tied
            # This ensures we generate sequences similar to the provided data
            if agonist_lengths:
                agonist_target_length = int(np.median(agonist_lengths))
            else:
                # Default to antagonist length if no agonist, or 50 if neither
                agonist_target_length = int(np.median(antagonist_lengths)) if antagonist_lengths else 50

            if antagonist_lengths:
                antagonist_target_length = int(np.median(antagonist_lengths))
            else:
                # Default to agonist length if no antagonist, or 50 if neither
                antagonist_target_length = int(np.median(agonist_lengths)) if agonist_lengths else 50

            self.targets[target_seq] = {
                'sequence': target_seq,
                'agonist_length': agonist_target_length,  # Target length for agonist generation
                'antagonist_length': antagonist_target_length,  # Target length for antagonist generation
                'agonist_count': len(agonist_binders),
                'antagonist_count': len(antagonist_binders)
            }

        logger.info(f"Found {len(self.targets)} unique targets")

    def _binder_length(self, binder_seq: str) -> int:
        smiles = peptide_seq_to_smiles(binder_seq)
        if self.tokenizer is None:
            return len(smiles)
        return smiles_token_length(smiles, self.tokenizer)

    def sample_targets(self, k: int, random_state: Optional[int] = None) -> List[str]:
        """
        Randomly sample K targets.

        Args:
            k: Number of targets to sample
            random_state: Random seed for reproducibility

        Returns:
            List of target sequences
        """
        if random_state is not None:
            np.random.seed(random_state)

        target_seqs = list(self.targets.keys())
        k = min(k, len(target_seqs))
        return np.random.choice(target_seqs, size=k, replace=False).tolist()

    def get_target_info(self, target_seq: str) -> Dict:
        """Get information for a specific target."""
        return self.targets[target_seq]

    def get_sequence_length(self, target_seq: str, direction: str) -> int:
        """
        Get the target sequence length for generation.

        Args:
            target_seq: Target protein sequence
            direction: 'agonist' or 'antagonist'

        Returns:
            Target binder sequence length
        """
        target_info = self.targets[target_seq]
        if direction == 'agonist' or direction == 1.0 or direction == '+1':
            return target_info['agonist_length']
        else:  # antagonist
            return target_info['antagonist_length']

    def get_all_targets(self) -> List[str]:
        """Get all target sequences."""
        return list(self.targets.keys())


def run_validation(
    policy_model: Diffusion,
    multi_target_affinity: MultiTargetBindingAffinity,
    directional_oracle: DirectionalOracle,
    tokenizer: SMILES_SPE_Tokenizer,
    val_dataset: TargetDataset,
    args: argparse.Namespace,
    epoch: int,
    device: torch.device,
    protein_token_cache: Optional[Dict[str, torch.Tensor]] = None
) -> Dict:
    """
    Run validation on all targets in validation dataset.

    Args:
        policy_model: Trained diffusion model
        affinity_predictor: Binding affinity predictor
        directional_oracle: Directional oracle
        tokenizer: Tokenizer
        val_dataset: Validation dataset
        args: Training arguments
        epoch: Current epoch
        device: Device

    Returns:
        Dictionary with validation metrics
    """
    logger.info(f"\n{SEPARATOR_LINE}")
    logger.info(f"Running validation at epoch {epoch}")
    logger.info(f"{SEPARATOR_LINE}")

    policy_model.eval()

    all_sequences = []
    all_affinities = []
    all_gated_rewards = []
    all_directions = []
    all_target_directions = []  # d* for each sequence
    all_valid_fractions = []
    all_valid_fractions_per_sample = []
    all_target_names = []

    val_targets = val_dataset.get_all_targets()

    if protein_token_cache is None:
        protein_token_cache = {}

    with torch.no_grad():
        for target_seq in tqdm(val_targets, desc="Validating targets"):
            target_info = val_dataset.get_target_info(target_seq)
            target_protein_tokens = protein_token_cache.get(target_seq)
            if target_protein_tokens is None:
                target_protein_tokens = directional_oracle.encode_protein(target_seq)
                protein_token_cache[target_seq] = target_protein_tokens

            # Generate for both agonist and antagonist
            for direction_name, d_star in [('agonist', 1.0), ('antagonist', -1.0)]:
                # Get the target sequence length for this direction
                target_length = val_dataset.get_sequence_length(target_seq, direction_name)

                # Temporarily set args.seq_length for this generation
                original_seq_length = args.seq_length
                args.seq_length = target_length

                # Create target-specific affinity predictor for this target
                target_affinity = TargetSpecificBindingAffinity(multi_target_affinity, target_seq)

                # Create reward model for this target+direction
                reward_model = create_reward_function(
                    affinity_predictor=target_affinity,
                    directional_oracle=directional_oracle,
                    target_direction=d_star,
                    target_protein_tokens=target_protein_tokens,
                    tokenizer=tokenizer,
                    device=device,
                    min_affinity_threshold=args.min_affinity_threshold,
                    use_confidence_weighting=True,
                    temperature=args.sigmoid_temperature
                )

                # Sample sequences with the correct length
                x_eval, eval_metrics = policy_model.sample_finetuned_td3b(
                    args,
                    reward_model,
                    batch_size=args.val_samples_per_target,
                    dataframe=False
                )

                # Restore original seq_length
                args.seq_length = original_seq_length

                # Decode sequences
                sequences = tokenizer.batch_decode(x_eval)

                # Get metrics
                affinities = eval_metrics.get('affinity', [])
                gated_rewards = eval_metrics.get('gated_reward', [])
                directions = eval_metrics.get('direction_predictions', [])
                valid_fraction = eval_metrics.get('valid_fraction', 0.0)

                # CRITICAL FIX: Metrics are only computed for valid sequences
                # So we should extend based on the length of metrics arrays, not all sequences
                num_valid = len(affinities)  # Number of valid sequences with metrics

                # Filter to only valid sequences (metrics are only for valid ones)
                from utils.app import PeptideAnalyzer
                analyzer = PeptideAnalyzer()
                valid_sequences = [seq for seq in sequences if analyzer.is_peptide(seq)][:num_valid]

                # Store (all arrays must have the same length = num_valid)
                all_sequences.extend(valid_sequences)  # Only valid sequences
                all_affinities.extend(affinities)
                all_gated_rewards.extend(gated_rewards)
                all_directions.extend(directions)
                all_target_directions.extend([d_star] * num_valid)
                all_valid_fractions.append(valid_fraction)
                all_valid_fractions_per_sample.extend([valid_fraction] * num_valid)
                all_target_names.extend([target_seq[:20]] * num_valid)

    # Compute validation metrics
    all_affinities = np.array(all_affinities)
    all_gated_rewards = np.array(all_gated_rewards)
    all_directions = np.array(all_directions)
    all_target_directions = np.array(all_target_directions)

    if all_directions.size == 0:
        direction_correct = np.array([], dtype=np.float32)
    else:
        direction_correct = np.where(
            all_target_directions > 0,
            all_directions >= 0.5,
            all_directions < 0.5
        ).astype(np.float32)

    # Consistency rewards: d* × (f_φ - 0.5)
    consistency_rewards = all_target_directions * (all_directions - 0.5)  # range from -1 to 1.
    success_rates = direction_correct * np.array(all_valid_fractions_per_sample, dtype=np.float32)

    # Separate by direction
    agonist_mask = all_target_directions == 1.0
    antagonist_mask = all_target_directions == -1.0

    consistency_agonist = consistency_rewards[agonist_mask]
    consistency_antagonist = consistency_rewards[antagonist_mask]

    val_metrics = {
        'affinity_mean': np.mean(all_affinities),
        'affinity_std': np.std(all_affinities),
        'gated_reward_mean': np.mean(all_gated_rewards),
        'gated_reward_std': np.std(all_gated_rewards),
        'direction_oracle_mean': np.mean(all_directions),
        'direction_oracle_std': np.std(all_directions),
        'consistency_reward_mean': np.mean(consistency_rewards),
        'consistency_reward_std': np.std(consistency_rewards),
        'consistency_agonist_mean': np.mean(consistency_agonist) if len(consistency_agonist) > 0 else 0.0,
        'consistency_agonist_std': np.std(consistency_agonist) if len(consistency_agonist) > 0 else 0.0,
        'consistency_antagonist_mean': np.mean(consistency_antagonist) if len(consistency_antagonist) > 0 else 0.0,
        'consistency_antagonist_std': np.std(consistency_antagonist) if len(consistency_antagonist) > 0 else 0.0,
        'valid_fraction_mean': np.mean(all_valid_fractions),
        'valid_fraction_std': np.std(all_valid_fractions),
        'direction_accuracy_mean': np.mean(direction_correct) if direction_correct.size else 0.0,
        'direction_accuracy_std': np.std(direction_correct) if direction_correct.size else 0.0,
        'success_rate_mean': np.mean(success_rates) if success_rates.size else 0.0,
        'success_rate_std': np.std(success_rates) if success_rates.size else 0.0
    }

    # Log validation metrics
    logger.info(f"\nValidation Results (Epoch {epoch}):")
    logger.info(f"  Affinity: {val_metrics['affinity_mean']:.4f} ± {val_metrics['affinity_std']:.4f}")
    logger.info(f"  Gated Reward: {val_metrics['gated_reward_mean']:.4f} ± {val_metrics['gated_reward_std']:.4f}")
    logger.info(f"  Direction Oracle: {val_metrics['direction_oracle_mean']:.4f} ± {val_metrics['direction_oracle_std']:.4f}")
    logger.info(f"  Consistency Reward: {val_metrics['consistency_reward_mean']:.4f} ± {val_metrics['consistency_reward_std']:.4f}")
    logger.info(f"  Consistency (d*=+1): {val_metrics['consistency_agonist_mean']:.4f} ± {val_metrics['consistency_agonist_std']:.4f}")
    logger.info(f"  Consistency (d*=-1): {val_metrics['consistency_antagonist_mean']:.4f} ± {val_metrics['consistency_antagonist_std']:.4f}")
    logger.info(f"  Valid Fraction: {val_metrics['valid_fraction_mean']:.4f} ± {val_metrics['valid_fraction_std']:.4f}")
    logger.info(f"  Direction Accuracy: {val_metrics['direction_accuracy_mean']:.4f} ± {val_metrics['direction_accuracy_std']:.4f}")
    logger.info(f"  Success Rate: {val_metrics['success_rate_mean']:.4f} ± {val_metrics['success_rate_std']:.4f}")

    # Save validation sequences to file
    val_df = pd.DataFrame({
        'target': all_target_names,
        'sequence': all_sequences,
        'target_direction': all_target_directions,
        'affinity': all_affinities,
        'gated_reward': all_gated_rewards,
        'direction_oracle': all_directions,
        'consistency_reward': consistency_rewards,
        'direction_accuracy': direction_correct,
        'success_rate': success_rates
    })

    val_output_path = os.path.join(args.save_path, f'validation_epoch_{epoch}.csv')
    val_df.to_csv(val_output_path, index=False)
    logger.info(f"Validation sequences saved to {val_output_path}")

    policy_model.train()

    return val_metrics


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Multi-Target TD3B Fine-Tuning')

    # Paths
    path_group = parser.add_argument_group('Paths')
    path_group.add_argument('--base_path', type=str,
                           default=os.path.dirname(os.path.abspath(__file__)),
                           help='Base path for TR2-D2 project (defaults to the repo root)')
    path_group.add_argument('--train_csv', type=str, required=True,
                           help='Path to training CSV file')
    path_group.add_argument('--val_csv', type=str, default=None,
                           help='Path to validation CSV file (optional)')
    path_group.add_argument('--pretrained_checkpoint', type=str, required=True,
                           help='Path to pretrained diffusion model checkpoint')
    path_group.add_argument('--run_name', type=str, required=True,
                           help='Name for this training run')
    path_group.add_argument('--device', type=str, default='cuda',
                           help='Device to use (cuda or cpu)')

    # Multi-target sampling
    target_group = parser.add_argument_group('Multi-Target Sampling')
    target_group.add_argument('--targets_per_mcts', type=int, default=5,
                             help='Number of targets to sample per MCTS round (K)')
    target_group.add_argument('--resample_targets_every', type=int, default=1,
                             help='Resample targets every N epochs')

    # Training hyperparameters
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--num_epochs', type=int, default=200,
                            help='Total number of training epochs')
    train_group.add_argument('--learning_rate', type=float, default=3e-4,
                            help='Learning rate for optimizer')
    train_group.add_argument('--train_batch_size', type=int, default=16,
                            help='Batch size for training (small to prevent OOM)')
    train_group.add_argument('--gradient_accumulation_steps', type=int, default=4,
                            help='Accumulate gradients over N steps')
    train_group.add_argument('--resample_every_n_step', type=int, default=10,
                            help='Resample MCTS every N epochs')
    train_group.add_argument('--save_every_n_epochs', type=int, default=20,
                            help='Save checkpoint every N epochs')
    train_group.add_argument('--validate_every_n_epochs', type=int, default=20,
                            help='Run validation every N epochs')
    train_group.add_argument('--num_epoch_for_sampling', type=int, default=5,
                            help='Run evaluation sampling every N epochs (set <=0 to disable)')
    train_group.add_argument('--reset_every_n_step', type=int, default=50,
                            help='Reset MCTS tree every N epochs')

    # MCTS hyperparameters
    mcts_group = parser.add_argument_group('MCTS')
    mcts_group.add_argument('--num_iter', type=int, default=50,
                           help='MCTS iterations per resample (v1 default: 50, reduce for multi-target)')
    mcts_group.add_argument('--num_children', type=int, default=30,
                           help='Children per MCTS expansion')
    mcts_group.add_argument('--buffer_size', type=int, default=50,
                           help='Pareto buffer size (v1 default: 50)')
    mcts_group.add_argument('--replay_buffer_size', type=int, default=0,
                           help='Max replay buffer size across resamples (0 disables replay)')
    mcts_group.add_argument('--replay_buffer_strategy', type=str, default='fifo',
                           choices=['fifo', 'random'],
                           help='Replay buffer eviction strategy when full')
    mcts_group.add_argument('--alpha', type=float, default=0.1,
                           help='Temperature for importance weighting')
    mcts_group.add_argument('--exploration', type=float, default=1.0,
                           help='UCB exploration constant')
    mcts_group.add_argument('--validity_reward', choices=['on', 'off'], default='on',
                           help="Validity gate during MCTS expansion. 'on' (default): "
                                "invalid decoded peptides are gated as zero-reward nodes "
                                "(PeptideAnalyzer.is_peptide filter). 'off': disable the gate "
                                "so invalid peptides are kept and scored on the pure "
                                "affinity x direction reward.")

    # TD3B loss hyperparameters
    loss_group = parser.add_argument_group('TD3B Loss')
    loss_group.add_argument('--contrastive_weight', type=float, default=0.1,
                           help='Weight for contrastive loss (v1 default: 0.1)')
    loss_group.add_argument('--contrastive_margin', type=float, default=1.0,
                           help='Margin for contrastive loss')
    loss_group.add_argument('--contrastive_type', type=str, default='triplet',
                           choices=['triplet', 'ntxent', 'supcon'],
                           help='Type of contrastive loss')
    loss_group.add_argument('--kl_beta', type=float, default=0.1,
                           help='KL divergence regularization coefficient (v1 default: 0.1)')
    loss_group.add_argument('--min_affinity_threshold', type=float, default=0.0,
                           help='Minimum affinity threshold for allosteric control (CRITICAL)')
    loss_group.add_argument('--sigmoid_temperature', type=float, default=0.1,
                           help='Temperature for sigmoid gating')

    # Validation
    val_group = parser.add_argument_group('Validation')
    val_group.add_argument('--val_samples_per_target', type=int, default=20,
                          help='Number of sequences to generate per target during validation')

    # Architecture
    arch_group = parser.add_argument_group('Architecture')
    arch_group.add_argument('--seq_length', type=int, default=200,
                           help='Maximum sequence length')
    arch_group.add_argument('--embedding_pool_method', type=str, default='cls',
                           choices=['cls', 'mean', 'max'],
                           help='Pooling method for embeddings')
    arch_group.add_argument('--hidden_dim', type=int, default=768,
                           help='Hidden dimension size')
    arch_group.add_argument('--num_layers', type=int, default=8,
                           help='Number of transformer layers (v1 default: 8)')
    arch_group.add_argument('--num_heads', type=int, default=8,
                           help='Number of attention heads (v1 default: 8)')
    arch_group.add_argument('--sampling_eps', type=float, default=1e-3,
                           help='Sampling epsilon (v1 default: 1e-3)')
    arch_group.add_argument('--total_num_steps', type=int, default=128,
                           help='Total number of diffusion steps (v1 default: 128)')

    # Optimization
    opt_group = parser.add_argument_group('Optimization')
    opt_group.add_argument('--grad_clip', action='store_true',
                          help='Enable gradient clipping')
    opt_group.add_argument('--gradnorm_clip', type=float, default=1.0,
                          help='Gradient norm clipping threshold')
    opt_group.add_argument('--wdce_num_replicates', type=int, default=16,
                          help='Number of replicates for WDCE loss (v1 default: 16)')
    opt_group.add_argument('--centering', action='store_true',
                          help='Enable centering in WDCE loss')

    # Logging
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument('--wandb_project', type=str, default='TD3B-multi-target',
                          help='W&B project name')
    log_group.add_argument('--wandb_entity', type=str, default=None,
                          help='W&B entity name (default: your personal/default entity)')

    # Directional oracle
    oracle_group = parser.add_argument_group('Directional Oracle')
    oracle_group.add_argument('--direction_oracle_ckpt', type=str, default=None,
                             help='Path to directional oracle checkpoint')
    oracle_group.add_argument('--direction_oracle_tr2d2_checkpoint', type=str, default=None,
                             help='Path to TR2D2 checkpoint used by the oracle')
    oracle_group.add_argument('--direction_oracle_tokenizer_vocab', type=str, default=None,
                             help='Path to SMILES tokenizer vocab for oracle')
    oracle_group.add_argument('--direction_oracle_tokenizer_splits', type=str, default=None,
                             help='Path to SMILES tokenizer splits for oracle')
    oracle_group.add_argument('--direction_oracle_esm_name', type=str,
                             default='facebook/esm2_t33_650M_UR50D',
                             help='ESM model name or local path')
    oracle_group.add_argument('--direction_oracle_esm_cache_dir', type=str, default=None,
                             help='Optional cache directory for ESM model')
    oracle_group.add_argument('--direction_oracle_esm_local_files_only', action='store_true',
                             help='Load ESM from local cache only (no network)')
    oracle_group.add_argument('--direction_oracle_max_ligand_length', type=int, default=768,
                             help='Max SMILES token length for oracle')
    oracle_group.add_argument('--direction_oracle_max_protein_length', type=int, default=1024,
                             help='Max protein token length for oracle')
    oracle_group.add_argument('--direction_oracle_d_model', type=int, default=256,
                             help='Oracle hidden dimension (must match checkpoint)')
    oracle_group.add_argument('--direction_oracle_n_heads', type=int, default=4,
                             help='Oracle attention heads (must match checkpoint)')
    oracle_group.add_argument('--direction_oracle_n_self_attn_layers', type=int, default=1,
                             help='Oracle self-attention layers (must match checkpoint)')
    oracle_group.add_argument('--direction_oracle_n_bmca_layers', type=int, default=2,
                             help='Oracle cross-attention layers (must match checkpoint)')
    oracle_group.add_argument('--direction_oracle_dropout', type=float, default=0.3,
                             help='Oracle dropout (must match checkpoint)')

    args = parser.parse_args()

    # Resolve default oracle paths relative to base_path
    base_tr2d2_path = os.path.join(args.base_path, 'tr2d2-pep')
    if args.direction_oracle_ckpt is None:
        args.direction_oracle_ckpt = os.path.join(
            base_tr2d2_path, 'best_model_tr2d2_gpcr_fixed.pt'
        )
    if args.direction_oracle_tr2d2_checkpoint is None:
        args.direction_oracle_tr2d2_checkpoint = os.path.join(
            base_tr2d2_path, 'pretrained', 'peptune-pretrained.ckpt'
        )
    if args.direction_oracle_tokenizer_vocab is None:
        args.direction_oracle_tokenizer_vocab = os.path.join(
            base_tr2d2_path, 'tokenizer', 'new_vocab.txt'
        )
    if args.direction_oracle_tokenizer_splits is None:
        args.direction_oracle_tokenizer_splits = os.path.join(
            base_tr2d2_path, 'tokenizer', 'new_splits.txt'
        )

    # Add derived attributes (required by MCTS)
    args.time_conditioning = False
    args.num_obj = 5  # Must match padded score vector size
    args.scalarization = "sum"
    # Validity gate toggle -> consumed by MCTS (mcts/peptide_mcts.py) via the
    # args threading through create_mcts_instance/create_td3b_mcts/TD3B_MCTS.
    args.enforce_validity = (args.validity_reward == "on")

    # Create save path
    args.save_path = create_output_directory(
        args.base_path,
        args.run_name,
        add_timestamp=True
    )

    return args


def main():
    args = parse_args()

    logger.info(f"\n{SEPARATOR_LINE}")
    logger.info("Multi-Target TD3B Fine-Tuning")
    logger.info(f"{SEPARATOR_LINE}\n")

    # Set device
    device = initialize_device(args.device)

    # Initialize W&B
    setup_wandb(
        project=args.wandb_project,
        name=args.run_name,
        config=vars(args),
        entity=args.wandb_entity
    )

    # Tokenizer
    tokenizer = load_tokenizer(args.base_path)

    # Load datasets
    logger.info("\n[1/6] Loading datasets...")
    train_dataset = TargetDataset(args.train_csv, tokenizer=tokenizer)
    val_dataset = TargetDataset(args.val_csv, tokenizer=tokenizer) if args.val_csv else None

    # Load models
    logger.info("\n[2/6] Loading models...")

    # Create diffusion config
    config = DiffusionConfig(
        roformer=RoFormerConfig(
            hidden_size=args.hidden_dim,
            n_layers=args.num_layers,
            n_heads=args.num_heads
        ),
        noise=NoiseConfig(),
        training=TrainingConfig(sampling_eps=args.sampling_eps),
        sampling=SamplingConfig(
            steps=args.total_num_steps,
            sampling_eps=args.sampling_eps
        ),
        eval_cfg=EvalConfig(),
        optim=OptimConfig(lr=args.learning_rate),
        mcts=MCTSConfig()
    )

    # Policy model
    policy_model = Diffusion(
        config=config,
        tokenizer=tokenizer,
        device=device
    ).to(device)

    # Load pretrained checkpoint
    checkpoint = torch.load(args.pretrained_checkpoint, map_location=device, weights_only=False)

    # Handle different checkpoint formats (like v1)
    CHECKPOINT_KEYS = ('state_dict', 'model_state_dict')
    state_dict = None
    for key in CHECKPOINT_KEYS:
        if key in checkpoint:
            state_dict = checkpoint[key]
            logger.info(f"Loading checkpoint from key: {key}")
            break

    if state_dict is None:
        # Assume checkpoint is already a state_dict
        state_dict = checkpoint
        logger.info("Loading checkpoint as direct state_dict")

    policy_model.load_state_dict(state_dict, strict=False)
    logger.info(f"Loaded pretrained checkpoint from {args.pretrained_checkpoint}")

    # Reference model (frozen)
    reference_model = Diffusion(
        config=config,
        tokenizer=tokenizer,
        device=device
    ).to(device)
    reference_model.load_state_dict(state_dict, strict=False)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False
    logger.info("Created reference model (frozen)")

    # Add TD3B sampling method, fix bugs, sampling sequences with w(t) as condition
    policy_model = add_td3b_sampling_to_model(policy_model)

    # Multi-target affinity predictor
    multi_target_affinity = MultiTargetBindingAffinity(
        tokenizer=tokenizer,
        base_path=args.base_path,
        device=device,
        emb_model=policy_model.backbone  # Use backbone Roformer model (matches v1)
    )
    logger.info("Created multi-target binding affinity predictor")

    # Directional oracle (GPCR classifier)
    for path_label, path in [
        ("direction_oracle_ckpt", args.direction_oracle_ckpt),
        ("direction_oracle_tr2d2_checkpoint", args.direction_oracle_tr2d2_checkpoint),
        ("direction_oracle_tokenizer_vocab", args.direction_oracle_tokenizer_vocab),
        ("direction_oracle_tokenizer_splits", args.direction_oracle_tokenizer_splits),
    ]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing {path_label}: {path}")

    directional_oracle = DirectionalOracle(
        model_ckpt=args.direction_oracle_ckpt,
        tr2d2_checkpoint=args.direction_oracle_tr2d2_checkpoint,
        tokenizer_vocab=args.direction_oracle_tokenizer_vocab,
        tokenizer_splits=args.direction_oracle_tokenizer_splits,
        esm_name=args.direction_oracle_esm_name,
        d_model=args.direction_oracle_d_model,
        n_heads=args.direction_oracle_n_heads,
        n_self_attn_layers=args.direction_oracle_n_self_attn_layers,
        n_bmca_layers=args.direction_oracle_n_bmca_layers,
        dropout=args.direction_oracle_dropout,
        max_ligand_length=args.direction_oracle_max_ligand_length,
        max_protein_length=args.direction_oracle_max_protein_length,
        device=device,
        esm_cache_dir=args.direction_oracle_esm_cache_dir,
        esm_local_files_only=args.direction_oracle_esm_local_files_only
    )
    directional_oracle.eval()

    protein_token_cache: Dict[str, torch.Tensor] = {}

    def get_protein_tokens(target_seq: str) -> torch.Tensor:
        cached = protein_token_cache.get(target_seq)
        if cached is None:
            cached = directional_oracle.encode_protein(target_seq)
            protein_token_cache[target_seq] = cached
        return cached

    # Loss function
    logger.info("\n[3/6] Creating loss function...")
    td3b_loss_fn = TD3BTotalLoss(
        contrastive_weight=args.contrastive_weight,
        contrastive_margin=args.contrastive_margin,
        kl_beta=args.kl_beta,
        reference_model=reference_model,
        adaptive_margin=True
    )

    # WDCE loss
    from training.finetune_utils import loss_wdce

    logger.info("\n[4/6] Setting up training...")
    policy_model.train()
    torch.set_grad_enabled(True)
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.learning_rate)

    # Training logs
    batch_losses = []
    batch_wdce_losses = []
    batch_contrastive_losses = []
    batch_kl_losses = []

    # Multi-target buffer
    # We'll store sequences from all sampled targets here
    buffer_sequences = []  # List of (x, log_rnd, reward, directional_label, confidence)
    current_targets = []

    def trim_replay_buffer(items, max_size, strategy):
        if max_size <= 0 or len(items) <= max_size:
            return items
        if strategy == "fifo":
            return items[-max_size:]
        indices = np.random.choice(len(items), size=max_size, replace=False)
        return [items[i] for i in indices]

    logger.info(f"\n{SEPARATOR_LINE}")
    logger.info("Starting Training")
    logger.info(f"{SEPARATOR_LINE}\n")

    # Training loop
    pbar = tqdm(range(args.num_epochs))

    for epoch in pbar:
        # Sample new targets if needed
        if epoch % args.resample_targets_every == 0 or len(current_targets) == 0:
            current_targets = train_dataset.sample_targets(
                k=args.targets_per_mcts,
                random_state=epoch
            )
            logger.info(f"\nEpoch {epoch}: Sampled {len(current_targets)} targets for training")

        # MCTS sampling phase (less frequent) - this is when we regenerate sequences
        if epoch % args.resample_every_n_step == 0:
            if args.replay_buffer_size <= 0:
                # Clear buffer only when regenerating with new MCTS if replay is disabled
                buffer_sequences = []
            else:
                logger.info(
                    f"Epoch {epoch}: Replay buffer enabled, keeping {len(buffer_sequences)} sequences before refresh"
                )
            logger.info(f"Epoch {epoch}: Running MCTS for {len(current_targets)} targets...")
            mcts_valid_total = 0
            mcts_run_count = 0
            mcts_empty_runs = 0

            with torch.no_grad():
                for target_seq in current_targets:
                    target_info = train_dataset.get_target_info(target_seq)

                    # Sample both agonist and antagonist
                    for direction_name, d_star in [('agonist', 1.0), ('antagonist', -1.0)]:
                        # Get the target sequence length for this direction
                        target_length = train_dataset.get_sequence_length(target_seq, direction_name)

                        # Temporarily set args.seq_length for this generation
                        original_seq_length = args.seq_length
                        args.seq_length = target_length

                        # Create target-specific affinity predictor for this target
                        target_affinity = TargetSpecificBindingAffinity(multi_target_affinity, target_seq)

                        # Create reward model for this target
                        reward_model = create_reward_function(
                            affinity_predictor=target_affinity,
                            directional_oracle=directional_oracle,
                            target_direction=d_star,
                            target_protein_tokens=get_protein_tokens(target_seq),
                            tokenizer=tokenizer,
                            device=device,
                            min_affinity_threshold=args.min_affinity_threshold,
                            use_confidence_weighting=True,
                            temperature=args.sigmoid_temperature
                        )

                        # Create MCTS using shared utility
                        mcts = create_mcts_instance(
                            args=args,
                            policy_model=policy_model,
                            reward_function=reward_model,
                            tokenizer=tokenizer,
                            buffer_size=args.buffer_size
                        )

                        # Run MCTS
                        reset_tree = (epoch % args.reset_every_n_step == 0)
                        results = mcts.forward(resetTree=reset_tree)

                        # Restore original seq_length
                        args.seq_length = original_seq_length

                        # Unpack results
                        if len(results) == 7:
                            x_final, log_rnd, final_rewards, score_vectors, sequences, directional_labels, confidences = results

                            # Skip if MCTS returned empty buffer (no valid sequences found)
                            if len(x_final) == 0:
                                logger.warning(f"MCTS returned empty buffer for target={target_seq[:20]}, direction={direction_name}")
                                mcts_run_count += 1
                                mcts_empty_runs += 1
                                continue
                            mcts_run_count += 1
                            mcts_valid_total += len(sequences)

                            # Add to buffer
                            for i in range(len(x_final)):
                                buffer_sequences.append({
                                    'x': x_final[i],
                                    'log_rnd': log_rnd[i],
                                    'reward': final_rewards[i],
                                    'directional_label': d_star,
                                    'confidence': confidences[i] if isinstance(confidences, np.ndarray) else 1.0
                                })

            if args.replay_buffer_size > 0:
                buffer_sequences = trim_replay_buffer(
                    buffer_sequences,
                    args.replay_buffer_size,
                    args.replay_buffer_strategy
                )

            logger.info(
                f"Epoch {epoch}: MCTS runs={mcts_run_count}, "
                f"valid_sequences={mcts_valid_total}, empty_runs={mcts_empty_runs}"
            )
            logger.info(f"Epoch {epoch}: Buffer size: {len(buffer_sequences)} sequences")

        # Training phase: sample mini-batches from buffer
        if len(buffer_sequences) == 0:
            logger.warning(f"Epoch {epoch}: Buffer is empty, skipping training")
            continue

        # Shuffle buffer
        np.random.shuffle(buffer_sequences)

        # Mini-batch training
        num_batches = max(1, len(buffer_sequences) // args.train_batch_size)
        epoch_loss = 0.0
        epoch_wdce_loss = 0.0
        epoch_contrastive_loss = 0.0
        epoch_kl_loss = 0.0

        optimizer.zero_grad()

        for batch_idx in range(num_batches):
            start_idx = batch_idx * args.train_batch_size
            end_idx = min(start_idx + args.train_batch_size, len(buffer_sequences))
            batch_data = buffer_sequences[start_idx:end_idx]

            # Pad sequences to the same length (efficient batching for variable-length sequences)
            # Use padding to handle different sequence lengths from different targets
            x_list = [item['x'] for item in batch_data]
            log_rnd_list = [item['log_rnd'] for item in batch_data]  # Scalars, not vectors!

            # Pad x_batch: pad with mask_index (typically 0 or a special token)
            mask_index = policy_model.mask_index if hasattr(policy_model, 'mask_index') else 0
            max_len = max(x.shape[0] for x in x_list)

            # Create padded tensors
            x_batch = torch.full(
                (len(x_list), max_len),
                fill_value=mask_index,
                dtype=x_list[0].dtype,
                device=device
            )

            # Create attention mask: 1 for real tokens, 0 for padding
            # This tells the model which positions are valid vs padded
            attn_mask = torch.zeros(
                (len(x_list), max_len),
                dtype=torch.long,
                device=device
            )

            # Fill in the real sequences and mark valid positions
            for i, x in enumerate(x_list):
                seq_len = x.shape[0]
                x_batch[i, :seq_len] = x.to(device)
                attn_mask[i, :seq_len] = 1  # Mark valid positions

            # log_rnd is a SCALAR per sequence, not a vector - just stack them
            log_rnd_batch = torch.stack([lr.to(device) if isinstance(lr, torch.Tensor) else torch.tensor(lr, device=device) for lr in log_rnd_list])

            directional_labels_batch = torch.tensor(
                [item['directional_label'] for item in batch_data],
                dtype=torch.float32,
                device=device
            )

            # WDCE loss (with attention mask to handle variable-length sequences)
            wdce_loss = loss_wdce(
                policy_model,
                log_rnd_batch,
                x_batch,
                num_replicates=args.wdce_num_replicates,
                centering=args.centering,
                attn_mask=attn_mask  # Pass attention mask to avoid computing loss on padding
            )

            # KL loss
            mask_index = policy_model.mask_index
            lamda = torch.rand(x_batch.shape[0], device=device)
            sigma_kl = -torch.log1p(-(1 - eps) * lamda)
            masked_index = torch.rand(*x_batch.shape, device=device) < lamda[..., None]
            perturbed_batch = torch.where(masked_index, mask_index, x_batch)
            # Use the actual attention mask (not all ones) to handle variable-length sequences
            attn_mask_kl = attn_mask.to(device)

            kl_loss = td3b_loss_fn.compute_kl_loss(
                policy_model,
                perturbed_batch,
                attn_mask_kl,
                sigma_kl
            )

            # Contrastive loss (if we have multiple directions)
            if len(torch.unique(directional_labels_batch)) > 1:
                embeddings = extract_embeddings_from_mdlm(
                    policy_model,
                    x_batch,
                    pool_method=args.embedding_pool_method
                )

                debug_mode = (epoch < 3) or (epoch > 0 and batch_contrastive_losses and batch_contrastive_losses[-1] < 1e-6)

                total_loss, loss_dict = td3b_loss_fn.compute_loss(
                    wdce_loss,
                    embeddings,
                    directional_labels_batch,
                    kl_loss=kl_loss,
                    debug=debug_mode
                )
            else:
                # Only WDCE + KL if no contrastive
                total_loss = wdce_loss + args.kl_beta * kl_loss
                loss_dict = {
                    'total_loss': total_loss.item(),
                    'wdce_loss': wdce_loss.item(),
                    'contrastive_loss': 0.0,
                    'kl_loss': kl_loss.item()
                }

            # Scale loss for gradient accumulation
            scaled_loss = total_loss / args.gradient_accumulation_steps
            scaled_loss.backward()

            # Accumulate losses
            epoch_loss += loss_dict['total_loss']
            epoch_wdce_loss += loss_dict['wdce_loss']
            epoch_contrastive_loss += loss_dict['contrastive_loss']
            epoch_kl_loss += loss_dict['kl_loss']

            # Gradient accumulation
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), args.gradnorm_clip)
                optimizer.step()
                optimizer.zero_grad()

        # Average losses
        epoch_loss /= num_batches
        epoch_wdce_loss /= num_batches
        epoch_contrastive_loss /= num_batches
        epoch_kl_loss /= num_batches

        batch_losses.append(epoch_loss)
        batch_wdce_losses.append(epoch_wdce_loss)
        batch_contrastive_losses.append(epoch_contrastive_loss)
        batch_kl_losses.append(epoch_kl_loss)

        # Validation
        if val_dataset is not None and (epoch + 1) % args.validate_every_n_epochs == 0:
            val_metrics = run_validation(
                policy_model,
                multi_target_affinity,
                directional_oracle,
                tokenizer,
                val_dataset,
                args,
                epoch,
                device,
                protein_token_cache=protein_token_cache
            )

            # Log to W&B
            wandb.log({
                "epoch": epoch,
                "val/affinity_mean": val_metrics['affinity_mean'],
                "val/affinity_std": val_metrics['affinity_std'],
                "val/gated_reward_mean": val_metrics['gated_reward_mean'],
                "val/gated_reward_std": val_metrics['gated_reward_std'],
                "val/direction_oracle_mean": val_metrics['direction_oracle_mean'],
                "val/direction_oracle_std": val_metrics['direction_oracle_std'],
                "val/consistency_reward_mean": val_metrics['consistency_reward_mean'],
                "val/consistency_reward_std": val_metrics['consistency_reward_std'],
                "val/consistency_agonist_mean": val_metrics['consistency_agonist_mean'],
                "val/consistency_antagonist_mean": val_metrics['consistency_antagonist_mean'],
                "val/valid_fraction_mean": val_metrics['valid_fraction_mean'],
                "val/direction_accuracy_mean": val_metrics['direction_accuracy_mean'],
                "val/direction_accuracy_std": val_metrics['direction_accuracy_std'],
                "val/success_rate_mean": val_metrics['success_rate_mean'],
                "val/success_rate_std": val_metrics['success_rate_std']
            })

        # Save checkpoint
        if (epoch + 1) % args.save_every_n_epochs == 0:
            model_path = os.path.join(args.save_path, f'model_epoch_{epoch}.ckpt')
            save_model(policy_model, model_path, config=vars(args), epoch=epoch)

    # Final save
    final_model_path = os.path.join(args.save_path, 'model_final.ckpt')
    save_model(policy_model, final_model_path, config=vars(args))

    cleanup_wandb()
    logger.info(f"\n{SEPARATOR_LINE}")
    logger.info("Training completed!")
    logger.info(f"{SEPARATOR_LINE}\n")


if __name__ == '__main__':
    main()
