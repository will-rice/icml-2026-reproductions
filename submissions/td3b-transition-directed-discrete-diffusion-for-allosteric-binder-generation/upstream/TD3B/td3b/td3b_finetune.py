"""
TD3B Finetuning Loop
Extends TR2-D2 training with contrastive loss and directional rewards.
"""

import numpy as np
import torch
import wandb
import os
# NOTE: `loss_wdce` is imported lazily inside td3b_finetune() (see below) to avoid a
# circular import: training.finetune_utils imports the td3b package at module load,
# so importing it here at module level would deadlock the finetune_utils <-> td3b cycle.
from .td3b_losses import TD3BTotalLoss, extract_embeddings_from_mdlm
from tqdm import tqdm
import pandas as pd
try:
    from plotting import plot_data_with_distribution_seaborn, plot_data
except ImportError:
    # `plotting` is an optional dev-only helper used solely by the legacy
    # single-target loop below. Guard the import so that `import td3b` (and the
    # inference/training entry points that depend on it) work without it.
    plot_data_with_distribution_seaborn = plot_data = None


def td3b_finetune(
    args,
    cfg,
    policy_model,
    reward_model,
    mcts=None,
    pretrained=None,
    filename=None,
    prot_name=None,
    eps=1e-5,
    # TD3B-specific arguments
    contrastive_weight=0.1,
    contrastive_margin=1.0,
    contrastive_type='margin',
    embedding_pool_method='mean',
    kl_beta=0.1
):
    """
    TD3B finetuning with combined WDCE + contrastive loss + KL regularization.

    Args:
        args: Configuration arguments
        cfg: Hydra config
        policy_model: Policy model (MDLM)
        reward_model: Reward scoring functions (TD3BRewardFunction)
        mcts: TD3B_MCTS instance
        pretrained: Pretrained model (for no-MCTS mode)
        filename: Output filename
        prot_name: Target protein name
        eps: Small epsilon
        contrastive_weight: λ for contrastive loss
        contrastive_margin: Margin for margin-based contrastive loss
        contrastive_type: 'margin' or 'infonce'
        embedding_pool_method: 'mean', 'max', or 'cls'
        kl_beta: β coefficient for KL divergence regularization
    Returns:
        batch_losses: List of training losses
    """
    base_path = args.base_path
    dt = (1 - eps) / args.total_num_steps

    if args.no_mcts:
        assert pretrained is not None, "pretrained model is required for no mcts"
    else:
        assert mcts is not None, "mcts is required for mcts"

    # Create reference model (frozen copy of policy model at start of training)
    # Cannot use copy.deepcopy() due to unpicklable objects (file handles, etc.)
    # Instead, create a new model instance and load CLONED state dict
    print("[TD3B] Creating reference model for KL regularization...")

    # Import Diffusion class
    from models.diffusion import Diffusion

    # Create new instance with same config
    reference_model = Diffusion(
        config=policy_model.config,
        tokenizer=policy_model.tokenizer,
        mode="eval",
        device=policy_model.device if hasattr(policy_model, 'device') else args.device
    )

    # Get the device from policy model
    device = policy_model.device if hasattr(policy_model, 'device') else args.device
    if device is None:
        device = next(policy_model.parameters()).device

    # IMPORTANT: Clone the state dict to create independent tensors
    # This ensures no memory sharing between policy and reference model
    state_dict_copy = {
        key: value.clone().detach()
        for key, value in policy_model.state_dict().items()
    }
    reference_model.load_state_dict(state_dict_copy)

    # Move reference model to same device as policy model
    reference_model = reference_model.to(device)

    # Freeze and set to eval mode
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    print(f"[TD3B] Reference model frozen with {sum(p.numel() for p in reference_model.parameters())} parameters")
    print(f"[TD3B] Reference model on device: {device}")

    # Verify no parameter sharing
    policy_params = {id(p) for p in policy_model.parameters()}
    ref_params = {id(p) for p in reference_model.parameters()}
    assert len(policy_params.intersection(ref_params)) == 0, \
        "ERROR: Reference model shares parameters with policy model!"
    print("[TD3B] ✓ Verified: No parameter sharing between policy and reference model")

    # Initialize TD3B total loss
    td3b_loss_fn = TD3BTotalLoss(
        contrastive_weight=contrastive_weight,
        contrastive_margin=contrastive_margin,
        contrastive_type=contrastive_type,
        kl_beta=kl_beta,
        reference_model=reference_model
    )

    # Set model to train mode
    policy_model.train()
    torch.set_grad_enabled(True)
    optim = torch.optim.AdamW(policy_model.parameters(), lr=args.learning_rate)

    # Record metrics
    batch_losses = []
    batch_wdce_losses = []
    batch_contrastive_losses = []
    batch_kl_losses = []

    # Initialize saved trajectories
    x_saved, log_rnd_saved, final_rewards_saved = None, None, None
    directional_labels_saved, confidences_saved = None, None

    # Logs
    valid_fraction_log = []
    affinity_log = []
    gated_reward_log = []
    confidence_log = []
    direction_prediction_log = []  # Oracle predictions f_φ ∈ [0, 1]
    consistency_reward_log = []  # d* × (f_φ - 0.5)

    ### Fine-Tuning Loop ###
    pbar = tqdm(range(args.num_epochs))

    for epoch in pbar:
        rewards = []
        losses = []

        policy_model.train()

        with torch.no_grad():
            if x_saved is None or epoch % args.resample_every_n_step == 0:
                # Generate trajectories
                if args.no_mcts:
                    # Direct sampling (not typical for TD3B, but keep for compatibility)
                    x_final, log_rnd, final_rewards = policy_model.sample_finetuned_with_rnd(
                        args, reward_model, pretrained
                    )
                    directional_labels = torch.zeros(x_final.size(0), dtype=torch.float32)
                    confidences = torch.ones(x_final.size(0), dtype=torch.float32)
                else:
                    # TD3B MCTS forward pass
                    # For dual-direction mode, sample BOTH directions in the same batch
                    if hasattr(args, 'target_direction') and args.target_direction == 'both':
                        print(f"[Dual-direction] Epoch {epoch}: Sampling BOTH agonist and antagonist binders")

                        # Sample agonist binders (d* = +1)
                        reward_model.target_direction = 1.0
                        if epoch % args.reset_every_n_step == 0:
                            results_agonist = mcts.forward(resetTree=True)
                        else:
                            results_agonist = mcts.forward(resetTree=False)

                        # Sample antagonist binders (d* = -1)
                        reward_model.target_direction = -1.0
                        # Don't reset tree for antagonist to save computation
                        results_antagonist = mcts.forward(resetTree=False)

                        # Unpack both results
                        if len(results_agonist) == 7 and len(results_antagonist) == 7:
                            x_agonist, log_rnd_agonist, rewards_agonist, _, _, labels_agonist, conf_agonist = results_agonist
                            x_antagonist, log_rnd_antagonist, rewards_antagonist, _, _, labels_antagonist, conf_antagonist = results_antagonist

                            # Force labels to be correct (in case oracle is wrong)
                            labels_agonist = torch.ones(x_agonist.size(0), dtype=torch.float32) * 1.0  # +1 for agonist
                            labels_antagonist = torch.ones(x_antagonist.size(0), dtype=torch.float32) * -1.0  # -1 for antagonist

                            # Combine both directions into single batch
                            x_final = torch.cat([x_agonist, x_antagonist], dim=0)
                            log_rnd = torch.cat([log_rnd_agonist, log_rnd_antagonist], dim=0)
                            final_rewards = np.concatenate([rewards_agonist, rewards_antagonist], axis=0)
                            directional_labels = torch.cat([labels_agonist, labels_antagonist], dim=0)
                            confidences = torch.cat([
                                conf_agonist if isinstance(conf_agonist, torch.Tensor) else torch.tensor(conf_agonist),
                                conf_antagonist if isinstance(conf_antagonist, torch.Tensor) else torch.tensor(conf_antagonist)
                            ], dim=0)

                            print(f"  → Combined batch: {x_agonist.size(0)} agonists + {x_antagonist.size(0)} antagonists = {x_final.size(0)} total")
                            print(f"  → Directional labels: {torch.unique(directional_labels).tolist()} (DIVERSITY CONFIRMED!)")
                        else:
                            raise ValueError("Dual-direction mode requires 7-value return from MCTS")
                    else:
                        # Single-direction mode
                        if epoch % args.reset_every_n_step == 0:
                            results = mcts.forward(resetTree=True)
                        else:
                            results = mcts.forward(resetTree=False)

                        # Unpack results (TD3B version includes directional labels and confidences)
                        if len(results) == 7:
                            x_final, log_rnd, final_rewards, score_vectors, sequences, directional_labels, confidences = results
                            # Convert numpy arrays to tensors immediately for consistency
                            if not isinstance(directional_labels, torch.Tensor):
                                directional_labels = torch.tensor(directional_labels, dtype=torch.float32)
                            if not isinstance(confidences, torch.Tensor):
                                confidences = torch.tensor(confidences, dtype=torch.float32)
                        else:
                            # Fallback for compatibility with base MCTS
                            x_final, log_rnd, final_rewards, score_vectors, sequences = results
                            directional_labels = torch.zeros(x_final.size(0), dtype=torch.float32)
                            confidences = torch.ones(x_final.size(0), dtype=torch.float32)

                # Save for next iteration
                x_saved = x_final
                log_rnd_saved = log_rnd
                final_rewards_saved = final_rewards
                directional_labels_saved = directional_labels
                confidences_saved = confidences
            else:
                # Reuse cached trajectories
                x_final = x_saved
                log_rnd = log_rnd_saved
                final_rewards = final_rewards_saved
                directional_labels = directional_labels_saved
                confidences = confidences_saved

        # Compute WDCE loss (lazy import breaks the finetune_utils <-> td3b cycle)
        from training.finetune_utils import loss_wdce
        wdce_loss = loss_wdce(
            policy_model,
            log_rnd,
            x_final,
            num_replicates=args.wdce_num_replicates,
            centering=args.centering
        )

        # Compute KL divergence loss
        # Use a random masking and forward pass for KL computation
        mask_index = policy_model.mask_index
        device = x_final.device

        # Sample random noise level
        lamda = torch.rand(x_final.shape[0], device=device)  # (B,)
        sigma_kl = -torch.log1p(-(1 - eps) * lamda)

        # Apply random masking
        masked_index = torch.rand(*x_final.shape, device=device) < lamda[..., None]  # (B, L)
        perturbed_batch = torch.where(masked_index, mask_index, x_final)
        attn_mask_kl = torch.ones_like(perturbed_batch).to(device)

        # Compute KL loss
        kl_loss = td3b_loss_fn.compute_kl_loss(
            policy_model,
            perturbed_batch,
            attn_mask_kl,
            sigma_kl
        )

        # Extract embeddings for contrastive loss
        # Only compute if we have directional labels
        if directional_labels is not None and len(torch.unique(directional_labels)) > 1:
            # Get device from backbone
            device = policy_model.backbone.device if hasattr(policy_model.backbone, 'device') else x_final.device

            embeddings = extract_embeddings_from_mdlm(
                policy_model,
                x_final.to(device),
                pool_method=embedding_pool_method
            )

            # Move directional labels to same device
            directional_labels = directional_labels.to(embeddings.device)

            # Enable debug mode for first 3 epochs or if loss was zero last epoch
            debug_mode = (epoch < 3) or (epoch > 0 and batch_contrastive_losses and batch_contrastive_losses[-1] < 1e-6)

            # Compute total TD3B loss
            total_loss, loss_dict = td3b_loss_fn.compute_loss(
                wdce_loss,
                embeddings,
                directional_labels,
                kl_loss=kl_loss,  # Pass KL loss
                debug=debug_mode  # Enable debugging when needed
            )
        else:
            # If no directional diversity, skip contrastive loss
            print(f"[WARNING] Epoch {epoch}: No directional diversity! Skipping contrastive loss.")
            print(f"  Labels: {directional_labels.cpu().tolist() if directional_labels is not None else 'None'}")
            total_loss = wdce_loss + td3b_loss_fn.kl_beta * kl_loss
            loss_dict = {
                'total_loss': total_loss.item(),
                'wdce_loss': wdce_loss.item(),
                'contrastive_loss': 0.0,
                'kl_loss': kl_loss.item()
            }

        # Gradient descent
        total_loss.backward()

        # Gradient clipping
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), args.gradnorm_clip)

        optim.step()
        optim.zero_grad()

        pbar.set_postfix(
            total_loss=loss_dict['total_loss'],
            wdce=loss_dict['wdce_loss'],
            ctr=loss_dict['contrastive_loss']
        )

        # Evaluation sampling
        x_eval, eval_metrics = policy_model.sample_finetuned_td3b(
            args,
            reward_model,
            batch_size=50,
            dataframe=False
        )

        # Extract metrics (TD3B-specific)
        affinity = eval_metrics.get('affinity', [0])
        gated_reward = eval_metrics.get('gated_reward', [0])
        confidence = eval_metrics.get('confidence', [1])
        valid_fraction = eval_metrics.get('valid_fraction', 0)

        # Extract direction predictions (f_φ ∈ [0, 1])
        direction_predictions = eval_metrics.get('direction_predictions', [0.5])

        # Compute consistency reward: d* × (f_φ - 0.5)
        # Get target direction d* from reward_model
        d_star = reward_model.target_direction  # +1 or -1
        consistency_rewards = [d_star * (f_phi - 0.5) for f_phi in direction_predictions]

        # Append to logs
        affinity_log.append(affinity)
        gated_reward_log.append(gated_reward)
        confidence_log.append(confidence)
        valid_fraction_log.append(valid_fraction)
        direction_prediction_log.append(direction_predictions)
        consistency_reward_log.append(consistency_rewards)

        batch_losses.append(loss_dict['total_loss'])
        batch_wdce_losses.append(loss_dict['wdce_loss'])
        batch_contrastive_losses.append(loss_dict['contrastive_loss'])
        batch_kl_losses.append(loss_dict.get('kl_loss', 0.0))

        # Compute search statistics
        if args.no_mcts:
            mean_reward_search = final_rewards.mean().item()
            min_reward_search = final_rewards.min().item()
            max_reward_search = final_rewards.max().item()
            median_reward_search = final_rewards.median().item()
        else:
            mean_reward_search = np.mean(final_rewards)
            min_reward_search = np.min(final_rewards)
            max_reward_search = np.max(final_rewards)
            median_reward_search = np.median(final_rewards)

        # Compute direction oracle and consistency reward statistics
        mean_direction = np.mean(direction_predictions) if len(direction_predictions) > 0 else 0.5
        std_direction = np.std(direction_predictions) if len(direction_predictions) > 0 else 0.0
        mean_consistency = np.mean(consistency_rewards) if len(consistency_rewards) > 0 else 0.0
        std_consistency = np.std(consistency_rewards) if len(consistency_rewards) > 0 else 0.0

        print(
            f"epoch {epoch} | "
            f"affinity {np.mean(affinity):.4f} | "
            f"gated_reward {np.mean(gated_reward):.4f} | "
            f"confidence {np.mean(confidence):.4f} | "
            f"valid_frac {valid_fraction:.4f} | "
            f"direction_oracle {mean_direction:.4f}±{std_direction:.4f} | "
            f"consistency_reward {mean_consistency:.4f}±{std_consistency:.4f} | "
            f"total_loss {loss_dict['total_loss']:.4f} | "
            f"wdce_loss {loss_dict['wdce_loss']:.4f} | "
            f"contrastive_loss {loss_dict['contrastive_loss']:.4f} | "
            f"kl_loss {loss_dict.get('kl_loss', 0.0):.4f}"
        )

        # W&B logging
        wandb.log({
            "epoch": epoch,
            "affinity": np.mean(affinity),
            "gated_reward": np.mean(gated_reward),
            "confidence": np.mean(confidence),
            "valid_fraction": valid_fraction,
            "direction_oracle/mean": mean_direction,
            "direction_oracle/std": std_direction,
            "consistency_reward/mean": mean_consistency,
            "consistency_reward/std": std_consistency,
            "total_loss": loss_dict['total_loss'],
            "wdce_loss": loss_dict['wdce_loss'],
            "contrastive_loss": loss_dict['contrastive_loss'],
            "kl_loss": loss_dict.get('kl_loss', 0.0),
            "mean_reward_search": mean_reward_search,
            "min_reward_search": min_reward_search,
            "max_reward_search": max_reward_search,
            "median_reward_search": median_reward_search
        })

        # Save checkpoint
        if (epoch + 1) % args.save_every_n_epochs == 0:
            model_path = os.path.join(args.save_path, f'model_{epoch}.ckpt')
            torch.save(policy_model.state_dict(), model_path)
            print(f"model saved at epoch {epoch}")

    ### End of Fine-Tuning Loop ###

    wandb.finish()

    # Save logs and plots
    plot_path = f'{base_path}/TR2-D2/tr2d2-pep/results/{args.run_name}'
    os.makedirs(plot_path, exist_ok=True)
    output_log_path = f'{base_path}/TR2-D2/tr2d2-pep/results/{args.run_name}/log_{filename}.csv'
    save_td3b_logs_to_file(
        valid_fraction_log,
        affinity_log,
        gated_reward_log,
        confidence_log,
        direction_prediction_log,
        consistency_reward_log,
        output_log_path
    )

    plot_data(valid_fraction_log,
              save_path=f'{base_path}/TR2-D2/tr2d2-pep/results/{args.run_name}/valid_{filename}.png')

    plot_data_with_distribution_seaborn(
        log1=affinity_log,
        save_path=f'{base_path}/TR2-D2/tr2d2-pep/results/{args.run_name}/affinity_{filename}.png',
        label1=f"Average Affinity to {prot_name}",
        title=f"Average Affinity to {prot_name} Over Iterations"
    )

    plot_data_with_distribution_seaborn(
        log1=gated_reward_log,
        save_path=f'{base_path}/TR2-D2/tr2d2-pep/results/{args.run_name}/gated_reward_{filename}.png',
        label1="Average Gated Reward",
        title="Average Gated Reward Over Iterations"
    )

    plot_data_with_distribution_seaborn(
        log1=confidence_log,
        save_path=f'{base_path}/TR2-D2/tr2d2-pep/results/{args.run_name}/confidence_{filename}.png',
        label1="Average Confidence",
        title="Average Confidence Over Iterations"
    )

    # Final evaluation
    x_eval, eval_metrics, df = policy_model.sample_finetuned_td3b(
        args,
        reward_model,
        batch_size=200,
        dataframe=True
    )
    df.to_csv(f'{base_path}/TR2-D2/tr2d2-pep/results/{args.run_name}/{prot_name}_generation_results.csv', index=False)

    return batch_losses


def save_td3b_logs_to_file(valid_fraction_log, affinity_log, gated_reward_log, confidence_log,
                           direction_prediction_log, consistency_reward_log, output_path):
    """
    Saves TD3B-specific logs to a CSV file.

    Parameters:
        valid_fraction_log (list): Log of valid fractions over iterations.
        affinity_log (list): Log of binding affinity over iterations.
        gated_reward_log (list): Log of gated rewards over iterations.
        confidence_log (list): Log of confidence scores over iterations.
        direction_prediction_log (list): Log of direction oracle predictions over iterations.
        consistency_reward_log (list): Log of consistency rewards over iterations.
        output_path (str): Path to save the log CSV file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Combine logs into a DataFrame
    log_data = {
        "Iteration": list(range(1, len(valid_fraction_log) + 1)),
        "Valid Fraction": valid_fraction_log,
        "Binding Affinity": affinity_log,
        "Gated Reward": gated_reward_log,
        "Confidence": confidence_log,
        "Direction Oracle": direction_prediction_log,
        "Consistency Reward": consistency_reward_log
    }

    df = pd.DataFrame(log_data)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Logs saved to {output_path}")


# Add sampling method to diffusion model (monkey patch or extend)
def add_td3b_sampling_to_model(model):
    """
    Adds TD3B-specific sampling method to the model.
    This is a helper function to extend the existing model.
    """
    def sample_finetuned_td3b(self, args, reward_model, batch_size=50, dataframe=False):
        """
        TD3B-specific sampling that returns directional metrics.
        """
        self.backbone.eval()
        self.noise.eval()

        if batch_size is None:
            batch_size = args.batch_size

        eps = getattr(args, "sampling_eps", 1e-5)
        num_steps = args.total_num_steps
        x_rollout = self.sample_prior(
            batch_size,
            args.seq_length).to(self.device, dtype=torch.long)

        timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
        dt = torch.tensor((1 - eps) / num_steps, device=self.device)

        for i in range(num_steps):
            t = timesteps[i] * torch.ones(x_rollout.shape[0], 1, device=self.device)
            log_p, x_next = self.single_reverse_step(x_rollout, t=t, dt=dt)
            x_rollout = x_next.to(self.device)

        mask_positions = (x_rollout == self.mask_index)
        if mask_positions.any().item():
            log_p, x_next = self.single_noise_removal(x_rollout, t=t, dt=dt)
            x_rollout = x_next.to(self.device)

        # Convert x to sequences to get valid ones
        from utils.app import PeptideAnalyzer
        analyzer = PeptideAnalyzer()
        sequences = self.tokenizer.batch_decode(x_rollout)
        valid_mask = torch.tensor([analyzer.is_peptide(seq) for seq in sequences], device=self.device)
        valid_sequences = [seq for seq, keep in zip(sequences, valid_mask.tolist()) if keep]
        valid_x_final = x_rollout[valid_mask] if valid_mask.any().item() else torch.empty(0, device=self.device)
        valid_fraction = len(valid_sequences) / batch_size

        if len(valid_sequences) > 0:
            result = reward_model(valid_sequences)
            if isinstance(result, tuple):
                total_rewards, info = result
                affinity = np.asarray(info.get('affinities', total_rewards))
                confidence = np.asarray(info.get('confidences', np.ones_like(affinity)))
                direction_predictions = np.asarray(info.get('directions', np.zeros_like(affinity)))
            else:
                total_rewards = np.asarray(result)
                if total_rewards.ndim > 1:
                    affinity = total_rewards[:, 0]
                else:
                    affinity = total_rewards
                confidence = np.ones_like(affinity)
                direction_predictions = np.zeros_like(affinity)

            rewards_t = torch.as_tensor(total_rewards, dtype=torch.float32, device=self.device)
            alpha = max(float(getattr(args, "alpha", 0.1)), 1e-6)
            weights = torch.softmax(rewards_t / alpha, dim=0)
            idx = torch.multinomial(weights, num_samples=batch_size, replacement=True)

            idx_np = idx.detach().cpu().numpy()
            x_resampled = valid_x_final[idx]
            sequences = [valid_sequences[i] for i in idx_np]
            total_rewards = total_rewards[idx_np]
            affinity = affinity[idx_np]
            confidence = confidence[idx_np]
            direction_predictions = direction_predictions[idx_np]
        else:
            x_resampled = x_rollout
            total_rewards = np.array([])
            affinity = np.array([])
            confidence = np.array([])
            direction_predictions = np.array([])

        eval_metrics = {
            'affinity': affinity,
            'gated_reward': total_rewards,
            'confidence': confidence,
            'direction_predictions': direction_predictions,
            'valid_fraction': valid_fraction
        }

        if dataframe:
            df = pd.DataFrame({
                'sequence': sequences if len(total_rewards) else [],
                'affinity': affinity,
                'gated_reward': total_rewards,
                'confidence': confidence
            })
            return x_resampled, eval_metrics, df
        else:
            return x_resampled, eval_metrics

    # Attach method to model
    model.sample_finetuned_td3b = sample_finetuned_td3b.__get__(model, type(model))
    return model
