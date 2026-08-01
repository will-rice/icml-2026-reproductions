#!/bin/bash

# Multi-Target TD3B Training Launch Script
# Trains TD3B on multiple protein targets with random sampling strategy

# ============================================================================
# Configuration
# ============================================================================

# Paths — BASE_PATH auto-detects the repo root (this script's own directory);
# the paths below derive from it, so they usually need no editing.
BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRETRAINED_CHECKPOINT="${BASE_PATH}/checkpoints/pretrained.ckpt"
TRAIN_CSV="${BASE_PATH}/data/train.csv"
VAL_CSV="${BASE_PATH}/data/test.csv"  # Optional: create validation split

# Run configuration
RUN_NAME="multi_target_td3b"  # Timestamp will be added automatically
DEVICE="cuda:0"
# Multi-target sampling
TARGETS_PER_MCTS=2        # Number of targets sampled per MCTS round (K)
RESAMPLE_TARGETS_EVERY=1  # Resample targets every N epochs

# Training hyperparameters
NUM_EPOCHS=200
LEARNING_RATE=3e-4
TRAIN_BATCH_SIZE=1            # Small batch size to prevent OOM
GRADIENT_ACCUMULATION_STEPS=32  # Effective batch size = 16 * 4 = 64
RESAMPLE_EVERY=10              # Run MCTS every N epochs
SAVE_EVERY=20
VALIDATE_EVERY=20
RESET_TREE_EVERY=50

# MCTS hyperparameters (aligned with v1, but can reduce for multi-target)
NUM_ITER=20          # MCTS iterations per resample (v1 default: 50, reduced for multi-target)
NUM_CHILDREN=16     # Children per MCTS expansion
BUFFER_SIZE=50       # Pareto buffer size (v1 default: 50)
REPLAY_BUFFER_SIZE=1000  # Recommended range: 500-5000 (0 disables replay)
REPLAY_BUFFER_STRATEGY="fifo"  # fifo or random
ALPHA=0.1           # Temperature for importance weighting
EXPLORATION=1.0     # UCB exploration constant

# TD3B hyperparameters (aligned with v1 defaults)
CONTRASTIVE_WEIGHT=0.1      # v1 default: 0.1
CONTRASTIVE_MARGIN=1.0
KL_BETA=0.1                 # v1 default: 0.1
MIN_AFFINITY_THRESHOLD=0.0  # CRITICAL: minimum affinity for allosteric control
SIGMOID_TEMPERATURE=0.1

# Validation
VAL_SAMPLES_PER_TARGET=20  # Number of sequences per target during validation

# Directional oracle (GPCR classifier)
ORACLE_CKPT="${BASE_PATH}/checkpoints/direction_oracle.pt"
ORACLE_TR2D2_CHECKPOINT="${BASE_PATH}/checkpoints/pretrained.ckpt"
ORACLE_TOKENIZER_VOCAB="${BASE_PATH}/tokenizer/new_vocab.txt"
ORACLE_TOKENIZER_SPLITS="${BASE_PATH}/tokenizer/new_splits.txt"
ORACLE_ESM_NAME="facebook/esm2_t33_650M_UR50D"
ORACLE_ESM_CACHE_DIR=""  # Optional: set to a cache dir path
ORACLE_ESM_LOCAL_FILES_ONLY=0  # Set to 1 to avoid network access
ORACLE_MAX_LIGAND_LENGTH=768
ORACLE_MAX_PROTEIN_LENGTH=1024
ORACLE_D_MODEL=256
ORACLE_N_HEADS=4
ORACLE_N_SELF_ATTN_LAYERS=1
ORACLE_N_BMCA_LAYERS=2
ORACLE_DROPOUT=0.3

EXTRA_ORACLE_ARGS=""
if [ -n "$ORACLE_ESM_CACHE_DIR" ]; then
    EXTRA_ORACLE_ARGS="$EXTRA_ORACLE_ARGS --direction_oracle_esm_cache_dir $ORACLE_ESM_CACHE_DIR"
fi
if [ "$ORACLE_ESM_LOCAL_FILES_ONLY" -eq 1 ]; then
    EXTRA_ORACLE_ARGS="$EXTRA_ORACLE_ARGS --direction_oracle_esm_local_files_only"
fi

# W&B (optional)
WANDB_PROJECT="tr2d2-multi-target"
WANDB_ENTITY=""  # set to your W&B entity/team (empty = your default entity)

# ============================================================================
# Launch Training
# ============================================================================

cd ${BASE_PATH}

echo "============================================================================"
echo "Multi-Target TD3B Training"
echo "============================================================================"
echo "Configuration:"
echo "  - Targets per MCTS: ${TARGETS_PER_MCTS}"
echo "  - Training batch size: ${TRAIN_BATCH_SIZE}"
echo "  - Gradient accumulation: ${GRADIENT_ACCUMULATION_STEPS}"
echo "  - Effective batch size: $((TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "  - Epochs: ${NUM_EPOCHS}"
echo "  - MCTS iterations: ${NUM_ITER}"
echo "  - MCTS children: ${NUM_CHILDREN}"
echo "  - Buffer size: ${BUFFER_SIZE}"
echo "  - Replay buffer size: ${REPLAY_BUFFER_SIZE} (${REPLAY_BUFFER_STRATEGY})"
echo "============================================================================"
echo ""

# Build command
CMD="python finetune_multi_target.py \
    --base_path ${BASE_PATH} \
    --train_csv ${TRAIN_CSV} \
    --pretrained_checkpoint ${PRETRAINED_CHECKPOINT} \
    --run_name ${RUN_NAME} \
    --device ${DEVICE} \
    \
    --targets_per_mcts ${TARGETS_PER_MCTS} \
    --resample_targets_every ${RESAMPLE_TARGETS_EVERY} \
    \
    --num_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --train_batch_size ${TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --resample_every_n_step ${RESAMPLE_EVERY} \
    --save_every_n_epochs ${SAVE_EVERY} \
    --validate_every_n_epochs ${VALIDATE_EVERY} \
    --reset_every_n_step ${RESET_TREE_EVERY} \
    \
    --num_iter ${NUM_ITER} \
    --num_children ${NUM_CHILDREN} \
    --buffer_size ${BUFFER_SIZE} \
    --replay_buffer_size ${REPLAY_BUFFER_SIZE} \
    --replay_buffer_strategy ${REPLAY_BUFFER_STRATEGY} \
    --alpha ${ALPHA} \
    --exploration ${EXPLORATION} \
    \
    --contrastive_weight ${CONTRASTIVE_WEIGHT} \
    --contrastive_margin ${CONTRASTIVE_MARGIN} \
    --kl_beta ${KL_BETA} \
    --min_affinity_threshold ${MIN_AFFINITY_THRESHOLD} \
    --sigmoid_temperature ${SIGMOID_TEMPERATURE} \
    \
    --direction_oracle_ckpt ${ORACLE_CKPT} \
    --direction_oracle_tr2d2_checkpoint ${ORACLE_TR2D2_CHECKPOINT} \
    --direction_oracle_tokenizer_vocab ${ORACLE_TOKENIZER_VOCAB} \
    --direction_oracle_tokenizer_splits ${ORACLE_TOKENIZER_SPLITS} \
    --direction_oracle_esm_name ${ORACLE_ESM_NAME} \
    --direction_oracle_max_ligand_length ${ORACLE_MAX_LIGAND_LENGTH} \
    --direction_oracle_max_protein_length ${ORACLE_MAX_PROTEIN_LENGTH} \
    --direction_oracle_d_model ${ORACLE_D_MODEL} \
    --direction_oracle_n_heads ${ORACLE_N_HEADS} \
    --direction_oracle_n_self_attn_layers ${ORACLE_N_SELF_ATTN_LAYERS} \
    --direction_oracle_n_bmca_layers ${ORACLE_N_BMCA_LAYERS} \
    --direction_oracle_dropout ${ORACLE_DROPOUT} \
    ${EXTRA_ORACLE_ARGS} \
    \
    --val_samples_per_target ${VAL_SAMPLES_PER_TARGET} \
    \
    --grad_clip \
    --gradnorm_clip 1.0 \
    --wandb_project ${WANDB_PROJECT}"

# Add validation CSV if it exists
if [ -f "${VAL_CSV}" ]; then
    CMD="${CMD} --val_csv ${VAL_CSV}"
    echo "Validation CSV: ${VAL_CSV}"
else
    echo "No validation CSV found (${VAL_CSV})"
    echo "Skipping validation during training"
fi

# Add W&B entity if specified
if [ -n "${WANDB_ENTITY}" ]; then
    CMD="${CMD} --wandb_entity ${WANDB_ENTITY}"
fi

echo ""
echo "Launching training..."
echo ""

# Execute
eval $CMD
