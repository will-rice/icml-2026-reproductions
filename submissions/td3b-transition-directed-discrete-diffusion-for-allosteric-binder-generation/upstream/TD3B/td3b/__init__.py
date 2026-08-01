"""
TD3B: Transition-Directed Discrete Diffusion for Binders
A module extending TR2-D2 with directional allosteric control.
"""

from .direction_oracle import DirectionalOracle
from .td3b_scoring import TD3BRewardFunction, TD3BConfidenceWeighting, create_td3b_reward_function
from .td3b_losses import ContrastiveLoss, InfoNCELoss, TD3BTotalLoss, extract_embeddings_from_mdlm
from .td3b_mcts import TD3B_MCTS, create_td3b_mcts
from .td3b_finetune import td3b_finetune, add_td3b_sampling_to_model
from .data_utils import TD3BDataset, load_td3b_data

__all__ = [
    'DirectionalOracle',
    'TD3BRewardFunction',
    'TD3BConfidenceWeighting',
    'create_td3b_reward_function',
    'ContrastiveLoss',
    'InfoNCELoss',
    'TD3BTotalLoss',
    'extract_embeddings_from_mdlm',
    'TD3B_MCTS',
    'create_td3b_mcts',
    'td3b_finetune',
    'add_td3b_sampling_to_model',
    'TD3BDataset',
    'load_td3b_data',
]

__version__ = '0.1.0'
