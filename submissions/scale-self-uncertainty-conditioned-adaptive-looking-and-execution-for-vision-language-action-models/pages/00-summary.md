# Summary: SCALE for Vision-Language-Action Models

- Attempt ID: `092b120a-9cf1-4351-b4fc-fcbf51f6565f`
- Paper ID: `7MlfE2Da2W`
- Slug: `scale-self-uncertainty-conditioned-adaptive-looking-and-execution-for-vision-language-action-models`
- Title: SCALE: Self-uncertainty Conditioned Adaptive Looking and Execution for Vision-Language-Action Models

## Overview

SCALE jointly modulates visual attention temperature and action decoding temperature using the VLA model's self-uncertainty during decoding.

- **Training-Free**: Requires zero additional training and no separate verifier module.
- **Single Pass**: Runs in a single forward pass per control step.
- **Backbone Compatibility**: Verified implementation for OpenVLA, pi0-FAST, and SpatialVLA backbones.

