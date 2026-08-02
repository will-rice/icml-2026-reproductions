# Summary: SCALE for Vision-Language-Action Models

- Attempt ID: `092b120a-9cf1-4351-b4fc-fcbf51f6565f`
- Paper ID: `7MlfE2Da2W`
- Slug: `scale-self-uncertainty-conditioned-adaptive-looking-and-execution-for-vision-language-action-models`
- Title: SCALE: Self-uncertainty Conditioned Adaptive Looking and Execution for Vision-Language-Action Models

## Overview

SCALE jointly modulates visual attention temperature and action decoding temperature using the VLA model's self-uncertainty during decoding.

- **Mechanism evidence**: Pinned SCALE source implements self-uncertainty, adaptive action temperature, adaptive visual attention temperature, and the SCALE decoding mode.
- **Training-free evidence**: Source-level checks support inference-only execution with no verifier, marked toy because this CPU audit does not time a deployed robot control step.
- **Unreplicated measurements**: LIBERO, SIMPLER-WidowX, and real-world robot success-rate claims require GPU simulation, pretrained VLA checkpoints, or physical robot hardware and are marked unavailable.
