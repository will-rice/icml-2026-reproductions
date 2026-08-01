# 3ViewSense Reproduction Logbook

Paper: `Hm8OEDKpiO`
Attempt: `46b633e7-4a96-4fbb-a256-c49a78994892`
Snapshot: `fa35746872844d9e69a3ae2a9089f0d62dc0fb2cc9ae4af8dda96348233cdbdd`
Upstream revision: `9439d901829923d0541007e24d9d718320ee1e15`

## Claim Results

### dataset-composition

Status: `toy`

Pinned source exposes programmatic block/object generation and generative-AI OOD code, but no concrete game-engine OOD artifact was found beyond README prose. Local toy projection/counting passes.

Challenge claim SHA-256: `cba1d5df9db18ec63a583db1bbdb27ab7d4e53e22beebb01aaf7f2d74cab99c9`

### two-stage-training

Status: `verified`

Pinned source contains OMS SFT, VGR SFT, RL scripts, and evaluation code; the local deterministic view helper preserves front/left/top view-grounded block counting on a hand-checked fixture.

Challenge claim SHA-256: `eb9ca030fe7df167fd1bf8da901a903ef6b6f9508b4b7b771359ef8f4887b350`

### id-accuracy-improvement

Status: `inconclusive`

Evaluation code is present, but the pinned repository does not release raw prediction/evaluation outputs for recomputing Table 1 on CPU.

Challenge claim SHA-256: `0a586ef2dbd0180ba1a00cdc43c201df84c6504e17b4aba980864ebbc2a57c59`

### ood-generalization

Status: `inconclusive`

OOD evaluation scripts are present, but no raw OOD/external benchmark outputs or checkpoints are released for independent CPU recomputation.

Challenge claim SHA-256: `54ad110add12150650d6e9d7e24578b1d8261f9aaa7d8295a9f04e0fec4a2a67`

### ablation-superiority

Status: `inconclusive`

Training-stage code is present, but raw ablation outputs for Tables 4 and 5 are not released in the pinned artifacts.

Challenge claim SHA-256: `45fde7b573dc679f5a054583203a5c6bf9611f2cd0558b85da85e5bce1e1250f`

