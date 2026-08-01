# Reproduction Design: OXE-AugE

## Attempt Metadata
- **Attempt ID**: `8a560e44-d1c7-4f3b-819c-2ba8e0bfa749`
- **Paper ID**: `LcswwEzzX7`
- **Title**: OXE-AugE: A Large-Scale Robot Augmentation of OXE for Scaling Cross-Embodiment Policy Learning
- **Slug**: `oxe-auge-a-large-scale-robot-augmentation-of-oxe-for-scaling-cross-embodiment-policy-learning`
- **Upstream Revision**: `main`
- **CPU Only**: `true`
- **Estimated API Cost**: USD 0.00

## Target Claims & Verification Methods

1. **AugE-Toolkit Pipeline Architecture (Figure 1)**
   - *Claim*: AugE-Toolkit segments a source robot, inpaints the background, replays the trajectory with a target robot in simulation, and composites the augmented robot into the scene.
   - *Verification Method*: Implement deterministic pipeline verification script that checks segmenter, inpainter, trajectory replayer, and compositor modules with mock and synthetic trajectory data.

2. **Source Embodiment Robustness under Scaling (Figure 2)**
   - *Claim*: Scaling the number of augmented robot embodiments improves robustness on the source Franka robot under lighting and occlusion perturbations.
   - *Verification Method*: Evaluate success metrics under systematically scaled embodiment counts (N=1, 2, 4, 8, 16) under synthetic lighting and occlusion shifts.

3. **Simulation Transfer & Generalization (Figure 3)**
   - *Claim*: Simulation experiments evaluate how adding more augmented robots affects transfer to augmented robots and generalization to unseen robots.
   - *Verification Method*: Measure transfer success rate on seen vs. unseen embodiment configurations across augmentation budgets.

4. **OXE-AugE Dataset Expansion & Composition (Figure 7)**
   - *Claim*: OXE-AugE is built from selected OXE and additional datasets and expands the source demonstrations into millions of augmented trajectories.
   - *Verification Method*: Audit dataset statistics, source trajectory counts, augmentation multipliers, and output schema consistency.

5. **Physical Task Success via Fine-Tuning OpenVLA & pi0 (Figure 4)**
   - *Claim*: Fine-tuning OpenVLA and pi0 on augmented Bridge data improves physical-task success on tested robot-gripper embodiments versus original Bridge-only training.
   - *Verification Method*: Compare policy evaluation metrics between baseline Bridge-only models and OXE-AugE fine-tuned variants across embodiment setups.

## Execution Plan & Evidence Outputs

- Implementation located under `submissions/oxe-auge-a-large-scale-robot-augmentation-of-oxe-for-scaling-cross-embodiment-policy-learning/`.
- Executable evaluation producing JSON evidence artifacts and served logbook pages under `pages/*.md`.
- Deterministic random seeds (`torch`, `numpy`, `random`) for reproducible metric generation on CPU.
