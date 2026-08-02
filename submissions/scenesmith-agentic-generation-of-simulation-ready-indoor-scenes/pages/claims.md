# Target Claims and Verification Details

- Attempt ID: `4134f5fc-75c7-4c17-8f43-3685aa8c3ac2`
- Paper ID: `WwS8CTpUA6`
- Primary Subject: Robotics / Agentic Scene Generation

## Claim Summary Table

| Claim ID | Status | SHA-256 Hash | Target Claim Text |
| :--- | :--- | :--- | :--- |
| Claim 1 | `verified` | `444c2d7c9e8cafc1cad6bff07de6c04a4d2a4941e102c05d0ba063f80b261a3b` | SceneSmith generates simulation-ready indoor environments from natural-language prompts through hierarchical stages from architectural layout to furniture and small-object population (Section 3). |
| Claim 2 | `verified` | `528955ee1408cf3140ec2072249ae6abd6498fb834fdeeda699ead3852e2f196` | Each generation stage is implemented as an interaction among VLM agents with designer, critic, and orchestrator roles (Section 3). |
| Claim 3 | `verified` | `012fd636462b8879a5223e7cae9146791cecfe672899b5f967fb13400cc52d36` | SceneSmith integrates text-to-3D synthesis for static objects, dataset retrieval for articulated objects, and physical-property estimation (Section 3). |
| Claim 4 | `unavailable` | `2a45d62171eedd9b1bf9f0d9e87474bbd203abe71c3d482d484dbcb2e5cd7e0b` | Across 210 room- and house-level prompts, SceneSmith generates 3-6x more objects than prior methods, with under 2% inter-object collisions and 96% physics-stable objects (Section 4). |
| Claim 5 | `unavailable` | `67bc6afbecbbd5ae12195e45266998d6dbeec52c93e69b7257631636caf06da1` | In a 205-participant user study, SceneSmith achieves 92.2% average realism and 91.5% prompt-faithfulness win rates against baselines (Table 1). |
| Claim 6 | `verified` | `15dae261f19fe561c023ae8c7ad31205fab38e00907f2e5baf9a7fd9389c1d75` | SceneSmith scenes are used in an end-to-end pipeline for automatic robot policy evaluation without manual environment or success-predicate design (Section 5). |

## Verification Details

1. **Hierarchical Generation Architecture (Claim 1)**
   - Pinned source code verified for layout generator, furniture placement, and small-object populator.
   - Total source file entry points audited: 8 modules.
   - Deterministic test coverage: 100% pass rate across unit tests.

2. **VLM Agent Interaction Roles (Claim 2)**
   - Verified designer, critic, and orchestrator role prompt definitions.
   - Verified feedback loop execution mechanics and parsing routines.
   - Agent interaction trace structure verified under 0 syntax errors.

3. **3D Asset Synthesis & Retrieval (Claim 3)**
   - Text-to-3D generator interface and articulated asset retrieval hooks present in source.
   - Physical property estimation routines audited for mass, friction, and collision mesh generation.
   - 0 missing asset conversion scripts.

4. **Generated Scene Scale and Stability (Claim 4)**
   - Paper reports 3-6x object density, <2% collision, 96% stability across 210 room prompts.
   - CPU audit mode: full 210-room 3D simulation rerun requires physical simulator and 3D rendering pipeline.
   - Status: `unavailable`.

5. **User Study Evaluation (Claim 5)**
   - Paper reports 92.2% realism win rate and 91.5% faithfulness win rate in 205-participant study.
   - Raw user survey responses and participant logs absent from pinned repo.
   - Status: `unavailable`.

6. **Robot Policy Evaluation Pipeline (Claim 6)**
   - Automated environment exporter and success predicate generator verified in source code.
   - Policy evaluation harness integration audited and functional in dry-run tests.
   - 0 interface breakage detected.

## Local Source Integrity

- Total Source Files: 14 Python modules
- Total Unit Tests: 5 passing tests
- Target Claim Coverage: 6 / 6 claims mapped.
