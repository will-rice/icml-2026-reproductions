# SceneSmith Reproduction Design

Attempt: `4134f5fc-75c7-4c17-8f43-3685aa8c3ac2`
Paper: `WwS8CTpUA6`
Worker: `codex-paper-owner-01`

## Upstream Pins

- Paper: arXiv `2602.09153`, OpenReview `WwS8CTpUA6`.
- Code: `https://github.com/nepfaff/scenesmith`, commit `67cc408fd38334b4a926efef45e284302ed5055b`.
- Public datasets found on Hugging Face: `nepfaff/scenesmith-example-scenes`, `nepfaff/scenesmith-preprocessed-data`, `nepfaff/scenesmith-sam3d-objects`.
- Local source inspection clone: `/tmp/scenesmith-upstream-67cc408.IxOHcU`.

## Reproduction Scope

The full SceneSmith generation pipeline is not CPU-only: the upstream README documents required API keys, SAM3D checkpoints, ArtVIP/AmbientCG data, and 32-45GB GPU memory for paper-quality scene generation. This attempt will therefore build a lightweight, independently executable evidence harness that vendors a small upstream source snapshot and statically verifies released-code support for the architectural claims.

The harness will not transcribe paper-reported benchmark/user-study values as reproduced metrics. Large-scale generation quality, object-count/collision/stability statistics, user-study win rates, and robot policy results will be marked unavailable unless released example-scene or result files in the public artifacts contain machine-readable evidence sufficient to recompute them.

## Evidence Plan

1. Bundle selected upstream files needed for inspection: `README.md`, `pyproject.toml`, core configuration YAML files, representative agent modules, asset/retrieval/generation modules, robot-evaluation modules, scripts, tests inventory, and a machine-readable upstream manifest.
2. Write tests first for a `generate_evidence.py` harness that must:
   - verify the pinned upstream commit and source manifest;
   - verify the configured ordered stage pipeline includes floor-plan, furniture, wall-mounted, ceiling-mounted, and manipuland stages;
   - verify stateful agent modules implement planner/designer/critic role construction;
   - verify asset generation/retrieval/physical-property modules and scripts exist for SAM3D/Hunyuan3D, HSSD/Objaverse/ArtVIP, AmbientCG, Drake/MuJoCo/USD export, and physics validation;
   - verify robot-evaluation task generation, policy interface, state tools, vision tools, and success validator modules exist;
   - mark claims requiring large GPU/API/user-study artifacts as unavailable with missing-artifact reasons.
3. Generate `evidence/scenesmith_results.json` with claim-level statuses, source-file evidence, missing artifacts, commands, and environment notes.
4. Provide a small Gradio-compatible report surface (`app.py`, `pages/report.md`) showing only independently recomputed/static audit results and unavailable evidence.
5. Validate with paper tests, root pytest, skill quick validation, and pre-commit from a clean validation checkout.

## Claim Handling

- Claims 1-3 and 6 can receive source-verified or partial-source-verified evidence if the released code contains the corresponding modules and configs.
- Claims 4-5 require generated scenes across 210 prompts and a 205-participant user study. These will be unavailable unless released result datasets expose recomputable object/collision/stability and user-study records.

## Expected Limitations

The harness verifies that the public release implements the claimed components and pipeline surfaces. It does not claim to reproduce paper-scale scene quality or benchmark/user-study numbers on this host.
