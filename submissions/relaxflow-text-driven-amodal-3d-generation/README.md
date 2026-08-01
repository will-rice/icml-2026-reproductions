# RelaxFlow: Text-Driven Amodal 3D Generation

Official reproduction package for **RelaxFlow** (ICML 2026 Paper ID `UamxHbDR3p`, arXiv:2603.05425).

## Summary
RelaxFlow introduces a training-free dual-branch framework fusing an observation branch and a semantic-prior branch via velocity blending for text-driven amodal 3D generation. The relaxation mechanism acts as low-pass filtering to reduce semantic vector field estimation errors.

## Running Verification
```bash
uv run python generate_evidence.py
uv run pytest
```
