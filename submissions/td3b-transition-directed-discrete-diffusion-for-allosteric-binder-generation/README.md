# TD3B Reproduction

This submission builds independently executable evidence for:

TD3B: Transition-Directed Discrete Diffusion for Allosteric Binder Generation

The evidence generator inspects the pinned upstream source release and Hub LFS metadata without downloading multi-GB checkpoints by default. It marks table-scale metric claims unavailable when the primary CSV/generated-binder artifacts are absent.

Run:

```bash
python generate_evidence.py --source-root upstream/TD3B --output evidence/td3b_results.json
pytest -q
```
