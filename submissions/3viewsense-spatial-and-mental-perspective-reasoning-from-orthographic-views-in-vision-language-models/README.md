# 3ViewSense Reproduction

This submission audits the pinned 3ViewSense repository and runs a small
deterministic orthographic-view check. It does not use paper-reported metrics
as reproduced measurements.

Pinned upstream:

- Repository: `https://github.com/Jasaxion/3ViewSense`
- Revision: `9439d901829923d0541007e24d9d718320ee1e15`

Run from the repository root:

```bash
VIEWSENSE_UPSTREAM_ROOT=scratch/3viewsense-upstream \
  uv run python submissions/3viewsense-spatial-and-mental-perspective-reasoning-from-orthographic-views-in-vision-language-models/generate_evidence.py
uv run pytest -q submissions/3viewsense-spatial-and-mental-perspective-reasoning-from-orthographic-views-in-vision-language-models/tests
```
