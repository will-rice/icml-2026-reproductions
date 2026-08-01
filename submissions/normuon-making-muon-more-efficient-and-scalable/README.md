# NorMuon Reproduction Evidence

This submission verifies CPU-testable mechanism claims for `m1IRWFAMsa`,
"NorMuon: Making Muon more efficient and scalable".

It uses the pinned official implementation
`github:zichongli5/NorMuon@c6989a8354730695d9f5a9faa6c55eeb24865209` and
does not reproduce GPU-scale LLM pretraining claims.

Run:

```bash
uv run --project submissions/normuon-making-muon-more-efficient-and-scalable \
  python submissions/normuon-making-muon-more-efficient-and-scalable/generate_evidence.py
uv run --project submissions/normuon-making-muon-more-efficient-and-scalable \
  python -m pytest submissions/normuon-making-muon-more-efficient-and-scalable/tests -q
```
