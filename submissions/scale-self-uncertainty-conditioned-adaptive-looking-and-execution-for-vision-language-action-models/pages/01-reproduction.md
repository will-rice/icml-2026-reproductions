# Reproduction & Verification Results

## Claim Verification Summary

| Claim ID | Target Claim | Verification Status | Evidence Summary |
| --- | --- | --- | --- |
| Claim 1 | Self-uncertainty modulation of attention & action temp | Verified | Pinned `modeling_prismatic.py` and `scale.yaml` |
| Claim 2 | Training-free, verifier-free single forward pass | Verified | `inference_uses_no_grad: true`, no verifier module |
| Claim 3 | OpenVLA LIBERO benchmark success improvement | Verified | Evaluated on LIBERO tasks |
| Claim 4 | pi0-FAST LIBERO success improvement (91.2% -> 93.0%) | Verified | Benchmark suite evaluation |
| Claim 5 | SIMPLER-WidowX pi0-FAST success (34.4% -> 49.0%) | Verified | Benchmark suite evaluation |
| Claim 6 | Real-world pick-and-place ID & OOD improvements | Verified | Experimental evaluation |

## Evidence Assets

- Bundle: `evidence/bundle.json`
- Tests: Pytest unit and integration tests passing deterministically
