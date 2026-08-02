# Reproduction & Verification Results

## Claim Verification Summary

| Claim ID | Target Claim | Verification Status | Evidence Summary |
| --- | --- | --- | --- |
| Claim 1 | Self-uncertainty modulation of attention & action temp | Verified | Pinned `modeling_prismatic.py` and `scale.yaml` |
| Claim 2 | Training-free, verifier-free single forward pass | Toy | Source-level checks support inference-only execution with no verifier, but this CPU audit does not time a deployed robot control step |
| Claim 3 | OpenVLA LIBERO benchmark success improvement | Unavailable | Benchmark success-rate comparisons were not rerun and no machine-readable raw result artifact was found in the pinned repo |
| Claim 4 | pi0-FAST LIBERO success improvement (91.2% -> 93.0%) | Unavailable | pi0-FAST/LIBERO average success rates were not rerun and are not backed by a raw result artifact |
| Claim 5 | SIMPLER-WidowX pi0-FAST success (34.4% -> 49.0%) | Unavailable | SIMPLER-WidowX claims require non-CPU benchmark reruns or raw logs absent from the pinned repo |
| Claim 6 | Real-world pick-and-place ID & OOD improvements | Unavailable | Real-world success rates require physical robot evaluations; no reproduced measurements are available here |

## Evidence Assets

- Bundle: `evidence/bundle.json`
- Tests: Pytest unit and integration tests passing deterministically
- Unreplicated: LIBERO, SIMPLER-WidowX, and real-world robot success-rate measurements
