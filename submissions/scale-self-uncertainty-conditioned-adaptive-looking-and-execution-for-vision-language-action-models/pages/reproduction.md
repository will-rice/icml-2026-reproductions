# SCALE Reproduction Evidence

Attempt `092b120a-9cf1-4351-b4fc-fcbf51f6565f` targets paper `7MlfE2Da2W` from snapshot `00263d6d9b331596d4be77a4cd17a4b1b6592f2ac7a72401cd62b751eaaef9bb`.

The evidence pins the official repository at `snumprlab/scale@b4ad2a69d14f91712704711e810cf9830e2b7121` and inspects source files instead of copying paper-reported benchmark values.

## Claim Status

- `claim-1`: verified at source level. The pinned code implements self-uncertainty, adaptive action temperature, adaptive visual attention temperature, SCALE defaults, and the SCALE decoding mode.
- `claim-2`: toy/source-level support only. The code and README support no-training/no-verifier inference, but this CPU audit does not measure a live robot control step.
- `claim-3`: unavailable. OpenVLA/LIBERO success-rate comparisons require benchmark reruns or raw result logs.
- `claim-4`: unavailable. pi0-FAST/LIBERO success-rate claims are not reproduced here.
- `claim-5`: unavailable. SIMPLER-WidowX claims are not reproduced here.
- `claim-6`: unavailable. Real-world pick-and-place success rates require physical robot experiments and are not reproduced here.

## Executable Surface

`generate_evidence.py` produces `evidence/bundle.json` with file hashes, upstream commit, source indicators, claim statuses, and unreplicated measurements. Pytest verifies that benchmark result claims are not promoted to reproduced evidence without raw artifacts.
