# TD3B Numeric Evidence Surface

This page exposes the concrete numbers in the machine-readable evidence
bundle. It separates code/file verification from table-scale metrics that
were not recomputed because the primary labels, generated binders, or
baseline result artifacts are absent.

## Claim Status Counts

| Status | Claim IDs | Count |
| --- | --- | ---: |
| verified | 1, 2 | 2 |
| unavailable | 3, 4, 5, 6 | 4 |

## Table Metrics Not Reproduced

| Paper metric claim | Paper-side number named by the challenge | Reproduction status |
| --- | ---: | --- |
| Direction Oracle accuracy | 0.93 | unavailable: no primary test labels or evaluation outputs |
| Direction Oracle precision | 0.90 | unavailable: no primary test labels or evaluation outputs |
| Direction Oracle recall | 0.91 | unavailable: no primary test labels or evaluation outputs |
| Direction Oracle F1 | 0.90 | unavailable: no primary test labels or evaluation outputs |
| Forward transition success | 61% | unavailable: generated binders and success labels absent |
| Reverse transition success | 100% | unavailable: generated binders and success labels absent |

## Checkpoint LFS Pointers

The evidence records the released checkpoint pointer sizes and hashes but
does not download or execute the multi-GB checkpoints in the CPU-only path.

| File | Pointer size | SHA-256 |
| --- | ---: | --- |
| checkpoints/direction_oracle.pt | 2850095568 | 5ee476c8100752caab069d17569beaece06728d3c8a92223b603c3cba6a9246d |
| checkpoints/pretrained.ckpt | 1386966244 | b259f022c21121f5c755fed61230d6fdf2626ee4ab8a23df479b3cf553fd4aef |
| checkpoints/td3b.ckpt | 231462144 | 9b8aeecbfe29b4652860028135c2d7abd2688cfa51aa939b419dd3aec41495d4 |

## Provenance

| Field | Value |
| --- | --- |
| upstream repository | ChatterjeeLab/TD3B |
| upstream revision | 7d3c9bfe171a1db77e7b5431c572dadce8520bb5 |
| challenge snapshot | d32beb9e79859f40a37e565155ef84fb3bdc6bf3679e8f79e8f5414cc3f60600 |
| challenge revision | 81166abbeb76e5f79ff87e51061b5a0306507203 |
| generated at | 2026-08-01T13:52:00+00:00 |
| paid API cost | 0.00 USD |

## Missing Primary Artifacts

The unavailable statuses come from missing primary artifacts, not from
failed metric reproduction:

- data/test.csv
- data/train.csv
- data/td3b_data_new.csv
- generated_binders/agonist
- generated_binders/antagonist.tar.gz
