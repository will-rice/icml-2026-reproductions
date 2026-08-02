# WarmServe Reproduction Claims & Evidence Details

Paper: `DVHpvumD60` | Attempt: `3d8acf83-97a0-43d3-a2fb-ea3e2b3c2b12`
Snapshot: `98fe583a0a55974d2d28e1beba12e398eb7e8b7f05fadb6d33fdd243b1988644`
Repository: `LLMServe/WarmServe@a60121519e077d2f128b597cbabc947e3e618aaf`

## Claim Assessment Summary

| Claim ID | Status | Challenge Claim SHA-256 | Summary |
| --- | --- | --- | --- |
| `claim-1` | `verified` | `0c0a74f9b2474f2a5c7861cd114a8540f6a35cad8376727574edc2deb6595b72` | Pinned source contains scheduler, prewarm manager, CUDA VMM, worker hooks, model config, and trace generator paths. |
| `claim-2` | `toy` | `74062b2b618fbcd71336f1910dc2ca3c093fb405878a352dc4025c9b85680135` | Trace-character generation and 5-minute windows present; AzureConv 7.3% error not recomputed without raw trace CSV. |
| `claim-3` | `unavailable` | `554d16b3aacae8ab9f26fda580eea541c9a22b74a2a4597602f63057eae3a5ac` | TTFT prewarming measurements require CUDA/Ray/vLLM deployment and raw logs. |
| `claim-4` | `unavailable` | `aa8aacfa03b2d61f40b5bc023835e12031c912f34805afa06ac9c0cb99970ad2` | End-to-end 50.8x tail TTFT claim was not rerun on GPU cluster. |
| `claim-5` | `unavailable` | `6182c76e4e228313cf3145aae8dc6e915d9645424f5b5f8d8032b07ec05264a0` | Ablation claims require benchmark logs or reruns not available in CPU audit. |
| `claim-6` | `unavailable` | `b6ad7a59b380d5d0e26e414fc8d831d0da015dc23ed944cd1e5d1bdd7f2e44c8` | 512-GPU P99 TTFT simulation was not rerun and no raw simulation artifact was found. |

## Source Verification Identifiers

- Tracked files in upstream repository: 1,294 files at commit `a60121519e077d2f128b597cbabc947e3e618aaf`
- `prewarm_manager.py`: 67,065 bytes (SHA-256: `7820b1aa74e56986e8eefa02124ca122d69b9da242ff8700b0d602f0f83d2bb5`)
- `scheduler.py`: 18,868 bytes (SHA-256: `8b68849475e9e9626384b9f299f730f0e41c7b6fa7a17bb843c93107b3ebe882`)
- `vmm.py`: 13,048 bytes (SHA-256: `498813c1335413b7a2c84d39147acc5ed0986ed9571e8db4b6904b3b3a96d320`)
- `utils.py` (Worker hooks): 35,340 bytes (SHA-256: `74ed4fccfce3ed048a9cf7fd8269fe1b581f3f6d6511a31c6cf8519d75f5d6cc`)
- `trace_generator.py`: 18,396 bytes (SHA-256: `874dd69cb109ceb0a9b9849d643f2ab45c08713e66a714f4d09d284bbff371e3`)
