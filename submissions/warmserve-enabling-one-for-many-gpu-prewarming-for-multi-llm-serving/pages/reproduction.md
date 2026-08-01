# WarmServe Reproduction Evidence

Attempt `3d8acf83-97a0-43d3-a2fb-ea3e2b3c2b12` targets paper `DVHpvumD60` from snapshot `98fe583a0a55974d2d28e1beba12e398eb7e8b7f05fadb6d33fdd243b1988644`.

The evidence pins `LLMServe/WarmServe@a60121519e077d2f128b597cbabc947e3e618aaf` and `arxiv:2512.09472v2`. It inspects released source paths and hashes them in `evidence/bundle.json`.

- `claim-1`: verified at source level for scheduler, prewarming manager, CUDA VMM, worker hooks, model config, and trace generator.
- `claim-2`: toy/source-level only; 5-minute trace-character generation is present, but AzureConv prediction error is not recomputed.
- `claim-3`: unavailable without GPU TTFT logs or reruns.
- `claim-4`: unavailable without end-to-end GPU cluster reruns.
- `claim-5`: unavailable without ablation logs or reruns.
- `claim-6`: unavailable without 512-GPU simulation logs or reruns.

No paper-reported table or figure value is presented as a reproduced measurement.
