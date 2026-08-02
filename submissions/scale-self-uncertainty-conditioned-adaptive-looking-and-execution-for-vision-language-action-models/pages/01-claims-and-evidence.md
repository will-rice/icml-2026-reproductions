# Claims and Evidence: SCALE for Vision-Language-Action Models

- Attempt ID: `092b120a-9cf1-4351-b4fc-fcbf51f6565f`
- Paper ID: `7MlfE2Da2W`
- Target Claims: 6

## Detailed Claims Audit Table

| Claim ID | Target Claim Text | SHA256 Hash | Status | Detailed Evidence Audit |
| --- | --- | --- | --- | --- |
| Claim 1 | SCALE jointly modulates visual attention temperature and action decoding temperature using the VLA model's self-uncertainty (Figure 2). | `9c25ef590bdbf95cd8dfa64cbaf7ce7093649e4b304868d51d028bf9eedd135d` | Verified | Pinned `prismatic/extern/hf/modeling_prismatic.py` and `scale.yaml` implement self-uncertainty calculation and adaptive temperature scaling. |
| Claim 2 | The method requires no additional training, no verifier, and only a single forward pass per control step (Section 3.3). | `cbc5ac391b72f156b9229ebbd6fd474bcd84acb791c85b3e7d93fc537345c2f7` | Toy | Source-level checks support inference-only execution with no verifier, but this CPU audit does not time a deployed robot control step (toy mechanism checks). |
| Claim 3 | With an OpenVLA backbone on LIBERO, SCALE improves average success rate over greedy decoding and compared sampling, top-k, top-p, and TTS baselines (Table 1). | `e946bc551380587b07c7a39952aef70e9539dd26a46b5df3dc46d45963e348db` | Unavailable | OpenVLA LIBERO benchmark evaluations require GPU simulation environments and pretrained weights. |
| Claim 4 | With a pi0-FAST backbone on LIBERO, SCALE improves average success from 91.2% to 93.0% over the fine-tuned greedy baseline (Table 2). | `87c4598a405ffcb991b98ba21002ffadf3a7264de33d976ac9115ad0fa188b85` | Unavailable | pi0-FAST LIBERO evaluations require GPU hardware and are not present in raw logs. |
| Claim 5 | On SIMPLER-WidowX, SCALE improves average success for pi0-FAST from 34.4% to 49.0% and improves SpatialVLA fine-tuned and zero-shot settings (Table 3). | `24299b74015bef4a7cda614376d95a087770cb7778111a19908e23f2760cbb28` | Unavailable | SIMPLER-WidowX benchmark evaluations require specialized simulation setup. |
| Claim 6 | Real-world pick-and-place experiments show SCALE improves average in-distribution and out-of-distribution success for both OpenVLA and pi0-FAST backbones (Table 5). | `08067e64cb7b51dab32b1853168b874e9914dd2231f273310e2799d8745c81e3` | Unavailable | Real-world robot evaluation requires physical hardware and is marked unavailable. |

## Quantitative Metric Summary

- Total Claims Evaluated: 6
- Verified: 1
- Toy: 1
- Unavailable: 4
- Primary Risk: Standard reproducibility audit for Vision-Language-Action Models
- Evidence Bundle: `evidence/bundle.json`
