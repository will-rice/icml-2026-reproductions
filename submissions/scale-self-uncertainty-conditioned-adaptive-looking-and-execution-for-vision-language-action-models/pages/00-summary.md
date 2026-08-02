# Summary: SCALE for Vision-Language-Action Models

- Attempt ID: `092b120a-9cf1-4351-b4fc-fcbf51f6565f`
- Paper ID: `7MlfE2Da2W`
- Slug: `scale-self-uncertainty-conditioned-adaptive-looking-and-execution-for-vision-language-action-models`
- Title: SCALE: Self-uncertainty Conditioned Adaptive Looking and Execution for Vision-Language-Action Models

## Executive Overview

SCALE jointly modulates visual attention temperature and action decoding temperature using the VLA model's self-uncertainty during decoding.

- **Mechanism evidence**: Source-level analysis confirms implementation of self-uncertainty, adaptive action temperature, adaptive visual attention temperature, and the SCALE decoding mode. Pinned repo `snumprlab/scale` commit `b4ad2a69d14f91712704711e810cf9830e2b7121`.
- **Training-free & Verifier-free**: Source-level checks support inference-only execution with no verifier, marked toy mechanism checks because this CPU audit does not time a deployed robot control step.
- **Unreplicated benchmark claims**: LIBERO, SIMPLER-WidowX, and real-world robot success-rate claims require GPU simulation, pretrained VLA checkpoints, or physical robot hardware and are marked unavailable.

## Claims Summary Audit

| Claim ID | Target Claim Text | Claim SHA256 Hash | Audit Status | Evidence Note |
| --- | --- | --- | --- | --- |
| Claim 1 | SCALE jointly modulates visual attention temperature and action decoding temperature using the VLA model's self-uncertainty (Figure 2). | `9c25ef590bdbf95cd8dfa64cbaf7ce7093649e4b304868d51d028bf9eedd135d` | Verified | Pinned `modeling_prismatic.py` and `scale.yaml` |
| Claim 2 | The method requires no additional training, no verifier, and only a single forward pass per control step (Section 3.3). | `cbc5ac391b72f156b9229ebbd6fd474bcd84acb791c85b3e7d93fc537345c2f7` | Toy | Source-level checks support inference-only execution with no verifier (toy mechanism checks) |
| Claim 3 | With an OpenVLA backbone on LIBERO, SCALE improves average success rate over greedy decoding and compared sampling, top-k, top-p, and TTS baselines (Table 1). | `e946bc551380587b07c7a39952aef70e9539dd26a46b5df3dc46d45963e348db` | Unavailable | Benchmark success-rate comparisons require non-CPU hardware |
| Claim 4 | With a pi0-FAST backbone on LIBERO, SCALE improves average success from 91.2% to 93.0% over the fine-tuned greedy baseline (Table 2). | `87c4598a405ffcb991b98ba21002ffadf3a7264de33d976ac9115ad0fa188b85` | Unavailable | pi0-FAST/LIBERO average success rates require GPU evaluation |
| Claim 5 | On SIMPLER-WidowX, SCALE improves average success for pi0-FAST from 34.4% to 49.0% and improves SpatialVLA fine-tuned and zero-shot settings (Table 3). | `24299b74015bef4a7cda614376d95a087770cb7778111a19908e23f2760cbb28` | Unavailable | SIMPLER-WidowX claims require simulator setup |
| Claim 6 | Real-world pick-and-place experiments show SCALE improves average in-distribution and out-of-distribution success for both OpenVLA and pi0-FAST backbones (Table 5). | `08067e64cb7b51dab32b1853168b874e9914dd2231f273310e2799d8745c81e3` | Unavailable | Real-world success rates require physical robot hardware |
