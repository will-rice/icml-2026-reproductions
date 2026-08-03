# RBench: CPU-only released-artifact reproduction

**Paper:** `p5QSlnwume`
**Attempt:** `8c21f2dc-a357-422e-9c1b-79a4d417e3dc`

| Question | Recomputed evidence | Status |
| --- | --- | --- |
| Five task domains and four embodiments? | Nine pinned prompt manifests: 5 task categories + 4 embodiment categories, 650 records total. The allowlisted source tracer did not recover complete metric routes. | Partial |
| 25 evaluated models? | Paper-era pin: 25/25 valid and unique. Later pin: 28/28, with 3 prepended models and the original 25 records field-equal. | Verified |
| Three named robotic-video failure modes captured? | All three exact phrases are missing from the pinned allowlisted artifacts; no parser or aggregation route can be attested. | Inconclusive |

## Cohort and consistency audit

- Paper-era ordered-name hash:
  `0997c2cd82bb96e065cb3f1f3606f451859491845bbca7492dce1ade10a8c9aa`
- Later ordered-name hash:
  `ed6242ddf922b260dfb5b96762444107dab0975626f0110b8e45d7f03924fba1`
- Later additions: `LingBot-Video`, `Cosmos3-Nano`, `Cosmos3-Super`.
- One later aggregate discrepancy: `LingBot-Video`, reported `0.620`,
  displayed-field mean `0.614`, absolute error `0.006`.

## Limits

The mean rule is inferred only as a consistency check over rounded committed
fields because neither pinned Space source computes `avg`. Video generation
was not rerun. Human correlation was not reproduced. No claim is made about
semantic video quality, metric validity on real videos, model capability, or
failure-mode prevalence.

Machine-readable provenance, commands, hashes, validation, and limitations are
in `evidence/`.
