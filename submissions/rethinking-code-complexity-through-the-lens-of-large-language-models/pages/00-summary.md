# LM-CC Reproduction Summary

This Space presents independently generated evidence for selected claims from
`tI5CFbRhmV`, "Rethinking Code Complexity Through the Lens of Large Language
Models".

The evidence command clones the pinned upstream repository
`xchen121/lm-cc@c38a26afdfc29ee517d734c6b677a4d6c65ec59b`, hashes the public
files it reads, recomputes LM-CC formula checks, recomputes correlation
statistics from released cached outputs, and writes `evidence/bundle.json`.

Selected status summary:

- LM-CC hierarchy/formula claim: `verified`.
- Partial-correlation claim: `toy`, because released cached outputs reproduce
  negative significant raw correlations for all selected task families but
  grouped partial-correlation significance only for program repair.
- Rewrite improvement claim: `toy`, because released simplified subsets show
  LM-CC reductions and mixed pass@1 deltas rather than a full reproduction of
  the headline maximum gain.
