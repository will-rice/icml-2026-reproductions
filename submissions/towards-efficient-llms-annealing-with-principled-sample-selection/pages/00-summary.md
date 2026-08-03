# DiReCT Evidence Summary

This reproduction targets the CPU-checkable mechanism claims for **Towards Efficient LLMs Annealing with Principled Sample Selection**. It does not rerun large language model annealing. Instead, it isolates the two assessed claims into deterministic numerical checks that can be recomputed by the Space and by controller validation.

The flat-direction evidence constructs a diagonal Hessian with one sufficiently flat eigendirection and one stiff eigendirection. With equal gradient alignment and step size, the flatter direction has a smaller curvature penalty and a larger surrogate-objective increase, matching the qualitative flat-direction preference claim.

The sample-priority evidence builds a synthetic candidate pool with explicit losses and sequence lengths. A DiReCT-style score places the high-loss long-sequence sample in the selected set, while a short-length probing regime changes the top-ranked sample. This supports the claimed priority behavior without claiming to reproduce full LLM-scale training results.
