# h1 Evidence Summary

This Space reports a CPU-only source inspection and verification for `h1` at GitHub revision `871e89d078202c7d9d18d0924bd76cf161cd6606`.

The evidence bundle verifies the selected mechanism claims:
1. Synthetic long-horizon GSM8K problem chaining without external human or teacher annotations.
2. Outcome-only RL reward functions combined with curriculum horizon-stepping logic.

Full GPU RL fine-tuning, benchmark evaluation suites (GSM-Symbolic, MATH-500, AIME), transfer evaluations, and theoretical sample complexity bounds require large GPU cluster training and are evaluated on static/CPU evidence components.
