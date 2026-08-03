# Summary

This Space provides CPU-only executable evidence for selected WeDLM method claims.

- Topological reordering moves observed logical tokens into a physical prefix.
- Strict causal attention lets masked positions attend to the observed prefix without bidirectional attention.
- Streaming decode commits confident left-edge predictions into a growing prefix while keeping a bounded active window.

The vLLM-served speedup claim is not reproduced by this CPU-only evidence bundle.
