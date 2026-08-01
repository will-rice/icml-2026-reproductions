# Stream RAG Evidence Report

This report covers paper `NMMmwSbzRx`, `Stream RAG: Instant and Accurate Spoken
Dialogue Systems with Streaming Tool Usage`.

The primary artifact is `arxiv:2510.02044v1`; the downloaded TeX source archive
has SHA256 `e40c9783bb9c6f9b5995d08fac509cce3d03102ea9e48d35b94a47d4a96725a7`.
No official training/evaluation repository or AudioCRAG-Human release was found.

The evidence bundle audits the released source for the streaming RAG formulation,
AudioCRAG construction counts, main accuracy/latency tables, and negative
sampling ablation. It also runs deterministic toy checks for parallel fixed
interval tool calls, model-triggered single-thread tool calls, and negative
sampling recovery labels.

The benchmark accuracy, human-audio dataset, and latency claims are not
reproduced measurements. They are marked inconclusive unless the evidence only
concerns a deterministic mechanism specified by the paper.
