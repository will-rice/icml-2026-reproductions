# ICML 2026 Reproduction: Rotary Position Encodings for Graphs (WIRE)

**Paper:** Rotary Position Encodings for Graphs
**Paper ID:** `trn64znfNx`
**Authors:** Isaac Reid, Arijit Sehanobish, Cederik Höfs, Bruno Mlodozeniec, Leonhard Vulpius, Federico Barbero, Adrian Weller, Krzysztof Choromanski, Richard E Turner, Petar Veličković
**Upstream Revision:** `arxiv:2509.22259v1+github:cederikhoefs/Graph-RoPE@4ac067eb38272543b0cdd7591d630399ff37bce4`
**License:** MIT License
**Space:** `will-rice/paper-wire-graph-rope`

---

## Reproducibility Claims & Computed Evidence

1. **Permutation Equivariance (Lemma 1):** Verified mathematically and empirically that arbitrary node permutations preserve relative WIRE rotary attention matrices.
2. **Grid RoPE Recovery (Theorem 2):** Verified that spectral coordinates on 1D/2D grid graphs correspond to discrete sinusoids, recovering standard RoPE behavior.
3. **Spectral Coordinates & Rotary Attention Integration (Section 3):** Evaluated full `WIREAttention` PyTorch module on synthetic graph topologies with deterministic outputs.

---

## Quickstart

Run the evidence generation pipeline:

```bash
python submissions/rotary_position_encodings_for_graphs/generate_evidence.py
```
