# Reproduction Design: Rotary Position Encodings for Graphs (WIRE)

**Paper ID:** `trn64znfNx`
**Title:** Rotary Position Encodings for Graphs
**Authors:** Isaac Reid, Arijit Sehanobish, Cederik Höfs, Bruno Mlodozeniec, Leonhard Vulpius, Federico Barbero, Adrian Weller, Krzysztof Choromanski, Richard E Turner, Petar Veličković
**Upstream Revision:** `arxiv:2509.22259v1+github:cederikhoefs/Graph-RoPE@4ac067eb38272543b0cdd7591d630399ff37bce4`
**Estimated API Cost:** USD 0.00 (Subscription model)
**Target Claims:**
1. `The WIRE transformation is equivariant to node-order permutations up to sign flips and rotations in degenerate eigenspaces (Lemma 1).`
2. `Regular RoPE is recovered as a special case of WIRE on grid graphs with appropriate learnable frequencies (Theorem 2, Figure 2).`
3. `WIRE applies rotary position encodings to graphs by using Laplacian spectral coordinates to define graph-dependent rotation angles (Figure 1, Section 3).`

---

## 1. System Architecture & Evidence Strategy

The goal of this reproduction is to provide independently executable evidence for the claims in *Rotary Position Encodings for Graphs* (ICML 2026 Spotlight).

### 1.1 Methodology Overview
WIRE (Wave-Induced Rotary Encodings) extends standard Rotary Position Encodings (RoPE) to arbitrary graphs.
For a graph \(G = (V, E)\) with unnormalized or normalized Graph Laplacian \(L = D - A = U \Lambda U^\top\), where \(U = [\mathbf{u}_1, \dots, \mathbf{u}_d]\) are Laplacian eigenvectors:
1. **Spectral Coordinates:** Each node \(v_i\) is assigned coordinates based on Laplacian eigenvectors \(\mathbf{u}_k(i)\).
2. **Rotary Encoding:** For a query vector \(\mathbf{q}_i\) or key vector \(\mathbf{k}_j\), multi-dimensional rotation matrices \(R(\theta_{i, m})\) are applied based on linear combinations of spectral coordinates:
   \[
   \theta_{i, m} = \sum_{k=1}^d \omega_{m, k} u_k(i)
   \]
3. **Inner Product Equivariance & Bias:** The inner product between rotated query \(\mathbf{q}_i\) and key \(\mathbf{k}_j\) depends only on spectral coordinate differences \(\mathbf{u}_k(i) - \mathbf{u}_k(j)\), yielding a translation-equivariant relative positional bias.

---

## 2. Target Claims & Independent Verification Protocol

### Claim 1: Permutation Equivariance (Lemma 1)
- **Statement:** Permuting the node ordering of graph \(G\) by permutation matrix \(P\) permutes the Laplacian eigenvectors \(U \mapsto P U\) (up to eigenvector sign flips \(\pm 1\) and orthogonal rotations within degenerate eigenspaces), preserving relative spectral distances \(\|\mathbf{u}(i) - \mathbf{u}(j)\|_2\) and leaving pairwise attention biases invariant under node reordering.
- **Evidence Test:** `test_wire_permutation_equivariance` constructs synthetic graph topologies (random graphs, cycles, trees), applies random node permutations \(P\), and verifies that WIRE query-key attention scores \(R(\theta_i) \mathbf{q}_i \cdot R(\theta_j) \mathbf{k}_j\) remain invariant (within tolerance \(10^{-5}\)).

### Claim 2: Recovery of 1D/2D RoPE on Grid Graphs (Theorem 2)
- **Statement:** On 1D ring/path graphs and 2D grid graphs, Laplacian eigenvectors correspond to discrete Fourier basis vectors (sine/cosine waves). Choosing standard frequency schedules \(\omega_m = 10000^{-2m/d}\) recovers standard 1D and 2D RoPE rotation matrices.
- **Evidence Test:** `test_wire_rope_grid_recovery` constructs 1D cycle graphs and 2D grid graphs, calculates Laplacian spectral coordinates, and asserts that the resulting WIRE rotation matrices match 1D/2D RoPE within tolerance \(10^{-5}\).

### Claim 3: Spectral Coordinates and Rotary Attention Integration (Figure 1, Section 3)
- **Statement:** WIRE computes graph Laplacian eigenvectors \(L = U \Lambda U^\top\) and applies 2D rotation blocks to query/key vectors, injecting graph structural information into attention layers while maintaining exact deterministic reproducibility.
- **Evidence Test:** `test_wire_spectral_rotary_attention` evaluates the full `WIREAttention` PyTorch module on synthetic graph benchmarks, verifying clean forward passes, gradient flow, and deterministic machine-readable outputs.

---

## 3. Execution & Deployment Plan

- **Project Path:** `submissions/rotary-position-encodings-for-graphs`
- **Worktree:** `.worktrees/wire-graph-rope` (branch `repro/wire-graph-rope`)
- **Hugging Face Space:** `will-rice/paper-wire-graph-rope`
- **Testing:** Pytest with strict TDD (test first, observe failure, implement, observe pass).
- **Validation:** Clean execution of pytest, pre-commit, exact git commit SHA verification on Space.
- **Verdict Handling:** Bounded polling, 1 improvement cycle if needed, final state complete.
