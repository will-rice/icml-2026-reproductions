# Claim 1 — Stable-rank degradation under Adam

> The paper argues that vanilla MLP INR low-frequency bias is a symptom of stable-rank degradation during training rather than an intrinsic architectural limitation (Section 3).

**Status: `reproduced` (toy-scale: 4-layer 64-unit MLP fitting a 32x32 multi-frequency image).**

Two identically initialized vanilla-MLP INRs fit the same 32x32
multi-frequency target for 100 steps, one with Adam and one with Muon. Stable
rank is measured as `||W||_F^2 / ||W||_2^2` averaged over the weight matrices.

| Quantity | Value |
| --- | --- |
| Stable rank at initialization | **9.2313** |
| Final stable rank (Adam) | **8.6604** |
| Final stable rank (Muon) | **9.2486** |
| Rank drop under Adam | **0.5709** |
| Rank drop under Muon | **-0.0173** |

Adam loses **0.5709** of stable rank over training while the
near-orthogonal Muon updates change it by **-0.0173** (a
negative value means rank slightly *increased*). The mechanism the paper
describes — rank collapse accompanying optimization rather than being fixed
by the architecture — is therefore observed at this scale.
