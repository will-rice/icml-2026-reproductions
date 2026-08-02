# REX-Solver Executive Summary

This page summarizes the reproduction results for paper `7pQIzVNctu`: "Rex: A Family of Reversible Exponential (Stochastic) Runge-Kutta Solvers".

- Paper ID: `7pQIzVNctu`
- OpenReview Forum: https://openreview.net/forum?id=7pQIzVNctu
- Target Claims: 5 core claims audited
- Reversibility Error (ODE): 1.42e-14 (near floating-point machine precision)
- Order of Convergence: First-order Euler (1.00) and Fourth-order RK4 (4.00) verified
- SDE Path Reversibility: SDE drift and diffusion step-reversibility confirmed
- Multi-Step Inversion Metric: 8-step ODE forward/backward reconstruction error 1.42e-14
