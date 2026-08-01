# Rex: A Family of Reversible Exponential (Stochastic) Runge-Kutta Solvers

- Attempt: `11b90d4c-61f2-4d93-949e-8d4618aca972`
- Paper: `7pQIzVNctu`
- Upstream: `arxiv:2502.08834+github:zblasingame/Rex-solver@e39b57415d5608b18d7c5631595f1d38f06813b8`
- Results SHA-256: `dcbccca6aff60ef6edc6b60f001f826357f50342991307595ddda33f8b441683`

## Claims

1. `verified` `06ee77e870a2c0447848e1f6159454496f17d144d02bc08fe44441f6b7ad332f`
   A NumPy implementation of the released Rex coupling inverts forward ODE sweeps exactly across couplings and step counts, and inverts a frozen-noise Euler-Maruyama SDE sweep, always to floating-point precision.

2. `verified` `69eedf49ae10686f77613801c126d0825e1a2ea7198e4d9f31c945e00670b8e0`
   Rex-coupled sweeps built from base increments of orders 1, 2, and 4 converge to the fine-step limit at measured rates matching the base order, demonstrating order inheritance; scalar integration independently recovers Euler and RK4 rates, and the RK4 negative-real stability radius is recovered numerically.

3. `verified` `be5532066024dda765f5b69ee4444b86c339c6adc9beeedeb4c995b2e61d0f13`
   An embedded Heun/Euler error estimator drives adaptive step selection inside the Rex coupling; replaying the accepted step sequence backward recovers the initial state to floating-point precision, demonstrating a reversible adaptive solve. The pinned canonical wrapper selects RK4 for fixed-step mode, DOPRI5 for adaptive mode, and rejects adaptive use without embedded error coefficients.

4. `verified` `311e73b22c834fd47107a52f11e90aa005067fa136e9b477108effe648d04cb2`
   A reversible DDIM is recovered numerically: pairing the eps-prediction DDIM base step through the Rex coupling inverts exactly, its single-step x-update reduces algebraically to the plain DDIM affine update, and the paired forward trajectory approaches the plain DDIM trajectory as steps refine.
