# Posterior Concentration Under Weak Priors (Theorem 3.1 direction)

Claim `4d2832c903b2d7d6e55947d20468d734b233f664c678deed688b9c37ae5b8aac`:
*"The theory gives conditions under which high-dimensional measurements make
the Bayesian posterior concentrate near the true signal despite weak priors."*

Executed analog: n = 128 dimensional Gaussian signal, noise std 0.05, exact
weak-prior posterior computed at increasing measurement ratios m/n. All
values copied verbatim from `evidence/evidence.json`:

| m/n | m | Weak-prior recon error | True-prior recon error | Posterior trace (weak) | Cosine sim (weak) | Error ratio weak/true |
|-----|---|------------------------|------------------------|------------------------|-------------------|-----------------------|
| 0.10 | 13  | 10.152101525861097 | 8.46101070069291   | 230.0036464201901  | 0.16862504211049809 | 1.1998686545839845 |
| 0.25 | 32  | 9.682278922846717  | 8.056861617198575  | 192.02602900573902 | 0.25675039714226855 | 1.2017432323993806 |
| 0.50 | 64  | 7.573160364594457  | 6.10834487944644   | 128.16517941493873 | 0.5995280954043982  | 1.2398056288663197 |
| 0.75 | 96  | 5.538800067719793  | 4.273722725994269  | 64.71949145739896  | 0.7983308210628925  | 1.29601296640769   |
| 0.90 | 115 | 3.24115770764266   | 2.92957862556829   | 28.328270144629442 | 0.9336177492208668  | 1.106356279143704  |
| 1.00 | 128 | 2.4379759374029555 | 2.4687490776133014 | 9.970739015897525  | 0.9631196347205059  | 0.9875349258904448 |

Reading: the weak-prior posterior trace collapses monotonically (230.0 to
9.97) as measurements accumulate, cosine similarity to the true signal rises
from 0.169 to 0.963, and by m/n = 1.0 the weak-prior estimator matches the
true-prior estimator (error ratio 0.988 < 1, within trial noise). This is
exactly Theorem 3.1's concentration mechanism, verified with exact posterior
algebra rather than sampling. The theorem's formal constants and the paper's
diffusion-model instantiation are not re-derived here.

Local outcome: **supported (analog scale)**.
