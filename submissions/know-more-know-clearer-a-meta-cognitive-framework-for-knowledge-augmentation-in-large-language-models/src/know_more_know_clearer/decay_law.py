"""Structural decay law implementation and statistical analysis."""

import numpy as np
from scipy import stats


class StructuralDecayLaw:
    def fit(self, accuracies: np.ndarray, uncertainties: np.ndarray) -> dict:
        accuracies = np.asarray(accuracies)
        uncertainties = np.asarray(uncertainties)
        
        # Calculate Spearman correlation coefficient
        res = stats.spearmanr(accuracies, uncertainties)
        
        # Fit exponential decay curve u = a * exp(-b * acc)
        # Using log linear regression for robust fit
        valid_mask = (uncertainties > 1e-6) & (accuracies > 1e-6)
        acc_valid = accuracies[valid_mask]
        unc_valid = uncertainties[valid_mask]
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(acc_valid, np.log(unc_valid))
        decay_rate = -slope
        
        return {
            "spearman_r": float(res.statistic),
            "p_value": float(res.pvalue),
            "decay_rate": float(decay_rate),
            "intercept": float(np.exp(intercept)),
            "r_squared": float(r_value ** 2),
        }
