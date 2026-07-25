import pytest
import numpy as np
from know_more_know_clearer.decay_law import StructuralDecayLaw

def test_structural_decay_law_fitting():
    decay_model = StructuralDecayLaw()
    
    # Generate synthetic decay data: higher accuracy -> lower uncertainty
    accuracies = np.linspace(0.1, 0.99, 20)
    uncertainties = 1.0 - np.sqrt(accuracies) + np.random.normal(0, 0.01, 20)
    uncertainties = np.clip(uncertainties, 0.0, 1.0)
    
    fit_results = decay_model.fit(accuracies, uncertainties)
    assert "spearman_r" in fit_results
    assert fit_results["spearman_r"] < 0  # Negative correlation: higher accuracy -> lower uncertainty
    assert fit_results["decay_rate"] > 0
