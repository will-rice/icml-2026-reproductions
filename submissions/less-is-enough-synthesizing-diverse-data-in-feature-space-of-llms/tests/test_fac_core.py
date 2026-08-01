from fac_evidence.core import compute_fac, missing_features


def test_fac_coverage_uses_anchor_denominator():
    value = compute_fac(anchor_task_features={1, 2, 3, 4}, generated_features={2, 4})

    assert value == 0.5


def test_missing_feature_selection_excludes_seed_covered_features():
    value = missing_features(anchor_relevant={1, 2, 3}, seed_relevant={2})

    assert value == {1, 3}
