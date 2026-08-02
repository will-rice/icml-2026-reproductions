from hive_repro.evidence import reconstruct_tiny_mix


def test_tiny_mixture_reconstruction_applies_weights_and_normalization():
    sources = [
        {"samples": [1.0, 0.0, -1.0], "applied_weight": 0.5},
        {"samples": [0.0, 1.0, 1.0], "applied_weight": 0.25},
    ]

    result = reconstruct_tiny_mix(
        sources,
        global_normalization_factor=2.0,
    )

    assert result["mix"] == [1.0, 0.5, -0.5]
    assert result["source_count"] == 2
