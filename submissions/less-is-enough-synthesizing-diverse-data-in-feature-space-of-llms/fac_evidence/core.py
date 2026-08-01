from collections.abc import Iterable


def _feature_set(values: Iterable[int]) -> set[int]:
    return {int(value) for value in values}


def compute_fac(anchor_task_features: Iterable[int], generated_features: Iterable[int]) -> float:
    anchor = _feature_set(anchor_task_features)
    if not anchor:
        raise ValueError("anchor_task_features must be non-empty")
    generated = _feature_set(generated_features)
    return len(anchor & generated) / len(anchor)


def missing_features(anchor_relevant: Iterable[int], seed_relevant: Iterable[int]) -> set[int]:
    return _feature_set(anchor_relevant) - _feature_set(seed_relevant)
