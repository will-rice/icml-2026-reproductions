from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rare_event_llm.process import SequenceRecord
from rare_event_llm.samplers import BiasedSample


@dataclass(frozen=True)
class HistogramComparison:
    bins: tuple[float, ...]
    exact: tuple[float, ...]
    reconstructed: tuple[float, ...]
    l1_error: float
    max_bin_error: float

    def as_dict(self) -> dict:
        return {
            "bins": list(self.bins),
            "exact": list(self.exact),
            "reconstructed": list(self.reconstructed),
            "l1_error": self.l1_error,
            "max_bin_error": self.max_bin_error,
        }


def exact_histogram(records: list[SequenceRecord], bins: np.ndarray) -> np.ndarray:
    probabilities = np.array([np.exp(record.log_probability) for record in records])
    observables = np.array([record.observable for record in records])
    histogram, _ = np.histogram(observables, bins=bins, weights=probabilities)
    return histogram / histogram.sum()


def reconstruct_unbiased_histogram(
    all_records: list[SequenceRecord],
    samples: list[BiasedSample],
    bins: np.ndarray,
) -> HistogramComparison:
    flat_records = [record for sample in samples for record in sample.records]
    betas = np.array([sample.beta for sample in samples], dtype=float)
    counts = np.array([len(sample.records) for sample in samples], dtype=float)
    log_partitions = np.array([sample.log_partition for sample in samples], dtype=float)

    weights = []
    observables = []
    for record in flat_records:
        denominator = np.sum(
            counts * np.exp(betas * record.observable - log_partitions)
        )
        weights.append(1.0 / denominator)
        observables.append(record.observable)

    reconstructed, _ = np.histogram(observables, bins=bins, weights=np.array(weights))
    reconstructed = reconstructed / reconstructed.sum()
    exact = exact_histogram(all_records, bins)
    delta = np.abs(reconstructed - exact)
    return HistogramComparison(
        bins=tuple(float(value) for value in bins),
        exact=tuple(float(value) for value in exact),
        reconstructed=tuple(float(value) for value in reconstructed),
        l1_error=float(delta.sum()),
        max_bin_error=float(delta.max()),
    )
