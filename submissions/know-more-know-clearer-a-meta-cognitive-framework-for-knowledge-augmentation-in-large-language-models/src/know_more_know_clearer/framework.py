"""Meta-cognitive framework for knowledge-intensive LLM tasks."""

from enum import Enum
import numpy as np


class KnowledgeRegion(Enum):
    MASTERED = "mastered"
    CONFUSED = "confused"
    MISSING = "missing"


class MetaCognitiveFramework:
    def __init__(self, confidence_threshold: float = 0.7, accuracy_threshold: float = 0.8):
        self.confidence_threshold = confidence_threshold
        self.accuracy_threshold = accuracy_threshold

    def partition_knowledge(self, confidence: float, accuracy: float) -> KnowledgeRegion:
        if confidence >= self.confidence_threshold and accuracy >= self.accuracy_threshold:
            return KnowledgeRegion.MASTERED
        elif confidence >= self.confidence_threshold and accuracy < self.accuracy_threshold:
            return KnowledgeRegion.CONFUSED
        else:
            return KnowledgeRegion.MISSING


class CognitionGuidedKnowledgeExpansion:
    def evaluate_expansion_targets(
        self, queries: list[str], confidences: list[float], threshold: float = 0.6
    ) -> list[str]:
        expansion_targets = []
        for query, conf in zip(queries, confidences):
            if conf < threshold:
                expansion_targets.append(query)
        return expansion_targets


class CognitionDrivenKnowledgeCalibration:
    def calibrate_confidence(
        self, confidences: np.ndarray, accuracies: np.ndarray
    ) -> np.ndarray:
        # Isotonic regression or temperature scaling shift to calibrate confidence scores towards accuracy
        mean_diff = np.mean(confidences) - np.mean(accuracies)
        calibrated = confidences - mean_diff * 0.8
        return np.clip(calibrated, 0.0, 1.0)
