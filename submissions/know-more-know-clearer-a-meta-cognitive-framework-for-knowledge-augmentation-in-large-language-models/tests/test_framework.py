import pytest
import numpy as np
from know_more_know_clearer.framework import (
    MetaCognitiveFramework,
    KnowledgeRegion,
    CognitionGuidedKnowledgeExpansion,
    CognitionDrivenKnowledgeCalibration,
)

def test_knowledge_region_partitioning():
    framework = MetaCognitiveFramework(confidence_threshold=0.7, accuracy_threshold=0.8)

    # Test Mastered
    region = framework.partition_knowledge(confidence=0.9, accuracy=0.95)
    assert region == KnowledgeRegion.MASTERED

    # Test Confused
    region = framework.partition_knowledge(confidence=0.85, accuracy=0.4)
    assert region == KnowledgeRegion.CONFUSED

    # Test Missing
    region = framework.partition_knowledge(confidence=0.2, accuracy=0.3)
    assert region == KnowledgeRegion.MISSING

def test_cognition_guided_knowledge_expansion():
    cgke = CognitionGuidedKnowledgeExpansion()
    queries = ["What is the capital of France?", "How does quantum computing work?"]
    confidences = [0.95, 0.3]

    expansion_needed = cgke.evaluate_expansion_targets(queries, confidences, threshold=0.6)
    assert len(expansion_needed) == 1
    assert expansion_needed[0] == "How does quantum computing work?"

def test_cognition_driven_knowledge_calibration():
    cdkc = CognitionDrivenKnowledgeCalibration()
    confidences = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    accuracies = np.array([1.0, 0.8, 0.6, 0.4, 0.2])

    calibrated = cdkc.calibrate_confidence(confidences, accuracies)
    assert len(calibrated) == len(confidences)
    # Calibrated scores should align better with accuracies
    ece_before = np.mean(np.abs(confidences - accuracies))
    ece_after = np.mean(np.abs(calibrated - accuracies))
    assert ece_after <= ece_before
