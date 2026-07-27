"""Know More, Know Clearer reproduction module."""

from .framework import (
    MetaCognitiveFramework,
    KnowledgeRegion,
    CognitionGuidedKnowledgeExpansion,
    CognitionDrivenKnowledgeCalibration,
)
from .decay_law import StructuralDecayLaw
from .evidence import generate_evidence

__all__ = [
    "MetaCognitiveFramework",
    "KnowledgeRegion",
    "CognitionGuidedKnowledgeExpansion",
    "CognitionDrivenKnowledgeCalibration",
    "StructuralDecayLaw",
    "generate_evidence",
]
