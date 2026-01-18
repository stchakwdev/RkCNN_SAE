"""
Interpretability analysis tools for Sparse Autoencoders.

This module provides tools to analyze and compare the interpretability
of baseline SAE vs RkCNN-initialized SAE latents, with a focus on
"revived" latents (dead in baseline, alive in RkCNN).
"""

from rkcnn_sae.interpretability.activation_store import (
    TokenRecord,
    TokenAwareActivationStore,
)
from rkcnn_sae.interpretability.top_activations import (
    TopActivation,
    TopActivationFinder,
)
from rkcnn_sae.interpretability.revived_detector import (
    RevivedLatentInfo,
    RevivedLatentDetector,
)
from rkcnn_sae.interpretability.metrics import (
    LatentInterpretabilityScore,
    InterpretabilityMetrics,
)

__all__ = [
    "TokenRecord",
    "TokenAwareActivationStore",
    "TopActivation",
    "TopActivationFinder",
    "RevivedLatentInfo",
    "RevivedLatentDetector",
    "LatentInterpretabilityScore",
    "InterpretabilityMetrics",
]
