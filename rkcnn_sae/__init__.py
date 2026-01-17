"""
RKCNN_SAE: Random k Conditional Nearest Neighbor methods for Sparse Autoencoders.

This package implements RkCNN probing methods for mechanistic interpretability
of neural network representations, with a focus on Sparse Autoencoders (SAEs).
"""

__version__ = "0.1.0"

from rkcnn_sae.core.rkcnn_probe import RkCNNProbe
from rkcnn_sae.core.separation_score import compute_separation_score
from rkcnn_sae.core.subset_sampler import SubsetSampler

__all__ = [
    "RkCNNProbe",
    "SubsetSampler",
    "compute_separation_score",
]
