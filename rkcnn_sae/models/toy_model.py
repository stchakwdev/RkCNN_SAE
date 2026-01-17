"""
Toy Model of Superposition for testing RkCNN methods.

This implements a simple model where:
- We have n_features sparse features, each represented by a random direction
- The model compresses these into a smaller d_hidden dimensional space
- This creates superposition: multiple features share the same dimensions

The goal is to test whether RkCNN can recover the original feature directions
from the compressed representations.

Based on: "Toy Models of Superposition" (Elhage et al., 2022)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ToyModelConfig:
    """Configuration for the toy model."""

    n_features: int = 10  # Number of ground truth features
    d_hidden: int = 5  # Hidden dimension (compressed space)
    feature_probability: float = 0.1  # Sparsity: P(feature is active)
    feature_scale: float = 1.0  # Scale of feature activations
    seed: Optional[int] = None


class ToyModel(nn.Module):
    """
    Toy model demonstrating superposition.

    The model maps n_features -> d_hidden -> n_features:
    - Encoder W: (n_features, d_hidden) - compresses features
    - Decoder W.T: (d_hidden, n_features) - reconstructs features

    Since d_hidden < n_features, multiple features must share dimensions
    (superposition).

    Parameters
    ----------
    config : ToyModelConfig
        Model configuration.
    """

    def __init__(self, config: ToyModelConfig):
        super().__init__()
        self.config = config

        # Set random seed
        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)

        # Feature directions: each row is a feature's direction in hidden space
        # Shape: (n_features, d_hidden)
        # These are orthogonal when n_features <= d_hidden, non-orthogonal otherwise
        W = torch.randn(config.n_features, config.d_hidden)
        # Normalize each feature direction
        W = F.normalize(W, dim=1)
        self.W = nn.Parameter(W, requires_grad=False)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encode sparse features into hidden representation.

        Parameters
        ----------
        features : torch.Tensor, shape (batch, n_features)
            Sparse feature activations.

        Returns
        -------
        hidden : torch.Tensor, shape (batch, d_hidden)
            Compressed hidden representation.
        """
        # hidden = features @ W
        # Each active feature adds its direction (scaled) to the hidden state
        return features @ self.W

    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Decode hidden representation back to features.

        Parameters
        ----------
        hidden : torch.Tensor, shape (batch, d_hidden)
            Hidden representation.

        Returns
        -------
        reconstructed : torch.Tensor, shape (batch, n_features)
            Reconstructed feature activations.
        """
        # reconstructed = hidden @ W.T
        return hidden @ self.W.T

    def forward(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode then decode.

        Parameters
        ----------
        features : torch.Tensor, shape (batch, n_features)

        Returns
        -------
        hidden : torch.Tensor, shape (batch, d_hidden)
        reconstructed : torch.Tensor, shape (batch, n_features)
        """
        hidden = self.encode(features)
        reconstructed = self.decode(hidden)
        return hidden, reconstructed

    def get_feature_directions(self) -> torch.Tensor:
        """
        Get the ground truth feature directions in hidden space.

        Returns
        -------
        directions : torch.Tensor, shape (n_features, d_hidden)
            Each row is a feature's direction.
        """
        return self.W.data.clone()


class ToyDataGenerator:
    """
    Generates sparse feature activation data for the toy model.

    Parameters
    ----------
    model : ToyModel
        The toy model.
    batch_size : int
        Number of samples per batch.
    device : str
        Device for tensors.
    """

    def __init__(
        self,
        model: ToyModel,
        batch_size: int = 1024,
        device: str = "cpu",
    ):
        self.model = model
        self.config = model.config
        self.batch_size = batch_size
        self.device = device

    def generate_features(self, n_samples: int) -> torch.Tensor:
        """
        Generate sparse feature activations.

        Each feature is active with probability feature_probability.
        When active, the activation is drawn from a uniform distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.

        Returns
        -------
        features : torch.Tensor, shape (n_samples, n_features)
            Sparse feature activations.
        """
        # Determine which features are active (Bernoulli)
        active = torch.bernoulli(
            torch.full(
                (n_samples, self.config.n_features),
                self.config.feature_probability,
            )
        ).to(self.device)

        # Sample activation magnitudes (uniform [0, 1])
        magnitudes = torch.rand(n_samples, self.config.n_features).to(self.device)

        # Combine: activation = active * magnitude * scale
        features = active * magnitudes * self.config.feature_scale

        return features

    def generate_batch(
        self,
        n_samples: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a batch of data through the toy model.

        Parameters
        ----------
        n_samples : Optional[int]
            Number of samples (default: self.batch_size).

        Returns
        -------
        features : torch.Tensor, shape (n_samples, n_features)
            Ground truth sparse features.
        hidden : torch.Tensor, shape (n_samples, d_hidden)
            Hidden representations (what RkCNN will analyze).
        reconstructed : torch.Tensor, shape (n_samples, n_features)
            Reconstructed features.
        """
        if n_samples is None:
            n_samples = self.batch_size

        features = self.generate_features(n_samples)

        with torch.no_grad():
            hidden, reconstructed = self.model(features)

        return features, hidden, reconstructed

    def generate_labeled_batch(
        self,
        n_samples: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate batch with labels indicating which feature is dominant.

        Parameters
        ----------
        n_samples : Optional[int]

        Returns
        -------
        features : torch.Tensor, shape (n_samples, n_features)
        hidden : torch.Tensor, shape (n_samples, d_hidden)
        labels : torch.Tensor, shape (n_samples,)
            Index of the dominant (most active) feature for each sample.
            -1 if no features are active.
        """
        features, hidden, _ = self.generate_batch(n_samples)

        # Label = index of max feature (or -1 if all zero)
        max_vals, labels = features.max(dim=1)
        labels[max_vals == 0] = -1

        return features, hidden, labels


def compute_feature_reconstruction_accuracy(
    model: ToyModel,
    features: torch.Tensor,
    threshold: float = 0.1,
) -> Tuple[float, float, float]:
    """
    Compute reconstruction accuracy for the toy model.

    Parameters
    ----------
    model : ToyModel
    features : torch.Tensor, shape (batch, n_features)
    threshold : float
        Threshold for considering a feature "active".

    Returns
    -------
    precision : float
        Fraction of reconstructed active features that were actually active.
    recall : float
        Fraction of true active features that were reconstructed.
    mse : float
        Mean squared error of reconstruction.
    """
    with torch.no_grad():
        hidden, reconstructed = model(features)

    # Ground truth active
    true_active = features > threshold
    # Predicted active
    pred_active = reconstructed > threshold

    # True positives, false positives, false negatives
    tp = (true_active & pred_active).sum().float()
    fp = (~true_active & pred_active).sum().float()
    fn = (true_active & ~pred_active).sum().float()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    mse = F.mse_loss(reconstructed, features).item()

    return precision.item(), recall.item(), mse


def analyze_superposition(model: ToyModel) -> dict:
    """
    Analyze the degree of superposition in the toy model.

    Parameters
    ----------
    model : ToyModel

    Returns
    -------
    analysis : dict
        Dictionary with superposition metrics:
        - feature_overlaps: pairwise dot products of feature directions
        - mean_overlap: average absolute overlap
        - max_overlap: maximum overlap
        - capacity_ratio: n_features / d_hidden
    """
    W = model.get_feature_directions()  # (n_features, d_hidden)

    # Pairwise dot products
    overlaps = W @ W.T  # (n_features, n_features)

    # Remove diagonal (self-overlaps are 1.0)
    n_features = overlaps.shape[0]
    mask = ~torch.eye(n_features, dtype=torch.bool)
    off_diagonal = overlaps[mask]

    return {
        "feature_overlaps": overlaps.numpy(),
        "mean_overlap": torch.abs(off_diagonal).mean().item(),
        "max_overlap": torch.abs(off_diagonal).max().item(),
        "capacity_ratio": model.config.n_features / model.config.d_hidden,
        "n_features": model.config.n_features,
        "d_hidden": model.config.d_hidden,
    }
