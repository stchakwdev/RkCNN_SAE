"""
Separation score computation for RkCNN probing.

The separation score measures how well a subset of dimensions
separates the data into meaningful clusters or directions.
Higher scores indicate that the subset captures feature-like structure.
"""

from typing import Literal, Optional

import numpy as np
import torch
from scipy.stats import kurtosis as scipy_kurtosis
from sklearn.neighbors import NearestNeighbors


ScoreMethod = Literal["knn", "kurtosis", "variance_ratio", "max_activation"]


def compute_separation_score(
    activations: torch.Tensor,
    method: ScoreMethod = "knn",
    k: int = 5,
    labels: Optional[torch.Tensor] = None,
) -> float:
    """
    Compute a separation score for activations.

    Parameters
    ----------
    activations : torch.Tensor, shape (n_samples, m)
        Activations restricted to a subset of m dimensions.
    method : ScoreMethod
        Scoring method to use:
        - "knn": K-nearest neighbor based separation (good for clusters)
        - "kurtosis": Excess kurtosis (high for sparse/peaky distributions)
        - "variance_ratio": Ratio of max to mean variance (detects dominant directions)
        - "max_activation": Maximum activation magnitude (simple baseline)
    k : int
        Number of neighbors for KNN-based methods.
    labels : Optional[torch.Tensor]
        Ground truth labels for supervised scoring (only used with "knn").

    Returns
    -------
    score : float
        Separation score (higher = better separation).
    """
    if method == "knn":
        return knn_separation_score(activations, k=k, labels=labels)
    elif method == "kurtosis":
        return kurtosis_score(activations)
    elif method == "variance_ratio":
        return variance_ratio_score(activations)
    elif method == "max_activation":
        return max_activation_score(activations)
    else:
        raise ValueError(f"Unknown scoring method: {method}")


def knn_separation_score(
    activations: torch.Tensor,
    k: int = 5,
    labels: Optional[torch.Tensor] = None,
) -> float:
    """
    KNN-based separation score.

    If labels are provided: measures how often k-nearest neighbors share the same label.
    If no labels: measures local density variation (higher variance = more structure).

    Parameters
    ----------
    activations : torch.Tensor, shape (n_samples, m)
    k : int
        Number of nearest neighbors.
    labels : Optional[torch.Tensor], shape (n_samples,)
        Ground truth labels.

    Returns
    -------
    score : float
        KNN separation score in [0, 1] if labels provided, else density variance.
    """
    acts_np = activations.detach().cpu().numpy().astype(np.float32)

    if acts_np.shape[0] <= k:
        return 0.0

    # Fit KNN
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean")
    nn.fit(acts_np)
    distances, indices = nn.kneighbors(acts_np)

    if labels is not None:
        # Supervised: measure label consistency among neighbors
        labels_np = labels.detach().cpu().numpy()
        same_label_count = 0
        total = 0

        for i in range(len(acts_np)):
            # indices[i, 0] is the point itself, so we skip it
            neighbor_indices = indices[i, 1:]  # k neighbors
            point_label = labels_np[i]

            for j in neighbor_indices:
                if labels_np[j] == point_label:
                    same_label_count += 1
                total += 1

        # Score: fraction of neighbors with same label
        score = same_label_count / total if total > 0 else 0.0
        return score

    else:
        # Unsupervised: measure variance of local densities
        # Higher variance suggests clustered structure
        # distances[:, 1:] excludes self-distance
        mean_neighbor_dist = distances[:, 1:].mean(axis=1)

        # Score: inverse of mean distance times variance
        # High variance + low mean distance = good clusters
        if mean_neighbor_dist.mean() < 1e-8:
            return 0.0

        density_variance = np.var(1.0 / (mean_neighbor_dist + 1e-8))
        return float(density_variance)


def kurtosis_score(activations: torch.Tensor) -> float:
    """
    Kurtosis-based separation score.

    High kurtosis indicates a distribution with heavy tails or a sharp peak,
    which is characteristic of sparse, feature-like activations.

    Parameters
    ----------
    activations : torch.Tensor, shape (n_samples, m)

    Returns
    -------
    score : float
        Mean excess kurtosis across dimensions.
    """
    acts_np = activations.detach().cpu().numpy().astype(np.float32)

    if acts_np.shape[0] < 4:
        return 0.0

    # Compute kurtosis for each dimension, take mean
    # Fisher=True gives excess kurtosis (normal = 0)
    kurt_per_dim = scipy_kurtosis(acts_np, axis=0, fisher=True)

    # Filter out NaN values (constant dimensions)
    kurt_per_dim = kurt_per_dim[~np.isnan(kurt_per_dim)]

    if len(kurt_per_dim) == 0:
        return 0.0

    # Return mean kurtosis (higher = more peaky/sparse)
    return float(np.mean(kurt_per_dim))


def variance_ratio_score(activations: torch.Tensor) -> float:
    """
    Variance ratio score.

    Measures the ratio of maximum variance to mean variance across dimensions.
    High ratio indicates a dominant direction (feature-like).

    Parameters
    ----------
    activations : torch.Tensor, shape (n_samples, m)

    Returns
    -------
    score : float
        Ratio of max to mean variance.
    """
    acts_np = activations.detach().cpu().numpy().astype(np.float32)

    if acts_np.shape[0] < 2:
        return 0.0

    variances = np.var(acts_np, axis=0)
    mean_var = np.mean(variances)

    if mean_var < 1e-10:
        return 0.0

    max_var = np.max(variances)
    return float(max_var / mean_var)


def max_activation_score(activations: torch.Tensor) -> float:
    """
    Maximum activation score (simple baseline).

    Higher max activation suggests stronger signal in this subset.

    Parameters
    ----------
    activations : torch.Tensor, shape (n_samples, m)

    Returns
    -------
    score : float
        Maximum absolute activation value.
    """
    return float(torch.abs(activations).max().item())


def batch_compute_scores(
    subset_activations: torch.Tensor,
    method: ScoreMethod = "knn",
    k: int = 5,
    labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute separation scores for multiple subsets in batch.

    Parameters
    ----------
    subset_activations : torch.Tensor, shape (h, n_samples, m)
        Activations for h subsets.
    method : ScoreMethod
        Scoring method.
    k : int
        Number of neighbors for KNN.
    labels : Optional[torch.Tensor]
        Ground truth labels.

    Returns
    -------
    scores : torch.Tensor, shape (h,)
        Score for each subset.
    """
    h = subset_activations.shape[0]
    scores = torch.zeros(h)

    for i in range(h):
        scores[i] = compute_separation_score(
            subset_activations[i],
            method=method,
            k=k,
            labels=labels,
        )

    return scores
