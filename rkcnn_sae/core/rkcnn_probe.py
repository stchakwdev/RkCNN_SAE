"""
RkCNN Probing: Random k Conditional Nearest Neighbor probing for feature discovery.

The RkCNN algorithm:
1. Sample h random subsets of m dimensions each
2. For each subset, compute activations restricted to those dimensions
3. Score each subset based on how well it separates the data
4. Keep the top-r subsets with highest scores
5. Use these top subsets to initialize SAE directions or analyze features

This is a method for unsupervised feature discovery in neural network activations.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from rkcnn_sae.core.separation_score import ScoreMethod, batch_compute_scores
from rkcnn_sae.core.subset_sampler import SubsetSampler


@dataclass
class RkCNNResult:
    """Results from RkCNN probing."""

    # Top-r subsets with highest scores
    top_subsets: torch.Tensor  # (r, m)
    top_scores: torch.Tensor  # (r,)

    # All subsets and scores (for analysis)
    all_subsets: torch.Tensor  # (h, m)
    all_scores: torch.Tensor  # (h,)

    # Derived directions (normalized mean activation for each top subset)
    top_directions: Optional[torch.Tensor] = None  # (r, d_model)

    def get_top_k(self, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the top-k subsets and their scores."""
        return self.top_subsets[:k], self.top_scores[:k]


class RkCNNProbe:
    """
    Random k Conditional Nearest Neighbor probing for feature discovery.

    Parameters
    ----------
    d_model : int
        Dimensionality of the activation space.
    m : int
        Size of each random subset.
    h : int
        Total number of random subsets to sample.
    r : int
        Number of top subsets to keep.
    score_method : ScoreMethod
        Method for computing separation scores.
    k_neighbors : int
        Number of neighbors for KNN-based scoring.
    seed : Optional[int]
        Random seed for reproducibility.
    device : str
        Device for computation.
    """

    def __init__(
        self,
        d_model: int,
        m: int,
        h: int = 100,
        r: int = 10,
        score_method: ScoreMethod = "knn",
        k_neighbors: int = 5,
        seed: Optional[int] = None,
        device: str = "cpu",
    ):
        self.d_model = d_model
        self.m = m
        self.h = h
        self.r = r
        self.score_method = score_method
        self.k_neighbors = k_neighbors
        self.device = device

        self.sampler = SubsetSampler(d_model, m, seed=seed)

    def probe(
        self,
        activations: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        compute_directions: bool = True,
        show_progress: bool = True,
    ) -> RkCNNResult:
        """
        Run RkCNN probing on activations.

        Parameters
        ----------
        activations : torch.Tensor, shape (n_samples, d_model)
            Activation matrix to probe.
        labels : Optional[torch.Tensor], shape (n_samples,)
            Ground truth labels (for supervised scoring).
        compute_directions : bool
            Whether to compute direction vectors for top subsets.
        show_progress : bool
            Whether to show progress bar.

        Returns
        -------
        result : RkCNNResult
            Probing results including top subsets and scores.
        """
        activations = activations.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)

        # Step 1: Sample random subsets
        subsets = self.sampler.sample_subsets_torch(self.h, device=self.device)

        # Step 2: Extract activations for each subset
        subset_acts = self.sampler.extract_subset_activations(activations, subsets)

        # Step 3: Score each subset
        scores = self._score_subsets(
            subset_acts, labels=labels, show_progress=show_progress
        )

        # Step 4: Select top-r subsets
        top_indices = torch.argsort(scores, descending=True)[: self.r]
        top_subsets = subsets[top_indices]
        top_scores = scores[top_indices]

        # Step 5: Optionally compute direction vectors
        top_directions = None
        if compute_directions:
            top_directions = self._compute_directions(
                activations, top_subsets, subset_acts[top_indices]
            )

        return RkCNNResult(
            top_subsets=top_subsets,
            top_scores=top_scores,
            all_subsets=subsets,
            all_scores=scores,
            top_directions=top_directions,
        )

    def _score_subsets(
        self,
        subset_activations: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Score all subsets."""
        h = subset_activations.shape[0]
        scores = torch.zeros(h, device=self.device)

        iterator = range(h)
        if show_progress:
            iterator = tqdm(iterator, desc="Scoring subsets")

        for i in iterator:
            scores[i] = self._score_single_subset(
                subset_activations[i], labels=labels
            )

        return scores

    def _score_single_subset(
        self,
        acts: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> float:
        """Score a single subset."""
        from rkcnn_sae.core.separation_score import compute_separation_score

        return compute_separation_score(
            acts,
            method=self.score_method,
            k=self.k_neighbors,
            labels=labels,
        )

    def _compute_directions(
        self,
        full_activations: torch.Tensor,
        top_subsets: torch.Tensor,
        top_subset_acts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute direction vectors for top subsets.

        For each top subset, we create a direction in the full d_model space
        by taking the principal direction within the subset dimensions.

        Parameters
        ----------
        full_activations : torch.Tensor, shape (n_samples, d_model)
        top_subsets : torch.Tensor, shape (r, m)
        top_subset_acts : torch.Tensor, shape (r, n_samples, m)

        Returns
        -------
        directions : torch.Tensor, shape (r, d_model)
            Normalized direction vectors in full activation space.
        """
        r = top_subsets.shape[0]
        directions = torch.zeros(r, self.d_model, device=self.device)

        for i in range(r):
            subset_indices = top_subsets[i]
            subset_acts = top_subset_acts[i]  # (n_samples, m)

            # Compute principal direction via SVD
            # Center the data
            centered = subset_acts - subset_acts.mean(dim=0, keepdim=True)

            try:
                # SVD to get principal direction
                U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
                principal_in_subset = Vh[0]  # First right singular vector (m,)

                # Embed into full space
                directions[i, subset_indices] = principal_in_subset
            except Exception:
                # Fallback: use mean activation direction
                mean_act = subset_acts.mean(dim=0)
                norm = torch.norm(mean_act)
                if norm > 1e-8:
                    directions[i, subset_indices] = mean_act / norm

        # Normalize full directions
        norms = torch.norm(directions, dim=1, keepdim=True)
        norms = torch.clamp(norms, min=1e-8)
        directions = directions / norms

        return directions

    def analyze_correlation_with_features(
        self,
        result: RkCNNResult,
        feature_activations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Analyze how top subsets correlate with known features.

        Parameters
        ----------
        result : RkCNNResult
            Output from probe().
        feature_activations : torch.Tensor, shape (n_samples, n_features)
            Known feature activations (e.g., from toy model ground truth).

        Returns
        -------
        correlations : torch.Tensor, shape (r, n_features)
            Correlation between each top subset's direction and each feature.
        """
        if result.top_directions is None:
            raise ValueError("Must call probe() with compute_directions=True")

        # Compute correlations
        # top_directions: (r, d_model)
        # feature_activations: (n_samples, n_features)

        # We need to project activations onto directions and correlate with features
        # This assumes we have the original activations...

        # For now, return a simple placeholder
        # In practice, this would involve computing:
        # projection_onto_direction = activations @ direction
        # correlation = corr(projection, feature_activation)

        n_features = feature_activations.shape[1]
        r = result.top_subsets.shape[0]
        return torch.zeros(r, n_features)


def mine_directions(
    activations: torch.Tensor,
    d_model: int,
    n_directions: int = 100,
    subset_size: Optional[int] = None,
    n_subsets: int = 1000,
    score_method: ScoreMethod = "kurtosis",
    device: str = "cpu",
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Convenience function to mine direction vectors from activations.

    Parameters
    ----------
    activations : torch.Tensor, shape (n_samples, d_model)
        Activation data.
    d_model : int
        Model dimension.
    n_directions : int
        Number of directions to mine.
    subset_size : Optional[int]
        Size of random subsets (default: sqrt(d_model)).
    n_subsets : int
        Number of random subsets to try.
    score_method : ScoreMethod
        Scoring method for subsets.
    device : str
        Computation device.
    seed : Optional[int]
        Random seed.

    Returns
    -------
    directions : torch.Tensor, shape (n_directions, d_model)
        Mined direction vectors.
    """
    if subset_size is None:
        subset_size = max(2, int(np.sqrt(d_model)))

    probe = RkCNNProbe(
        d_model=d_model,
        m=subset_size,
        h=n_subsets,
        r=n_directions,
        score_method=score_method,
        device=device,
        seed=seed,
    )

    result = probe.probe(activations, compute_directions=True, show_progress=True)
    return result.top_directions
