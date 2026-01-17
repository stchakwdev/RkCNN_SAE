"""Random subset sampling for RkCNN probing."""

from typing import List, Optional, Tuple

import numpy as np
import torch


class SubsetSampler:
    """
    Generates random subsets of dimensions for RkCNN probing.

    The key idea: sample many random subsets of the activation dimensions,
    then score each subset based on how well it separates the data.
    Top-scoring subsets reveal meaningful structure in the representations.

    Parameters
    ----------
    d_model : int
        Total number of dimensions in the activation space.
    m : int
        Size of each random subset (number of dimensions per subset).
    seed : Optional[int]
        Random seed for reproducibility.
    """

    def __init__(self, d_model: int, m: int, seed: Optional[int] = None):
        self.d_model = d_model
        self.m = m
        self.rng = np.random.default_rng(seed)

        if m > d_model:
            raise ValueError(f"Subset size m={m} cannot exceed d_model={d_model}")
        if m < 1:
            raise ValueError(f"Subset size m={m} must be positive")

    def sample_subsets(self, h: int) -> np.ndarray:
        """
        Generate h random subsets of m dimensions each.

        Parameters
        ----------
        h : int
            Number of subsets to generate.

        Returns
        -------
        subsets : np.ndarray, shape (h, m)
            Array where each row contains the indices of one subset.
        """
        subsets = np.zeros((h, self.m), dtype=np.int64)
        for i in range(h):
            subsets[i] = self.rng.choice(self.d_model, size=self.m, replace=False)
        return subsets

    def sample_subsets_torch(self, h: int, device: str = "cpu") -> torch.Tensor:
        """
        Generate h random subsets as a torch tensor.

        Parameters
        ----------
        h : int
            Number of subsets to generate.
        device : str
            Device to place the tensor on.

        Returns
        -------
        subsets : torch.Tensor, shape (h, m)
            Tensor where each row contains the indices of one subset.
        """
        subsets = self.sample_subsets(h)
        return torch.from_numpy(subsets).to(device)

    def extract_subset_activations(
        self,
        activations: torch.Tensor,
        subsets: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract activations for each subset.

        Parameters
        ----------
        activations : torch.Tensor, shape (n_samples, d_model)
            Full activation matrix.
        subsets : torch.Tensor, shape (h, m)
            Indices of dimensions for each subset.

        Returns
        -------
        subset_acts : torch.Tensor, shape (h, n_samples, m)
            Activations restricted to each subset's dimensions.
        """
        h = subsets.shape[0]
        n_samples = activations.shape[0]
        m = subsets.shape[1]

        # Use advanced indexing to gather subset activations
        # activations: (n_samples, d_model)
        # subsets: (h, m)
        # output: (h, n_samples, m)
        subset_acts = activations[:, subsets]  # (n_samples, h, m)
        subset_acts = subset_acts.permute(1, 0, 2)  # (h, n_samples, m)

        return subset_acts


def generate_orthogonal_subsets(
    d_model: int,
    m: int,
    seed: Optional[int] = None
) -> Tuple[List[np.ndarray], int]:
    """
    Generate non-overlapping (orthogonal) subsets that partition the dimensions.

    Useful for testing: when subsets don't overlap, we can analyze
    which dimensions contribute to separation independently.

    Parameters
    ----------
    d_model : int
        Total number of dimensions.
    m : int
        Size of each subset.
    seed : Optional[int]
        Random seed.

    Returns
    -------
    subsets : List[np.ndarray]
        List of non-overlapping subsets.
    n_subsets : int
        Number of subsets generated (= d_model // m).
    """
    rng = np.random.default_rng(seed)

    # Shuffle all indices
    all_indices = np.arange(d_model)
    rng.shuffle(all_indices)

    # Partition into non-overlapping subsets
    n_subsets = d_model // m
    subsets = []
    for i in range(n_subsets):
        subsets.append(all_indices[i * m : (i + 1) * m])

    return subsets, n_subsets
