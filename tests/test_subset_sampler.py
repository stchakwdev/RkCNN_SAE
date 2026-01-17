"""Unit tests for subset_sampler module."""

import numpy as np
import pytest
import torch

from rkcnn_sae.core.subset_sampler import (
    SubsetSampler,
    generate_orthogonal_subsets,
)


class TestSubsetSampler:
    """Tests for SubsetSampler class."""

    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        sampler = SubsetSampler(d_model=100, m=10)
        assert sampler.d_model == 100
        assert sampler.m == 10

    def test_init_invalid_m_greater_than_d(self):
        """Test that m > d_model raises ValueError."""
        with pytest.raises(ValueError, match="cannot exceed"):
            SubsetSampler(d_model=10, m=20)

    def test_init_invalid_m_zero(self):
        """Test that m <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            SubsetSampler(d_model=10, m=0)

    def test_sample_subsets_shape(self):
        """Test that sample_subsets returns correct shape."""
        sampler = SubsetSampler(d_model=100, m=10, seed=42)
        subsets = sampler.sample_subsets(h=50)

        assert subsets.shape == (50, 10)
        assert subsets.dtype == np.int64

    def test_sample_subsets_valid_indices(self):
        """Test that all indices are valid (0 to d_model-1)."""
        sampler = SubsetSampler(d_model=100, m=10, seed=42)
        subsets = sampler.sample_subsets(h=50)

        assert np.all(subsets >= 0)
        assert np.all(subsets < 100)

    def test_sample_subsets_no_duplicates_within_subset(self):
        """Test that each subset has no duplicate indices."""
        sampler = SubsetSampler(d_model=100, m=10, seed=42)
        subsets = sampler.sample_subsets(h=50)

        for i in range(50):
            unique_in_subset = np.unique(subsets[i])
            assert len(unique_in_subset) == 10, f"Subset {i} has duplicates"

    def test_sample_subsets_reproducibility(self):
        """Test that same seed produces same subsets."""
        sampler1 = SubsetSampler(d_model=100, m=10, seed=42)
        sampler2 = SubsetSampler(d_model=100, m=10, seed=42)

        subsets1 = sampler1.sample_subsets(h=20)
        subsets2 = sampler2.sample_subsets(h=20)

        np.testing.assert_array_equal(subsets1, subsets2)

    def test_sample_subsets_different_seeds(self):
        """Test that different seeds produce different subsets."""
        sampler1 = SubsetSampler(d_model=100, m=10, seed=42)
        sampler2 = SubsetSampler(d_model=100, m=10, seed=123)

        subsets1 = sampler1.sample_subsets(h=20)
        subsets2 = sampler2.sample_subsets(h=20)

        # Should not be equal (with very high probability)
        assert not np.array_equal(subsets1, subsets2)

    def test_sample_subsets_torch(self):
        """Test torch tensor output."""
        sampler = SubsetSampler(d_model=100, m=10, seed=42)
        subsets = sampler.sample_subsets_torch(h=50)

        assert isinstance(subsets, torch.Tensor)
        assert subsets.shape == (50, 10)
        assert subsets.dtype == torch.int64

    def test_extract_subset_activations_shape(self):
        """Test activation extraction produces correct shape."""
        sampler = SubsetSampler(d_model=100, m=10, seed=42)

        activations = torch.randn(500, 100)  # 500 samples, 100 dims
        subsets = sampler.sample_subsets_torch(h=20)

        subset_acts = sampler.extract_subset_activations(activations, subsets)

        assert subset_acts.shape == (20, 500, 10)  # (h, n_samples, m)

    def test_extract_subset_activations_values(self):
        """Test that extracted values match original activations."""
        sampler = SubsetSampler(d_model=10, m=3, seed=42)

        activations = torch.arange(50).reshape(5, 10).float()  # 5 samples, 10 dims
        subsets = torch.tensor([[0, 2, 4], [1, 3, 5]])  # 2 subsets

        subset_acts = sampler.extract_subset_activations(activations, subsets)

        # Check first subset extracts columns 0, 2, 4
        expected_0 = activations[:, [0, 2, 4]]
        torch.testing.assert_close(subset_acts[0], expected_0)

        # Check second subset extracts columns 1, 3, 5
        expected_1 = activations[:, [1, 3, 5]]
        torch.testing.assert_close(subset_acts[1], expected_1)


class TestGenerateOrthogonalSubsets:
    """Tests for generate_orthogonal_subsets function."""

    def test_orthogonal_subsets_count(self):
        """Test correct number of subsets generated."""
        subsets, n_subsets = generate_orthogonal_subsets(d_model=100, m=10)

        assert n_subsets == 10  # 100 // 10
        assert len(subsets) == 10

    def test_orthogonal_subsets_size(self):
        """Test each subset has correct size."""
        subsets, _ = generate_orthogonal_subsets(d_model=100, m=10)

        for subset in subsets:
            assert len(subset) == 10

    def test_orthogonal_subsets_no_overlap(self):
        """Test that subsets don't overlap."""
        subsets, _ = generate_orthogonal_subsets(d_model=100, m=10, seed=42)

        all_indices = np.concatenate(subsets)
        unique_indices = np.unique(all_indices)

        # All indices should be unique (no overlap)
        assert len(unique_indices) == len(all_indices)

    def test_orthogonal_subsets_cover_all(self):
        """Test that subsets cover all indices when d_model is divisible by m."""
        subsets, _ = generate_orthogonal_subsets(d_model=100, m=10, seed=42)

        all_indices = np.sort(np.concatenate(subsets))
        expected = np.arange(100)

        np.testing.assert_array_equal(all_indices, expected)

    def test_orthogonal_subsets_reproducibility(self):
        """Test reproducibility with same seed."""
        subsets1, _ = generate_orthogonal_subsets(d_model=100, m=10, seed=42)
        subsets2, _ = generate_orthogonal_subsets(d_model=100, m=10, seed=42)

        for s1, s2 in zip(subsets1, subsets2):
            np.testing.assert_array_equal(s1, s2)
