"""Unit tests for separation_score module."""

import numpy as np
import pytest
import torch

from rkcnn_sae.core.separation_score import (
    compute_separation_score,
    knn_separation_score,
    kurtosis_score,
    variance_ratio_score,
    max_activation_score,
    batch_compute_scores,
)


class TestKNNSeparationScore:
    """Tests for KNN-based separation score."""

    def test_knn_with_labels_perfect_separation(self):
        """Test KNN score with perfectly separated clusters."""
        # Create two well-separated clusters
        cluster1 = torch.randn(50, 5) + torch.tensor([5.0, 0, 0, 0, 0])
        cluster2 = torch.randn(50, 5) + torch.tensor([-5.0, 0, 0, 0, 0])
        activations = torch.cat([cluster1, cluster2], dim=0)
        labels = torch.cat([torch.zeros(50), torch.ones(50)])

        score = knn_separation_score(activations, k=5, labels=labels)

        # Should be close to 1.0 (perfect separation)
        assert score > 0.9

    def test_knn_with_labels_random_data(self):
        """Test KNN score with random (no structure) data."""
        activations = torch.randn(100, 5)
        labels = torch.randint(0, 2, (100,))

        score = knn_separation_score(activations, k=5, labels=labels)

        # Should be close to 0.5 (random baseline for 2 classes)
        assert 0.3 < score < 0.7

    def test_knn_without_labels(self):
        """Test KNN score without labels (unsupervised mode)."""
        activations = torch.randn(100, 5)
        score = knn_separation_score(activations, k=5, labels=None)

        # Should return some density variance measure
        assert isinstance(score, float)
        assert score >= 0

    def test_knn_too_few_samples(self):
        """Test KNN with fewer samples than k."""
        activations = torch.randn(3, 5)
        score = knn_separation_score(activations, k=5, labels=None)

        assert score == 0.0


class TestKurtosisScore:
    """Tests for kurtosis-based separation score."""

    def test_kurtosis_sparse_data(self):
        """Test kurtosis score on sparse (peaky) data."""
        # Create sparse data (mostly zeros with some large values)
        activations = torch.zeros(1000, 5)
        activations[:50] = torch.randn(50, 5) * 5  # Only 5% non-zero

        score = kurtosis_score(activations)

        # Sparse data should have high kurtosis
        assert score > 3.0  # Excess kurtosis > 0 for leptokurtic

    def test_kurtosis_gaussian_data(self):
        """Test kurtosis score on Gaussian data."""
        activations = torch.randn(1000, 5)
        score = kurtosis_score(activations)

        # Gaussian has excess kurtosis ≈ 0
        assert -1.0 < score < 1.0

    def test_kurtosis_uniform_data(self):
        """Test kurtosis score on uniform data."""
        activations = torch.rand(1000, 5)  # Uniform [0, 1]
        score = kurtosis_score(activations)

        # Uniform has negative excess kurtosis (platykurtic)
        assert score < 0

    def test_kurtosis_too_few_samples(self):
        """Test kurtosis with too few samples."""
        activations = torch.randn(2, 5)  # Need at least 4
        score = kurtosis_score(activations)

        assert score == 0.0


class TestVarianceRatioScore:
    """Tests for variance ratio score."""

    def test_variance_ratio_dominant_direction(self):
        """Test variance ratio when one dimension dominates."""
        activations = torch.randn(100, 5)
        activations[:, 0] *= 10  # Make first dimension have 100x variance
        # With 5 dims: 1 at ~100 variance, 4 at ~1 variance
        # mean_var ≈ (100 + 4) / 5 ≈ 20.8
        # ratio ≈ 100 / 20.8 ≈ 4.8

        score = variance_ratio_score(activations)

        # Ratio should be notably higher than 1 (equal variance case)
        assert score > 3.0

    def test_variance_ratio_equal_variance(self):
        """Test variance ratio with equal variance in all dimensions."""
        activations = torch.randn(100, 5)
        score = variance_ratio_score(activations)

        # With equal variance, ratio should be close to 1
        assert 0.5 < score < 3.0

    def test_variance_ratio_constant_data(self):
        """Test variance ratio with constant data."""
        activations = torch.ones(100, 5)
        score = variance_ratio_score(activations)

        assert score == 0.0


class TestMaxActivationScore:
    """Tests for max activation score."""

    def test_max_activation_basic(self):
        """Test max activation score returns correct max."""
        activations = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        score = max_activation_score(activations)

        assert score == 6.0

    def test_max_activation_negative(self):
        """Test max activation with negative values."""
        activations = torch.tensor([[-10.0, 2.0], [3.0, -5.0]])
        score = max_activation_score(activations)

        # Should return max absolute value
        assert score == 10.0


class TestComputeSeparationScore:
    """Tests for the main compute_separation_score function."""

    def test_dispatch_knn(self):
        """Test that 'knn' method dispatches correctly."""
        activations = torch.randn(100, 5)
        score = compute_separation_score(activations, method="knn", k=5)
        assert isinstance(score, float)

    def test_dispatch_kurtosis(self):
        """Test that 'kurtosis' method dispatches correctly."""
        activations = torch.randn(100, 5)
        score = compute_separation_score(activations, method="kurtosis")
        assert isinstance(score, float)

    def test_dispatch_variance_ratio(self):
        """Test that 'variance_ratio' method dispatches correctly."""
        activations = torch.randn(100, 5)
        score = compute_separation_score(activations, method="variance_ratio")
        assert isinstance(score, float)

    def test_dispatch_max_activation(self):
        """Test that 'max_activation' method dispatches correctly."""
        activations = torch.randn(100, 5)
        score = compute_separation_score(activations, method="max_activation")
        assert isinstance(score, float)

    def test_invalid_method(self):
        """Test that invalid method raises ValueError."""
        activations = torch.randn(100, 5)
        with pytest.raises(ValueError, match="Unknown scoring method"):
            compute_separation_score(activations, method="invalid_method")


class TestBatchComputeScores:
    """Tests for batch_compute_scores function."""

    def test_batch_scores_shape(self):
        """Test that batch scoring returns correct shape."""
        # 10 subsets, 100 samples each, 5 dimensions
        subset_activations = torch.randn(10, 100, 5)

        scores = batch_compute_scores(subset_activations, method="kurtosis")

        assert scores.shape == (10,)

    def test_batch_scores_values(self):
        """Test that batch scores match individual scores."""
        subset_activations = torch.randn(5, 100, 5)

        batch_scores = batch_compute_scores(subset_activations, method="kurtosis")

        # Compute individually and compare
        for i in range(5):
            individual_score = compute_separation_score(
                subset_activations[i], method="kurtosis"
            )
            assert abs(batch_scores[i].item() - individual_score) < 1e-5
