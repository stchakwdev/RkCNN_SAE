"""Unit tests for evaluation metrics module."""

import numpy as np
import pytest
import torch

from rkcnn_sae.evaluation.metrics import (
    compute_dead_latent_rate,
    compute_l0_sparsity,
    compute_reconstruction_loss,
    compute_explained_variance,
    evaluate_sae,
    compute_feature_recovery_rate,
    SAEMetrics,
)


class TestComputeDeadLatentRate:
    """Tests for compute_dead_latent_rate function."""

    def test_no_dead_latents(self):
        """Test when all latents are active."""
        # All latents have at least one activation > 0
        latents = torch.rand(100, 50)  # All positive

        rate, n_dead, n_total = compute_dead_latent_rate(latents)

        assert rate == 0.0
        assert n_dead == 0
        assert n_total == 50

    def test_all_dead_latents(self):
        """Test when all latents are dead."""
        latents = torch.zeros(100, 50)

        rate, n_dead, n_total = compute_dead_latent_rate(latents)

        assert rate == 1.0
        assert n_dead == 50
        assert n_total == 50

    def test_partial_dead_latents(self):
        """Test with some dead latents."""
        latents = torch.zeros(100, 50)
        latents[:, :25] = torch.rand(100, 25)  # First 25 are active

        rate, n_dead, n_total = compute_dead_latent_rate(latents)

        assert rate == 0.5  # 25/50
        assert n_dead == 25
        assert n_total == 50

    def test_with_threshold(self):
        """Test with custom threshold."""
        latents = torch.zeros(100, 50)
        latents[:, :25] = 0.01  # Below threshold
        latents[:, 25:35] = 0.1  # Above threshold

        rate, n_dead, _ = compute_dead_latent_rate(latents, threshold=0.05)

        assert n_dead == 40  # 25 + 15 that never exceed threshold


class TestComputeL0Sparsity:
    """Tests for compute_l0_sparsity function."""

    def test_fully_active(self):
        """Test when all latents are active."""
        latents = torch.ones(100, 50)

        l0 = compute_l0_sparsity(latents)

        assert l0 == 50.0

    def test_fully_sparse(self):
        """Test when all latents are zero."""
        latents = torch.zeros(100, 50)

        l0 = compute_l0_sparsity(latents)

        assert l0 == 0.0

    def test_partial_sparsity(self):
        """Test with partial sparsity."""
        latents = torch.zeros(100, 50)
        latents[:, :10] = 1.0  # 10 active per sample

        l0 = compute_l0_sparsity(latents)

        assert l0 == 10.0

    def test_with_threshold(self):
        """Test with custom threshold."""
        latents = torch.zeros(100, 50)
        latents[:, :10] = 0.01  # Below threshold
        latents[:, 10:20] = 0.1  # Above threshold

        l0 = compute_l0_sparsity(latents, threshold=0.05)

        assert l0 == 10.0  # Only 10 above threshold


class TestComputeReconstructionLoss:
    """Tests for compute_reconstruction_loss function."""

    def test_perfect_reconstruction(self):
        """Test with perfect reconstruction."""
        original = torch.randn(100, 50)
        reconstructed = original.clone()

        loss = compute_reconstruction_loss(original, reconstructed)

        assert loss == 0.0

    def test_nonzero_loss(self):
        """Test with imperfect reconstruction."""
        original = torch.randn(100, 50)
        reconstructed = original + torch.randn_like(original) * 0.1

        loss = compute_reconstruction_loss(original, reconstructed)

        assert loss > 0
        assert loss < 0.02  # Should be small given noise scale


class TestComputeExplainedVariance:
    """Tests for compute_explained_variance function."""

    def test_perfect_reconstruction(self):
        """Test with perfect reconstruction."""
        original = torch.randn(100, 50)
        reconstructed = original.clone()

        ev = compute_explained_variance(original, reconstructed)

        assert abs(ev - 1.0) < 1e-5

    def test_zero_reconstruction(self):
        """Test with zero reconstruction."""
        original = torch.randn(100, 50)
        reconstructed = torch.zeros_like(original)

        ev = compute_explained_variance(original, reconstructed)

        # EV should be low (possibly negative)
        assert ev < 0.5

    def test_partial_reconstruction(self):
        """Test with partial reconstruction."""
        original = torch.randn(100, 50)
        # Reconstruct with some noise
        reconstructed = original + torch.randn_like(original) * 0.3

        ev = compute_explained_variance(original, reconstructed)

        # Should be between 0 and 1
        assert 0 < ev < 1


class TestEvaluateSAE:
    """Tests for evaluate_sae function."""

    def test_returns_sae_metrics(self):
        """Test that function returns SAEMetrics."""
        original = torch.randn(100, 50)
        latents = torch.rand(100, 200)
        reconstructed = original + torch.randn_like(original) * 0.1

        metrics = evaluate_sae(original, latents, reconstructed)

        assert isinstance(metrics, SAEMetrics)

    def test_metrics_fields(self):
        """Test that all metrics fields are populated."""
        original = torch.randn(100, 50)
        latents = torch.rand(100, 200)
        reconstructed = original + torch.randn_like(original) * 0.1

        metrics = evaluate_sae(original, latents, reconstructed)

        assert hasattr(metrics, "dead_latent_rate")
        assert hasattr(metrics, "l0_sparsity")
        assert hasattr(metrics, "reconstruction_loss")
        assert hasattr(metrics, "explained_variance")
        assert hasattr(metrics, "n_dead_latents")
        assert hasattr(metrics, "n_total_latents")

    def test_metrics_consistency(self):
        """Test that metrics are internally consistent."""
        original = torch.randn(100, 50)
        latents = torch.zeros(100, 200)
        latents[:, :50] = torch.rand(100, 50)  # 50 active, 150 dead
        reconstructed = original.clone()

        metrics = evaluate_sae(original, latents, reconstructed)

        assert metrics.n_dead_latents == 150
        assert metrics.n_total_latents == 200
        assert metrics.dead_latent_rate == 0.75


class TestComputeFeatureRecoveryRate:
    """Tests for compute_feature_recovery_rate function."""

    def test_perfect_recovery(self):
        """Test with identical directions."""
        true_directions = torch.randn(10, 50)
        true_directions = true_directions / true_directions.norm(dim=1, keepdim=True)

        mined_directions = true_directions.clone()

        rate, similarity = compute_feature_recovery_rate(
            mined_directions, true_directions, similarity_threshold=0.9
        )

        assert rate == 1.0

    def test_no_recovery(self):
        """Test with orthogonal directions."""
        # Create orthogonal sets
        true_directions = torch.eye(10, 50)  # 10 orthogonal directions
        mined_directions = torch.eye(10, 50)
        # Shift mined to be in different dimensions
        mined_directions = torch.roll(mined_directions, shifts=10, dims=1)

        rate, similarity = compute_feature_recovery_rate(
            mined_directions, true_directions, similarity_threshold=0.5
        )

        # Should be 0 since they're orthogonal
        assert rate == 0.0

    def test_partial_recovery(self):
        """Test with partial recovery."""
        true_directions = torch.randn(10, 50)
        true_directions = true_directions / true_directions.norm(dim=1, keepdim=True)

        # Mine only half the directions correctly
        mined_directions = torch.randn(10, 50)
        mined_directions[:5] = true_directions[:5]  # Match first 5
        mined_directions = mined_directions / mined_directions.norm(dim=1, keepdim=True)

        rate, similarity = compute_feature_recovery_rate(
            mined_directions, true_directions, similarity_threshold=0.9
        )

        # Should recover at least 5/10 = 50%
        assert rate >= 0.5

    def test_similarity_matrix_shape(self):
        """Test that similarity matrix has correct shape."""
        true_directions = torch.randn(10, 50)
        mined_directions = torch.randn(15, 50)

        _, similarity = compute_feature_recovery_rate(
            mined_directions, true_directions
        )

        assert similarity.shape == (15, 10)  # (n_mined, n_true)


class TestSAEMetrics:
    """Tests for SAEMetrics dataclass."""

    def test_creation(self):
        """Test dataclass creation."""
        metrics = SAEMetrics(
            dead_latent_rate=0.3,
            l0_sparsity=50.0,
            reconstruction_loss=0.01,
            explained_variance=0.95,
            n_dead_latents=300,
            n_total_latents=1000,
        )

        assert metrics.dead_latent_rate == 0.3
        assert metrics.l0_sparsity == 50.0
        assert metrics.reconstruction_loss == 0.01
        assert metrics.explained_variance == 0.95
        assert metrics.n_dead_latents == 300
        assert metrics.n_total_latents == 1000
