"""Unit tests for rkcnn_probe module."""

import numpy as np
import pytest
import torch

from rkcnn_sae.core.rkcnn_probe import RkCNNProbe, RkCNNResult, mine_directions


class TestRkCNNResult:
    """Tests for RkCNNResult dataclass."""

    def test_get_top_k(self):
        """Test get_top_k method."""
        result = RkCNNResult(
            top_subsets=torch.arange(20).reshape(10, 2),
            top_scores=torch.arange(10, 0, -1).float(),
            all_subsets=torch.arange(40).reshape(20, 2),
            all_scores=torch.randn(20),
        )

        subsets, scores = result.get_top_k(5)

        assert subsets.shape == (5, 2)
        assert scores.shape == (5,)
        assert scores[0] == 10.0  # Highest score


class TestRkCNNProbe:
    """Tests for RkCNNProbe class."""

    def test_init(self):
        """Test probe initialization."""
        probe = RkCNNProbe(
            d_model=100,
            m=10,
            h=50,
            r=5,
            score_method="kurtosis",
            seed=42,
        )

        assert probe.d_model == 100
        assert probe.m == 10
        assert probe.h == 50
        assert probe.r == 5

    def test_probe_returns_result(self):
        """Test that probe returns RkCNNResult."""
        probe = RkCNNProbe(
            d_model=50,
            m=5,
            h=20,
            r=5,
            score_method="kurtosis",
            seed=42,
        )

        activations = torch.randn(100, 50)
        result = probe.probe(activations, show_progress=False)

        assert isinstance(result, RkCNNResult)

    def test_probe_result_shapes(self):
        """Test that probe result has correct shapes."""
        probe = RkCNNProbe(
            d_model=50,
            m=5,
            h=20,
            r=5,
            score_method="kurtosis",
            seed=42,
        )

        activations = torch.randn(100, 50)
        result = probe.probe(activations, show_progress=False)

        assert result.top_subsets.shape == (5, 5)  # (r, m)
        assert result.top_scores.shape == (5,)  # (r,)
        assert result.all_subsets.shape == (20, 5)  # (h, m)
        assert result.all_scores.shape == (20,)  # (h,)

    def test_probe_top_scores_sorted(self):
        """Test that top scores are sorted descending."""
        probe = RkCNNProbe(
            d_model=50,
            m=5,
            h=20,
            r=5,
            score_method="kurtosis",
            seed=42,
        )

        activations = torch.randn(100, 50)
        result = probe.probe(activations, show_progress=False)

        # Check descending order
        for i in range(len(result.top_scores) - 1):
            assert result.top_scores[i] >= result.top_scores[i + 1]

    def test_probe_with_labels(self):
        """Test probe with labels (supervised mode)."""
        probe = RkCNNProbe(
            d_model=50,
            m=5,
            h=20,
            r=5,
            score_method="knn",
            k_neighbors=5,
            seed=42,
        )

        # Create labeled data with some structure
        activations = torch.randn(100, 50)
        labels = torch.randint(0, 3, (100,))

        result = probe.probe(activations, labels=labels, show_progress=False)

        assert isinstance(result, RkCNNResult)
        assert result.top_scores.shape == (5,)

    def test_probe_computes_directions(self):
        """Test that directions are computed when requested."""
        probe = RkCNNProbe(
            d_model=50,
            m=5,
            h=20,
            r=5,
            score_method="kurtosis",
            seed=42,
        )

        activations = torch.randn(100, 50)
        result = probe.probe(activations, compute_directions=True, show_progress=False)

        assert result.top_directions is not None
        assert result.top_directions.shape == (5, 50)  # (r, d_model)

    def test_probe_directions_normalized(self):
        """Test that computed directions are normalized."""
        probe = RkCNNProbe(
            d_model=50,
            m=5,
            h=20,
            r=5,
            score_method="kurtosis",
            seed=42,
        )

        activations = torch.randn(100, 50)
        result = probe.probe(activations, compute_directions=True, show_progress=False)

        # Check normalization (L2 norm ≈ 1)
        norms = torch.norm(result.top_directions, dim=1)
        torch.testing.assert_close(norms, torch.ones(5), atol=1e-5, rtol=1e-5)

    def test_probe_no_directions_when_disabled(self):
        """Test that directions are not computed when disabled."""
        probe = RkCNNProbe(
            d_model=50,
            m=5,
            h=20,
            r=5,
            score_method="kurtosis",
            seed=42,
        )

        activations = torch.randn(100, 50)
        result = probe.probe(activations, compute_directions=False, show_progress=False)

        assert result.top_directions is None

    def test_probe_reproducibility(self):
        """Test that same seed produces same results."""
        activations = torch.randn(100, 50)

        probe1 = RkCNNProbe(d_model=50, m=5, h=20, r=5, seed=42)
        probe2 = RkCNNProbe(d_model=50, m=5, h=20, r=5, seed=42)

        result1 = probe1.probe(activations, show_progress=False)
        result2 = probe2.probe(activations, show_progress=False)

        torch.testing.assert_close(result1.top_scores, result2.top_scores)
        torch.testing.assert_close(result1.all_subsets, result2.all_subsets)

    def test_probe_different_methods(self):
        """Test probe with different scoring methods."""
        activations = torch.randn(100, 50)

        for method in ["knn", "kurtosis", "variance_ratio"]:
            probe = RkCNNProbe(
                d_model=50,
                m=5,
                h=20,
                r=5,
                score_method=method,
                seed=42,
            )

            result = probe.probe(activations, show_progress=False)
            assert result.top_scores.shape == (5,)


class TestMineDirections:
    """Tests for mine_directions convenience function."""

    def test_mine_directions_shape(self):
        """Test that mine_directions returns correct shape."""
        activations = torch.randn(100, 50)

        directions = mine_directions(
            activations=activations,
            d_model=50,
            n_directions=10,
            n_subsets=30,
            seed=42,
        )

        assert directions.shape == (10, 50)

    def test_mine_directions_normalized(self):
        """Test that mined directions are normalized."""
        activations = torch.randn(100, 50)

        directions = mine_directions(
            activations=activations,
            d_model=50,
            n_directions=10,
            n_subsets=30,
            seed=42,
        )

        norms = torch.norm(directions, dim=1)
        torch.testing.assert_close(norms, torch.ones(10), atol=1e-5, rtol=1e-5)

    def test_mine_directions_default_subset_size(self):
        """Test that default subset size is sqrt(d_model)."""
        activations = torch.randn(100, 100)

        # Should use m = sqrt(100) = 10 by default
        directions = mine_directions(
            activations=activations,
            d_model=100,
            n_directions=5,
            n_subsets=20,
            seed=42,
        )

        assert directions.shape == (5, 100)

    def test_mine_directions_with_structured_data(self):
        """Test mining on data with known structure."""
        # Create data with a dominant direction
        n_samples = 200
        d_model = 20

        # Create activations with a clear signal in first 5 dimensions
        activations = torch.randn(n_samples, d_model) * 0.1
        signal = torch.randn(n_samples, 1) * 5
        activations[:, :5] += signal  # Add correlated signal

        directions = mine_directions(
            activations=activations,
            d_model=d_model,
            n_directions=5,
            subset_size=5,
            n_subsets=50,
            score_method="variance_ratio",
            seed=42,
        )

        # Top direction should have significant weight in first 5 dimensions
        top_direction = directions[0]
        weight_in_signal = torch.abs(top_direction[:5]).sum()
        weight_outside = torch.abs(top_direction[5:]).sum()

        # Most weight should be in signal dimensions
        assert weight_in_signal > weight_outside
