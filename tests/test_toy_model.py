"""Unit tests for toy_model module."""

import numpy as np
import pytest
import torch

from rkcnn_sae.models.toy_model import (
    ToyModel,
    ToyModelConfig,
    ToyDataGenerator,
    analyze_superposition,
    compute_feature_reconstruction_accuracy,
)


class TestToyModelConfig:
    """Tests for ToyModelConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ToyModelConfig()

        assert config.n_features == 10
        assert config.d_hidden == 5
        assert config.feature_probability == 0.1
        assert config.feature_scale == 1.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ToyModelConfig(
            n_features=20,
            d_hidden=10,
            feature_probability=0.2,
        )

        assert config.n_features == 20
        assert config.d_hidden == 10
        assert config.feature_probability == 0.2


class TestToyModel:
    """Tests for ToyModel class."""

    def test_init(self):
        """Test model initialization."""
        config = ToyModelConfig(n_features=10, d_hidden=5)
        model = ToyModel(config)

        assert model.config == config
        assert model.W.shape == (10, 5)

    def test_feature_directions_normalized(self):
        """Test that feature directions are normalized."""
        config = ToyModelConfig(n_features=10, d_hidden=5)
        model = ToyModel(config)

        norms = torch.norm(model.W, dim=1)
        torch.testing.assert_close(norms, torch.ones(10), atol=1e-5, rtol=1e-5)

    def test_encode_shape(self):
        """Test encode produces correct shape."""
        config = ToyModelConfig(n_features=10, d_hidden=5)
        model = ToyModel(config)

        features = torch.randn(32, 10)
        hidden = model.encode(features)

        assert hidden.shape == (32, 5)

    def test_decode_shape(self):
        """Test decode produces correct shape."""
        config = ToyModelConfig(n_features=10, d_hidden=5)
        model = ToyModel(config)

        hidden = torch.randn(32, 5)
        reconstructed = model.decode(hidden)

        assert reconstructed.shape == (32, 10)

    def test_forward_shapes(self):
        """Test forward pass produces correct shapes."""
        config = ToyModelConfig(n_features=10, d_hidden=5)
        model = ToyModel(config)

        features = torch.randn(32, 10)
        hidden, reconstructed = model(features)

        assert hidden.shape == (32, 5)
        assert reconstructed.shape == (32, 10)

    def test_encode_decode_consistency(self):
        """Test that encode then decode is consistent with forward."""
        config = ToyModelConfig(n_features=10, d_hidden=5)
        model = ToyModel(config)

        features = torch.randn(32, 10)

        # Forward pass
        hidden1, reconstructed1 = model(features)

        # Manual encode/decode
        hidden2 = model.encode(features)
        reconstructed2 = model.decode(hidden2)

        torch.testing.assert_close(hidden1, hidden2)
        torch.testing.assert_close(reconstructed1, reconstructed2)

    def test_get_feature_directions(self):
        """Test get_feature_directions returns copy."""
        config = ToyModelConfig(n_features=10, d_hidden=5)
        model = ToyModel(config)

        directions = model.get_feature_directions()

        assert directions.shape == (10, 5)
        assert not directions.requires_grad  # Should be detached

        # Modifying returned tensor shouldn't affect model
        original = model.W.data.clone()
        directions[0] = 999
        torch.testing.assert_close(model.W.data, original)

    def test_reproducibility(self):
        """Test model is reproducible with same seed."""
        config = ToyModelConfig(n_features=10, d_hidden=5, seed=42)
        model1 = ToyModel(config)
        model2 = ToyModel(config)

        torch.testing.assert_close(model1.W, model2.W)


class TestToyDataGenerator:
    """Tests for ToyDataGenerator class."""

    def test_generate_features_shape(self):
        """Test feature generation shape."""
        config = ToyModelConfig(n_features=10, d_hidden=5)
        model = ToyModel(config)
        generator = ToyDataGenerator(model)

        features = generator.generate_features(n_samples=100)

        assert features.shape == (100, 10)

    def test_generate_features_sparsity(self):
        """Test that features are sparse."""
        config = ToyModelConfig(
            n_features=10,
            d_hidden=5,
            feature_probability=0.1,
        )
        model = ToyModel(config)
        generator = ToyDataGenerator(model)

        features = generator.generate_features(n_samples=1000)

        # Average sparsity should be close to feature_probability
        actual_sparsity = (features > 0).float().mean().item()
        assert 0.05 < actual_sparsity < 0.15  # Within reasonable range

    def test_generate_batch(self):
        """Test batch generation."""
        config = ToyModelConfig(n_features=10, d_hidden=5)
        model = ToyModel(config)
        generator = ToyDataGenerator(model, batch_size=64)

        features, hidden, reconstructed = generator.generate_batch()

        assert features.shape == (64, 10)
        assert hidden.shape == (64, 5)
        assert reconstructed.shape == (64, 10)

    def test_generate_batch_custom_size(self):
        """Test batch generation with custom size."""
        config = ToyModelConfig(n_features=10, d_hidden=5)
        model = ToyModel(config)
        generator = ToyDataGenerator(model, batch_size=64)

        features, hidden, reconstructed = generator.generate_batch(n_samples=128)

        assert features.shape == (128, 10)

    def test_generate_labeled_batch(self):
        """Test labeled batch generation."""
        config = ToyModelConfig(n_features=10, d_hidden=5)
        model = ToyModel(config)
        generator = ToyDataGenerator(model)

        features, hidden, labels = generator.generate_labeled_batch(n_samples=100)

        assert features.shape == (100, 10)
        assert hidden.shape == (100, 5)
        assert labels.shape == (100,)

        # Labels should be in range [-1, n_features-1]
        assert labels.min() >= -1
        assert labels.max() < 10


class TestAnalyzeSuperposition:
    """Tests for analyze_superposition function."""

    def test_analysis_keys(self):
        """Test that analysis returns expected keys."""
        config = ToyModelConfig(n_features=10, d_hidden=5)
        model = ToyModel(config)

        analysis = analyze_superposition(model)

        expected_keys = [
            "feature_overlaps",
            "mean_overlap",
            "max_overlap",
            "capacity_ratio",
            "n_features",
            "d_hidden",
        ]
        for key in expected_keys:
            assert key in analysis

    def test_capacity_ratio(self):
        """Test capacity ratio calculation."""
        config = ToyModelConfig(n_features=10, d_hidden=5)
        model = ToyModel(config)

        analysis = analyze_superposition(model)

        assert analysis["capacity_ratio"] == 2.0  # 10 / 5

    def test_overlap_range(self):
        """Test that overlaps are in valid range."""
        config = ToyModelConfig(n_features=10, d_hidden=5)
        model = ToyModel(config)

        analysis = analyze_superposition(model)

        # Overlaps should be in [-1, 1] for normalized vectors
        assert 0 <= analysis["mean_overlap"] <= 1
        assert 0 <= analysis["max_overlap"] <= 1


class TestComputeReconstructionAccuracy:
    """Tests for compute_feature_reconstruction_accuracy function."""

    def test_returns_three_values(self):
        """Test that function returns precision, recall, mse."""
        config = ToyModelConfig(n_features=10, d_hidden=5)
        model = ToyModel(config)
        generator = ToyDataGenerator(model)

        features, _, _ = generator.generate_batch(n_samples=100)

        precision, recall, mse = compute_feature_reconstruction_accuracy(
            model, features
        )

        assert isinstance(precision, float)
        assert isinstance(recall, float)
        assert isinstance(mse, float)

    def test_metrics_in_valid_range(self):
        """Test that metrics are in valid ranges."""
        config = ToyModelConfig(n_features=10, d_hidden=5)
        model = ToyModel(config)
        generator = ToyDataGenerator(model)

        features, _, _ = generator.generate_batch(n_samples=100)

        precision, recall, mse = compute_feature_reconstruction_accuracy(
            model, features
        )

        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert mse >= 0
