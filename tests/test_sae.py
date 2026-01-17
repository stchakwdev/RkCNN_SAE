"""Unit tests for SAE models."""

import numpy as np
import pytest
import torch

from rkcnn_sae.models.sae import SAEConfig, SparseAutoencoder, SAETrainer
from rkcnn_sae.models.rkcnn_sae import (
    RkCNNSAEConfig,
    RkCNNSparseAutoencoder,
    create_rkcnn_sae,
)


class TestSAEConfig:
    """Tests for SAEConfig."""

    def test_required_params(self):
        """Test that required parameters are set."""
        config = SAEConfig(d_model=768, n_latents=6144)

        assert config.d_model == 768
        assert config.n_latents == 6144

    def test_default_values(self):
        """Test default configuration values."""
        config = SAEConfig(d_model=768, n_latents=6144)

        assert config.l1_coefficient == 1e-3
        assert config.tied_weights is False
        assert config.normalize_decoder is True
        assert config.activation == "relu"
        assert config.bias is True


class TestSparseAutoencoder:
    """Tests for SparseAutoencoder class."""

    def test_init(self):
        """Test SAE initialization."""
        config = SAEConfig(d_model=64, n_latents=256)
        sae = SparseAutoencoder(config)

        assert sae.encoder.in_features == 64
        assert sae.encoder.out_features == 256
        assert sae.decoder.in_features == 256
        assert sae.decoder.out_features == 64

    def test_encode_shape(self):
        """Test encode produces correct shape."""
        config = SAEConfig(d_model=64, n_latents=256)
        sae = SparseAutoencoder(config)

        x = torch.randn(32, 64)
        latents = sae.encode(x)

        assert latents.shape == (32, 256)

    def test_decode_shape(self):
        """Test decode produces correct shape."""
        config = SAEConfig(d_model=64, n_latents=256)
        sae = SparseAutoencoder(config)

        latents = torch.randn(32, 256)
        reconstructed = sae.decode(latents)

        assert reconstructed.shape == (32, 64)

    def test_activate_relu(self):
        """Test ReLU activation."""
        config = SAEConfig(d_model=64, n_latents=256, activation="relu")
        sae = SparseAutoencoder(config)

        latents = torch.randn(32, 256)
        activated = sae.activate(latents)

        # All values should be >= 0
        assert (activated >= 0).all()
        # Positive values should be unchanged
        positive_mask = latents > 0
        torch.testing.assert_close(
            activated[positive_mask], latents[positive_mask]
        )

    def test_activate_topk(self):
        """Test top-k activation."""
        config = SAEConfig(d_model=64, n_latents=256, activation="topk", k=10)
        sae = SparseAutoencoder(config)

        latents = torch.randn(32, 256)
        activated = sae.activate(latents)

        # Should have at most k non-zero entries per sample
        for i in range(32):
            n_active = (activated[i] != 0).sum().item()
            assert n_active <= 10

    def test_forward_shapes(self):
        """Test forward pass shapes."""
        config = SAEConfig(d_model=64, n_latents=256)
        sae = SparseAutoencoder(config)

        x = torch.randn(32, 64)
        latents, reconstructed, pre_activation = sae(x)

        assert latents.shape == (32, 256)
        assert reconstructed.shape == (32, 64)
        assert pre_activation.shape == (32, 256)

    def test_forward_latents_sparse(self):
        """Test that latents are sparse after activation."""
        config = SAEConfig(d_model=64, n_latents=256)
        sae = SparseAutoencoder(config)

        x = torch.randn(32, 64)
        latents, _, _ = sae(x)

        # After ReLU, roughly half should be zero
        sparsity = (latents == 0).float().mean().item()
        assert sparsity > 0.3  # At least 30% zeros

    def test_compute_loss(self):
        """Test loss computation."""
        config = SAEConfig(d_model=64, n_latents=256, l1_coefficient=0.01)
        sae = SparseAutoencoder(config)

        x = torch.randn(32, 64)
        latents, reconstructed, _ = sae(x)
        loss, loss_dict = sae.compute_loss(x, latents, reconstructed)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert "total" in loss_dict
        assert "reconstruction" in loss_dict
        assert "l1" in loss_dict

    def test_loss_decreases_with_training(self):
        """Test that loss decreases with gradient descent."""
        config = SAEConfig(d_model=32, n_latents=128)
        sae = SparseAutoencoder(config)
        optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)

        x = torch.randn(100, 32)

        # Initial loss
        latents, reconstructed, _ = sae(x)
        initial_loss, _ = sae.compute_loss(x, latents, reconstructed)

        # Train for a few steps
        for _ in range(50):
            optimizer.zero_grad()
            latents, reconstructed, _ = sae(x)
            loss, _ = sae.compute_loss(x, latents, reconstructed)
            loss.backward()
            optimizer.step()

        # Final loss should be lower
        latents, reconstructed, _ = sae(x)
        final_loss, _ = sae.compute_loss(x, latents, reconstructed)

        assert final_loss < initial_loss

    def test_normalize_decoder(self):
        """Test decoder normalization."""
        config = SAEConfig(d_model=64, n_latents=256, normalize_decoder=True)
        sae = SparseAutoencoder(config)

        sae.normalize_decoder()

        # Each column of decoder weight should have unit norm
        norms = sae.decoder.weight.norm(dim=0)
        torch.testing.assert_close(
            norms, torch.ones(256), atol=1e-5, rtol=1e-5
        )

    def test_get_decoder_directions(self):
        """Test getting decoder directions."""
        config = SAEConfig(d_model=64, n_latents=256)
        sae = SparseAutoencoder(config)

        directions = sae.get_decoder_directions()

        assert directions.shape == (256, 64)  # (n_latents, d_model)

    def test_tied_weights(self):
        """Test tied weights configuration."""
        config = SAEConfig(d_model=64, n_latents=256, tied_weights=True)
        sae = SparseAutoencoder(config)

        assert sae.decoder is None

        # Forward should still work
        x = torch.randn(32, 64)
        latents, reconstructed, _ = sae(x)

        assert reconstructed.shape == (32, 64)


class TestRkCNNSparseAutoencoder:
    """Tests for RkCNN-initialized SAE."""

    def test_init(self):
        """Test RkCNN SAE initialization."""
        config = RkCNNSAEConfig(
            d_model=64,
            n_latents=256,
            rkcnn_directions_fraction=0.5,
            rkcnn_n_subsets=50,
        )
        sae = RkCNNSparseAutoencoder(config)

        assert sae.rkcnn_config.rkcnn_directions_fraction == 0.5
        assert sae.rkcnn_initialized_mask is None  # Not yet initialized

    def test_initialize_with_rkcnn(self):
        """Test RkCNN initialization."""
        config = RkCNNSAEConfig(
            d_model=64,
            n_latents=256,
            rkcnn_directions_fraction=0.5,
            rkcnn_n_subsets=50,
        )
        sae = RkCNNSparseAutoencoder(config)

        activations = torch.randn(100, 64)
        n_initialized = sae.initialize_with_rkcnn(
            activations, seed=42, show_progress=False
        )

        # Should initialize min(256*0.5, 50) = 50 latents
        assert n_initialized == 50
        assert sae.rkcnn_initialized_mask is not None
        assert sae.rkcnn_initialized_mask.sum() == 50
        assert sae.mined_directions is not None
        assert sae.mined_directions.shape == (50, 64)

    def test_get_rkcnn_latent_stats(self):
        """Test RkCNN latent statistics."""
        config = RkCNNSAEConfig(
            d_model=64,
            n_latents=256,
            rkcnn_directions_fraction=0.5,
            rkcnn_n_subsets=50,
        )
        sae = RkCNNSparseAutoencoder(config)

        activations = torch.randn(100, 64)
        sae.initialize_with_rkcnn(activations, seed=42, show_progress=False)

        # Generate some latent activations
        x = torch.randn(50, 64)
        latents, _, _ = sae(x)

        stats = sae.get_rkcnn_latent_stats(latents)

        assert "rkcnn_n" in stats
        assert "random_n" in stats
        assert "rkcnn_dead_rate" in stats
        assert "random_dead_rate" in stats

    def test_forward_after_init(self):
        """Test that forward works after RkCNN initialization."""
        config = RkCNNSAEConfig(
            d_model=64,
            n_latents=256,
            rkcnn_directions_fraction=0.5,
            rkcnn_n_subsets=50,
        )
        sae = RkCNNSparseAutoencoder(config)

        activations = torch.randn(100, 64)
        sae.initialize_with_rkcnn(activations, seed=42, show_progress=False)

        x = torch.randn(32, 64)
        latents, reconstructed, _ = sae(x)

        assert latents.shape == (32, 256)
        assert reconstructed.shape == (32, 64)


class TestCreateRkCNNSAE:
    """Tests for create_rkcnn_sae convenience function."""

    def test_create_with_defaults(self):
        """Test creation with default parameters."""
        sae = create_rkcnn_sae(d_model=64)

        assert sae.config.d_model == 64
        assert sae.config.n_latents == 64 * 8  # default expansion = 8

    def test_create_with_custom_params(self):
        """Test creation with custom parameters."""
        sae = create_rkcnn_sae(
            d_model=64,
            expansion_factor=4,
            l1_coefficient=0.01,
            rkcnn_fraction=0.3,
        )

        assert sae.config.d_model == 64
        assert sae.config.n_latents == 256
        assert sae.config.l1_coefficient == 0.01
        assert sae.rkcnn_config.rkcnn_directions_fraction == 0.3


class TestSAETrainer:
    """Tests for SAETrainer class."""

    def test_train_step(self):
        """Test single training step."""
        config = SAEConfig(d_model=32, n_latents=128)
        sae = SparseAutoencoder(config)
        trainer = SAETrainer(sae, lr=1e-3)

        batch = torch.randn(64, 32)
        loss_dict = trainer.train_step(batch)

        assert "total" in loss_dict
        assert "reconstruction" in loss_dict
        assert "l1" in loss_dict

    def test_evaluate(self):
        """Test evaluation."""
        config = SAEConfig(d_model=32, n_latents=128)
        sae = SparseAutoencoder(config)
        trainer = SAETrainer(sae, lr=1e-3)

        # Create simple dataloader
        data = torch.randn(100, 32)
        dataset = torch.utils.data.TensorDataset(data)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

        avg_losses, latents, recons, inputs = trainer.evaluate(dataloader)

        assert "total" in avg_losses
        assert latents.shape[0] == 100
        assert recons.shape == (100, 32)
        assert inputs.shape == (100, 32)
