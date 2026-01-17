"""
Standard Sparse Autoencoder implementation.

This is the baseline SAE that will be compared against RkCNN-initialized SAE.
Based on: "Scaling Monosemanticity" (Anthropic, 2024)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAEConfig:
    """Configuration for Sparse Autoencoder."""

    d_model: int  # Input dimension (e.g., 768 for GPT-2)
    n_latents: int  # Number of SAE latents (typically 4x to 64x d_model)
    l1_coefficient: float = 1e-3  # L1 sparsity penalty
    tied_weights: bool = False  # Whether encoder/decoder weights are tied
    normalize_decoder: bool = True  # Normalize decoder columns
    activation: str = "relu"  # Activation function ("relu" or "topk")
    k: Optional[int] = None  # For top-k activation, number of active latents
    bias: bool = True  # Use bias terms


class SparseAutoencoder(nn.Module):
    """
    Standard Sparse Autoencoder.

    Architecture:
        input (d_model) -> encoder (n_latents) -> activation -> decoder (d_model)

    The encoder projects inputs to a high-dimensional sparse space.
    L1 regularization encourages sparsity in the latent activations.

    Parameters
    ----------
    config : SAEConfig
        Model configuration.
    """

    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config

        # Encoder: d_model -> n_latents
        self.encoder = nn.Linear(config.d_model, config.n_latents, bias=config.bias)

        # Decoder: n_latents -> d_model
        if config.tied_weights:
            # Tied weights: decoder is transpose of encoder
            self.decoder = None
        else:
            self.decoder = nn.Linear(config.n_latents, config.d_model, bias=config.bias)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        nn.init.xavier_uniform_(self.encoder.weight)
        if self.encoder.bias is not None:
            nn.init.zeros_(self.encoder.bias)

        if self.decoder is not None:
            nn.init.xavier_uniform_(self.decoder.weight)
            if self.decoder.bias is not None:
                nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent space.

        Parameters
        ----------
        x : torch.Tensor, shape (..., d_model)
            Input activations.

        Returns
        -------
        latents : torch.Tensor, shape (..., n_latents)
            Latent activations (before sparsity).
        """
        return self.encoder(x)

    def activate(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Apply activation function (ReLU or TopK).

        Parameters
        ----------
        latents : torch.Tensor, shape (..., n_latents)

        Returns
        -------
        activated : torch.Tensor, shape (..., n_latents)
            Sparse latent activations.
        """
        if self.config.activation == "relu":
            return F.relu(latents)
        elif self.config.activation == "topk":
            if self.config.k is None:
                raise ValueError("k must be specified for topk activation")
            return self._topk_activation(latents, self.config.k)
        else:
            raise ValueError(f"Unknown activation: {self.config.activation}")

    def _topk_activation(self, latents: torch.Tensor, k: int) -> torch.Tensor:
        """Top-k activation: keep only the k largest activations."""
        # Get top-k values and indices
        values, indices = torch.topk(latents, k, dim=-1)

        # Create sparse output
        activated = torch.zeros_like(latents)
        activated.scatter_(-1, indices, F.relu(values))

        return activated

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents back to input space.

        Parameters
        ----------
        latents : torch.Tensor, shape (..., n_latents)
            Sparse latent activations.

        Returns
        -------
        reconstructed : torch.Tensor, shape (..., d_model)
            Reconstructed input.
        """
        if self.config.tied_weights:
            # Encoder weight is (n_latents, d_model), need transpose for decode
            return F.linear(latents, self.encoder.weight.T)
        else:
            return self.decoder(latents)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (..., d_model)
            Input activations.

        Returns
        -------
        latents : torch.Tensor, shape (..., n_latents)
            Sparse latent activations.
        reconstructed : torch.Tensor, shape (..., d_model)
            Reconstructed input.
        pre_activation : torch.Tensor, shape (..., n_latents)
            Latents before activation (for auxiliary losses).
        """
        pre_activation = self.encode(x)
        latents = self.activate(pre_activation)
        reconstructed = self.decode(latents)

        return latents, reconstructed, pre_activation

    def compute_loss(
        self,
        x: torch.Tensor,
        latents: torch.Tensor,
        reconstructed: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute SAE training loss.

        Loss = MSE(reconstruction) + l1_coefficient * L1(latents)

        Parameters
        ----------
        x : torch.Tensor
            Original input.
        latents : torch.Tensor
            Sparse latent activations.
        reconstructed : torch.Tensor
            Reconstructed input.

        Returns
        -------
        total_loss : torch.Tensor
            Total loss.
        loss_dict : dict
            Dictionary with individual loss components.
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstructed, x)

        # Sparsity loss (L1)
        l1_loss = latents.abs().mean()

        # Total loss
        total_loss = recon_loss + self.config.l1_coefficient * l1_loss

        return total_loss, {
            "total": total_loss.item(),
            "reconstruction": recon_loss.item(),
            "l1": l1_loss.item(),
        }

    def normalize_decoder(self):
        """Normalize decoder weight columns to unit norm."""
        if self.config.normalize_decoder and self.decoder is not None:
            with torch.no_grad():
                # decoder.weight shape: (d_model, n_latents)
                # Normalize each column (latent direction)
                norms = self.decoder.weight.norm(dim=0, keepdim=True)
                self.decoder.weight.data = self.decoder.weight.data / (norms + 1e-8)

    def get_decoder_directions(self) -> torch.Tensor:
        """
        Get decoder weight vectors (learned feature directions).

        Returns
        -------
        directions : torch.Tensor, shape (n_latents, d_model)
            Each row is a learned feature direction.
        """
        if self.config.tied_weights:
            return self.encoder.weight.data  # (n_latents, d_model)
        else:
            return self.decoder.weight.data.T  # Transpose to (n_latents, d_model)


class SAETrainer:
    """
    Trainer for Sparse Autoencoder.

    Parameters
    ----------
    sae : SparseAutoencoder
        The SAE to train.
    lr : float
        Learning rate.
    device : str
        Device for training.
    """

    def __init__(
        self,
        sae: SparseAutoencoder,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.sae = sae.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    def train_step(self, batch: torch.Tensor) -> dict:
        """
        Single training step.

        Parameters
        ----------
        batch : torch.Tensor, shape (batch_size, d_model)
            Batch of activations.

        Returns
        -------
        loss_dict : dict
            Dictionary with loss components.
        """
        batch = batch.to(self.device)

        self.optimizer.zero_grad()
        latents, reconstructed, _ = self.sae(batch)
        loss, loss_dict = self.sae.compute_loss(batch, latents, reconstructed)

        loss.backward()
        self.optimizer.step()

        # Optionally normalize decoder
        if self.sae.config.normalize_decoder:
            self.sae.normalize_decoder()

        return loss_dict

    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        epoch: int = 0,
    ) -> dict:
        """
        Train for one epoch.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader providing activation batches.
        epoch : int
            Current epoch number.

        Returns
        -------
        avg_losses : dict
            Average losses for the epoch.
        """
        self.sae.train()
        total_losses = {"total": 0.0, "reconstruction": 0.0, "l1": 0.0}
        n_batches = 0

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]  # Handle (data, labels) format

            loss_dict = self.train_step(batch)
            for key in total_losses:
                total_losses[key] += loss_dict[key]
            n_batches += 1

        # Average losses
        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
        return avg_losses

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> Tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate SAE on data.

        Parameters
        ----------
        dataloader : DataLoader
            Evaluation data.

        Returns
        -------
        avg_losses : dict
        all_latents : torch.Tensor
        all_reconstructed : torch.Tensor
        all_inputs : torch.Tensor
        """
        self.sae.eval()
        total_losses = {"total": 0.0, "reconstruction": 0.0, "l1": 0.0}
        all_latents = []
        all_reconstructed = []
        all_inputs = []
        n_batches = 0

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(self.device)

            latents, reconstructed, _ = self.sae(batch)
            _, loss_dict = self.sae.compute_loss(batch, latents, reconstructed)

            for key in total_losses:
                total_losses[key] += loss_dict[key]
            n_batches += 1

            all_latents.append(latents.cpu())
            all_reconstructed.append(reconstructed.cpu())
            all_inputs.append(batch.cpu())

        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
        all_latents = torch.cat(all_latents, dim=0)
        all_reconstructed = torch.cat(all_reconstructed, dim=0)
        all_inputs = torch.cat(all_inputs, dim=0)

        return avg_losses, all_latents, all_reconstructed, all_inputs
