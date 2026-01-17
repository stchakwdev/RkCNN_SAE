"""
RkCNN-Initialized Sparse Autoencoder.

This SAE uses directions discovered by RkCNN to initialize the decoder weights,
with the hypothesis that this reduces dead latents and improves feature recovery.

Key idea:
- Standard SAE: Random initialization leads to many dead latents (~30-60%)
- RkCNN SAE: Initialize decoder with meaningful directions from RkCNN probing
- Expected benefit: Lower dead latent rate, better feature coverage
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rkcnn_sae.core.rkcnn_probe import mine_directions
from rkcnn_sae.models.sae import SAEConfig, SparseAutoencoder


@dataclass
class RkCNNSAEConfig(SAEConfig):
    """Configuration for RkCNN-initialized SAE."""

    # RkCNN probing parameters
    rkcnn_subset_size: Optional[int] = None  # m (default: sqrt(d_model))
    rkcnn_n_subsets: int = 1000  # h
    rkcnn_score_method: str = "kurtosis"  # Scoring method
    rkcnn_directions_fraction: float = 0.5  # Fraction of latents to init with RkCNN

    # Initialization parameters
    init_scale: float = 1.0  # Scale for initialized weights


class RkCNNSparseAutoencoder(SparseAutoencoder):
    """
    Sparse Autoencoder with RkCNN-initialized decoder.

    This extends the standard SAE by:
    1. Mining directions from activation data using RkCNN
    2. Initializing a fraction of decoder columns with these directions
    3. Remaining columns are initialized randomly

    The hypothesis is that starting with meaningful directions reduces
    the "dead latent" problem and improves feature coverage.

    Parameters
    ----------
    config : RkCNNSAEConfig
        Model configuration.
    """

    def __init__(self, config: RkCNNSAEConfig):
        # Initialize parent (standard SAE)
        super().__init__(config)
        self.rkcnn_config = config

        # Track which latents were initialized with RkCNN
        self.rkcnn_initialized_mask = None
        self.mined_directions = None

    def initialize_with_rkcnn(
        self,
        activations: torch.Tensor,
        device: str = "cpu",
        seed: Optional[int] = None,
        show_progress: bool = True,
    ) -> int:
        """
        Initialize decoder with RkCNN-mined directions.

        Parameters
        ----------
        activations : torch.Tensor, shape (n_samples, d_model)
            Activation data to mine directions from.
        device : str
            Device for computation.
        seed : Optional[int]
            Random seed for reproducibility.
        show_progress : bool
            Show progress bar.

        Returns
        -------
        n_initialized : int
            Number of latents initialized with RkCNN directions.
        """
        config = self.rkcnn_config

        # Determine how many latents to initialize with RkCNN
        # Cap at n_subsets since we can only get one direction per subset
        n_rkcnn = int(config.n_latents * config.rkcnn_directions_fraction)
        n_rkcnn = min(n_rkcnn, config.n_latents, config.rkcnn_n_subsets)

        if n_rkcnn == 0:
            return 0

        print(f"Mining {n_rkcnn} directions with RkCNN (from {config.rkcnn_n_subsets} subsets)...")

        # Mine directions
        self.mined_directions = mine_directions(
            activations=activations,
            d_model=config.d_model,
            n_directions=n_rkcnn,
            subset_size=config.rkcnn_subset_size,
            n_subsets=config.rkcnn_n_subsets,
            score_method=config.rkcnn_score_method,
            device=device,
            seed=seed,
        )

        # Initialize decoder with mined directions
        with torch.no_grad():
            if config.tied_weights:
                # For tied weights, modify encoder
                # encoder.weight: (n_latents, d_model)
                self.encoder.weight[:n_rkcnn] = self.mined_directions * config.init_scale
            else:
                # decoder.weight: (d_model, n_latents)
                # Set first n_rkcnn columns
                self.decoder.weight[:, :n_rkcnn] = (
                    self.mined_directions.T * config.init_scale
                )

        # Track which latents were RkCNN-initialized
        self.rkcnn_initialized_mask = torch.zeros(config.n_latents, dtype=torch.bool)
        self.rkcnn_initialized_mask[:n_rkcnn] = True

        print(f"Initialized {n_rkcnn}/{config.n_latents} latents with RkCNN directions")
        return n_rkcnn

    def get_rkcnn_latent_stats(
        self,
        latents: torch.Tensor,
        threshold: float = 0.0,
    ) -> dict:
        """
        Compare statistics between RkCNN-initialized and random latents.

        Parameters
        ----------
        latents : torch.Tensor, shape (n_samples, n_latents)
        threshold : float
            Activation threshold.

        Returns
        -------
        stats : dict
            Statistics comparing RkCNN vs random latents.
        """
        if self.rkcnn_initialized_mask is None:
            return {"error": "Not initialized with RkCNN"}

        rkcnn_mask = self.rkcnn_initialized_mask.to(latents.device)
        random_mask = ~rkcnn_mask

        # Extract latent subsets
        rkcnn_latents = latents[:, rkcnn_mask]  # (n_samples, n_rkcnn)
        random_latents = latents[:, random_mask]  # (n_samples, n_random)

        def compute_stats(lat: torch.Tensor, name: str) -> dict:
            n_latents = lat.shape[1]
            if n_latents == 0:
                return {f"{name}_n": 0}

            # Dead latent rate
            max_per_latent = lat.max(dim=0).values
            dead_rate = (max_per_latent <= threshold).float().mean().item()

            # Activity rate
            active = (lat > threshold).float()
            activity_rate = active.mean().item()

            # Mean activation when active
            active_mask = lat > threshold
            if active_mask.any():
                mean_when_active = lat[active_mask].mean().item()
            else:
                mean_when_active = 0.0

            return {
                f"{name}_n": n_latents,
                f"{name}_dead_rate": dead_rate,
                f"{name}_activity_rate": activity_rate,
                f"{name}_mean_when_active": mean_when_active,
            }

        stats = {}
        stats.update(compute_stats(rkcnn_latents, "rkcnn"))
        stats.update(compute_stats(random_latents, "random"))

        return stats


def create_rkcnn_sae(
    d_model: int,
    expansion_factor: int = 8,
    l1_coefficient: float = 1e-3,
    rkcnn_fraction: float = 0.5,
    **kwargs,
) -> RkCNNSparseAutoencoder:
    """
    Convenience function to create RkCNN SAE.

    Parameters
    ----------
    d_model : int
        Input dimension.
    expansion_factor : int
        n_latents = d_model * expansion_factor.
    l1_coefficient : float
        Sparsity penalty.
    rkcnn_fraction : float
        Fraction of latents to initialize with RkCNN.
    **kwargs
        Additional RkCNNSAEConfig parameters.

    Returns
    -------
    sae : RkCNNSparseAutoencoder
        Configured SAE.
    """
    n_latents = d_model * expansion_factor

    config = RkCNNSAEConfig(
        d_model=d_model,
        n_latents=n_latents,
        l1_coefficient=l1_coefficient,
        rkcnn_directions_fraction=rkcnn_fraction,
        **kwargs,
    )

    return RkCNNSparseAutoencoder(config)


def compare_initialization_methods(
    activations: torch.Tensor,
    d_model: int,
    n_latents: int,
    l1_coefficient: float = 1e-3,
    n_train_steps: int = 1000,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 42,
) -> dict:
    """
    Compare baseline SAE vs RkCNN-initialized SAE.

    Parameters
    ----------
    activations : torch.Tensor
        Training data.
    d_model : int
        Input dimension.
    n_latents : int
        Number of SAE latents.
    l1_coefficient : float
        L1 sparsity penalty.
    n_train_steps : int
        Number of training steps.
    batch_size : int
        Batch size.
    lr : float
        Learning rate.
    device : str
        Device.
    seed : int
        Random seed.

    Returns
    -------
    results : dict
        Comparison results including dead latent rates and metrics.
    """
    from rkcnn_sae.evaluation.metrics import evaluate_sae
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create datasets
    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    results = {}

    # --- Train Baseline SAE ---
    print("\n" + "=" * 50)
    print("Training Baseline SAE")
    print("=" * 50)

    baseline_config = SAEConfig(
        d_model=d_model,
        n_latents=n_latents,
        l1_coefficient=l1_coefficient,
    )
    baseline_sae = SparseAutoencoder(baseline_config).to(device)
    baseline_optimizer = torch.optim.Adam(baseline_sae.parameters(), lr=lr)

    baseline_losses = []
    step = 0
    while step < n_train_steps:
        for batch in dataloader:
            if step >= n_train_steps:
                break
            batch = batch[0].to(device)
            baseline_optimizer.zero_grad()
            latents, reconstructed, _ = baseline_sae(batch)
            loss, _ = baseline_sae.compute_loss(batch, latents, reconstructed)
            loss.backward()
            baseline_optimizer.step()
            baseline_losses.append(loss.item())
            step += 1

            if step % 200 == 0:
                print(f"  Step {step}: loss = {loss.item():.4f}")

    # Evaluate baseline
    with torch.no_grad():
        all_latents = []
        all_recons = []
        for batch in dataloader:
            batch = batch[0].to(device)
            latents, reconstructed, _ = baseline_sae(batch)
            all_latents.append(latents)
            all_recons.append(reconstructed)
        all_latents = torch.cat(all_latents, dim=0)
        all_recons = torch.cat(all_recons, dim=0)

    baseline_metrics = evaluate_sae(activations.to(device), all_latents, all_recons)
    results["baseline"] = {
        "dead_latent_rate": baseline_metrics.dead_latent_rate,
        "l0_sparsity": baseline_metrics.l0_sparsity,
        "reconstruction_loss": baseline_metrics.reconstruction_loss,
        "explained_variance": baseline_metrics.explained_variance,
        "final_loss": baseline_losses[-1],
    }
    print(f"\nBaseline Results:")
    print(f"  Dead latent rate: {baseline_metrics.dead_latent_rate:.2%}")
    print(f"  L0 sparsity: {baseline_metrics.l0_sparsity:.2f}")
    print(f"  Reconstruction loss: {baseline_metrics.reconstruction_loss:.6f}")

    # --- Train RkCNN SAE ---
    print("\n" + "=" * 50)
    print("Training RkCNN-Initialized SAE")
    print("=" * 50)

    rkcnn_config = RkCNNSAEConfig(
        d_model=d_model,
        n_latents=n_latents,
        l1_coefficient=l1_coefficient,
        rkcnn_directions_fraction=0.5,
        rkcnn_n_subsets=500,
        rkcnn_score_method="kurtosis",
    )
    rkcnn_sae = RkCNNSparseAutoencoder(rkcnn_config)

    # Initialize with RkCNN
    rkcnn_sae.initialize_with_rkcnn(
        activations,
        device=device,
        seed=seed,
        show_progress=True,
    )
    rkcnn_sae = rkcnn_sae.to(device)

    rkcnn_optimizer = torch.optim.Adam(rkcnn_sae.parameters(), lr=lr)

    rkcnn_losses = []
    step = 0
    while step < n_train_steps:
        for batch in dataloader:
            if step >= n_train_steps:
                break
            batch = batch[0].to(device)
            rkcnn_optimizer.zero_grad()
            latents, reconstructed, _ = rkcnn_sae(batch)
            loss, _ = rkcnn_sae.compute_loss(batch, latents, reconstructed)
            loss.backward()
            rkcnn_optimizer.step()
            rkcnn_losses.append(loss.item())
            step += 1

            if step % 200 == 0:
                print(f"  Step {step}: loss = {loss.item():.4f}")

    # Evaluate RkCNN SAE
    with torch.no_grad():
        all_latents = []
        all_recons = []
        for batch in dataloader:
            batch = batch[0].to(device)
            latents, reconstructed, _ = rkcnn_sae(batch)
            all_latents.append(latents)
            all_recons.append(reconstructed)
        all_latents = torch.cat(all_latents, dim=0)
        all_recons = torch.cat(all_recons, dim=0)

    rkcnn_metrics = evaluate_sae(activations.to(device), all_latents, all_recons)

    # Get per-subset stats
    rkcnn_latent_stats = rkcnn_sae.get_rkcnn_latent_stats(all_latents)

    results["rkcnn"] = {
        "dead_latent_rate": rkcnn_metrics.dead_latent_rate,
        "l0_sparsity": rkcnn_metrics.l0_sparsity,
        "reconstruction_loss": rkcnn_metrics.reconstruction_loss,
        "explained_variance": rkcnn_metrics.explained_variance,
        "final_loss": rkcnn_losses[-1],
        "latent_stats": rkcnn_latent_stats,
    }

    print(f"\nRkCNN SAE Results:")
    print(f"  Dead latent rate: {rkcnn_metrics.dead_latent_rate:.2%}")
    print(f"  L0 sparsity: {rkcnn_metrics.l0_sparsity:.2f}")
    print(f"  Reconstruction loss: {rkcnn_metrics.reconstruction_loss:.6f}")
    print(f"  RkCNN latents dead rate: {rkcnn_latent_stats.get('rkcnn_dead_rate', 'N/A'):.2%}")
    print(f"  Random latents dead rate: {rkcnn_latent_stats.get('random_dead_rate', 'N/A'):.2%}")

    # --- Summary ---
    print("\n" + "=" * 50)
    print("COMPARISON SUMMARY")
    print("=" * 50)
    print(f"Dead Latent Rate:  Baseline {baseline_metrics.dead_latent_rate:.2%} -> RkCNN {rkcnn_metrics.dead_latent_rate:.2%}")
    print(f"L0 Sparsity:       Baseline {baseline_metrics.l0_sparsity:.2f} -> RkCNN {rkcnn_metrics.l0_sparsity:.2f}")
    print(f"Recon Loss:        Baseline {baseline_metrics.reconstruction_loss:.6f} -> RkCNN {rkcnn_metrics.reconstruction_loss:.6f}")

    # Success criteria
    dead_latent_improved = rkcnn_metrics.dead_latent_rate < baseline_metrics.dead_latent_rate
    l0_not_worse = rkcnn_metrics.l0_sparsity <= baseline_metrics.l0_sparsity * 1.1  # 10% tolerance
    recon_not_worse = rkcnn_metrics.reconstruction_loss <= baseline_metrics.reconstruction_loss * 1.1

    results["success"] = {
        "dead_latent_improved": dead_latent_improved,
        "l0_not_worse": l0_not_worse,
        "recon_not_worse": recon_not_worse,
        "overall": dead_latent_improved and l0_not_worse and recon_not_worse,
    }

    return results
