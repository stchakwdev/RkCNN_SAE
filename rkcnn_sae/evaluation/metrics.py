"""
Evaluation metrics for Sparse Autoencoders.

Key metrics:
- Dead latent rate: fraction of SAE latents that never activate
- L0 sparsity: average number of active latents per sample
- Reconstruction loss: how well the SAE reconstructs inputs
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class SAEMetrics:
    """Container for SAE evaluation metrics."""

    dead_latent_rate: float  # Fraction of latents that never fire
    l0_sparsity: float  # Average number of active latents
    reconstruction_loss: float  # MSE reconstruction error
    explained_variance: float  # Fraction of variance explained
    n_dead_latents: int  # Absolute count of dead latents
    n_total_latents: int  # Total latent count


def compute_dead_latent_rate(
    latent_activations: torch.Tensor,
    threshold: float = 0.0,
) -> tuple[float, int, int]:
    """
    Compute the dead latent rate.

    A latent is "dead" if it never activates above threshold across all samples.

    Parameters
    ----------
    latent_activations : torch.Tensor, shape (n_samples, n_latents)
        Latent activations from SAE encoder.
    threshold : float
        Activation threshold (latents below this are considered inactive).

    Returns
    -------
    dead_rate : float
        Fraction of dead latents.
    n_dead : int
        Number of dead latents.
    n_total : int
        Total number of latents.
    """
    # Check if each latent ever activates above threshold
    max_per_latent = latent_activations.max(dim=0).values  # (n_latents,)
    is_dead = max_per_latent <= threshold

    n_dead = is_dead.sum().item()
    n_total = latent_activations.shape[1]
    dead_rate = n_dead / n_total

    return dead_rate, n_dead, n_total


def compute_l0_sparsity(
    latent_activations: torch.Tensor,
    threshold: float = 0.0,
) -> float:
    """
    Compute L0 sparsity (average number of active latents per sample).

    Parameters
    ----------
    latent_activations : torch.Tensor, shape (n_samples, n_latents)
        Latent activations.
    threshold : float
        Threshold for considering a latent "active".

    Returns
    -------
    l0 : float
        Average number of active latents per sample.
    """
    active = latent_activations > threshold  # (n_samples, n_latents)
    l0_per_sample = active.sum(dim=1).float()  # (n_samples,)
    return l0_per_sample.mean().item()


def compute_reconstruction_loss(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    reduction: str = "mean",
) -> float:
    """
    Compute reconstruction loss (MSE).

    Parameters
    ----------
    original : torch.Tensor, shape (n_samples, d_model)
        Original activations.
    reconstructed : torch.Tensor, shape (n_samples, d_model)
        Reconstructed activations from SAE.
    reduction : str
        How to reduce: "mean", "sum", or "none".

    Returns
    -------
    loss : float
        Reconstruction loss.
    """
    return F.mse_loss(reconstructed, original, reduction=reduction).item()


def compute_explained_variance(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> float:
    """
    Compute explained variance ratio.

    EV = 1 - Var(original - reconstructed) / Var(original)

    Parameters
    ----------
    original : torch.Tensor, shape (n_samples, d_model)
    reconstructed : torch.Tensor, shape (n_samples, d_model)

    Returns
    -------
    ev : float
        Explained variance ratio (1.0 = perfect reconstruction).
    """
    residual = original - reconstructed
    var_residual = torch.var(residual)
    var_original = torch.var(original)

    if var_original < 1e-10:
        return 1.0 if var_residual < 1e-10 else 0.0

    ev = 1.0 - (var_residual / var_original)
    return ev.item()


def evaluate_sae(
    original: torch.Tensor,
    latent_activations: torch.Tensor,
    reconstructed: torch.Tensor,
    activation_threshold: float = 0.0,
) -> SAEMetrics:
    """
    Compute all SAE metrics.

    Parameters
    ----------
    original : torch.Tensor, shape (n_samples, d_model)
        Original activations.
    latent_activations : torch.Tensor, shape (n_samples, n_latents)
        SAE encoder outputs (hidden states).
    reconstructed : torch.Tensor, shape (n_samples, d_model)
        SAE decoder outputs (reconstructions).
    activation_threshold : float
        Threshold for counting active latents.

    Returns
    -------
    metrics : SAEMetrics
        All evaluation metrics.
    """
    dead_rate, n_dead, n_total = compute_dead_latent_rate(
        latent_activations, threshold=activation_threshold
    )
    l0 = compute_l0_sparsity(latent_activations, threshold=activation_threshold)
    recon_loss = compute_reconstruction_loss(original, reconstructed)
    ev = compute_explained_variance(original, reconstructed)

    return SAEMetrics(
        dead_latent_rate=dead_rate,
        l0_sparsity=l0,
        reconstruction_loss=recon_loss,
        explained_variance=ev,
        n_dead_latents=n_dead,
        n_total_latents=n_total,
    )


def compute_feature_recovery_rate(
    mined_directions: torch.Tensor,
    true_directions: torch.Tensor,
    similarity_threshold: float = 0.8,
) -> tuple[float, torch.Tensor]:
    """
    Compute how many true features are recovered by mined directions.

    A true feature is "recovered" if there exists a mined direction
    with cosine similarity above threshold.

    Parameters
    ----------
    mined_directions : torch.Tensor, shape (n_mined, d_model)
        Directions discovered by RkCNN.
    true_directions : torch.Tensor, shape (n_true, d_model)
        Ground truth feature directions.
    similarity_threshold : float
        Cosine similarity threshold for considering a match.

    Returns
    -------
    recovery_rate : float
        Fraction of true features recovered.
    similarity_matrix : torch.Tensor, shape (n_mined, n_true)
        Pairwise cosine similarities.
    """
    # Normalize directions
    mined_norm = F.normalize(mined_directions, dim=1)
    true_norm = F.normalize(true_directions, dim=1)

    # Compute cosine similarity matrix
    similarity = mined_norm @ true_norm.T  # (n_mined, n_true)

    # For each true feature, find max similarity with any mined direction
    max_sim_per_true = similarity.abs().max(dim=0).values  # (n_true,)

    # Count recovered features
    n_recovered = (max_sim_per_true >= similarity_threshold).sum().item()
    n_true = true_directions.shape[0]
    recovery_rate = n_recovered / n_true

    return recovery_rate, similarity
