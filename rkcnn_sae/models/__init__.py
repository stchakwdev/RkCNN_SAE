"""Model implementations for RKCNN-SAE experiments."""

from rkcnn_sae.models.toy_model import ToyModel, ToyModelConfig, ToyDataGenerator
from rkcnn_sae.models.sae import SAEConfig, SparseAutoencoder, SAETrainer
from rkcnn_sae.models.rkcnn_sae import (
    RkCNNSAEConfig,
    RkCNNSparseAutoencoder,
    create_rkcnn_sae,
)

__all__ = [
    # Toy model
    "ToyModel",
    "ToyModelConfig",
    "ToyDataGenerator",
    # SAE
    "SAEConfig",
    "SparseAutoencoder",
    "SAETrainer",
    # RkCNN SAE
    "RkCNNSAEConfig",
    "RkCNNSparseAutoencoder",
    "create_rkcnn_sae",
]
