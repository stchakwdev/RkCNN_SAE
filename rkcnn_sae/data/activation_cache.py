"""
Activation caching utilities for GPT-2.

Efficiently caches MLP activations from GPT-2 for SAE training.
Uses TransformerLens for hooking into model internals.
"""

import os
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

try:
    from transformer_lens import HookedTransformer
    from transformer_lens.utils import tokenize_and_concatenate
    TRANSFORMER_LENS_AVAILABLE = True
except ImportError:
    TRANSFORMER_LENS_AVAILABLE = False
    print("Warning: transformer_lens not available. GPT-2 activation caching disabled.")


class ActivationCache:
    """
    Cache activations from GPT-2 model.

    This class handles:
    - Loading and running GPT-2 with TransformerLens
    - Hooking MLP activations at specified layers
    - Batching and caching activations to disk

    Parameters
    ----------
    model_name : str
        HuggingFace model name (e.g., "gpt2", "gpt2-medium").
    layer : int
        Which layer to extract activations from.
    hook_point : str
        Where to hook (e.g., "mlp_out", "post", "resid_post").
    device : str
        Device to run model on.
    cache_dir : Optional[str]
        Directory to cache activations to disk.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        layer: int = 6,
        hook_point: str = "mlp_out",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cache_dir: Optional[str] = None,
    ):
        if not TRANSFORMER_LENS_AVAILABLE:
            raise ImportError(
                "transformer_lens is required for activation caching. "
                "Install with: pip install transformer-lens"
            )

        self.model_name = model_name
        self.layer = layer
        self.hook_point = hook_point
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load model
        print(f"Loading model: {model_name}")
        self.model = HookedTransformer.from_pretrained(model_name, device=device)
        self.d_model = self.model.cfg.d_model
        self.d_mlp = self.model.cfg.d_mlp

        # Construct hook name
        # Common hook points: "blocks.{layer}.hook_mlp_out", "blocks.{layer}.hook_resid_post"
        if hook_point == "mlp_out":
            self.hook_name = f"blocks.{layer}.hook_mlp_out"
            self.activation_dim = self.d_mlp
        elif hook_point == "resid_post":
            self.hook_name = f"blocks.{layer}.hook_resid_post"
            self.activation_dim = self.d_model
        else:
            self.hook_name = f"blocks.{layer}.{hook_point}"
            # Try to infer dimension
            self.activation_dim = self.d_model

        print(f"Hook point: {self.hook_name}")
        print(f"Activation dimension: {self.activation_dim}")

    def get_activations(
        self,
        tokens: torch.Tensor,
        return_loss: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get activations for a batch of tokens.

        Parameters
        ----------
        tokens : torch.Tensor, shape (batch, seq_len)
            Token IDs.
        return_loss : bool
            Whether to also return the model's loss.

        Returns
        -------
        activations : torch.Tensor, shape (batch * seq_len, activation_dim)
            Flattened activations.
        loss : torch.Tensor (optional)
            Model loss on the tokens.
        """
        tokens = tokens.to(self.device)

        # Run model with hooks
        _, cache = self.model.run_with_cache(
            tokens,
            names_filter=self.hook_name,
            return_type="loss" if return_loss else None,
        )

        # Extract activations
        activations = cache[self.hook_name]  # (batch, seq_len, activation_dim)

        # Flatten to (batch * seq_len, activation_dim)
        activations = activations.reshape(-1, activations.shape[-1])

        if return_loss:
            loss = self.model(tokens, return_type="loss")
            return activations, loss
        return activations

    def cache_dataset(
        self,
        dataset_name: str = "openwebtext",
        max_tokens: int = 1_000_000,
        seq_len: int = 128,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Cache activations from a dataset.

        Parameters
        ----------
        dataset_name : str
            HuggingFace dataset name.
        max_tokens : int
            Maximum number of tokens to process.
        seq_len : int
            Sequence length.
        batch_size : int
            Batch size for processing.
        show_progress : bool
            Show progress bar.

        Returns
        -------
        activations : torch.Tensor, shape (n_tokens, activation_dim)
            Cached activations.
        """
        from datasets import load_dataset

        # Check if cached
        cache_file = None
        if self.cache_dir:
            cache_file = (
                self.cache_dir
                / f"{self.model_name.replace('/', '_')}_layer{self.layer}_{self.hook_point}_{max_tokens}.pt"
            )
            if cache_file.exists():
                print(f"Loading cached activations from {cache_file}")
                return torch.load(cache_file)

        # Load dataset (non-streaming for tokenize_and_concatenate compatibility)
        print(f"Loading dataset: {dataset_name}")
        try:
            # Try loading a subset directly (non-streaming)
            dataset = load_dataset(
                dataset_name,
                split=f"train[:{min(max_tokens * 2, 100000)}]",  # Load subset
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"Warning: Could not load {dataset_name}: {e}")
            print("Falling back to wikitext...")
            try:
                dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            except Exception:
                print("Using wikitext-103 raw...")
                dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

        # Tokenize
        print("Tokenizing dataset...")
        tokens_data = tokenize_and_concatenate(
            dataset,
            self.model.tokenizer,
            max_length=seq_len,
            add_bos_token=True,
        )

        # Limit tokens
        n_sequences = min(max_tokens // seq_len, len(tokens_data))
        tokens_data = tokens_data.select(range(n_sequences))

        # Extract activations
        all_activations = []
        total_tokens = 0

        iterator = range(0, n_sequences, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Caching activations")

        with torch.no_grad():
            for i in iterator:
                batch_indices = range(i, min(i + batch_size, n_sequences))
                tokens = torch.stack([
                    tokens_data[j]["tokens"] for j in batch_indices
                ]).to(self.device)

                activations = self.get_activations(tokens)
                all_activations.append(activations.cpu())

                total_tokens += activations.shape[0]
                if total_tokens >= max_tokens:
                    break

        all_activations = torch.cat(all_activations, dim=0)[:max_tokens]

        # Save cache
        if cache_file:
            print(f"Saving cached activations to {cache_file}")
            torch.save(all_activations, cache_file)

        return all_activations


class ActivationDataLoader:
    """
    DataLoader for cached activations.

    Parameters
    ----------
    activations : torch.Tensor
        Cached activations, shape (n_samples, d_model).
    batch_size : int
        Batch size.
    shuffle : bool
        Whether to shuffle data.
    device : str
        Device to load batches to.
    """

    def __init__(
        self,
        activations: torch.Tensor,
        batch_size: int = 256,
        shuffle: bool = True,
        device: str = "cpu",
    ):
        self.activations = activations
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.n_samples = activations.shape[0]

    def __len__(self) -> int:
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[torch.Tensor]:
        indices = torch.arange(self.n_samples)
        if self.shuffle:
            indices = indices[torch.randperm(self.n_samples)]

        for i in range(0, self.n_samples, self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            batch = self.activations[batch_indices].to(self.device)
            yield batch


def create_synthetic_gpt2_activations(
    n_samples: int = 10000,
    d_model: int = 768,
    n_features: int = 2000,
    sparsity: float = 0.05,
    seed: int = 42,
) -> torch.Tensor:
    """
    Create synthetic activations that mimic GPT-2 statistics.

    This is useful for testing without GPU or GPT-2 access.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    d_model : int
        Activation dimension.
    n_features : int
        Number of sparse features.
    sparsity : float
        Probability a feature is active.
    seed : int
        Random seed.

    Returns
    -------
    activations : torch.Tensor, shape (n_samples, d_model)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create feature directions (random orthogonal-ish)
    features = torch.randn(n_features, d_model)
    features = features / features.norm(dim=1, keepdim=True)

    # Create sparse feature activations
    active = torch.bernoulli(torch.full((n_samples, n_features), sparsity))
    magnitudes = torch.rand(n_samples, n_features) * active

    # Project to activation space
    activations = magnitudes @ features

    # Add noise
    noise = torch.randn_like(activations) * 0.1
    activations = activations + noise

    return activations
