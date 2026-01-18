"""
Token-aware activation storage for interpretability analysis.

Stores activations alongside their source tokens and context,
enabling mapping from latent activations back to source tokens.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Iterator
import torch
import numpy as np
from tqdm import tqdm


@dataclass
class TokenRecord:
    """Record of a single token with its context and position."""

    token_id: int  # Token ID
    token_text: str  # Decoded token text
    context_before: str  # Text before this token
    context_after: str  # Text after this token
    position: int  # Position in sequence
    sequence_idx: int  # Which sequence this came from

    def get_highlighted_context(self, max_chars: int = 50) -> str:
        """Get context with token highlighted in brackets."""
        before = self.context_before[-max_chars:] if len(self.context_before) > max_chars else self.context_before
        after = self.context_after[:max_chars] if len(self.context_after) > max_chars else self.context_after
        return f"...{before}[{self.token_text}]{after}..."


class TokenAwareActivationStore:
    """
    Stores activations with associated token information.

    This enables mapping from latent activations back to the specific
    tokens that caused them, which is essential for interpretability.

    Parameters
    ----------
    activations : torch.Tensor
        Activations of shape (n_tokens, activation_dim).
    token_records : List[TokenRecord]
        Token information for each activation.
    """

    def __init__(
        self,
        activations: torch.Tensor,
        token_records: List[TokenRecord],
    ):
        assert len(token_records) == activations.shape[0], (
            f"Mismatch: {len(token_records)} token records vs {activations.shape[0]} activations"
        )
        self.activations = activations
        self.token_records = token_records
        self.n_tokens = len(token_records)
        self.activation_dim = activations.shape[1]

    @classmethod
    def from_model_and_dataset(
        cls,
        model,  # HookedTransformer
        tokenizer,
        dataset,
        layer: int,
        hook_point: str = "mlp_post",
        max_tokens: int = 50000,
        seq_len: int = 128,
        batch_size: int = 16,
        device: str = "cuda",
        show_progress: bool = True,
    ) -> "TokenAwareActivationStore":
        """
        Create a token-aware store from a model and dataset.

        Parameters
        ----------
        model : HookedTransformer
            TransformerLens model.
        tokenizer
            HuggingFace tokenizer.
        dataset
            HuggingFace dataset with 'text' column.
        layer : int
            Layer to extract activations from.
        hook_point : str
            Hook point type ("mlp_post", "resid_post", etc.).
        max_tokens : int
            Maximum number of tokens to cache.
        seq_len : int
            Sequence length.
        batch_size : int
            Batch size for processing.
        device : str
            Device for model.
        show_progress : bool
            Show progress bar.

        Returns
        -------
        store : TokenAwareActivationStore
            Token-aware activation store.
        """
        from transformer_lens.utils import tokenize_and_concatenate

        # Construct hook name
        if hook_point == "mlp_post":
            hook_name = f"blocks.{layer}.mlp.hook_post"
        elif hook_point == "resid_post":
            hook_name = f"blocks.{layer}.hook_resid_post"
        else:
            hook_name = f"blocks.{layer}.{hook_point}"

        # Tokenize dataset
        print("Tokenizing dataset...")
        tokens_data = tokenize_and_concatenate(
            dataset,
            tokenizer,
            max_length=seq_len,
            add_bos_token=True,
        )

        n_sequences = min(max_tokens // seq_len, len(tokens_data))
        tokens_data = tokens_data.select(range(n_sequences))

        all_activations = []
        all_token_records = []
        total_tokens = 0

        iterator = range(0, n_sequences, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Caching token-aware activations")

        with torch.no_grad():
            for i in iterator:
                batch_indices = list(range(i, min(i + batch_size, n_sequences)))
                tokens_batch = torch.stack([
                    tokens_data[j]["tokens"] for j in batch_indices
                ]).to(device)

                # Run model with hooks
                _, cache = model.run_with_cache(
                    tokens_batch,
                    names_filter=hook_name,
                    return_type=None,
                )

                activations = cache[hook_name]  # (batch, seq_len, dim)

                # Create token records for each token in batch
                for batch_idx, seq_idx in enumerate(batch_indices):
                    tokens = tokens_batch[batch_idx].cpu().tolist()
                    decoded = [tokenizer.decode([t]) for t in tokens]
                    full_text = tokenizer.decode(tokens)

                    for pos in range(seq_len):
                        # Build context
                        context_before = "".join(decoded[:pos])
                        context_after = "".join(decoded[pos+1:])

                        record = TokenRecord(
                            token_id=tokens[pos],
                            token_text=decoded[pos],
                            context_before=context_before,
                            context_after=context_after,
                            position=pos,
                            sequence_idx=seq_idx,
                        )
                        all_token_records.append(record)

                # Flatten activations: (batch, seq, dim) -> (batch*seq, dim)
                flat_acts = activations.reshape(-1, activations.shape[-1])
                all_activations.append(flat_acts.cpu())

                total_tokens += flat_acts.shape[0]
                if total_tokens >= max_tokens:
                    break

        all_activations = torch.cat(all_activations, dim=0)[:max_tokens]
        all_token_records = all_token_records[:max_tokens]

        print(f"Cached {len(all_token_records)} tokens with activations")

        return cls(all_activations, all_token_records)

    def get_token_for_index(self, idx: int) -> TokenRecord:
        """Get the token record for a given activation index."""
        return self.token_records[idx]

    def get_activations_for_token_text(
        self,
        token_text: str,
        exact_match: bool = True,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Get all activations for a specific token text.

        Parameters
        ----------
        token_text : str
            Token text to search for.
        exact_match : bool
            If True, require exact match. Otherwise, substring match.

        Returns
        -------
        activations : torch.Tensor
            Activations for matching tokens.
        indices : List[int]
            Indices of matching tokens.
        """
        indices = []
        for i, record in enumerate(self.token_records):
            if exact_match:
                if record.token_text == token_text:
                    indices.append(i)
            else:
                if token_text in record.token_text:
                    indices.append(i)

        if not indices:
            return torch.empty(0, self.activation_dim), []

        return self.activations[indices], indices

    def compute_latent_activations(
        self,
        sae,  # SparseAutoencoder
        batch_size: int = 512,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Compute SAE latent activations for all stored activations.

        Parameters
        ----------
        sae : SparseAutoencoder
            Trained SAE model.
        batch_size : int
            Batch size for processing.
        device : str
            Device for SAE.

        Returns
        -------
        latent_activations : torch.Tensor
            Latent activations, shape (n_tokens, n_latents).
        """
        sae = sae.to(device)
        sae.eval()

        all_latents = []

        with torch.no_grad():
            for i in range(0, self.n_tokens, batch_size):
                batch = self.activations[i:i+batch_size].to(device)
                latents, _, _ = sae(batch)
                all_latents.append(latents.cpu())

        return torch.cat(all_latents, dim=0)

    def save(self, path: str):
        """Save the store to disk."""
        import pickle
        data = {
            "activations": self.activations,
            "token_records": self.token_records,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved TokenAwareActivationStore to {path}")

    @classmethod
    def load(cls, path: str) -> "TokenAwareActivationStore":
        """Load a store from disk."""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(data["activations"], data["token_records"])
