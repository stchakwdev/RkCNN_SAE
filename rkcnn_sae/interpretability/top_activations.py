"""
Top-K activation retrieval for interpretability analysis.

Finds the tokens that most strongly activate each SAE latent,
enabling interpretation of what each latent represents.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch
from collections import defaultdict

from rkcnn_sae.interpretability.activation_store import TokenRecord, TokenAwareActivationStore


@dataclass
class TopActivation:
    """Record of a top-activating token for a latent."""

    latent_idx: int  # Which latent
    token_record: TokenRecord  # Token info
    activation_value: float  # Activation strength
    rank: int  # Rank (1 = highest)


class TopActivationFinder:
    """
    Find top-activating tokens for each SAE latent.

    This is the core tool for interpreting what each latent
    has learned to represent.

    Parameters
    ----------
    token_store : TokenAwareActivationStore
        Store with tokens and their activations.
    latent_activations : torch.Tensor
        SAE latent activations, shape (n_tokens, n_latents).
    """

    def __init__(
        self,
        token_store: TokenAwareActivationStore,
        latent_activations: torch.Tensor,
    ):
        self.token_store = token_store
        self.latent_activations = latent_activations
        self.n_tokens = latent_activations.shape[0]
        self.n_latents = latent_activations.shape[1]

        # Precompute max activations per latent for efficiency
        self._max_per_latent = latent_activations.max(dim=0).values

    def get_top_k_for_latent(
        self,
        latent_idx: int,
        k: int = 20,
    ) -> List[TopActivation]:
        """
        Get the top-k activating tokens for a specific latent.

        Parameters
        ----------
        latent_idx : int
            Index of the latent to analyze.
        k : int
            Number of top tokens to return.

        Returns
        -------
        top_activations : List[TopActivation]
            Top-k activations, sorted by activation value (descending).
        """
        if latent_idx < 0 or latent_idx >= self.n_latents:
            raise ValueError(f"Invalid latent_idx: {latent_idx}")

        # Get activations for this latent
        latent_acts = self.latent_activations[:, latent_idx]

        # Get top-k indices and values
        k = min(k, self.n_tokens)
        top_values, top_indices = torch.topk(latent_acts, k)

        results = []
        for rank, (idx, value) in enumerate(zip(top_indices.tolist(), top_values.tolist())):
            record = self.token_store.get_token_for_index(idx)
            results.append(TopActivation(
                latent_idx=latent_idx,
                token_record=record,
                activation_value=value,
                rank=rank + 1,
            ))

        return results

    def get_max_activation(self, latent_idx: int) -> float:
        """Get maximum activation value for a latent."""
        return self._max_per_latent[latent_idx].item()

    def get_activity_rate(self, latent_idx: int, threshold: float = 0.0) -> float:
        """Get fraction of tokens where latent activates above threshold."""
        acts = self.latent_activations[:, latent_idx]
        return (acts > threshold).float().mean().item()

    def is_dead(self, latent_idx: int, threshold: float = 0.0) -> bool:
        """Check if a latent is dead (never activates above threshold)."""
        return self.get_max_activation(latent_idx) <= threshold

    def format_examples(
        self,
        latent_idx: int,
        k: int = 10,
        max_context_chars: int = 40,
    ) -> str:
        """
        Format top activations as a human-readable string.

        Parameters
        ----------
        latent_idx : int
            Latent to analyze.
        k : int
            Number of examples to include.
        max_context_chars : int
            Max characters for context on each side.

        Returns
        -------
        formatted : str
            Human-readable summary.
        """
        top_acts = self.get_top_k_for_latent(latent_idx, k)

        if not top_acts:
            return f"Latent #{latent_idx}: No activations found"

        max_act = self.get_max_activation(latent_idx)
        activity = self.get_activity_rate(latent_idx) * 100

        lines = [
            f"Latent #{latent_idx}",
            f"Max activation: {max_act:.3f}",
            f"Activity rate: {activity:.2f}%",
            "",
            "Top activating tokens:",
        ]

        for act in top_acts:
            context = act.token_record.get_highlighted_context(max_context_chars)
            lines.append(
                f"  {act.rank}. \"{act.token_record.token_text}\" ({act.activation_value:.3f})"
            )
            lines.append(f"     {context}")

        return "\n".join(lines)

    def get_common_tokens(
        self,
        latent_idx: int,
        k: int = 20,
    ) -> List[Tuple[str, int, float]]:
        """
        Get most common tokens among top activations.

        Returns
        -------
        common_tokens : List[Tuple[str, count, avg_activation]]
            Token text, count, and average activation.
        """
        top_acts = self.get_top_k_for_latent(latent_idx, k)

        token_stats = defaultdict(lambda: {"count": 0, "total_act": 0.0})
        for act in top_acts:
            text = act.token_record.token_text.strip()
            token_stats[text]["count"] += 1
            token_stats[text]["total_act"] += act.activation_value

        results = []
        for text, stats in token_stats.items():
            avg_act = stats["total_act"] / stats["count"]
            results.append((text, stats["count"], avg_act))

        # Sort by count, then by avg activation
        results.sort(key=lambda x: (-x[1], -x[2]))
        return results

    def find_latents_for_token(
        self,
        token_text: str,
        top_n_latents: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Find which latents most strongly respond to a specific token.

        Parameters
        ----------
        token_text : str
            Token to search for.
        top_n_latents : int
            Number of top latents to return.

        Returns
        -------
        latent_activations : List[Tuple[latent_idx, avg_activation]]
            Latents that respond to this token, sorted by strength.
        """
        # Find all indices of this token
        _, indices = self.token_store.get_activations_for_token_text(token_text)

        if not indices:
            return []

        # Average latent activations for this token
        token_latent_acts = self.latent_activations[indices].mean(dim=0)

        # Get top latents
        top_values, top_indices = torch.topk(token_latent_acts, top_n_latents)

        return list(zip(top_indices.tolist(), top_values.tolist()))

    def batch_analyze(
        self,
        latent_indices: List[int],
        k: int = 10,
    ) -> dict:
        """
        Analyze multiple latents efficiently.

        Returns dict with latent_idx -> analysis info.
        """
        results = {}
        for idx in latent_indices:
            results[idx] = {
                "max_activation": self.get_max_activation(idx),
                "activity_rate": self.get_activity_rate(idx),
                "is_dead": self.is_dead(idx),
                "top_tokens": self.get_common_tokens(idx, k),
            }
        return results
