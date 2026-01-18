"""
Interpretability metrics for SAE latents.

Quantitative measures of how interpretable/monosemantic each latent is,
enabling comparison between baseline and RkCNN-initialized SAEs.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import torch
import numpy as np
from collections import Counter


@dataclass
class LatentInterpretabilityScore:
    """Interpretability scores for a single latent."""

    latent_idx: int

    # Core metrics
    activation_entropy: float  # Lower = more monosemantic
    top_k_concentration: float  # Higher = activations focused on few tokens
    token_type_diversity: float  # Lower = more specific token type

    # Activity metrics
    max_activation: float
    activity_rate: float
    is_dead: bool

    # Context metrics (optional, requires more compute)
    context_similarity: Optional[float] = None  # Higher = consistent contexts

    @property
    def monosemanticity_score(self) -> float:
        """
        Combined monosemanticity score (0-1, higher = more interpretable).

        Combines entropy, concentration, and diversity into single score.
        """
        if self.is_dead:
            return 0.0

        # Normalize entropy (assuming max entropy around 10 for typical distributions)
        entropy_score = 1.0 - min(self.activation_entropy / 10.0, 1.0)

        # Concentration is already 0-1
        concentration_score = self.top_k_concentration

        # Diversity inverted (lower is better)
        diversity_score = 1.0 - min(self.token_type_diversity, 1.0)

        # Weighted combination
        score = (
            0.4 * entropy_score +
            0.4 * concentration_score +
            0.2 * diversity_score
        )

        return float(np.clip(score, 0.0, 1.0))


class InterpretabilityMetrics:
    """
    Compute interpretability metrics for SAE latents.

    Parameters
    ----------
    latent_activations : torch.Tensor
        SAE latent activations, shape (n_tokens, n_latents).
    token_texts : List[str]
        Token text for each activation.
    """

    def __init__(
        self,
        latent_activations: torch.Tensor,
        token_texts: List[str],
    ):
        self.latent_activations = latent_activations
        self.token_texts = token_texts
        self.n_tokens = latent_activations.shape[0]
        self.n_latents = latent_activations.shape[1]

    def compute_activation_entropy(
        self,
        latent_idx: int,
        epsilon: float = 1e-10,
    ) -> float:
        """
        Compute entropy of activation distribution.

        Lower entropy = more focused on specific tokens = more monosemantic.

        Parameters
        ----------
        latent_idx : int
            Latent to analyze.
        epsilon : float
            Small value for numerical stability.

        Returns
        -------
        entropy : float
            Shannon entropy of normalized activation distribution.
        """
        acts = self.latent_activations[:, latent_idx]

        # Only consider positive activations
        positive_acts = torch.clamp(acts, min=0)

        total = positive_acts.sum()
        if total < epsilon:
            return 0.0  # Dead latent

        # Normalize to probability distribution
        probs = positive_acts / total

        # Compute entropy (only for non-zero probs)
        log_probs = torch.log(probs + epsilon)
        entropy = -(probs * log_probs).sum()

        return entropy.item()

    def compute_top_k_concentration(
        self,
        latent_idx: int,
        k: int = 10,
    ) -> float:
        """
        Compute what fraction of total activation is in top-k tokens.

        Higher = more concentrated = potentially more interpretable.

        Parameters
        ----------
        latent_idx : int
            Latent to analyze.
        k : int
            Number of top tokens.

        Returns
        -------
        concentration : float
            Fraction of activation in top-k tokens.
        """
        acts = self.latent_activations[:, latent_idx]
        positive_acts = torch.clamp(acts, min=0)

        total = positive_acts.sum()
        if total < 1e-10:
            return 0.0  # Dead latent

        # Get top-k activations
        top_k_values, _ = torch.topk(positive_acts, min(k, self.n_tokens))
        top_k_sum = top_k_values.sum()

        return (top_k_sum / total).item()

    def compute_token_type_diversity(
        self,
        latent_idx: int,
        top_k: int = 20,
    ) -> float:
        """
        Compute diversity of token types in top activations.

        Lower diversity = more specific = more interpretable.

        Uses normalized type count: unique_types / top_k.

        Parameters
        ----------
        latent_idx : int
            Latent to analyze.
        top_k : int
            Number of top tokens to consider.

        Returns
        -------
        diversity : float
            Normalized token type diversity (0-1).
        """
        acts = self.latent_activations[:, latent_idx]
        top_values, top_indices = torch.topk(acts, min(top_k, self.n_tokens))

        # Get unique token types
        top_tokens = [self.token_texts[i].strip().lower() for i in top_indices.tolist()]

        # Count unique tokens
        unique_count = len(set(top_tokens))

        # Normalize by top_k
        diversity = unique_count / len(top_tokens) if top_tokens else 1.0

        return diversity

    def compute_context_similarity(
        self,
        latent_idx: int,
        contexts: List[str],
        top_k: int = 20,
        method: str = "jaccard",
    ) -> float:
        """
        Compute similarity of contexts for top-activating tokens.

        Higher similarity = more consistent context = more interpretable.

        Parameters
        ----------
        latent_idx : int
            Latent to analyze.
        contexts : List[str]
            Context string for each token.
        top_k : int
            Number of top tokens to consider.
        method : str
            "jaccard" for word overlap, "embedding" for semantic (requires sentence-transformers).

        Returns
        -------
        similarity : float
            Average pairwise context similarity (0-1).
        """
        acts = self.latent_activations[:, latent_idx]
        top_values, top_indices = torch.topk(acts, min(top_k, self.n_tokens))

        # Filter to actually active tokens
        active_mask = top_values > 0
        active_indices = top_indices[active_mask].tolist()

        if len(active_indices) < 2:
            return 0.0

        # Get contexts for top tokens
        top_contexts = [contexts[i] for i in active_indices]

        if method == "jaccard":
            return self._jaccard_similarity(top_contexts)
        elif method == "embedding":
            return self._embedding_similarity(top_contexts)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _jaccard_similarity(self, contexts: List[str]) -> float:
        """Compute average pairwise Jaccard similarity of context word sets."""
        # Convert contexts to word sets
        word_sets = [set(ctx.lower().split()) for ctx in contexts]

        # Compute pairwise Jaccard
        similarities = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                intersection = len(word_sets[i] & word_sets[j])
                union = len(word_sets[i] | word_sets[j])
                if union > 0:
                    similarities.append(intersection / union)

        return np.mean(similarities) if similarities else 0.0

    def _embedding_similarity(self, contexts: List[str]) -> float:
        """Compute semantic similarity using sentence embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(contexts)

            # Compute pairwise cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            sim_matrix = cosine_similarity(embeddings)

            # Average off-diagonal elements
            n = len(contexts)
            total = 0
            count = 0
            for i in range(n):
                for j in range(i + 1, n):
                    total += sim_matrix[i, j]
                    count += 1

            return total / count if count > 0 else 0.0

        except ImportError:
            print("Warning: sentence-transformers not available, using Jaccard fallback")
            return self._jaccard_similarity(contexts)

    def score_latent(
        self,
        latent_idx: int,
        contexts: Optional[List[str]] = None,
        dead_threshold: float = 0.0,
    ) -> LatentInterpretabilityScore:
        """
        Compute all interpretability metrics for a latent.

        Parameters
        ----------
        latent_idx : int
            Latent to analyze.
        contexts : Optional[List[str]]
            Context strings for computing context similarity.
        dead_threshold : float
            Threshold for considering latent dead.

        Returns
        -------
        score : LatentInterpretabilityScore
            All interpretability metrics.
        """
        acts = self.latent_activations[:, latent_idx]
        max_act = acts.max().item()
        activity_rate = (acts > dead_threshold).float().mean().item()
        is_dead = max_act <= dead_threshold

        return LatentInterpretabilityScore(
            latent_idx=latent_idx,
            activation_entropy=self.compute_activation_entropy(latent_idx),
            top_k_concentration=self.compute_top_k_concentration(latent_idx),
            token_type_diversity=self.compute_token_type_diversity(latent_idx),
            max_activation=max_act,
            activity_rate=activity_rate,
            is_dead=is_dead,
            context_similarity=(
                self.compute_context_similarity(latent_idx, contexts)
                if contexts else None
            ),
        )

    def score_all_latents(
        self,
        contexts: Optional[List[str]] = None,
        dead_threshold: float = 0.0,
        show_progress: bool = True,
    ) -> List[LatentInterpretabilityScore]:
        """
        Score all latents.

        Returns
        -------
        scores : List[LatentInterpretabilityScore]
            Scores for all latents.
        """
        from tqdm import tqdm

        iterator = range(self.n_latents)
        if show_progress:
            iterator = tqdm(iterator, desc="Scoring latents")

        scores = []
        for idx in iterator:
            scores.append(self.score_latent(idx, contexts, dead_threshold))

        return scores

    @staticmethod
    def compare_saes(
        baseline_scores: List[LatentInterpretabilityScore],
        rkcnn_scores: List[LatentInterpretabilityScore],
    ) -> Dict:
        """
        Compare interpretability metrics between two SAEs.

        Parameters
        ----------
        baseline_scores : List[LatentInterpretabilityScore]
            Scores from baseline SAE.
        rkcnn_scores : List[LatentInterpretabilityScore]
            Scores from RkCNN SAE.

        Returns
        -------
        comparison : dict
            Statistical comparison of metrics.
        """
        def get_alive_scores(scores):
            return [s for s in scores if not s.is_dead]

        baseline_alive = get_alive_scores(baseline_scores)
        rkcnn_alive = get_alive_scores(rkcnn_scores)

        def mean_metric(scores, attr):
            if not scores:
                return 0.0
            return np.mean([getattr(s, attr) for s in scores])

        return {
            "n_latents": len(baseline_scores),
            "baseline": {
                "n_alive": len(baseline_alive),
                "n_dead": len(baseline_scores) - len(baseline_alive),
                "mean_entropy": mean_metric(baseline_alive, "activation_entropy"),
                "mean_concentration": mean_metric(baseline_alive, "top_k_concentration"),
                "mean_diversity": mean_metric(baseline_alive, "token_type_diversity"),
                "mean_monosemanticity": mean_metric(baseline_alive, "monosemanticity_score"),
            },
            "rkcnn": {
                "n_alive": len(rkcnn_alive),
                "n_dead": len(rkcnn_scores) - len(rkcnn_alive),
                "mean_entropy": mean_metric(rkcnn_alive, "activation_entropy"),
                "mean_concentration": mean_metric(rkcnn_alive, "top_k_concentration"),
                "mean_diversity": mean_metric(rkcnn_alive, "token_type_diversity"),
                "mean_monosemanticity": mean_metric(rkcnn_alive, "monosemanticity_score"),
            },
            "differences": {
                "alive_diff": len(rkcnn_alive) - len(baseline_alive),
                "entropy_diff": mean_metric(rkcnn_alive, "activation_entropy") - mean_metric(baseline_alive, "activation_entropy"),
                "concentration_diff": mean_metric(rkcnn_alive, "top_k_concentration") - mean_metric(baseline_alive, "top_k_concentration"),
                "diversity_diff": mean_metric(rkcnn_alive, "token_type_diversity") - mean_metric(baseline_alive, "token_type_diversity"),
                "monosemanticity_diff": mean_metric(rkcnn_alive, "monosemanticity_score") - mean_metric(baseline_alive, "monosemanticity_score"),
            },
        }

    @staticmethod
    def format_comparison(comparison: Dict) -> str:
        """Format comparison dict as human-readable string."""
        lines = [
            "=" * 60,
            "INTERPRETABILITY COMPARISON: Baseline vs RkCNN",
            "=" * 60,
            "",
            f"Total latents: {comparison['n_latents']:,}",
            "",
            "ALIVE LATENTS:",
            f"  Baseline: {comparison['baseline']['n_alive']:,}",
            f"  RkCNN:    {comparison['rkcnn']['n_alive']:,}",
            f"  Diff:     {comparison['differences']['alive_diff']:+,}",
            "",
            "MEAN ACTIVATION ENTROPY (lower = more monosemantic):",
            f"  Baseline: {comparison['baseline']['mean_entropy']:.4f}",
            f"  RkCNN:    {comparison['rkcnn']['mean_entropy']:.4f}",
            f"  Diff:     {comparison['differences']['entropy_diff']:+.4f}",
            "",
            "MEAN TOP-K CONCENTRATION (higher = more focused):",
            f"  Baseline: {comparison['baseline']['mean_concentration']:.4f}",
            f"  RkCNN:    {comparison['rkcnn']['mean_concentration']:.4f}",
            f"  Diff:     {comparison['differences']['concentration_diff']:+.4f}",
            "",
            "MEAN TOKEN DIVERSITY (lower = more specific):",
            f"  Baseline: {comparison['baseline']['mean_diversity']:.4f}",
            f"  RkCNN:    {comparison['rkcnn']['mean_diversity']:.4f}",
            f"  Diff:     {comparison['differences']['diversity_diff']:+.4f}",
            "",
            "MEAN MONOSEMANTICITY SCORE (higher = better):",
            f"  Baseline: {comparison['baseline']['mean_monosemanticity']:.4f}",
            f"  RkCNN:    {comparison['rkcnn']['mean_monosemanticity']:.4f}",
            f"  Diff:     {comparison['differences']['monosemanticity_diff']:+.4f}",
        ]

        return "\n".join(lines)
