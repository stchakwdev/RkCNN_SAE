"""
Revived latent detection for comparing baseline vs RkCNN SAEs.

A "revived" latent is one that is dead in the baseline SAE
but alive in the RkCNN-initialized SAE. These are especially
interesting for understanding what RkCNN initialization captures.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch
from collections import Counter

from rkcnn_sae.interpretability.activation_store import TokenAwareActivationStore
from rkcnn_sae.interpretability.top_activations import TopActivationFinder, TopActivation


@dataclass
class RevivedLatentInfo:
    """Information about a latent that was revived by RkCNN initialization."""

    latent_idx: int  # Index in the SAE

    # Baseline SAE stats
    baseline_max_activation: float
    baseline_activity_rate: float

    # RkCNN SAE stats
    rkcnn_max_activation: float
    rkcnn_activity_rate: float

    # Top activations from RkCNN
    top_tokens: List[TopActivation]

    # Interpretation hints
    common_token_texts: List[Tuple[str, int]]  # (token, count)

    @property
    def improvement_factor(self) -> float:
        """How much more active is RkCNN vs baseline."""
        if self.baseline_max_activation <= 0:
            return float('inf') if self.rkcnn_max_activation > 0 else 0
        return self.rkcnn_max_activation / self.baseline_max_activation


class RevivedLatentDetector:
    """
    Detect latents that are "revived" by RkCNN initialization.

    Parameters
    ----------
    token_store : TokenAwareActivationStore
        Token-aware activation store.
    baseline_sae : SparseAutoencoder
        Baseline (randomly initialized) SAE.
    rkcnn_sae : SparseAutoencoder
        RkCNN-initialized SAE.
    device : str
        Device for computation.
    """

    def __init__(
        self,
        token_store: TokenAwareActivationStore,
        baseline_sae,
        rkcnn_sae,
        device: str = "cpu",
    ):
        self.token_store = token_store
        self.baseline_sae = baseline_sae.to(device)
        self.rkcnn_sae = rkcnn_sae.to(device)
        self.device = device

        # Compute latent activations for both SAEs
        print("Computing baseline SAE latent activations...")
        self.baseline_latents = token_store.compute_latent_activations(
            baseline_sae, device=device
        )

        print("Computing RkCNN SAE latent activations...")
        self.rkcnn_latents = token_store.compute_latent_activations(
            rkcnn_sae, device=device
        )

        # Create finders for both
        self.baseline_finder = TopActivationFinder(token_store, self.baseline_latents)
        self.rkcnn_finder = TopActivationFinder(token_store, self.rkcnn_latents)

        self.n_latents = self.baseline_latents.shape[1]

    def find_revived_latents(
        self,
        dead_threshold: float = 0.0,
        alive_threshold: float = 0.0,
        min_activity_rate: float = 0.001,
    ) -> List[RevivedLatentInfo]:
        """
        Find latents that are dead in baseline but alive in RkCNN.

        Parameters
        ----------
        dead_threshold : float
            Max activation threshold for considering a latent "dead".
        alive_threshold : float
            Min activation threshold for considering a latent "alive".
        min_activity_rate : float
            Minimum activity rate for a latent to be considered alive.

        Returns
        -------
        revived : List[RevivedLatentInfo]
            List of revived latent information, sorted by RkCNN activity.
        """
        revived = []

        for latent_idx in range(self.n_latents):
            baseline_max = self.baseline_finder.get_max_activation(latent_idx)
            baseline_rate = self.baseline_finder.get_activity_rate(latent_idx, dead_threshold)

            rkcnn_max = self.rkcnn_finder.get_max_activation(latent_idx)
            rkcnn_rate = self.rkcnn_finder.get_activity_rate(latent_idx, alive_threshold)

            # Check if revived: dead in baseline, alive in RkCNN
            is_baseline_dead = baseline_max <= dead_threshold
            is_rkcnn_alive = rkcnn_max > alive_threshold and rkcnn_rate >= min_activity_rate

            if is_baseline_dead and is_rkcnn_alive:
                # Get top activations
                top_acts = self.rkcnn_finder.get_top_k_for_latent(latent_idx, k=20)

                # Get common tokens
                common_tokens = Counter()
                for act in top_acts:
                    token_text = act.token_record.token_text.strip()
                    if token_text:  # Skip empty tokens
                        common_tokens[token_text] += 1

                revived.append(RevivedLatentInfo(
                    latent_idx=latent_idx,
                    baseline_max_activation=baseline_max,
                    baseline_activity_rate=baseline_rate,
                    rkcnn_max_activation=rkcnn_max,
                    rkcnn_activity_rate=rkcnn_rate,
                    top_tokens=top_acts,
                    common_token_texts=common_tokens.most_common(10),
                ))

        # Sort by RkCNN activity rate (most active first)
        revived.sort(key=lambda x: -x.rkcnn_activity_rate)

        return revived

    def find_killed_latents(
        self,
        dead_threshold: float = 0.0,
        alive_threshold: float = 0.0,
        min_activity_rate: float = 0.001,
    ) -> List[Tuple[int, float, float]]:
        """
        Find latents that are alive in baseline but dead in RkCNN.

        Returns
        -------
        killed : List[Tuple[latent_idx, baseline_max, rkcnn_max]]
        """
        killed = []

        for latent_idx in range(self.n_latents):
            baseline_max = self.baseline_finder.get_max_activation(latent_idx)
            baseline_rate = self.baseline_finder.get_activity_rate(latent_idx)

            rkcnn_max = self.rkcnn_finder.get_max_activation(latent_idx)
            rkcnn_rate = self.rkcnn_finder.get_activity_rate(latent_idx)

            is_baseline_alive = baseline_max > alive_threshold and baseline_rate >= min_activity_rate
            is_rkcnn_dead = rkcnn_max <= dead_threshold

            if is_baseline_alive and is_rkcnn_dead:
                killed.append((latent_idx, baseline_max, rkcnn_max))

        return killed

    def get_comparison_summary(
        self,
        dead_threshold: float = 0.0,
    ) -> dict:
        """
        Get summary statistics comparing baseline vs RkCNN.

        Returns
        -------
        summary : dict
            Summary statistics.
        """
        baseline_dead = sum(
            1 for i in range(self.n_latents)
            if self.baseline_finder.is_dead(i, dead_threshold)
        )
        rkcnn_dead = sum(
            1 for i in range(self.n_latents)
            if self.rkcnn_finder.is_dead(i, dead_threshold)
        )

        revived = self.find_revived_latents(dead_threshold=dead_threshold)
        killed = self.find_killed_latents(dead_threshold=dead_threshold)

        return {
            "n_latents": self.n_latents,
            "baseline_dead": baseline_dead,
            "baseline_dead_rate": baseline_dead / self.n_latents,
            "rkcnn_dead": rkcnn_dead,
            "rkcnn_dead_rate": rkcnn_dead / self.n_latents,
            "n_revived": len(revived),
            "n_killed": len(killed),
            "net_improvement": len(revived) - len(killed),
            "dead_latent_reduction": baseline_dead - rkcnn_dead,
        }

    def format_revived_report(
        self,
        n_examples: int = 10,
        top_k_per_latent: int = 5,
    ) -> str:
        """
        Generate a human-readable report of revived latents.

        Parameters
        ----------
        n_examples : int
            Number of revived latents to show.
        top_k_per_latent : int
            Number of top tokens per latent.

        Returns
        -------
        report : str
            Formatted report.
        """
        summary = self.get_comparison_summary()
        revived = self.find_revived_latents()[:n_examples]

        lines = [
            "=" * 60,
            "REVIVED LATENT ANALYSIS",
            "=" * 60,
            "",
            f"Total latents: {summary['n_latents']:,}",
            f"Baseline dead: {summary['baseline_dead']:,} ({summary['baseline_dead_rate']:.1%})",
            f"RkCNN dead: {summary['rkcnn_dead']:,} ({summary['rkcnn_dead_rate']:.1%})",
            f"",
            f"Latents revived: {summary['n_revived']:,}",
            f"Latents killed: {summary['n_killed']:,}",
            f"Net improvement: {summary['net_improvement']:+,}",
            "",
            "=" * 60,
            f"TOP {n_examples} REVIVED LATENTS",
            "=" * 60,
        ]

        for i, info in enumerate(revived):
            lines.extend([
                "",
                f"{'─' * 50}",
                f"Revived Latent #{info.latent_idx}",
                f"{'─' * 50}",
                f"Baseline max: {info.baseline_max_activation:.4f} (dead)",
                f"RkCNN max: {info.rkcnn_max_activation:.4f}",
                f"RkCNN activity rate: {info.rkcnn_activity_rate:.2%}",
                "",
                "Top activating tokens:",
            ])

            for j, act in enumerate(info.top_tokens[:top_k_per_latent]):
                context = act.token_record.get_highlighted_context(30)
                lines.append(
                    f"  {j+1}. \"{act.token_record.token_text}\" ({act.activation_value:.2f})"
                )
                lines.append(f"     {context}")

            if info.common_token_texts:
                common = ", ".join(f'"{t}" ({c}x)' for t, c in info.common_token_texts[:5])
                lines.extend([
                    "",
                    f"Common tokens: {common}",
                ])

        return "\n".join(lines)
