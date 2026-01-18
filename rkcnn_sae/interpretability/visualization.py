"""
Visualization tools for interpretability analysis.

Generates plots and reports comparing baseline vs RkCNN SAE interpretability.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json

import matplotlib.pyplot as plt
import numpy as np

from rkcnn_sae.interpretability.metrics import LatentInterpretabilityScore
from rkcnn_sae.interpretability.revived_detector import RevivedLatentInfo


def plot_revived_analysis(
    revived_latents: List[RevivedLatentInfo],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """
    Plot analysis of revived latents.

    Parameters
    ----------
    revived_latents : List[RevivedLatentInfo]
        List of revived latent information.
    save_path : Optional[str]
        Path to save the figure.
    figsize : Tuple[int, int]
        Figure size.

    Returns
    -------
    fig : plt.Figure
        The matplotlib figure.
    """
    if not revived_latents:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, "No revived latents found",
                ha='center', va='center', fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Distribution of RkCNN max activations for revived latents
    ax = axes[0, 0]
    max_acts = [r.rkcnn_max_activation for r in revived_latents]
    ax.hist(max_acts, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax.set_xlabel('RkCNN Max Activation')
    ax.set_ylabel('Count')
    ax.set_title(f'Revived Latent Max Activations (n={len(revived_latents)})')
    ax.axvline(np.median(max_acts), color='red', linestyle='--', label=f'Median: {np.median(max_acts):.2f}')
    ax.legend()

    # 2. Distribution of activity rates
    ax = axes[0, 1]
    activity_rates = [r.rkcnn_activity_rate * 100 for r in revived_latents]
    ax.hist(activity_rates, bins=30, color='forestgreen', edgecolor='white', alpha=0.8)
    ax.set_xlabel('RkCNN Activity Rate (%)')
    ax.set_ylabel('Count')
    ax.set_title('Activity Rates of Revived Latents')
    ax.axvline(np.median(activity_rates), color='red', linestyle='--', label=f'Median: {np.median(activity_rates):.2f}%')
    ax.legend()

    # 3. Top 10 most active revived latents
    ax = axes[1, 0]
    top_10 = sorted(revived_latents, key=lambda x: -x.rkcnn_activity_rate)[:10]
    indices = [r.latent_idx for r in top_10]
    rates = [r.rkcnn_activity_rate * 100 for r in top_10]
    ax.barh(range(len(indices)), rates, color='coral', edgecolor='white')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([f'Latent {i}' for i in indices])
    ax.set_xlabel('Activity Rate (%)')
    ax.set_title('Top 10 Most Active Revived Latents')
    ax.invert_yaxis()

    # 4. Common tokens across revived latents
    ax = axes[1, 1]
    token_counts = {}
    for r in revived_latents:
        for token, count in r.common_token_texts[:5]:
            token = token.strip()
            if token:
                token_counts[token] = token_counts.get(token, 0) + count

    if token_counts:
        sorted_tokens = sorted(token_counts.items(), key=lambda x: -x[1])[:15]
        tokens = [t[0] for t in sorted_tokens]
        counts = [t[1] for t in sorted_tokens]
        ax.barh(range(len(tokens)), counts, color='mediumpurple', edgecolor='white')
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels([f'"{t}"' for t in tokens])
        ax.set_xlabel('Frequency in Top Activations')
        ax.set_title('Most Common Tokens in Revived Latents')
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "No token data available",
                ha='center', va='center', fontsize=12)
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved revived analysis plot to {save_path}")

    return fig


def plot_interpretability_comparison(
    baseline_scores: List[LatentInterpretabilityScore],
    rkcnn_scores: List[LatentInterpretabilityScore],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """
    Plot comparison of interpretability metrics between SAEs.

    Parameters
    ----------
    baseline_scores : List[LatentInterpretabilityScore]
        Scores from baseline SAE.
    rkcnn_scores : List[LatentInterpretabilityScore]
        Scores from RkCNN SAE.
    save_path : Optional[str]
        Path to save the figure.
    figsize : Tuple[int, int]
        Figure size.

    Returns
    -------
    fig : plt.Figure
        The matplotlib figure.
    """
    # Filter to alive latents
    baseline_alive = [s for s in baseline_scores if not s.is_dead]
    rkcnn_alive = [s for s in rkcnn_scores if not s.is_dead]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Entropy comparison
    ax = axes[0, 0]
    baseline_entropy = [s.activation_entropy for s in baseline_alive]
    rkcnn_entropy = [s.activation_entropy for s in rkcnn_alive]

    bins = np.linspace(0, max(max(baseline_entropy) if baseline_entropy else 1,
                              max(rkcnn_entropy) if rkcnn_entropy else 1), 30)
    ax.hist(baseline_entropy, bins=bins, alpha=0.6, label='Baseline', color='blue')
    ax.hist(rkcnn_entropy, bins=bins, alpha=0.6, label='RkCNN', color='orange')
    ax.set_xlabel('Activation Entropy')
    ax.set_ylabel('Count')
    ax.set_title('Entropy Distribution (lower = more monosemantic)')
    ax.legend()

    # 2. Top-K concentration comparison
    ax = axes[0, 1]
    baseline_conc = [s.top_k_concentration for s in baseline_alive]
    rkcnn_conc = [s.top_k_concentration for s in rkcnn_alive]

    ax.hist(baseline_conc, bins=30, alpha=0.6, label='Baseline', color='blue')
    ax.hist(rkcnn_conc, bins=30, alpha=0.6, label='RkCNN', color='orange')
    ax.set_xlabel('Top-K Concentration')
    ax.set_ylabel('Count')
    ax.set_title('Concentration Distribution (higher = more focused)')
    ax.legend()

    # 3. Monosemanticity score comparison
    ax = axes[1, 0]
    baseline_mono = [s.monosemanticity_score for s in baseline_alive]
    rkcnn_mono = [s.monosemanticity_score for s in rkcnn_alive]

    ax.hist(baseline_mono, bins=30, alpha=0.6, label='Baseline', color='blue')
    ax.hist(rkcnn_mono, bins=30, alpha=0.6, label='RkCNN', color='orange')
    ax.set_xlabel('Monosemanticity Score')
    ax.set_ylabel('Count')
    ax.set_title('Monosemanticity Distribution (higher = better)')
    ax.legend()

    # 4. Summary bar chart
    ax = axes[1, 1]

    def safe_mean(vals):
        return np.mean(vals) if vals else 0

    metrics = ['Entropy\n(lower=better)', 'Concentration\n(higher=better)',
               'Monosemanticity\n(higher=better)']
    baseline_vals = [
        safe_mean(baseline_entropy),
        safe_mean(baseline_conc),
        safe_mean(baseline_mono),
    ]
    rkcnn_vals = [
        safe_mean(rkcnn_entropy),
        safe_mean(rkcnn_conc),
        safe_mean(rkcnn_mono),
    ]

    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, baseline_vals, width, label='Baseline', color='blue', alpha=0.7)
    ax.bar(x + width/2, rkcnn_vals, width, label='RkCNN', color='orange', alpha=0.7)
    ax.set_ylabel('Mean Value')
    ax.set_title('Average Metric Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    # Add value labels
    for i, (bv, rv) in enumerate(zip(baseline_vals, rkcnn_vals)):
        ax.text(i - width/2, bv + 0.01, f'{bv:.2f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, rv + 0.01, f'{rv:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved interpretability comparison plot to {save_path}")

    return fig


def generate_analysis_report(
    summary: Dict,
    comparison: Dict,
    revived_latents: List[RevivedLatentInfo],
    output_path: str,
    top_n_examples: int = 5,
) -> str:
    """
    Generate a comprehensive text report.

    Parameters
    ----------
    summary : Dict
        Summary statistics from RevivedLatentDetector.
    comparison : Dict
        Comparison from InterpretabilityMetrics.
    revived_latents : List[RevivedLatentInfo]
        Revived latent information.
    output_path : str
        Path to save the report.
    top_n_examples : int
        Number of example latents to include.

    Returns
    -------
    report : str
        The full report text.
    """
    lines = [
        "=" * 70,
        "INTERPRETABILITY ANALYSIS REPORT",
        "RkCNN-Initialized SAE vs Baseline SAE",
        "=" * 70,
        "",
        "EXECUTIVE SUMMARY",
        "-" * 40,
        f"Total latents analyzed: {summary['n_latents']:,}",
        "",
        "Dead Latent Comparison:",
        f"  Baseline: {summary['baseline_dead']:,} ({summary['baseline_dead_rate']:.1%})",
        f"  RkCNN:    {summary['rkcnn_dead']:,} ({summary['rkcnn_dead_rate']:.1%})",
        f"  Reduction: {summary['dead_latent_reduction']:,} latents ({summary['dead_latent_reduction']/summary['n_latents']:.1%})",
        "",
        "Latent Revival Statistics:",
        f"  Latents revived (dead->alive): {summary['n_revived']:,}",
        f"  Latents killed (alive->dead):  {summary['n_killed']:,}",
        f"  Net improvement:               {summary['net_improvement']:+,}",
        "",
    ]

    # Add interpretability comparison
    if comparison:
        lines.extend([
            "",
            "INTERPRETABILITY METRICS",
            "-" * 40,
            "",
            "Alive Latent Statistics:",
            f"  Baseline alive: {comparison['baseline']['n_alive']:,}",
            f"  RkCNN alive:    {comparison['rkcnn']['n_alive']:,}",
            "",
            "Mean Activation Entropy (lower = more monosemantic):",
            f"  Baseline: {comparison['baseline']['mean_entropy']:.4f}",
            f"  RkCNN:    {comparison['rkcnn']['mean_entropy']:.4f}",
            f"  Diff:     {comparison['differences']['entropy_diff']:+.4f}",
            "",
            "Mean Top-K Concentration (higher = more focused):",
            f"  Baseline: {comparison['baseline']['mean_concentration']:.4f}",
            f"  RkCNN:    {comparison['rkcnn']['mean_concentration']:.4f}",
            f"  Diff:     {comparison['differences']['concentration_diff']:+.4f}",
            "",
            "Mean Monosemanticity Score (higher = better):",
            f"  Baseline: {comparison['baseline']['mean_monosemanticity']:.4f}",
            f"  RkCNN:    {comparison['rkcnn']['mean_monosemanticity']:.4f}",
            f"  Diff:     {comparison['differences']['monosemanticity_diff']:+.4f}",
        ])

    # Add revived latent examples
    if revived_latents:
        lines.extend([
            "",
            "",
            "REVIVED LATENT EXAMPLES",
            "-" * 40,
            "",
            f"Showing top {min(top_n_examples, len(revived_latents))} revived latents:",
        ])

        for i, info in enumerate(revived_latents[:top_n_examples]):
            lines.extend([
                "",
                f"{'━' * 50}",
                f"Revived Latent #{info.latent_idx}",
                f"{'━' * 50}",
                f"Baseline max: {info.baseline_max_activation:.4f} (dead)",
                f"RkCNN max: {info.rkcnn_max_activation:.4f}",
                f"RkCNN activity rate: {info.rkcnn_activity_rate:.2%}",
                "",
                "Top activating tokens:",
            ])

            for j, act in enumerate(info.top_tokens[:5]):
                context = act.token_record.get_highlighted_context(30)
                lines.append(f"  {j+1}. \"{act.token_record.token_text}\" ({act.activation_value:.2f})")
                lines.append(f"     {context}")

            if info.common_token_texts:
                common = ", ".join(f'"{t}" ({c}x)' for t, c in info.common_token_texts[:5])
                lines.extend([
                    "",
                    f"Common tokens: {common}",
                ])

    # Footer
    lines.extend([
        "",
        "",
        "=" * 70,
        "END OF REPORT",
        "=" * 70,
    ])

    report = "\n".join(lines)

    # Save report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
    print(f"Saved analysis report to {output_path}")

    return report


def save_results_json(
    summary: Dict,
    comparison: Dict,
    revived_latents: List[RevivedLatentInfo],
    output_path: str,
):
    """
    Save all results to JSON for later analysis.

    Parameters
    ----------
    summary : Dict
        Summary statistics.
    comparison : Dict
        Interpretability comparison.
    revived_latents : List[RevivedLatentInfo]
        Revived latent information.
    output_path : str
        Path to save JSON.
    """
    # Convert revived latents to serializable format
    revived_data = []
    for info in revived_latents:
        revived_data.append({
            "latent_idx": info.latent_idx,
            "baseline_max_activation": info.baseline_max_activation,
            "baseline_activity_rate": info.baseline_activity_rate,
            "rkcnn_max_activation": info.rkcnn_max_activation,
            "rkcnn_activity_rate": info.rkcnn_activity_rate,
            "common_token_texts": info.common_token_texts,
            "top_tokens": [
                {
                    "token_text": act.token_record.token_text,
                    "activation_value": act.activation_value,
                    "context": act.token_record.get_highlighted_context(50),
                }
                for act in info.top_tokens[:10]
            ],
        })

    results = {
        "summary": summary,
        "comparison": comparison,
        "revived_latents": revived_data,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results JSON to {output_path}")
