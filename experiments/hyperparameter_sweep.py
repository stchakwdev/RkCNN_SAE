#!/usr/bin/env python3
"""
Hyperparameter sweep for RkCNN-SAE.

Sweeps over key parameters to find optimal configuration:
- rkcnn_n_subsets (h): Number of random subsets
- rkcnn_fraction: Fraction of latents to initialize with RkCNN
- rkcnn_score_method: Scoring method for subsets
- l1_coefficient: Sparsity penalty

Based on the implementation plan recommendations:
- m (subset size): √d_model as baseline
- r (top subsets): 200-400
- h (total subsets): 3r to 10r (600-2000)
"""

import argparse
import itertools
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rkcnn_sae.models.sae import SAEConfig, SparseAutoencoder, SAETrainer
from rkcnn_sae.models.rkcnn_sae import RkCNNSAEConfig, RkCNNSparseAutoencoder
from rkcnn_sae.evaluation.metrics import evaluate_sae


def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for RkCNN-SAE")

    # Model settings
    parser.add_argument("--layer", type=int, default=6,
                        help="GPT-2 layer to use (default: 6, best from multi-layer analysis)")
    parser.add_argument("--hook-point", type=str, default="mlp_out")

    # Data settings
    parser.add_argument("--max-tokens", type=int, default=50000,
                        help="Tokens to use (reduced for sweep speed)")
    parser.add_argument("--use-synthetic", action="store_true",
                        help="Use synthetic data for quick testing")

    # Fixed SAE settings
    parser.add_argument("--expansion-factor", type=int, default=8)
    parser.add_argument("--n-train-steps", type=int, default=3000,
                        help="Training steps per config (reduced for sweep)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)

    # Sweep settings
    parser.add_argument("--sweep-mode", type=str, default="quick",
                        choices=["quick", "full", "custom"],
                        help="Sweep mode: quick (few configs), full (all combos), custom")

    # Execution
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="./results/hyperparam_sweep")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def get_sweep_configs(mode: str) -> List[Dict[str, Any]]:
    """Generate hyperparameter configurations to sweep."""

    if mode == "quick":
        # Quick sweep: Test key variations
        configs = [
            # Baseline variations
            {"rkcnn_n_subsets": 600, "rkcnn_fraction": 0.5, "score_method": "kurtosis", "l1_coeff": 1e-3},
            # More subsets
            {"rkcnn_n_subsets": 1000, "rkcnn_fraction": 0.5, "score_method": "kurtosis", "l1_coeff": 1e-3},
            # Less subsets
            {"rkcnn_n_subsets": 300, "rkcnn_fraction": 0.5, "score_method": "kurtosis", "l1_coeff": 1e-3},
            # Higher fraction
            {"rkcnn_n_subsets": 600, "rkcnn_fraction": 0.75, "score_method": "kurtosis", "l1_coeff": 1e-3},
            # Lower fraction
            {"rkcnn_n_subsets": 600, "rkcnn_fraction": 0.25, "score_method": "kurtosis", "l1_coeff": 1e-3},
            # KNN scoring
            {"rkcnn_n_subsets": 600, "rkcnn_fraction": 0.5, "score_method": "knn", "l1_coeff": 1e-3},
            # Variance ratio scoring
            {"rkcnn_n_subsets": 600, "rkcnn_fraction": 0.5, "score_method": "variance_ratio", "l1_coeff": 1e-3},
            # Lower L1
            {"rkcnn_n_subsets": 600, "rkcnn_fraction": 0.5, "score_method": "kurtosis", "l1_coeff": 5e-4},
            # Higher L1
            {"rkcnn_n_subsets": 600, "rkcnn_fraction": 0.5, "score_method": "kurtosis", "l1_coeff": 5e-3},
        ]

    elif mode == "full":
        # Full grid sweep
        n_subsets_values = [300, 600, 1000, 1500]
        fraction_values = [0.25, 0.5, 0.75]
        score_methods = ["kurtosis", "knn", "variance_ratio"]
        l1_values = [5e-4, 1e-3, 5e-3]

        configs = []
        for n_sub, frac, method, l1 in itertools.product(
            n_subsets_values, fraction_values, score_methods, l1_values
        ):
            configs.append({
                "rkcnn_n_subsets": n_sub,
                "rkcnn_fraction": frac,
                "score_method": method,
                "l1_coeff": l1,
            })

    else:  # custom - focused on best from quick sweep
        configs = [
            # Best config variations
            {"rkcnn_n_subsets": 600, "rkcnn_fraction": 0.5, "score_method": "kurtosis", "l1_coeff": 1e-3},
            {"rkcnn_n_subsets": 800, "rkcnn_fraction": 0.5, "score_method": "kurtosis", "l1_coeff": 1e-3},
            {"rkcnn_n_subsets": 600, "rkcnn_fraction": 0.6, "score_method": "kurtosis", "l1_coeff": 1e-3},
            {"rkcnn_n_subsets": 600, "rkcnn_fraction": 0.5, "score_method": "kurtosis", "l1_coeff": 8e-4},
        ]

    return configs


def load_activations(args, verbose: bool = True) -> torch.Tensor:
    """Load or generate activations for the sweep."""

    # Determine dimension
    if args.hook_point == "mlp_out":
        d_model = 3072
    else:
        d_model = 768

    if args.use_synthetic:
        if verbose:
            print(f"Generating synthetic activations (d={d_model})...")
        # Generate synthetic data with similar statistics to real GPT-2
        n_samples = args.max_tokens
        activations = torch.randn(n_samples, d_model) * 0.5
        # Add some structure (sparse features)
        for i in range(d_model // 10):
            mask = torch.rand(n_samples) > 0.9
            activations[mask, i * 10:(i + 1) * 10] += torch.randn(mask.sum(), 10) * 2
        return activations
    else:
        if verbose:
            print(f"Loading real GPT-2 activations (layer {args.layer})...")

        try:
            from rkcnn_sae.data.activation_cache import ActivationCache

            cache = ActivationCache(
                model_name="gpt2",
                layer=args.layer,
                hook_point="mlp_out",  # Will use mlp.hook_post internally (3072d)
                device=args.device,
            )

            activations = cache.cache_dataset(
                dataset_name="wikitext",
                max_tokens=args.max_tokens,
            )

            if verbose:
                print(f"Loaded {activations.shape[0]} activation vectors")
            return activations

        except Exception as e:
            print(f"Failed to load real data: {e}")
            print("Falling back to synthetic data...")
            return load_activations(
                argparse.Namespace(**{**vars(args), 'use_synthetic': True}),
                verbose=verbose
            )


def run_single_config(
    config: Dict[str, Any],
    activations: torch.Tensor,
    args,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a single hyperparameter configuration."""

    d_model = activations.shape[1]
    n_latents = d_model * args.expansion_factor

    config_str = (
        f"h={config['rkcnn_n_subsets']}, "
        f"frac={config['rkcnn_fraction']}, "
        f"score={config['score_method']}, "
        f"l1={config['l1_coeff']}"
    )

    if verbose:
        print(f"\n  Config: {config_str}")

    # Train baseline SAE
    baseline_config = SAEConfig(
        d_model=d_model,
        n_latents=n_latents,
        l1_coefficient=config['l1_coeff'],
    )
    baseline_sae = SparseAutoencoder(baseline_config).to(args.device)
    baseline_trainer = SAETrainer(baseline_sae, lr=args.lr, device=args.device)

    activations_device = activations.to(args.device)
    n_samples = activations.shape[0]

    for step in range(args.n_train_steps):
        idx = torch.randint(0, n_samples, (args.batch_size,))
        batch = activations_device[idx]
        baseline_trainer.train_step(batch)

    # Forward pass to get latents and reconstructions (batched to avoid OOM)
    eval_batch_size = min(10000, n_samples)  # Use subset for evaluation to avoid OOM
    eval_idx = torch.randperm(n_samples)[:eval_batch_size]
    eval_activations = activations_device[eval_idx]

    with torch.no_grad():
        baseline_latents, baseline_recons, _ = baseline_sae(eval_activations)
    baseline_metrics = evaluate_sae(eval_activations, baseline_latents, baseline_recons)

    # Train RkCNN SAE
    rkcnn_config = RkCNNSAEConfig(
        d_model=d_model,
        n_latents=n_latents,
        l1_coefficient=config['l1_coeff'],
        rkcnn_directions_fraction=config['rkcnn_fraction'],
        rkcnn_n_subsets=config['rkcnn_n_subsets'],
        rkcnn_score_method=config['score_method'],
    )
    rkcnn_sae = RkCNNSparseAutoencoder(rkcnn_config).to(args.device)
    rkcnn_sae.initialize_with_rkcnn(activations_device, device=args.device)
    rkcnn_trainer = SAETrainer(rkcnn_sae, lr=args.lr, device=args.device)

    for step in range(args.n_train_steps):
        idx = torch.randint(0, n_samples, (args.batch_size,))
        batch = activations_device[idx]
        rkcnn_trainer.train_step(batch)

    # Forward pass to get latents and reconstructions (use same eval subset for fair comparison)
    with torch.no_grad():
        rkcnn_latents, rkcnn_recons, _ = rkcnn_sae(eval_activations)
    rkcnn_metrics = evaluate_sae(eval_activations, rkcnn_latents, rkcnn_recons)

    # Calculate improvement
    baseline_dead = baseline_metrics.n_dead_latents
    rkcnn_dead = rkcnn_metrics.n_dead_latents

    if baseline_dead > 0:
        improvement = (baseline_dead - rkcnn_dead) / baseline_dead * 100
    else:
        improvement = 0.0

    result = {
        "config": config,
        "baseline": {
            "dead_latent_rate": baseline_metrics.dead_latent_rate,
            "n_dead_latents": baseline_metrics.n_dead_latents,
            "l0_sparsity": baseline_metrics.l0_sparsity,
            "reconstruction_loss": baseline_metrics.reconstruction_loss,
            "explained_variance": baseline_metrics.explained_variance,
        },
        "rkcnn": {
            "dead_latent_rate": rkcnn_metrics.dead_latent_rate,
            "n_dead_latents": rkcnn_metrics.n_dead_latents,
            "l0_sparsity": rkcnn_metrics.l0_sparsity,
            "reconstruction_loss": rkcnn_metrics.reconstruction_loss,
            "explained_variance": rkcnn_metrics.explained_variance,
        },
        "improvement": {
            "dead_latent_reduction_pct": improvement,
            "dead_latent_diff": baseline_dead - rkcnn_dead,
            "explained_variance_diff": rkcnn_metrics.explained_variance - baseline_metrics.explained_variance,
        }
    }

    if verbose:
        print(f"    Baseline dead: {baseline_dead}, RkCNN dead: {rkcnn_dead}, Improvement: {improvement:.1f}%")

    return result


def create_sweep_plots(results: List[Dict], output_dir: str):
    """Create visualization of sweep results."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Extract data
    improvements = [r['improvement']['dead_latent_reduction_pct'] for r in results]
    n_subsets = [r['config']['rkcnn_n_subsets'] for r in results]
    fractions = [r['config']['rkcnn_fraction'] for r in results]
    l1_coeffs = [r['config']['l1_coeff'] for r in results]
    score_methods = [r['config']['score_method'] for r in results]

    # Plot 1: Improvement vs n_subsets
    ax1 = axes[0, 0]
    for method in ['kurtosis', 'knn', 'variance_ratio']:
        mask = [s == method for s in score_methods]
        if any(mask):
            x = [n for n, m in zip(n_subsets, mask) if m]
            y = [i for i, m in zip(improvements, mask) if m]
            ax1.scatter(x, y, label=method, alpha=0.7, s=50)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Number of Subsets (h)')
    ax1.set_ylabel('Dead Latent Reduction (%)')
    ax1.set_title('Impact of Subset Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Improvement vs fraction
    ax2 = axes[0, 1]
    for method in ['kurtosis', 'knn', 'variance_ratio']:
        mask = [s == method for s in score_methods]
        if any(mask):
            x = [f for f, m in zip(fractions, mask) if m]
            y = [i for i, m in zip(improvements, mask) if m]
            ax2.scatter(x, y, label=method, alpha=0.7, s=50)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('RkCNN Fraction')
    ax2.set_ylabel('Dead Latent Reduction (%)')
    ax2.set_title('Impact of Initialization Fraction')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Improvement by score method (box plot style)
    ax3 = axes[1, 0]
    method_improvements = {m: [] for m in ['kurtosis', 'knn', 'variance_ratio']}
    for r in results:
        method_improvements[r['config']['score_method']].append(
            r['improvement']['dead_latent_reduction_pct']
        )

    methods = list(method_improvements.keys())
    values = [method_improvements[m] for m in methods]

    positions = range(len(methods))
    for i, (method, vals) in enumerate(zip(methods, values)):
        if vals:
            ax3.bar(i, np.mean(vals), alpha=0.7, label=f'{method} (n={len(vals)})')
            ax3.scatter([i] * len(vals), vals, color='black', alpha=0.5, s=20)

    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xticks(positions)
    ax3.set_xticklabels(methods)
    ax3.set_xlabel('Score Method')
    ax3.set_ylabel('Dead Latent Reduction (%)')
    ax3.set_title('Comparison by Score Method')
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Improvement vs L1 coefficient
    ax4 = axes[1, 1]
    for method in ['kurtosis', 'knn', 'variance_ratio']:
        mask = [s == method for s in score_methods]
        if any(mask):
            x = [l for l, m in zip(l1_coeffs, mask) if m]
            y = [i for i, m in zip(improvements, mask) if m]
            ax4.scatter(x, y, label=method, alpha=0.7, s=50)
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xscale('log')
    ax4.set_xlabel('L1 Coefficient')
    ax4.set_ylabel('Dead Latent Reduction (%)')
    ax4.set_title('Impact of L1 Regularization')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hyperparam_sweep.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved sweep visualization to {output_dir}/hyperparam_sweep.png")


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("RkCNN-SAE Hyperparameter Sweep")
    print("=" * 60)
    print(f"Mode: {args.sweep_mode}")
    print(f"Layer: {args.layer}")
    print(f"Device: {args.device}")
    print(f"Synthetic data: {args.use_synthetic}")

    # Get configurations
    configs = get_sweep_configs(args.sweep_mode)
    print(f"\nTotal configurations to test: {len(configs)}")

    # Load activations once
    activations = load_activations(args, verbose=args.verbose)
    print(f"Activation shape: {activations.shape}")

    # Run sweep
    results = []
    for i, config in enumerate(tqdm(configs, desc="Sweeping configs")):
        print(f"\n[{i+1}/{len(configs)}]", end="")
        result = run_single_config(config, activations, args, verbose=args.verbose)
        results.append(result)

    # Find best configuration
    best_idx = np.argmax([r['improvement']['dead_latent_reduction_pct'] for r in results])
    best_result = results[best_idx]

    print("\n" + "=" * 60)
    print("SWEEP RESULTS")
    print("=" * 60)

    print("\nTop 5 configurations by dead latent reduction:")
    sorted_results = sorted(results, key=lambda x: x['improvement']['dead_latent_reduction_pct'], reverse=True)
    for i, r in enumerate(sorted_results[:5]):
        cfg = r['config']
        imp = r['improvement']['dead_latent_reduction_pct']
        print(f"  {i+1}. h={cfg['rkcnn_n_subsets']}, frac={cfg['rkcnn_fraction']}, "
              f"score={cfg['score_method']}, l1={cfg['l1_coeff']:.0e} → {imp:.1f}%")

    print(f"\nBest configuration:")
    print(f"  rkcnn_n_subsets: {best_result['config']['rkcnn_n_subsets']}")
    print(f"  rkcnn_fraction: {best_result['config']['rkcnn_fraction']}")
    print(f"  score_method: {best_result['config']['score_method']}")
    print(f"  l1_coefficient: {best_result['config']['l1_coeff']}")
    print(f"  Dead latent reduction: {best_result['improvement']['dead_latent_reduction_pct']:.1f}%")
    print(f"  Explained variance diff: {best_result['improvement']['explained_variance_diff']:.4f}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "n_configs": len(configs),
        "best_config": best_result['config'],
        "best_improvement": best_result['improvement'],
        "all_results": results,
    }

    output_file = os.path.join(args.output_dir, "hyperparam_sweep_results.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nSaved results to {output_file}")

    # Create visualizations
    create_sweep_plots(results, args.output_dir)

    print("\n" + "=" * 60)
    print("SWEEP COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
