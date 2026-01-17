#!/usr/bin/env python3
"""
Multi-layer analysis: Run RkCNN-SAE experiments across all GPT-2 layers.

This script compares baseline SAE vs RkCNN-SAE at each layer to identify:
1. Which layers benefit most from RkCNN initialization
2. How dead latent rates vary across layers
3. Optimal layer for feature extraction
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# Import from main experiment
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rkcnn_sae.models.sae import SAEConfig, SparseAutoencoder, SAETrainer
from rkcnn_sae.models.rkcnn_sae import RkCNNSAEConfig, RkCNNSparseAutoencoder
from rkcnn_sae.evaluation.metrics import evaluate_sae
from rkcnn_sae.data.activation_cache import ActivationCache, ActivationDataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-layer GPT-2 SAE analysis")

    # Model settings
    parser.add_argument("--model-name", type=str, default="gpt2")
    parser.add_argument("--layers", type=str, default="all",
                        help="Layers to analyze: 'all' or comma-separated (e.g., '0,3,6,9,11')")
    parser.add_argument("--hook-point", type=str, default="mlp_out")

    # Data settings
    parser.add_argument("--max-tokens", type=int, default=100000,
                        help="Tokens per layer (reduced for speed)")
    parser.add_argument("--use-synthetic", action="store_true",
                        help="Use synthetic data (for testing)")

    # SAE settings
    parser.add_argument("--expansion-factor", type=int, default=8)
    parser.add_argument("--l1-coefficient", type=float, default=1e-3)
    parser.add_argument("--n-train-steps", type=int, default=5000,
                        help="Training steps per layer (reduced for multi-layer)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)

    # RkCNN settings
    parser.add_argument("--rkcnn-fraction", type=float, default=0.5)
    parser.add_argument("--rkcnn-n-subsets", type=int, default=600)
    parser.add_argument("--rkcnn-score-method", type=str, default="kurtosis",
                        choices=["knn", "kurtosis", "variance_ratio"])

    # Execution
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="./results/multi_layer")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def get_layers(args, n_layers: int = 12) -> List[int]:
    """Parse layer specification."""
    if args.layers == "all":
        return list(range(n_layers))
    else:
        return [int(x.strip()) for x in args.layers.split(",")]


def run_layer_experiment(
    layer: int,
    args,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run baseline vs RkCNN SAE experiment for a single layer."""

    if verbose:
        print(f"\n{'='*60}")
        print(f"Layer {layer}")
        print(f"{'='*60}")

    # Determine activation dimension based on hook point
    # For GPT-2: d_model=768, d_mlp=3072
    if args.hook_point == "mlp_out":
        expected_d_model = 3072  # MLP output dimension
    else:
        expected_d_model = 768  # Residual stream dimension

    # Get activations
    if args.use_synthetic:
        from rkcnn_sae.data.activation_cache import create_synthetic_gpt2_activations
        activations = create_synthetic_gpt2_activations(
            n_samples=args.max_tokens,
            d_model=expected_d_model,
            seed=args.seed + layer,
        )
        d_model = activations.shape[1]
    else:
        try:
            cache = ActivationCache(
                model_name=args.model_name,
                layer=layer,
                hook_point=args.hook_point,
                device=args.device,
            )
            activations = cache.cache_dataset(
                max_tokens=args.max_tokens,
                show_progress=verbose,
            )
            d_model = cache.activation_dim

            # Verify dimension matches expected
            actual_dim = activations.shape[1]
            if actual_dim != expected_d_model:
                if verbose:
                    print(f"  Warning: Activation dimension mismatch: got {actual_dim}, expected {expected_d_model}")
                    print(f"  Falling back to synthetic data...")
                from rkcnn_sae.data.activation_cache import create_synthetic_gpt2_activations
                activations = create_synthetic_gpt2_activations(
                    n_samples=args.max_tokens,
                    d_model=expected_d_model,
                    seed=args.seed + layer,
                )
                d_model = activations.shape[1]
        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to load real data: {e}")
                print(f"  Falling back to synthetic data...")
            from rkcnn_sae.data.activation_cache import create_synthetic_gpt2_activations
            activations = create_synthetic_gpt2_activations(
                n_samples=args.max_tokens,
                d_model=expected_d_model,
                seed=args.seed + layer,
            )
            d_model = activations.shape[1]

    if verbose:
        print(f"  Activations: {activations.shape}")

    n_latents = d_model * args.expansion_factor

    # Split data
    n_train = int(len(activations) * 0.9)
    train_acts = activations[:n_train]
    eval_acts = activations[n_train:]

    train_loader = ActivationDataLoader(train_acts, batch_size=args.batch_size, device=args.device)
    eval_loader = ActivationDataLoader(eval_acts, batch_size=args.batch_size, shuffle=False, device=args.device)

    results = {"layer": layer, "d_model": d_model, "n_latents": n_latents}

    # ========== Baseline SAE ==========
    if verbose:
        print(f"  Training baseline SAE...")

    baseline_config = SAEConfig(
        d_model=d_model,
        n_latents=n_latents,
        l1_coefficient=args.l1_coefficient,
    )
    baseline_sae = SparseAutoencoder(baseline_config)
    baseline_trainer = SAETrainer(baseline_sae, lr=args.lr, device=args.device)

    for step in range(args.n_train_steps):
        for batch in train_loader:
            baseline_trainer.train_step(batch)
            break  # One batch per step

    # Evaluate baseline
    _, baseline_latents, baseline_recons, baseline_inputs = baseline_trainer.evaluate(eval_loader)
    baseline_metrics = evaluate_sae(baseline_inputs, baseline_latents, baseline_recons)

    results["baseline"] = {
        "dead_latent_rate": baseline_metrics.dead_latent_rate,
        "n_dead_latents": baseline_metrics.n_dead_latents,
        "l0_sparsity": baseline_metrics.l0_sparsity,
        "reconstruction_loss": baseline_metrics.reconstruction_loss,
        "explained_variance": baseline_metrics.explained_variance,
    }

    # ========== RkCNN SAE ==========
    if verbose:
        print(f"  Training RkCNN SAE...")

    rkcnn_config = RkCNNSAEConfig(
        d_model=d_model,
        n_latents=n_latents,
        l1_coefficient=args.l1_coefficient,
        rkcnn_directions_fraction=args.rkcnn_fraction,
        rkcnn_n_subsets=args.rkcnn_n_subsets,
        rkcnn_score_method=args.rkcnn_score_method,
    )
    rkcnn_sae = RkCNNSparseAutoencoder(rkcnn_config)

    # Initialize with RkCNN
    n_init = rkcnn_sae.initialize_with_rkcnn(
        train_acts[:10000].to(args.device),
        seed=args.seed,
        show_progress=False,
    )

    rkcnn_trainer = SAETrainer(rkcnn_sae, lr=args.lr, device=args.device)

    for step in range(args.n_train_steps):
        for batch in train_loader:
            rkcnn_trainer.train_step(batch)
            break

    # Evaluate RkCNN
    _, rkcnn_latents, rkcnn_recons, rkcnn_inputs = rkcnn_trainer.evaluate(eval_loader)
    rkcnn_metrics = evaluate_sae(rkcnn_inputs, rkcnn_latents, rkcnn_recons)

    results["rkcnn"] = {
        "dead_latent_rate": rkcnn_metrics.dead_latent_rate,
        "n_dead_latents": rkcnn_metrics.n_dead_latents,
        "l0_sparsity": rkcnn_metrics.l0_sparsity,
        "reconstruction_loss": rkcnn_metrics.reconstruction_loss,
        "explained_variance": rkcnn_metrics.explained_variance,
        "n_rkcnn_initialized": n_init,
    }

    # Compute improvement
    results["improvement"] = {
        "dead_latent_reduction": (
            (baseline_metrics.n_dead_latents - rkcnn_metrics.n_dead_latents)
            / max(baseline_metrics.n_dead_latents, 1) * 100
        ),
        "dead_latent_rate_diff": baseline_metrics.dead_latent_rate - rkcnn_metrics.dead_latent_rate,
        "l0_diff": baseline_metrics.l0_sparsity - rkcnn_metrics.l0_sparsity,
        "recon_diff": baseline_metrics.reconstruction_loss - rkcnn_metrics.reconstruction_loss,
    }

    if verbose:
        print(f"  Baseline: {baseline_metrics.n_dead_latents} dead ({baseline_metrics.dead_latent_rate*100:.2f}%)")
        print(f"  RkCNN:    {rkcnn_metrics.n_dead_latents} dead ({rkcnn_metrics.dead_latent_rate*100:.2f}%)")
        print(f"  Improvement: {results['improvement']['dead_latent_reduction']:.1f}% fewer dead latents")

    return results


def create_visualizations(all_results: List[Dict], output_dir: Path):
    """Create visualization plots for multi-layer analysis."""

    layers = [r["layer"] for r in all_results]
    baseline_dead = [r["baseline"]["dead_latent_rate"] * 100 for r in all_results]
    rkcnn_dead = [r["rkcnn"]["dead_latent_rate"] * 100 for r in all_results]
    improvement = [r["improvement"]["dead_latent_reduction"] for r in all_results]

    baseline_l0 = [r["baseline"]["l0_sparsity"] for r in all_results]
    rkcnn_l0 = [r["rkcnn"]["l0_sparsity"] for r in all_results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Dead latent rate by layer
    ax1 = axes[0, 0]
    x = np.arange(len(layers))
    width = 0.35
    ax1.bar(x - width/2, baseline_dead, width, label='Baseline', color='#ff6b6b', alpha=0.8)
    ax1.bar(x + width/2, rkcnn_dead, width, label='RkCNN', color='#4ecdc4', alpha=0.8)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Dead Latent Rate (%)')
    ax1.set_title('Dead Latent Rate by Layer')
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Improvement by layer
    ax2 = axes[0, 1]
    colors = ['#4ecdc4' if v > 0 else '#ff6b6b' for v in improvement]
    ax2.bar(layers, improvement, color=colors, alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Dead Latent Reduction (%)')
    ax2.set_title('RkCNN Improvement by Layer')
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: L0 Sparsity by layer
    ax3 = axes[1, 0]
    ax3.plot(layers, baseline_l0, 'o-', label='Baseline', color='#ff6b6b', markersize=8)
    ax3.plot(layers, rkcnn_l0, 's-', label='RkCNN', color='#4ecdc4', markersize=8)
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('L0 Sparsity')
    ax3.set_title('L0 Sparsity by Layer')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # Plot 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Find best layer
    best_layer_idx = np.argmax(improvement)
    best_layer = layers[best_layer_idx]

    summary_text = f"""
    MULTI-LAYER ANALYSIS SUMMARY
    ══════════════════════════════════════

    Layers analyzed: {len(layers)}

    Best layer for RkCNN: Layer {best_layer}
      • Dead latent reduction: {improvement[best_layer_idx]:.1f}%
      • Baseline dead rate: {baseline_dead[best_layer_idx]:.2f}%
      • RkCNN dead rate: {rkcnn_dead[best_layer_idx]:.2f}%

    Average improvement: {np.mean(improvement):.1f}%

    Layers with >0% improvement: {sum(1 for i in improvement if i > 0)}/{len(layers)}
    """

    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / "multi_layer_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nVisualization saved to: {output_dir / 'multi_layer_analysis.png'}")


def main():
    args = parse_args()

    print("=" * 60)
    print("Multi-Layer GPT-2 SAE Analysis")
    print("=" * 60)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get layers to analyze
    layers = get_layers(args)
    print(f"\nLayers to analyze: {layers}")
    print(f"Training steps per layer: {args.n_train_steps}")
    print(f"Tokens per layer: {args.max_tokens}")
    print(f"Device: {args.device}")

    # Run experiments
    all_results = []

    for layer in tqdm(layers, desc="Analyzing layers"):
        try:
            result = run_layer_experiment(layer, args, verbose=args.verbose)
            all_results.append(result)
        except Exception as e:
            print(f"\nError at layer {layer}: {e}")
            continue

    # Save results
    results_file = output_dir / "multi_layer_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": vars(args),
            "layers": all_results,
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Create visualizations
    if len(all_results) > 1:
        create_visualizations(all_results, output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\n{'Layer':<8} {'Baseline Dead':<15} {'RkCNN Dead':<15} {'Improvement':<12}")
    print("-" * 50)

    for r in all_results:
        print(f"{r['layer']:<8} {r['baseline']['dead_latent_rate']*100:>6.2f}%        "
              f"{r['rkcnn']['dead_latent_rate']*100:>6.2f}%        "
              f"{r['improvement']['dead_latent_reduction']:>+6.1f}%")

    # Find best layer
    if all_results:
        improvements = [r["improvement"]["dead_latent_reduction"] for r in all_results]
        best_idx = np.argmax(improvements)
        best_layer = all_results[best_idx]["layer"]

        print(f"\n✓ Best layer for RkCNN initialization: Layer {best_layer}")
        print(f"  Dead latent reduction: {improvements[best_idx]:.1f}%")


if __name__ == "__main__":
    main()
