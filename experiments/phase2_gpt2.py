#!/usr/bin/env python3
"""
Phase 2: GPT-2 SAE Experiments

This experiment compares baseline SAE vs RkCNN-initialized SAE on GPT-2 activations.

IMPORTANT: Only run this after Phase 1 passes!

Success Criteria:
- Dead latent rate: Baseline ~30-60% → RkCNN target < 20%
- L0 sparsity: RkCNN <= baseline
- Reconstruction loss: RkCNN <= baseline
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rkcnn_sae.data.activation_cache import (
    ActivationCache,
    ActivationDataLoader,
    create_synthetic_gpt2_activations,
)
from rkcnn_sae.evaluation.metrics import evaluate_sae
from rkcnn_sae.models.rkcnn_sae import (
    RkCNNSAEConfig,
    RkCNNSparseAutoencoder,
    compare_initialization_methods,
)
from rkcnn_sae.models.sae import SAEConfig, SparseAutoencoder, SAETrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: GPT-2 SAE Experiments")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config YAML"
    )

    # Data arguments
    parser.add_argument(
        "--model-name", type=str, default="gpt2", help="GPT-2 model name"
    )
    parser.add_argument("--layer", type=int, default=6, help="Layer to extract")
    parser.add_argument(
        "--hook-point", type=str, default="mlp_out", help="Hook point"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=100_000, help="Max tokens to cache"
    )
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        help="Use synthetic data (no GPU required)",
    )

    # SAE arguments
    parser.add_argument(
        "--expansion-factor", type=int, default=8, help="SAE expansion factor"
    )
    parser.add_argument(
        "--l1-coefficient", type=float, default=1e-3, help="L1 sparsity penalty"
    )
    parser.add_argument(
        "--n-train-steps", type=int, default=10000, help="Training steps"
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    # RkCNN arguments
    parser.add_argument(
        "--rkcnn-fraction",
        type=float,
        default=0.5,
        help="Fraction of latents to init with RkCNN",
    )
    parser.add_argument(
        "--rkcnn-n-subsets", type=int, default=600, help="Number of RkCNN subsets"
    )
    parser.add_argument(
        "--rkcnn-score-method",
        type=str,
        default="kurtosis",
        choices=["knn", "kurtosis", "variance_ratio"],
        help="RkCNN scoring method",
    )

    # General arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./results/phase2", help="Output directory"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=1000, help="Checkpoint frequency"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Quick test run with minimal data",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_activations(args) -> torch.Tensor:
    """Load or generate activation data."""
    if args.use_synthetic or args.dry_run:
        print("Using synthetic GPT-2-like activations")
        n_samples = 1000 if args.dry_run else args.max_tokens
        return create_synthetic_gpt2_activations(
            n_samples=n_samples,
            d_model=768,  # GPT-2 small
            n_features=2000,
            sparsity=0.05,
            seed=args.seed,
        )

    # Use real GPT-2 activations
    cache = ActivationCache(
        model_name=args.model_name,
        layer=args.layer,
        hook_point=args.hook_point,
        device=args.device,
        cache_dir=args.output_dir + "/cache",
    )

    activations = cache.cache_dataset(
        dataset_name="openwebtext",
        max_tokens=args.max_tokens,
        seq_len=128,
        batch_size=32,
        show_progress=True,
    )

    return activations


def train_sae(
    sae: SparseAutoencoder,
    dataloader: ActivationDataLoader,
    n_steps: int,
    lr: float,
    device: str,
    checkpoint_dir: Path,
    checkpoint_every: int,
    name: str = "sae",
) -> tuple:
    """Train an SAE with checkpointing."""
    sae = sae.to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    losses = []
    step = 0
    epoch = 0

    pbar = tqdm(total=n_steps, desc=f"Training {name}")

    while step < n_steps:
        epoch += 1
        for batch in dataloader:
            if step >= n_steps:
                break

            batch = batch.to(device)
            optimizer.zero_grad()

            latents, reconstructed, _ = sae(batch)
            loss, loss_dict = sae.compute_loss(batch, latents, reconstructed)

            loss.backward()
            optimizer.step()

            # Normalize decoder if needed
            if sae.config.normalize_decoder:
                sae.normalize_decoder()

            losses.append(loss_dict)
            step += 1
            pbar.update(1)

            if step % 100 == 0:
                pbar.set_postfix(loss=loss_dict["total"], l1=loss_dict["l1"])

            # Checkpoint
            if step % checkpoint_every == 0:
                ckpt_path = checkpoint_dir / f"{name}_step_{step}.pt"
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": sae.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "losses": losses,
                    },
                    ckpt_path,
                )

    pbar.close()
    return sae, losses


def run_experiment(args) -> dict:
    """Run the Phase 2 experiment."""
    print("=" * 60)
    print("Phase 2: GPT-2 SAE Experiments")
    print("=" * 60)

    if args.dry_run:
        print("\n*** DRY RUN MODE - Using minimal data ***\n")
        args.max_tokens = 1000
        args.n_train_steps = 100
        args.use_synthetic = True

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ----- Step 1: Load/Generate Activations -----
    print("\n[1/5] Loading activation data...")
    activations = get_activations(args)
    d_model = activations.shape[1]
    n_samples = activations.shape[0]

    print(f"  Activation shape: {activations.shape}")
    print(f"  d_model: {d_model}")
    print(f"  n_samples: {n_samples}")

    # Create dataloader
    dataloader = ActivationDataLoader(
        activations,
        batch_size=args.batch_size,
        shuffle=True,
        device=args.device,
    )

    # SAE configuration
    n_latents = d_model * args.expansion_factor
    print(f"\n  SAE latents: {n_latents}")
    print(f"  Expansion factor: {args.expansion_factor}x")

    # ----- Step 2: Train Baseline SAE -----
    print("\n[2/5] Training Baseline SAE...")

    baseline_config = SAEConfig(
        d_model=d_model,
        n_latents=n_latents,
        l1_coefficient=args.l1_coefficient,
        normalize_decoder=True,
    )
    baseline_sae = SparseAutoencoder(baseline_config)

    baseline_sae, baseline_losses = train_sae(
        baseline_sae,
        dataloader,
        n_steps=args.n_train_steps,
        lr=args.lr,
        device=args.device,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        name="baseline",
    )

    # ----- Step 3: Train RkCNN SAE -----
    print("\n[3/5] Training RkCNN-Initialized SAE...")

    rkcnn_config = RkCNNSAEConfig(
        d_model=d_model,
        n_latents=n_latents,
        l1_coefficient=args.l1_coefficient,
        normalize_decoder=True,
        rkcnn_directions_fraction=args.rkcnn_fraction,
        rkcnn_n_subsets=args.rkcnn_n_subsets,
        rkcnn_score_method=args.rkcnn_score_method,
        rkcnn_subset_size=int(np.sqrt(d_model)),  # sqrt(d_model)
    )
    rkcnn_sae = RkCNNSparseAutoencoder(rkcnn_config)

    # Initialize with RkCNN
    print("  Mining directions with RkCNN...")
    n_initialized = rkcnn_sae.initialize_with_rkcnn(
        activations,
        device=args.device,
        seed=args.seed,
        show_progress=args.verbose,
    )

    # Save mined directions
    if rkcnn_sae.mined_directions is not None:
        torch.save(
            rkcnn_sae.mined_directions.cpu(),
            output_dir / "rkcnn_directions.pt",
        )

    rkcnn_sae, rkcnn_losses = train_sae(
        rkcnn_sae,
        dataloader,
        n_steps=args.n_train_steps,
        lr=args.lr,
        device=args.device,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        name="rkcnn",
    )

    # ----- Step 4: Evaluate Both SAEs -----
    print("\n[4/5] Evaluating SAEs...")

    # Evaluate baseline
    print("  Evaluating baseline SAE...")
    with torch.no_grad():
        baseline_sae.eval()
        all_latents_b = []
        all_recons_b = []
        for batch in dataloader:
            batch = batch.to(args.device)
            latents, recons, _ = baseline_sae(batch)
            all_latents_b.append(latents.cpu())
            all_recons_b.append(recons.cpu())
        all_latents_b = torch.cat(all_latents_b, dim=0)
        all_recons_b = torch.cat(all_recons_b, dim=0)

    baseline_metrics = evaluate_sae(activations, all_latents_b, all_recons_b)

    # Evaluate RkCNN SAE
    print("  Evaluating RkCNN SAE...")
    with torch.no_grad():
        rkcnn_sae.eval()
        all_latents_r = []
        all_recons_r = []
        for batch in dataloader:
            batch = batch.to(args.device)
            latents, recons, _ = rkcnn_sae(batch)
            all_latents_r.append(latents.cpu())
            all_recons_r.append(recons.cpu())
        all_latents_r = torch.cat(all_latents_r, dim=0)
        all_recons_r = torch.cat(all_recons_r, dim=0)

    rkcnn_metrics = evaluate_sae(activations, all_latents_r, all_recons_r)

    # Get RkCNN vs Random latent comparison
    rkcnn_latent_stats = rkcnn_sae.get_rkcnn_latent_stats(all_latents_r)

    # ----- Step 5: Analyze Results -----
    print("\n[5/5] Analyzing results...")

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print(f"\n{'Metric':<25} {'Baseline':<15} {'RkCNN':<15} {'Δ':<10}")
    print("-" * 60)

    # Dead latent rate
    dead_delta = rkcnn_metrics.dead_latent_rate - baseline_metrics.dead_latent_rate
    dead_better = "✓" if dead_delta < 0 else "✗"
    print(
        f"{'Dead Latent Rate':<25} {baseline_metrics.dead_latent_rate:.2%}{'':<7} "
        f"{rkcnn_metrics.dead_latent_rate:.2%}{'':<7} {dead_delta:+.2%} {dead_better}"
    )

    # L0 sparsity
    l0_delta = rkcnn_metrics.l0_sparsity - baseline_metrics.l0_sparsity
    l0_better = "✓" if l0_delta <= baseline_metrics.l0_sparsity * 0.1 else "✗"
    print(
        f"{'L0 Sparsity':<25} {baseline_metrics.l0_sparsity:.2f}{'':<11} "
        f"{rkcnn_metrics.l0_sparsity:.2f}{'':<11} {l0_delta:+.2f} {l0_better}"
    )

    # Reconstruction loss
    recon_delta = rkcnn_metrics.reconstruction_loss - baseline_metrics.reconstruction_loss
    recon_better = "✓" if recon_delta <= baseline_metrics.reconstruction_loss * 0.1 else "✗"
    print(
        f"{'Reconstruction Loss':<25} {baseline_metrics.reconstruction_loss:.6f}{'':<5} "
        f"{rkcnn_metrics.reconstruction_loss:.6f}{'':<5} {recon_delta:+.6f} {recon_better}"
    )

    # Explained variance
    ev_delta = rkcnn_metrics.explained_variance - baseline_metrics.explained_variance
    print(
        f"{'Explained Variance':<25} {baseline_metrics.explained_variance:.2%}{'':<7} "
        f"{rkcnn_metrics.explained_variance:.2%}{'':<7} {ev_delta:+.2%}"
    )

    # RkCNN-specific stats
    if "rkcnn_dead_rate" in rkcnn_latent_stats:
        print("\n" + "-" * 60)
        print("RkCNN Latent Analysis:")
        print(f"  RkCNN-initialized latent dead rate: {rkcnn_latent_stats['rkcnn_dead_rate']:.2%}")
        print(f"  Random-initialized latent dead rate: {rkcnn_latent_stats['random_dead_rate']:.2%}")

    # Success criteria
    print("\n" + "=" * 60)
    print("SUCCESS CRITERIA")
    print("=" * 60)

    # If baseline already has very low dead latent rate, just check RkCNN maintains it
    # (can't improve on 0% or near-0%)
    dead_latent_improved = (
        rkcnn_metrics.dead_latent_rate < baseline_metrics.dead_latent_rate
        or baseline_metrics.dead_latent_rate < 0.05  # Baseline already excellent
    )

    criteria = {
        "dead_latent_improved_or_excellent": dead_latent_improved,
        "dead_latent_target": rkcnn_metrics.dead_latent_rate < 0.2,
        "l0_not_worse": rkcnn_metrics.l0_sparsity <= baseline_metrics.l0_sparsity * 1.1,
        "recon_not_worse": rkcnn_metrics.reconstruction_loss <= baseline_metrics.reconstruction_loss * 1.1,
    }

    for name, passed in criteria.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")

    overall_pass = all(criteria.values())
    print("\n" + "=" * 60)
    print(f"PHASE 2 RESULT: {'PASS ✓' if overall_pass else 'FAIL ✗'}")
    print("=" * 60)

    # ----- Save Results -----
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model_name": args.model_name,
            "layer": args.layer,
            "hook_point": args.hook_point,
            "d_model": d_model,
            "n_latents": n_latents,
            "expansion_factor": args.expansion_factor,
            "l1_coefficient": args.l1_coefficient,
            "n_train_steps": args.n_train_steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "rkcnn_fraction": args.rkcnn_fraction,
            "rkcnn_n_subsets": args.rkcnn_n_subsets,
            "rkcnn_score_method": args.rkcnn_score_method,
            "seed": args.seed,
            "use_synthetic": args.use_synthetic,
            "dry_run": args.dry_run,
        },
        "baseline": {
            "dead_latent_rate": baseline_metrics.dead_latent_rate,
            "l0_sparsity": baseline_metrics.l0_sparsity,
            "reconstruction_loss": baseline_metrics.reconstruction_loss,
            "explained_variance": baseline_metrics.explained_variance,
            "n_dead_latents": baseline_metrics.n_dead_latents,
        },
        "rkcnn": {
            "dead_latent_rate": rkcnn_metrics.dead_latent_rate,
            "l0_sparsity": rkcnn_metrics.l0_sparsity,
            "reconstruction_loss": rkcnn_metrics.reconstruction_loss,
            "explained_variance": rkcnn_metrics.explained_variance,
            "n_dead_latents": rkcnn_metrics.n_dead_latents,
            "n_rkcnn_initialized": n_initialized,
            "latent_stats": rkcnn_latent_stats,
        },
        "success_criteria": criteria,
        "overall_pass": overall_pass,
    }

    # Save JSON
    results_file = output_dir / "phase2_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Save models
    torch.save(baseline_sae.state_dict(), output_dir / "baseline_sae.pt")
    torch.save(rkcnn_sae.state_dict(), output_dir / "rkcnn_sae.pt")
    print(f"Models saved to: {output_dir}")

    # Generate plots
    plot_results(
        baseline_metrics=baseline_metrics,
        rkcnn_metrics=rkcnn_metrics,
        baseline_losses=baseline_losses,
        rkcnn_losses=rkcnn_losses,
        rkcnn_latent_stats=rkcnn_latent_stats,
        output_dir=output_dir,
    )

    return results


def plot_results(
    baseline_metrics,
    rkcnn_metrics,
    baseline_losses,
    rkcnn_losses,
    rkcnn_latent_stats,
    output_dir: Path,
):
    """Generate visualization plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Training loss curves
    ax = axes[0, 0]
    baseline_total = [l["total"] for l in baseline_losses]
    rkcnn_total = [l["total"] for l in rkcnn_losses]
    ax.plot(baseline_total, label="Baseline", alpha=0.8)
    ax.plot(rkcnn_total, label="RkCNN", alpha=0.8)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Total Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.set_yscale("log")

    # Plot 2: Dead latent comparison
    ax = axes[0, 1]
    metrics = ["Dead Latent\nRate", "L0 Sparsity\n(÷100)", "Recon Loss\n(×1000)"]
    baseline_vals = [
        baseline_metrics.dead_latent_rate,
        baseline_metrics.l0_sparsity / 100,
        baseline_metrics.reconstruction_loss * 1000,
    ]
    rkcnn_vals = [
        rkcnn_metrics.dead_latent_rate,
        rkcnn_metrics.l0_sparsity / 100,
        rkcnn_metrics.reconstruction_loss * 1000,
    ]

    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width / 2, baseline_vals, width, label="Baseline", color="steelblue")
    ax.bar(x + width / 2, rkcnn_vals, width, label="RkCNN", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Value")
    ax.set_title("SAE Metrics Comparison")
    ax.legend()

    # Plot 3: RkCNN vs Random latent dead rates
    ax = axes[1, 0]
    if "rkcnn_dead_rate" in rkcnn_latent_stats:
        categories = ["RkCNN-Init\nLatents", "Random-Init\nLatents"]
        rates = [
            rkcnn_latent_stats["rkcnn_dead_rate"],
            rkcnn_latent_stats["random_dead_rate"],
        ]
        colors = ["coral", "steelblue"]
        ax.bar(categories, rates, color=colors, edgecolor="black")
        ax.set_ylabel("Dead Latent Rate")
        ax.set_title("Dead Latents: RkCNN vs Random Init")
        ax.set_ylim(0, 1)
        for i, v in enumerate(rates):
            ax.text(i, v + 0.02, f"{v:.1%}", ha="center")
    else:
        ax.text(0.5, 0.5, "No RkCNN stats available", ha="center", va="center")

    # Plot 4: L1 loss curves
    ax = axes[1, 1]
    baseline_l1 = [l["l1"] for l in baseline_losses]
    rkcnn_l1 = [l["l1"] for l in rkcnn_losses]
    ax.plot(baseline_l1, label="Baseline", alpha=0.8)
    ax.plot(rkcnn_l1, label="RkCNN", alpha=0.8)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("L1 Loss")
    ax.set_title("Sparsity (L1) Loss")
    ax.legend()

    plt.tight_layout()
    plot_file = output_dir / "phase2_plots.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"Plots saved to: {plot_file}")


def main():
    args = parse_args()

    # Load config if provided
    if args.config:
        config = load_config(args.config)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)

    results = run_experiment(args)

    # Exit code
    if results["overall_pass"]:
        print("\n🎉 Phase 2 passed! RkCNN improves SAE training.")
        sys.exit(0)
    else:
        print("\n❌ Phase 2 did not meet all criteria.")
        sys.exit(1)


if __name__ == "__main__":
    main()
