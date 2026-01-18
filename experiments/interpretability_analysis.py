#!/usr/bin/env python3
"""
Feature Interpretability Analysis

Compares interpretability of baseline SAE vs RkCNN-initialized SAE latents.
Focuses on "revived" latents (dead in baseline, alive in RkCNN) to understand
what features RkCNN uniquely captures.

Key Questions:
1. How many latents does RkCNN "revive" (dead→alive)?
2. What do these revived latents represent (top-activating tokens)?
3. Are RkCNN latents more monosemantic/interpretable?
4. Do RkCNN-initialized latents perform better than randomly-initialized ones?

Usage:
    python experiments/interpretability_analysis.py \
        --layer 6 \
        --max-tokens 50000 \
        --device cuda \
        --output-dir ./results/interpretability
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Feature Interpretability Analysis: Baseline vs RkCNN SAE"
    )

    # Data arguments
    parser.add_argument(
        "--model-name", type=str, default="gpt2", help="GPT-2 model name"
    )
    parser.add_argument("--layer", type=int, default=6, help="Layer to analyze")
    parser.add_argument(
        "--hook-point", type=str, default="mlp_out", help="Hook point"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=50000, help="Max tokens for analysis"
    )
    parser.add_argument(
        "--dataset", type=str, default="wikitext", help="Dataset to use"
    )

    # SAE arguments
    parser.add_argument(
        "--expansion-factor", type=int, default=8, help="SAE expansion factor"
    )
    parser.add_argument(
        "--l1-coefficient", type=float, default=1e-3, help="L1 sparsity penalty"
    )
    parser.add_argument(
        "--n-train-steps", type=int, default=5000, help="Training steps"
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

    # Analysis arguments
    parser.add_argument(
        "--top-k-tokens", type=int, default=20, help="Top-K tokens per latent"
    )
    parser.add_argument(
        "--n-example-latents", type=int, default=10, help="Example latents in report"
    )

    # Load pre-trained SAEs instead of training
    parser.add_argument(
        "--baseline-sae-path", type=str, default=None, help="Path to pre-trained baseline SAE"
    )
    parser.add_argument(
        "--rkcnn-sae-path", type=str, default=None, help="Path to pre-trained RkCNN SAE"
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
        "--output-dir",
        type=str,
        default="./results/interpretability",
        help="Output directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Quick test run with minimal data",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    return parser.parse_args()


def run_analysis(args) -> dict:
    """Run the interpretability analysis."""
    from datasets import load_dataset

    from rkcnn_sae.data.activation_cache import ActivationCache, ActivationDataLoader
    from rkcnn_sae.evaluation.metrics import evaluate_sae
    from rkcnn_sae.interpretability.activation_store import TokenAwareActivationStore
    from rkcnn_sae.interpretability.top_activations import TopActivationFinder
    from rkcnn_sae.interpretability.revived_detector import RevivedLatentDetector
    from rkcnn_sae.interpretability.metrics import InterpretabilityMetrics
    from rkcnn_sae.interpretability.visualization import (
        plot_revived_analysis,
        plot_interpretability_comparison,
        generate_analysis_report,
        save_results_json,
    )
    from rkcnn_sae.models.sae import SAEConfig, SparseAutoencoder
    from rkcnn_sae.models.rkcnn_sae import RkCNNSAEConfig, RkCNNSparseAutoencoder

    print("=" * 70)
    print("FEATURE INTERPRETABILITY ANALYSIS")
    print("Baseline SAE vs RkCNN-Initialized SAE")
    print("=" * 70)

    if args.dry_run:
        print("\n*** DRY RUN MODE - Using minimal data ***\n")
        args.max_tokens = 2000
        args.n_train_steps = 100

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===== Step 1: Load Model and Create Token-Aware Store =====
    print("\n[1/6] Loading model and caching activations with tokens...")

    cache = ActivationCache(
        model_name=args.model_name,
        layer=args.layer,
        hook_point=args.hook_point,
        device=args.device,
    )

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    try:
        if args.dataset == "wikitext":
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        else:
            dataset = load_dataset(args.dataset, split="train")
    except Exception as e:
        print(f"Warning: Could not load {args.dataset}: {e}")
        print("Falling back to wikitext-2...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    # Create token-aware store
    token_store = TokenAwareActivationStore.from_model_and_dataset(
        model=cache.model,
        tokenizer=cache.model.tokenizer,
        dataset=dataset,
        layer=args.layer,
        hook_point="mlp_post",
        max_tokens=args.max_tokens,
        seq_len=128,
        batch_size=16,
        device=args.device,
        show_progress=True,
    )

    d_model = token_store.activation_dim
    n_tokens = token_store.n_tokens
    n_latents = d_model * args.expansion_factor

    print(f"  Tokens cached: {n_tokens:,}")
    print(f"  Activation dim: {d_model}")
    print(f"  SAE latents: {n_latents:,}")

    # Get standard activations for training
    activations = token_store.activations

    dataloader = ActivationDataLoader(
        activations,
        batch_size=args.batch_size,
        shuffle=True,
        device=args.device,
    )

    # ===== Step 2: Train or Load SAEs =====
    print("\n[2/6] Preparing SAEs...")

    baseline_config = SAEConfig(
        d_model=d_model,
        n_latents=n_latents,
        l1_coefficient=args.l1_coefficient,
        normalize_decoder=True,
    )

    rkcnn_config = RkCNNSAEConfig(
        d_model=d_model,
        n_latents=n_latents,
        l1_coefficient=args.l1_coefficient,
        normalize_decoder=True,
        rkcnn_directions_fraction=args.rkcnn_fraction,
        rkcnn_n_subsets=args.rkcnn_n_subsets,
        rkcnn_score_method=args.rkcnn_score_method,
        rkcnn_subset_size=int(np.sqrt(d_model)),
    )

    if args.baseline_sae_path and args.rkcnn_sae_path:
        # Load pre-trained SAEs
        print("  Loading pre-trained SAEs...")
        baseline_sae = SparseAutoencoder(baseline_config)
        baseline_sae.load_state_dict(torch.load(args.baseline_sae_path, map_location=args.device))

        rkcnn_sae = RkCNNSparseAutoencoder(rkcnn_config)
        rkcnn_sae.load_state_dict(torch.load(args.rkcnn_sae_path, map_location=args.device))
    else:
        # Train SAEs
        print("  Training Baseline SAE...")
        baseline_sae = SparseAutoencoder(baseline_config)
        baseline_sae = train_sae_simple(
            baseline_sae, dataloader, args.n_train_steps, args.lr, args.device, "baseline"
        )

        print("  Training RkCNN SAE...")
        rkcnn_sae = RkCNNSparseAutoencoder(rkcnn_config)
        rkcnn_sae.initialize_with_rkcnn(
            activations,
            device=args.device,
            seed=args.seed,
            show_progress=args.verbose,
        )
        rkcnn_sae = train_sae_simple(
            rkcnn_sae, dataloader, args.n_train_steps, args.lr, args.device, "rkcnn"
        )

        # Save trained SAEs
        torch.save(baseline_sae.state_dict(), output_dir / "baseline_sae.pt")
        torch.save(rkcnn_sae.state_dict(), output_dir / "rkcnn_sae.pt")

    baseline_sae = baseline_sae.to(args.device)
    rkcnn_sae = rkcnn_sae.to(args.device)

    # ===== Step 3: Compute SAE Latent Activations =====
    print("\n[3/6] Computing latent activations...")

    baseline_latents = token_store.compute_latent_activations(
        baseline_sae, batch_size=512, device=args.device
    )
    rkcnn_latents = token_store.compute_latent_activations(
        rkcnn_sae, batch_size=512, device=args.device
    )

    print(f"  Baseline latents shape: {baseline_latents.shape}")
    print(f"  RkCNN latents shape: {rkcnn_latents.shape}")

    # ===== Step 4: Detect Revived Latents =====
    print("\n[4/6] Detecting revived latents...")

    detector = RevivedLatentDetector(
        token_store=token_store,
        baseline_sae=baseline_sae,
        rkcnn_sae=rkcnn_sae,
        device=args.device,
    )

    summary = detector.get_comparison_summary()
    revived_latents = detector.find_revived_latents()

    print(f"\n  Summary:")
    print(f"    Total latents: {summary['n_latents']:,}")
    print(f"    Baseline dead: {summary['baseline_dead']:,} ({summary['baseline_dead_rate']:.1%})")
    print(f"    RkCNN dead: {summary['rkcnn_dead']:,} ({summary['rkcnn_dead_rate']:.1%})")
    print(f"    Revived (dead→alive): {summary['n_revived']:,}")
    print(f"    Killed (alive→dead): {summary['n_killed']:,}")
    print(f"    Net improvement: {summary['net_improvement']:+,}")

    # ===== Step 5: Compute Interpretability Metrics =====
    print("\n[5/6] Computing interpretability metrics...")

    # Get token texts for metrics
    token_texts = [r.token_text for r in token_store.token_records]

    # Compute metrics for both SAEs
    baseline_metrics_calc = InterpretabilityMetrics(baseline_latents, token_texts)
    rkcnn_metrics_calc = InterpretabilityMetrics(rkcnn_latents, token_texts)

    print("  Scoring baseline latents...")
    baseline_scores = baseline_metrics_calc.score_all_latents(show_progress=True)

    print("  Scoring RkCNN latents...")
    rkcnn_scores = rkcnn_metrics_calc.score_all_latents(show_progress=True)

    # Compare SAEs
    comparison = InterpretabilityMetrics.compare_saes(baseline_scores, rkcnn_scores)

    print(f"\n  Interpretability Comparison:")
    print(f"    Alive latents - Baseline: {comparison['baseline']['n_alive']:,}, RkCNN: {comparison['rkcnn']['n_alive']:,}")
    print(f"    Mean monosemanticity - Baseline: {comparison['baseline']['mean_monosemanticity']:.4f}, RkCNN: {comparison['rkcnn']['mean_monosemanticity']:.4f}")
    print(f"    Mean entropy - Baseline: {comparison['baseline']['mean_entropy']:.4f}, RkCNN: {comparison['rkcnn']['mean_entropy']:.4f}")

    # ===== Step 6: Generate Visualizations and Report =====
    print("\n[6/6] Generating visualizations and report...")

    # Plot revived analysis
    plot_revived_analysis(
        revived_latents,
        save_path=str(output_dir / "revived_analysis.png"),
    )

    # Plot interpretability comparison
    plot_interpretability_comparison(
        baseline_scores,
        rkcnn_scores,
        save_path=str(output_dir / "interp_comparison.png"),
    )

    # Generate text report
    report = generate_analysis_report(
        summary=summary,
        comparison=comparison,
        revived_latents=revived_latents,
        output_path=str(output_dir / "analysis_report.txt"),
        top_n_examples=args.n_example_latents,
    )

    # Save JSON results
    save_results_json(
        summary=summary,
        comparison=comparison,
        revived_latents=revived_latents,
        output_path=str(output_dir / "interpretability_results.json"),
    )

    # Save token store for later analysis
    token_store.save(str(output_dir / "token_store.pkl"))

    # ===== Final Summary =====
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(f"\nKey Findings:")
    print(f"  1. Dead latent reduction: {summary['dead_latent_reduction']:+,} ({summary['dead_latent_reduction']/summary['n_latents']:.1%})")
    print(f"  2. Revived latents: {summary['n_revived']:,} (now active features)")
    print(f"  3. Monosemanticity change: {comparison['differences']['monosemanticity_diff']:+.4f}")
    print(f"  4. Entropy change: {comparison['differences']['entropy_diff']:+.4f} (lower = better)")

    # Determine success
    is_improved = (
        summary['dead_latent_reduction'] > 0 or
        comparison['differences']['monosemanticity_diff'] > 0 or
        comparison['differences']['entropy_diff'] < 0
    )

    print(f"\n  RkCNN Improvement: {'YES' if is_improved else 'NO'}")

    print(f"\nOutputs saved to: {output_dir}/")
    print(f"  - interpretability_results.json")
    print(f"  - revived_analysis.png")
    print(f"  - interp_comparison.png")
    print(f"  - analysis_report.txt")
    print(f"  - token_store.pkl")

    return {
        "summary": summary,
        "comparison": comparison,
        "n_revived": len(revived_latents),
        "is_improved": is_improved,
    }


def train_sae_simple(
    sae,
    dataloader,
    n_steps: int,
    lr: float,
    device: str,
    name: str = "sae",
):
    """Simple SAE training loop without checkpointing."""
    sae = sae.to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    step = 0
    pbar = tqdm(total=n_steps, desc=f"Training {name}")

    while step < n_steps:
        for batch in dataloader:
            if step >= n_steps:
                break

            batch = batch.to(device)
            optimizer.zero_grad()

            latents, reconstructed, _ = sae(batch)
            loss, loss_dict = sae.compute_loss(batch, latents, reconstructed)

            loss.backward()
            optimizer.step()

            if sae.config.normalize_decoder:
                sae.normalize_decoder()

            step += 1
            pbar.update(1)

            if step % 100 == 0:
                pbar.set_postfix(loss=loss_dict["total"])

    pbar.close()
    return sae


def main():
    args = parse_args()
    results = run_analysis(args)

    if results["is_improved"]:
        print("\nRkCNN initialization shows interpretability improvements!")
        sys.exit(0)
    else:
        print("\nNo clear interpretability improvement detected.")
        sys.exit(0)  # Still success, just informational


if __name__ == "__main__":
    main()
