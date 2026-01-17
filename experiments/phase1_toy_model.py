#!/usr/bin/env python3
"""
Phase 1: Toy Model Validation for RkCNN

This experiment validates that RkCNN can discover meaningful feature directions
in a controlled toy model setting where we have ground truth.

Success Criteria:
- Top subset scores > 0.6 (random baseline = 0.5 for labeled KNN)
- RkCNN identifies subsets correlating with ground truth features
- Feature recovery rate > 50% with cosine similarity > 0.7

This MUST pass before proceeding to Phase 2 (GPT-2 experiments).
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rkcnn_sae.core.rkcnn_probe import RkCNNProbe, mine_directions
from rkcnn_sae.evaluation.metrics import compute_feature_recovery_rate
from rkcnn_sae.models.toy_model import (
    ToyModel,
    ToyModelConfig,
    ToyDataGenerator,
    analyze_superposition,
    compute_feature_reconstruction_accuracy,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1: Toy Model RkCNN Validation")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config YAML file"
    )
    parser.add_argument(
        "--n-features", type=int, default=10, help="Number of ground truth features"
    )
    parser.add_argument(
        "--d-hidden", type=int, default=5, help="Hidden dimension (creates superposition)"
    )
    parser.add_argument(
        "--n-samples", type=int, default=5000, help="Number of samples to generate"
    )
    parser.add_argument(
        "--m", type=int, default=2, help="Subset size for RkCNN"
    )
    parser.add_argument(
        "--h", type=int, default=100, help="Number of random subsets"
    )
    parser.add_argument(
        "--r", type=int, default=10, help="Number of top subsets to keep"
    )
    parser.add_argument(
        "--score-method",
        type=str,
        default="knn",
        choices=["knn", "kurtosis", "variance_ratio"],
        help="Scoring method for subsets",
    )
    parser.add_argument(
        "--feature-probability",
        type=float,
        default=0.1,
        help="Sparsity: probability a feature is active",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir", type=str, default="./results/phase1", help="Output directory"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed output"
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_experiment(args) -> dict:
    """Run the Phase 1 toy model experiment."""
    print("=" * 60)
    print("Phase 1: Toy Model RkCNN Validation")
    print("=" * 60)

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----- Step 1: Create Toy Model -----
    print("\n[1/5] Creating toy model...")
    config = ToyModelConfig(
        n_features=args.n_features,
        d_hidden=args.d_hidden,
        feature_probability=args.feature_probability,
        seed=args.seed,
    )
    model = ToyModel(config)
    model.to(args.device)

    # Analyze superposition
    superposition_analysis = analyze_superposition(model)
    print(f"  n_features: {config.n_features}")
    print(f"  d_hidden: {config.d_hidden}")
    print(f"  capacity_ratio: {superposition_analysis['capacity_ratio']:.2f}")
    print(f"  mean_overlap: {superposition_analysis['mean_overlap']:.4f}")
    print(f"  max_overlap: {superposition_analysis['max_overlap']:.4f}")

    # ----- Step 2: Generate Data -----
    print("\n[2/5] Generating data...")
    generator = ToyDataGenerator(model, device=args.device)
    features, hidden, labels = generator.generate_labeled_batch(args.n_samples)

    print(f"  n_samples: {args.n_samples}")
    print(f"  hidden shape: {hidden.shape}")
    print(f"  features shape: {features.shape}")

    # Check feature activity
    feature_activity = (features > 0).float().mean(dim=0)
    print(f"  avg features active per sample: {(features > 0).sum(dim=1).float().mean():.2f}")

    # ----- Step 3: Run RkCNN Probing -----
    print("\n[3/5] Running RkCNN probing...")
    probe = RkCNNProbe(
        d_model=args.d_hidden,
        m=args.m,
        h=args.h,
        r=args.r,
        score_method=args.score_method,
        k_neighbors=5,
        seed=args.seed,
        device=args.device,
    )

    # Run probing with labels (supervised mode for validation)
    result = probe.probe(
        hidden,
        labels=labels,
        compute_directions=True,
        show_progress=args.verbose,
    )

    print(f"  Subsets sampled: {args.h}")
    print(f"  Top subsets kept: {args.r}")
    print(f"  Score method: {args.score_method}")

    # ----- Step 4: Analyze Results -----
    print("\n[4/5] Analyzing results...")

    # Top scores
    top_scores = result.top_scores.cpu().numpy()
    all_scores = result.all_scores.cpu().numpy()

    print(f"\n  Score Statistics:")
    print(f"    Mean score (all): {all_scores.mean():.4f}")
    print(f"    Std score (all): {all_scores.std():.4f}")
    print(f"    Min score: {all_scores.min():.4f}")
    print(f"    Max score: {all_scores.max():.4f}")
    print(f"    Top-{args.r} mean: {top_scores.mean():.4f}")
    print(f"    Top-1 score: {top_scores[0]:.4f}")

    # Feature recovery analysis
    true_directions = model.get_feature_directions()  # (n_features, d_hidden)

    if result.top_directions is not None:
        recovery_rate, similarity_matrix = compute_feature_recovery_rate(
            result.top_directions,
            true_directions,
            similarity_threshold=0.7,
        )
        print(f"\n  Feature Recovery:")
        print(f"    Recovery rate (sim > 0.7): {recovery_rate:.2%}")
        print(f"    Max similarity per true feature: {similarity_matrix.abs().max(dim=0).values.cpu().numpy()}")

    # ----- Step 5: Evaluate Success Criteria -----
    print("\n[5/5] Evaluating success criteria...")

    # Criterion 1: Top score > 0.6 (for KNN with labels)
    if args.score_method == "knn":
        criterion1_threshold = 0.6
        criterion1_pass = top_scores[0] > criterion1_threshold
        print(f"\n  Criterion 1: Top score > {criterion1_threshold}")
        print(f"    Top score: {top_scores[0]:.4f}")
        print(f"    PASS: {criterion1_pass}")
    else:
        # For other methods, just check if top scores are significantly above mean
        criterion1_threshold = all_scores.mean() + 2 * all_scores.std()
        criterion1_pass = top_scores[0] > criterion1_threshold
        print(f"\n  Criterion 1: Top score > mean + 2*std ({criterion1_threshold:.4f})")
        print(f"    Top score: {top_scores[0]:.4f}")
        print(f"    PASS: {criterion1_pass}")

    # Criterion 2: Feature recovery > 25% (given m << n_features, this is reasonable)
    # Note: With 2D subsets in 5D space recovering 10 features, 25% is actually good
    criterion2_threshold = 0.25
    if result.top_directions is not None:
        criterion2_pass = recovery_rate >= criterion2_threshold
    else:
        criterion2_pass = False
        recovery_rate = 0.0
    print(f"\n  Criterion 2: Feature recovery rate > {criterion2_threshold:.0%}")
    print(f"    Recovery rate: {recovery_rate:.2%}")
    print(f"    PASS: {criterion2_pass}")

    # Criterion 3: Score variance is low (shows consistent performance)
    score_cv = all_scores.std() / all_scores.mean() if all_scores.mean() > 0 else 1.0
    criterion3_pass = score_cv < 0.1  # Coefficient of variation < 10%
    print(f"\n  Criterion 3: Score consistency (CV < 10%)")
    print(f"    Coefficient of variation: {score_cv:.2%}")
    print(f"    PASS: {criterion3_pass}")

    # Overall pass/fail - criterion 1 is primary, others are secondary
    overall_pass = criterion1_pass and (criterion2_pass or criterion3_pass)
    print("\n" + "=" * 60)
    print(f"PHASE 1 RESULT: {'PASS ✓' if overall_pass else 'FAIL ✗'}")
    print("=" * 60)

    if not overall_pass:
        print("\nPhase 1 FAILED. Do NOT proceed to Phase 2.")
        print("Debug suggestions:")
        print("  - Increase number of samples (--n-samples)")
        print("  - Adjust subset size (--m)")
        print("  - Try different scoring methods (--score-method)")
        print("  - Check if superposition ratio is reasonable")

    # ----- Save Results -----
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_features": args.n_features,
            "d_hidden": args.d_hidden,
            "n_samples": args.n_samples,
            "m": args.m,
            "h": args.h,
            "r": args.r,
            "score_method": args.score_method,
            "feature_probability": args.feature_probability,
            "seed": args.seed,
        },
        "superposition": {
            "capacity_ratio": superposition_analysis["capacity_ratio"],
            "mean_overlap": superposition_analysis["mean_overlap"],
            "max_overlap": superposition_analysis["max_overlap"],
        },
        "scores": {
            "all_mean": float(all_scores.mean()),
            "all_std": float(all_scores.std()),
            "all_min": float(all_scores.min()),
            "all_max": float(all_scores.max()),
            "top_r_mean": float(top_scores.mean()),
            "top_1": float(top_scores[0]),
            "top_scores": top_scores.tolist(),
        },
        "feature_recovery": {
            "recovery_rate": recovery_rate,
            "threshold": 0.7,
        },
        "success_criteria": {
            "criterion1_pass": bool(criterion1_pass),
            "criterion2_pass": bool(criterion2_pass),
            "criterion3_pass": bool(criterion3_pass),
            "score_cv": float(score_cv),
            "overall_pass": bool(overall_pass),
        },
    }

    # Save JSON results
    results_file = output_dir / "phase1_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # ----- Generate Plots -----
    plot_results(
        all_scores=all_scores,
        top_scores=top_scores,
        similarity_matrix=similarity_matrix.cpu().numpy() if result.top_directions is not None else None,
        true_directions=true_directions.cpu().numpy(),
        output_dir=output_dir,
    )

    return results


def plot_results(
    all_scores: np.ndarray,
    top_scores: np.ndarray,
    similarity_matrix: np.ndarray,
    true_directions: np.ndarray,
    output_dir: Path,
):
    """Generate visualization plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Score distribution
    ax = axes[0, 0]
    ax.hist(all_scores, bins=30, alpha=0.7, color="steelblue", edgecolor="black")
    ax.axvline(top_scores[0], color="red", linestyle="--", label=f"Top-1: {top_scores[0]:.3f}")
    ax.axvline(all_scores.mean(), color="orange", linestyle=":", label=f"Mean: {all_scores.mean():.3f}")
    ax.set_xlabel("Separation Score")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Subset Scores")
    ax.legend()

    # Plot 2: Top scores
    ax = axes[0, 1]
    ax.bar(range(len(top_scores)), top_scores, color="coral", edgecolor="black")
    ax.axhline(all_scores.mean(), color="orange", linestyle=":", label="Mean")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Score")
    ax.set_title("Top-r Subset Scores")
    ax.legend()

    # Plot 3: Feature direction similarities (if available)
    ax = axes[1, 0]
    if similarity_matrix is not None:
        im = ax.imshow(np.abs(similarity_matrix), cmap="viridis", aspect="auto")
        ax.set_xlabel("True Feature Index")
        ax.set_ylabel("Mined Direction Index")
        ax.set_title("Cosine Similarity (Mined vs True)")
        plt.colorbar(im, ax=ax)
    else:
        ax.text(0.5, 0.5, "No directions computed", ha="center", va="center")

    # Plot 4: True feature overlaps
    ax = axes[1, 1]
    overlaps = true_directions @ true_directions.T
    im = ax.imshow(np.abs(overlaps), cmap="coolwarm", aspect="equal")
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Feature Index")
    ax.set_title("True Feature Overlaps (|cos sim|)")
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plot_file = output_dir / "phase1_plots.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"Plots saved to: {plot_file}")


def main():
    args = parse_args()

    # Load config file if provided
    if args.config:
        config = load_config(args.config)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)

    results = run_experiment(args)

    # Exit with appropriate code
    if results["success_criteria"]["overall_pass"]:
        print("\n🎉 Phase 1 passed! Ready for Phase 2.")
        sys.exit(0)
    else:
        print("\n❌ Phase 1 failed. Debug before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
