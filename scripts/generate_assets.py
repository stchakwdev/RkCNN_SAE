#!/usr/bin/env python3
"""Generate placeholder assets for README documentation."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path


def create_banner():
    """Create project banner image."""
    fig, ax = plt.subplots(figsize=(12, 4))

    # Background gradient
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    gradient = np.vstack([gradient] * 100)
    ax.imshow(gradient, aspect='auto', cmap='Blues', extent=[0, 12, 0, 4], alpha=0.3)

    # Title
    ax.text(6, 2.5, 'RkCNN-SAE', fontsize=48, fontweight='bold',
            ha='center', va='center', color='#1a365d')
    ax.text(6, 1.3, 'Random k Conditional Nearest Neighbor Methods\nfor Sparse Autoencoders',
            fontsize=16, ha='center', va='center', color='#2d3748', style='italic')

    # Decorative elements
    circle1 = plt.Circle((1.5, 2), 0.5, color='#3182ce', alpha=0.6)
    circle2 = plt.Circle((10.5, 2), 0.5, color='#e53e3e', alpha=0.6)
    ax.add_patch(circle1)
    ax.add_patch(circle2)

    # Connecting line
    ax.plot([2, 10], [2, 2], 'k--', alpha=0.3, linewidth=2)

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('assets/rkcnn_sae_banner.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: assets/rkcnn_sae_banner.png")


def create_method_diagram():
    """Create method diagram showing RkCNN pipeline."""
    fig, ax = plt.subplots(figsize=(12, 5))

    # Boxes for pipeline stages
    boxes = [
        (0.5, 2, 'Activations\nX ∈ ℝⁿˣᵈ', '#e2e8f0'),
        (2.5, 2, 'Random\nSubsets', '#bee3f8'),
        (4.5, 2, 'Separation\nScoring', '#90cdf4'),
        (6.5, 2, 'Top-k\nSelection', '#63b3ed'),
        (8.5, 2, 'Direction\nExtraction', '#4299e1'),
        (10.5, 2, 'SAE\nInit', '#3182ce'),
    ]

    for x, y, text, color in boxes:
        rect = mpatches.FancyBboxPatch((x-0.7, y-0.6), 1.4, 1.2,
                                        boxstyle="round,pad=0.05",
                                        facecolor=color, edgecolor='#2d3748', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrows
    for i in range(len(boxes) - 1):
        ax.annotate('', xy=(boxes[i+1][0]-0.7, 2), xytext=(boxes[i][0]+0.7, 2),
                   arrowprops=dict(arrowstyle='->', color='#2d3748', lw=2))

    # Labels
    ax.text(6, 3.5, 'RkCNN Probing Pipeline', fontsize=16, fontweight='bold',
            ha='center', color='#1a365d')

    ax.set_xlim(-0.5, 12)
    ax.set_ylim(0.5, 4)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('assets/method_diagram.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: assets/method_diagram.png")


def create_training_curves_placeholder():
    """Create placeholder training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Generate fake training data
    steps = np.arange(0, 10000, 100)
    baseline_loss = 0.1 * np.exp(-steps/3000) + 0.001 + np.random.randn(len(steps)) * 0.002
    rkcnn_loss = 0.08 * np.exp(-steps/2500) + 0.0008 + np.random.randn(len(steps)) * 0.0015

    # Loss curves
    ax = axes[0]
    ax.plot(steps, baseline_loss, label='Baseline SAE', color='#3182ce', linewidth=2)
    ax.plot(steps, rkcnn_loss, label='RkCNN SAE', color='#e53e3e', linewidth=2)
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Total Loss', fontsize=12)
    ax.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Dead latent rate over time
    ax = axes[1]
    baseline_dead = 0.4 - 0.1 * (1 - np.exp(-steps/5000)) + np.random.randn(len(steps)) * 0.02
    rkcnn_dead = 0.15 - 0.05 * (1 - np.exp(-steps/4000)) + np.random.randn(len(steps)) * 0.01
    baseline_dead = np.clip(baseline_dead, 0, 1)
    rkcnn_dead = np.clip(rkcnn_dead, 0, 1)

    ax.plot(steps, baseline_dead * 100, label='Baseline SAE', color='#3182ce', linewidth=2)
    ax.plot(steps, rkcnn_dead * 100, label='RkCNN SAE', color='#e53e3e', linewidth=2)
    ax.axhline(y=20, color='gray', linestyle='--', label='Target (20%)', alpha=0.7)
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Dead Latent Rate (%)', fontsize=12)
    ax.set_title('Dead Latent Rate', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('assets/training_curves_placeholder.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: assets/training_curves_placeholder.png")


def create_dead_latent_comparison():
    """Create dead latent comparison bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))

    categories = ['Baseline\nSAE', 'RkCNN\nSAE']
    dead_rates = [42, 15]  # Placeholder values
    colors = ['#3182ce', '#e53e3e']

    bars = ax.bar(categories, dead_rates, color=colors, edgecolor='#2d3748', linewidth=2)

    # Add value labels
    for bar, rate in zip(bars, dead_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Target line
    ax.axhline(y=20, color='green', linestyle='--', linewidth=2, label='Target (< 20%)')

    ax.set_ylabel('Dead Latent Rate (%)', fontsize=12)
    ax.set_title('Dead Latent Rate Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 60)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Improvement annotation
    ax.annotate('', xy=(1, 15), xytext=(0, 42),
               arrowprops=dict(arrowstyle='->', color='green', lw=3))
    ax.text(0.5, 30, '↓ 64%\nimprovement', ha='center', fontsize=11,
            color='green', fontweight='bold')

    plt.tight_layout()
    plt.savefig('assets/dead_latent_comparison_placeholder.png', dpi=150,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Created: assets/dead_latent_comparison_placeholder.png")


def main():
    """Generate all assets."""
    # Change to project root
    import os
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)

    # Create assets directory if needed
    Path('assets').mkdir(exist_ok=True)

    print("Generating README assets...")
    print("-" * 40)

    create_banner()
    create_method_diagram()
    create_training_curves_placeholder()
    create_dead_latent_comparison()

    print("-" * 40)
    print("All assets generated!")


if __name__ == "__main__":
    main()
