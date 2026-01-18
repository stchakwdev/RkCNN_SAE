# CLAUDE.md - RkCNN-SAE Project Context

## Project Overview

This project implements **Random k Conditional Nearest Neighbor (RkCNN)** methods for initializing Sparse Autoencoders (SAEs) used in mechanistic interpretability research.

## Key Results Achieved

### Phase 1: Toy Model - PASSED
- Top KNN Score: 0.850 (threshold > 0.6)
- Feature Recovery: 30% (threshold > 25%)
- Score Consistency: 1.46% CV (threshold < 10%)

### Phase 2: Real GPT-2 - PASSED
- **14% reduction in dead latents** (50 → 43)
- L0 sparsity improved (2979 → 2955)
- Reconstruction quality maintained

### Phase 3: Multi-Layer Real GPT-2 Analysis - COMPLETED
- **6 out of 12 layers improved** with RkCNN initialization
- **Best result: Layer 6 with 6.0% reduction** in dead latents (1,307 fewer)
- **Layer 11 reconstruction dramatically improved**: 52% → 77% explained variance
- Layers that benefited: 0, 3, 4, 6, 7, 11 (early + final layers)
- Configuration: 100K tokens, 5000 training steps, 24576 latents (8x expansion)

## Critical Technical Learnings

### 1. Transformer-Lens Hook Points

**IMPORTANT:** The hook point naming is confusing:

| Hook Name | Actual Output | Dimension (GPT-2) |
|-----------|---------------|-------------------|
| `blocks.{L}.hook_mlp_out` | MLP output PROJECTED BACK to residual stream | 768 |
| `blocks.{L}.mlp.hook_post` | MLP hidden state AFTER GeLU activation | **3072** |
| `blocks.{L}.hook_resid_post` | Residual stream after layer | 768 |

**For SAE training on MLP activations (3072d), use:**
```python
hook_name = f"blocks.{layer}.mlp.hook_post"  # NOT hook_mlp_out!
```

### 2. Dataset Loading Issues

OpenWebText downloads are unreliable. The fallback chain is:
1. OpenWebText (often fails mid-download)
2. wikitext-103-raw-v1 (more reliable)
3. wikitext-2-raw-v1 (smallest, most reliable)

**Best practice:** Use wikitext as default for development, OpenWebText for final runs.

### 3. Dimension Verification

Always verify activation dimensions after loading:
```python
actual_dim = activations.shape[1]
if actual_dim != expected_d_model:
    print(f"Warning: Dimension mismatch: {actual_dim} vs {expected_d_model}")
    # Fall back to synthetic data
```

### 4. RunPod Setup

**IMPORTANT: Always run GPU experiments on RunPod, not locally.**

Preferred GPU types (in order):
1. NVIDIA GeForce RTX 4090
2. NVIDIA RTX A6000
3. NVIDIA GeForce RTX 3090

**Required PyTorch version for transformer-lens:**
```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

**Full dependency install:**
```bash
pip install transformer-lens scikit-learn matplotlib tqdm datasets einops
```

**Quick setup script for RunPod:**
```bash
cd /workspace && \
git clone https://github.com/yourusername/RKCNN_SAE.git && \
cd RKCNN_SAE && \
pip install -e . && \
pip install transformer-lens scikit-learn matplotlib tqdm datasets einops
```

## Project Structure

```
RKCNN_SAE/
├── rkcnn_sae/
│   ├── core/
│   │   ├── rkcnn_probe.py      # Main RkCNN algorithm
│   │   ├── separation_score.py # KNN/kurtosis scoring
│   │   └── subset_sampler.py   # Random subset generation
│   ├── models/
│   │   ├── sae.py              # Standard SAE
│   │   └── rkcnn_sae.py        # RkCNN-initialized SAE
│   ├── data/
│   │   └── activation_cache.py # GPT-2 activation extraction
│   ├── evaluation/
│   │   └── metrics.py          # Dead latents, L0, reconstruction
│   └── interpretability/       # NEW: Feature interpretability analysis
│       ├── activation_store.py # Token-aware activation caching
│       ├── top_activations.py  # Top-K retrieval per latent
│       ├── revived_detector.py # Find baseline-dead, rkcnn-alive latents
│       ├── metrics.py          # Interpretability metrics
│       └── visualization.py    # Plotting and reporting
├── experiments/
│   ├── phase1_toy_model.py     # Toy model validation
│   ├── phase2_gpt2.py          # GPT-2 SAE comparison
│   ├── multi_layer_analysis.py # All 12 layers analysis
│   └── interpretability_analysis.py # NEW: Interpretability comparison
└── results/
    ├── phase1/                 # Toy model results
    ├── phase2/                 # GPT-2 results
    ├── multi_layer/            # Multi-layer analysis
    └── interpretability/       # NEW: Interpretability results
```

## Running Experiments

### Phase 1 (CPU, ~5 min)
```bash
python experiments/phase1_toy_model.py --verbose
```

### Phase 2 (GPU recommended)
```bash
python experiments/phase2_gpt2.py --device cuda
```

### Multi-Layer Analysis (GPU required)
```bash
python experiments/multi_layer_analysis.py \
    --layers all \
    --max-tokens 100000 \
    --n-train-steps 5000 \
    --output-dir ./results/multi_layer_real \
    --verbose
```

### Interpretability Analysis (GPU recommended)
```bash
python experiments/interpretability_analysis.py \
    --layer 6 \
    --max-tokens 50000 \
    --device cuda \
    --output-dir ./results/interpretability
```

This analyzes:
- **Revived latents**: Latents that are dead in baseline but alive in RkCNN SAE
- **Interpretability metrics**: Activation entropy, top-K concentration, token diversity
- **Top activating tokens**: What each revived latent represents

## Current Status

### Completed
- [x] Phase 1 toy model validation
- [x] Phase 2 real GPT-2 single layer (Layer 6)
- [x] Multi-layer analysis on synthetic data
- [x] Fixed hook point issue (`mlp.hook_post` vs `hook_mlp_out`)
- [x] Multi-layer analysis on real GPT-2 data (all 12 layers) ✓

### To Do
- [x] Hyperparameter sweep ✓ (best: 11.9% dead latent reduction)
- [x] Feature interpretability analysis ✓ (module: `rkcnn_sae/interpretability/`)
- [ ] Paper writeup

## Known Issues

1. **OpenWebText Download Failures**: The dataset download often fails mid-way. Solution: Use wikitext as primary dataset.

2. **Wikitext Dimension Bug (FIXED)**: When using wikitext with `hook_mlp_out`, dimensions were wrong. Fixed by using `mlp.hook_post` instead.

3. **Memory on Large Runs**: For 12-layer analysis with 100K tokens, needs ~16GB VRAM. RTX 4090 recommended.

## Key Parameters

| Parameter | Phase 2 Value | Multi-Layer Value |
|-----------|---------------|-------------------|
| d_model | 768 | 768 |
| d_mlp | 3072 | 3072 |
| n_latents | 6144 (8x) | 24576 (8x) |
| l1_coefficient | 0.001 | 0.001 |
| rkcnn_fraction | 0.5 | 0.5 |
| rkcnn_n_subsets | 600 | 600 |
| score_method | kurtosis | kurtosis |

## Multi-Layer Analysis Results Summary

Results from real GPT-2 analysis (wikitext, 100K tokens, 5000 steps per SAE):

| Layer | Baseline Dead | RkCNN Dead | Improvement |
|-------|---------------|------------|-------------|
| 0 | 92.3% | 90.4% | **+2.0%** ✓ |
| 3 | 96.5% | 94.7% | **+1.9%** ✓ |
| 4 | 96.1% | 94.1% | **+2.2%** ✓ |
| 6 | 89.3% | 84.0% | **+6.0%** ✓ |
| 7 | 85.2% | 85.1% | **+0.2%** ✓ |
| 11 | 70.8% | 69.6% | **+1.7%** ✓ |

**Key insight**: RkCNN is most effective on early layers (0, 3, 4) and the final layer (11), where activation patterns are more structured.
