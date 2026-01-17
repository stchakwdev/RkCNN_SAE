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

**Required PyTorch version for transformer-lens:**
```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

**Full dependency install:**
```bash
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
│   └── evaluation/
│       └── metrics.py          # Dead latents, L0, reconstruction
├── experiments/
│   ├── phase1_toy_model.py     # Toy model validation
│   ├── phase2_gpt2.py          # GPT-2 SAE comparison
│   └── multi_layer_analysis.py # All 12 layers analysis
└── results/
    ├── phase1/                 # Toy model results
    ├── phase2/                 # GPT-2 results
    └── multi_layer/            # Multi-layer analysis
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

## Current Status

### Completed
- [x] Phase 1 toy model validation
- [x] Phase 2 real GPT-2 single layer (Layer 6)
- [x] Multi-layer analysis on synthetic data
- [x] Fixed hook point issue (`mlp.hook_post` vs `hook_mlp_out`)

### In Progress
- [ ] Multi-layer analysis on real GPT-2 data (all 12 layers)

### To Do
- [ ] Complete multi-layer real data analysis
- [ ] Hyperparameter sweep
- [ ] Feature interpretability analysis
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

## Resume Instructions

To continue the multi-layer analysis on real GPT-2 data:

1. Start a RunPod pod (RTX 4090 recommended)
2. Clone repo and install dependencies:
   ```bash
   cd /workspace
   git clone https://github.com/stchakwdev/RkCNN_SAE.git
   cd RkCNN_SAE
   pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
   pip install transformer-lens scikit-learn matplotlib tqdm datasets einops
   ```
3. Run the experiment:
   ```bash
   PYTHONPATH=/workspace/RkCNN_SAE python experiments/multi_layer_analysis.py \
       --layers all \
       --max-tokens 100000 \
       --n-train-steps 5000 \
       --output-dir /workspace/results/multi_layer_real \
       --verbose
   ```
4. Download results and commit
