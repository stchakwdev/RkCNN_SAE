# RkCNN-Inspired Methods for Mechanistic Interpretability

**Date:** 2026-01-16  
**Source Paper:** [Random k Conditional Nearest Neighbor for High-Dimensional Data](https://peerj.com/articles/cs-2497/)

---

## Executive Summary

This document outlines an experimental plan to apply **Random k Conditional Nearest Neighbor (RkCNN)** concepts to mechanistic interpretability (mech interp), specifically targeting the curse of dimensionality in neural network feature spaces.

**Core insight:** RkCNN mitigates dimensionality issues by:
1. Training classifiers on random feature subsets (not all dimensions)
2. Using "separation scores" to filter informative subsets
3. Ensembling over many subspace models to reduce variance

This maps naturally to mech interp challenges in Sparse Autoencoders (SAEs) and circuit discovery.

---

## Problem Statement

### The Curse of Dimensionality in Mech Interp

From Anthropic's "Towards Monosemanticity":
> "A key challenge to our agenda of reverse engineering neural networks is the curse of dimensionality: as we study ever-larger models, the volume of the latent space representing the model's internal state that we need to interpret grows exponentially."

**Current SAE challenges where RkCNN ideas could help:**
- **Dead features**: Latents that never activate (up to 60% in some SAEs)
- **Shrinkage**: L1 penalty pulls all activations toward zero
- **Scaling**: Need massive SAEs (100K+ features) to capture all concepts
- **Feature splitting**: Same concept fragmented across multiple features
- **Polysemanticity**: Features responding to unrelated concepts

---

## Related Work

### Directly Relevant
1. **"Neural random subspace" (Pattern Recognition, 2020)** - Implements random subspace method using CNN layers; achieves higher accuracy than conventional forest methods
2. **"Random Subspace Ensembles of FCN for Time Series"** - Applies RSE to deep learning successfully
3. **"Sparse Feature Circuits" (Marks et al.)** - Uses sparse features for causal circuit discovery

### Recent SAE Innovations
1. **TopK SAEs** (OpenAI, 2024) - Direct sparsity control, eliminates L1 shrinkage
2. **Crosscoders** (Anthropic, 2024) - Cross-layer features, discovers redundant structure across layers
3. **Transcoders** (2025) - Outperform SAEs on interpretability benchmarks
4. **Active Subspace Initialization** - Addresses dead features in low-rank attention outputs

### Gap in Literature
**No one has applied RkCNN-style random subspace ensembles with informativeness scoring to SAE training or mech interp feature discovery.**

---

## Proposed Experiments

### Experiment 1: Random Subspace Ensemble SAEs (RSE-SAE)

**Hypothesis:** Training multiple smaller SAEs on random subspaces of activation space, then ensembling, will produce more interpretable features with fewer dead latents than a single large SAE.

**Method:**
```python
# Pseudocode for RSE-SAE
def train_rse_sae(activations, num_subsets=h, subset_size=m, top_k=r):
    """
    activations: [batch, d_model]
    h: total random subsets to generate
    m: dimensions per subset  
    r: number of contributing SAEs after filtering
    """
    subsets = []
    for _ in range(h):
        # Random feature subset
        dims = random.sample(range(d_model), m)
        subset_activations = activations[:, dims]
        
        # Train small SAE
        sae = SparseAutoencoder(input_dim=m, hidden_dim=m*4)
        sae.train(subset_activations)
        
        # Compute separation score (adapted for SAE features)
        score = compute_separation_score(sae, subset_activations)
        subsets.append((sae, dims, score))
    
    # Keep top-r by separation score
    top_subsets = sorted(subsets, key=lambda x: x[2], reverse=True)[:r]
    
    return EnsembleSAE(top_subsets)
```

**Separation Score for SAEs:**
Adapt RkCNN's separation score to measure how well an SAE's features discriminate between activation patterns:
- Option A: Inter-class variance of SAE reconstructions
- Option B: Mutual information between latents and known concept labels
- Option C: Feature activation consistency across related inputs

**Evaluation Metrics:**
1. Dead feature rate (lower is better)
2. Feature interpretability (human eval + automated metrics)
3. Reconstruction loss
4. Feature monosemanticity score

**Baseline:** Standard single-layer SAE with equivalent total parameters

---

### Experiment 2: Separation Score Feature Filtering

**Hypothesis:** Applying RkCNN-style separation scoring to post-hoc filter SAE features will identify the most causally relevant features for downstream circuit analysis.

**Method:**
1. Train standard SAE on model activations
2. Compute separation score for each latent feature:
   - Score = how well feature activations separate known concepts/behaviors
3. Keep top-k features by separation score
4. Compare circuit discovery results (à la Sparse Feature Circuits)

**Concrete Implementation:**
```python
def separation_score_for_feature(feature_idx, sae, activation_dataset, labels):
    """
    Compute how well a single SAE feature separates labeled activation clusters.
    """
    feature_acts = sae.encode(activation_dataset)[:, feature_idx]
    
    # Between-class variance / within-class variance
    scores = []
    for class_label in unique(labels):
        class_acts = feature_acts[labels == class_label]
        between_var = (class_acts.mean() - feature_acts.mean())**2
        within_var = class_acts.var()
        scores.append(between_var / (within_var + eps))
    
    return mean(scores)
```

---

### Experiment 3: Random Subspace Circuit Discovery

**Hypothesis:** Discovering circuits in random subspaces of the residual stream, then ensembling, will find more robust and generalizable computational structures.

**Method:**
1. For each random subspace:
   - Project residual stream to subspace
   - Run activation patching / causal tracing
   - Identify circuits in that subspace
2. Aggregate circuits across subspaces
3. Circuits appearing in multiple subspaces = more robust

**Key Metric:** Circuit generalization across distributions

---

## Implementation Plan

### Phase 1: Infrastructure (Week 1-2)
- [ ] Set up SAE training codebase (use [Language-Model-SAEs](https://github.com/OpenMOSS/Language-Model-SAEs) or [sparse_autoencoder](https://github.com/ai-safety-foundation/sparse_autoencoder))
- [ ] Implement random subspace sampling
- [ ] Implement separation score computation (adapt from RkCNN paper)
- [ ] Set up evaluation harness

### Phase 2: RSE-SAE Training (Week 3-4)
- [ ] Train RSE-SAE on Pythia-70M residual stream (layer 3)
- [ ] Hyperparameter sweep: m (subset size), h (total subsets), r (top-k)
- [ ] Compare against baseline SAE (same total parameters)
- [ ] Analyze dead feature rates

### Phase 3: Feature Quality Analysis (Week 5-6)
- [ ] Compute interpretability scores (manual + automated)
- [ ] Run monosemanticity evaluation
- [ ] Compare separation-score-filtered features vs random features
- [ ] Visualization: feature clustering in subspaces

### Phase 4: Circuit Discovery (Week 7-8)
- [ ] Apply RSE-identified features to Sparse Feature Circuits methodology
- [ ] Test circuit robustness across subspaces
- [ ] Write up results

---

## Key Parameters (from RkCNN paper)

| Parameter | RkCNN Meaning | Mech Interp Mapping |
|-----------|--------------|---------------------|
| k | Neighbors in kCNN | Could map to SAE sparsity k |
| m | Features per subset | Dimensions of activation subspace |
| r | Contributing subsets | SAEs in final ensemble |
| h | Total subsets sampled | Candidate SAEs before filtering |

**Recommended starting values (adapted from paper):**
- k: Small (1-3) works best for overlapping classes
- m: √(d_model) as baseline
- r: 200-400
- h: 3r to 10r

For GPT-2 small (d_model=768):
- m = 28 (√768)
- r = 200
- h = 600-2000

---

## Expected Outcomes

1. **Reduced dead features**: Random subspace training should naturally prevent features from dying by focusing on smaller, more manageable spaces
2. **Better interpretability**: Separation score filtering should surface genuinely informative features
3. **More robust circuits**: Ensemble approach should find circuits that generalize better

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Computational cost of training many SAEs | Use small subsets, efficient implementations; could parallelize |
| Separation score doesn't translate well | Try multiple score variants; compare with other feature quality metrics |
| Ensemble aggregation loses fine-grained structure | Experiment with different aggregation methods (voting, averaging, union) |

---

## References

1. Lin, Y.-T., & Kuo, L. (2024). Random k conditional nearest neighbor for high-dimensional data. *PeerJ Computer Science*, 10, e2497. https://peerj.com/articles/cs-2497/
2. Anthropic. (2023). Towards Monosemanticity. https://transformer-circuits.pub/2023/monosemantic-features
3. Cunningham, H., et al. (2023). Sparse Autoencoders Find Highly Interpretable Features in Language Models. https://arxiv.org/abs/2309.08600
4. Marks, S., et al. (2024). Sparse Feature Circuits. https://arxiv.org/abs/2403.19647
5. Gao, L., et al. (2024). Scaling and Evaluating Sparse Autoencoders. https://arxiv.org/abs/2406.04093
6. Anthropic. (2024). Sparse Crosscoders for Cross-Layer Features. https://transformer-circuits.pub/2024/crosscoders/
7. Paulo, G., et al. (2025). Transcoders Beat Sparse Autoencoders for Interpretability. https://arxiv.org/abs/2501.18823

---

## Next Steps

1. **Confirm direction** with Timo
2. **Clone/fork** SAE training repo
3. **Implement** separation score module
4. **Run** initial feasibility test on small model

---

*Document generated 2026-01-16 by Clawd 🦀*
