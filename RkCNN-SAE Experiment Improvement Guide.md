# **RkCNN-SAE Experiment Improvement Guide**

**Strategy:** De-risking & Iterative Complexity

**Based on:** *Steinhardt's Research Heuristics* & *Nanda's Mech Interp Framework*

## **1\. Strategic Pivot: Why SAEs & Toy Models?**

**The Core Risk:** You are testing a new method (RkCNN) on a domain (LLM internals) where the "ground truth" is unknown. If your experiment fails on GPT-2, you won't know if RkCNN is flawed or if the model just represents features differently than expected.

**The Fix:**

1. **Avoid Transcoders (For Now):** Transcoders introduce layer-to-layer dynamics (input $\\rightarrow$ output). This adds confounding variables.  
2. **Stick to SAEs:** SAEs are "static" feature extractors. They are the simplest unit test for RkCNN.  
3. **Start with "Toy Models of Superposition":** This is the **De-risking Step**. In a toy model, you *mathematically define* the features. If RkCNN cannot find features here, it is mathematically incapable of working on GPT-2.

## **Phase 1: The "Unit Test" (Toy Models)**

**Goal:** Prove that "Random Subsets \+ Separation Score" can actually identify feature directions in a compressed space (superposition).

### **A. The Setup**

Replicate the setup from Anthropic's *Toy Models of Superposition*:

* **Input:** $x \\in \\mathbb{R}^{20}$ (Sparse features, e.g., only 2 active at once).  
* **Bottleneck:** $h \= Wx$, where $h \\in \\mathbb{R}^{5}$. (The model *must* use superposition).  
* **Task:** The model tries to reconstruct $x$ from $h$.

### **B. The RkCNN Hypothesis Test**

Instead of training an SAE immediately, run RkCNN on the hidden states $h$:

1. **Sample:** Take random subsets of dimensions from $h$ (e.g., subsets of size $k=2$).  
2. **Label:** Since this is synthetic, you *know* when Feature 1 is active. Use "Feature 1 Active" vs "Feature 1 Inactive" as your classification target.  
3. **Score:** Train a KNN/Linear probe on the subset. Does it get high accuracy?

### **C. Success Criteria (Go/No-Go)**

* **Pass:** RkCNN identifies a specific subset of $h$ that correlates highly with "Feature 1."  
* **Fail:** RkCNN performs no better than a random baseline, or requires $k$ to be equal to the full dimension size to work.  
* **Action:** If it fails, refine the "Separation Score" metric. Do not move to Phase 2\.

## **Phase 2: The "Integration Test" (GPT-2 Small)**

**Goal:** Solve the **"Dead Latent"** problem in SAEs using RkCNN initialization.

**Hypothesis:** Standard SAEs initialize encoder weights randomly. This leads to many "dead" neurons that never activate. RkCNN can identify "dense" regions of the activation space to seed the SAE.

### **A. Methodology: RkCNN-Initialized SAE**

Instead of random initialization ($W\_{enc} \\sim \\mathcal{N}(0, 1)$), use RkCNN to "mine" meaningful directions first.

1. **Data Collection:** Run GPT-2 Small on 1M tokens. Cache activations from Layer 6\.  
2. **RkCNN Mining:**  
   * Sample random direction subsets.  
   * Compute "Variance" or "Kurtosis" of the projection onto these subsets (a proxy for "Separation Score" when we don't have labels).  
   * Keep the top $N$ directions that show high non-Gaussianity (spikes).  
3. **Initialization:** Set the weights of the SAE encoder ($W\_{enc}$) to these mined directions.

### **B. Evaluation Metrics**

Compare **Standard SAE** vs. **RkCNN-SAE** on the following:

| Metric | Definition | Hypothesis |
| :---- | :---- | :---- |
| **L0 (Sparsity)** | Average number of active features per token. | RkCNN should achieve *lower* L0 for the same reconstruction error. |
| **Dead Latents** | % of neurons that never fire. | RkCNN should have **significantly fewer** dead latents (primary success metric). |
| **CE Loss Recovered** | How well the SAE output restores model performance. | Should be equal to or better than baseline. |

## **3\. Implementation Checklist (TransformerLens)**

**Tools Required:** TransformerLens (for hooks), SAELens (optional, for standard SAE training code).

* \[ \] **Step 1:** Write a ToyModel class (simple PyTorch linear layers) that forces superposition.  
* \[ \] **Step 2:** Implement RkCNN\_Probe:  
  * Function that takes activations (tensor).  
  * Loops $N$ times selecting random indices.  
  * Fits a simple LogisticRegression or KNN.  
  * Returns score and indices.  
* \[ \] **Step 3:** Run RkCNN\_Probe on Toy Model activations. Visualize the "winning" subsets—do they align with the ground truth feature vectors?  
* \[ \] **Step 4:** (Only after Step 3 works) Download gpt2-small.  
* \[ \] **Step 5:** Implement custom SAE initialization using the vectors found in Step 2\.

## **4\. Pitfalls & "Interpretability Illusions"**

**Warning:** Just because a subset separates data doesn't mean it's a *causal feature*.

* **The Correlation Trap:** In high dimensions, random vectors can correlate with almost anything by chance (Bleaching).  
* **Mitigation:** When evaluating on GPT-2, use **Causal Scrubbing**.  
  * If RkCNN says "Subset A encodes 'The Eiffel Tower'", then ablating *everything except* Subset A should still allow the model to predict "Paris".