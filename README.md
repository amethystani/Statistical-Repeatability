# Statistical-Repeatability-Analysis

## Abstract

This repository encompasses the full implementation of advanced statistical methodologies focused on quantifying and analyzing repeatability metrics within high-dimensional datasets. Grounded in the framework proposed by Zeyi Wang et al. (2020), the project uses discriminability measures, rank-based metrics, and permutation-driven hypothesis testing across diverse dataset configurations. The objective is to use systemic patterns in data reproducibility while optimizing preprocessing pipelines and benchmarking robustness under extreme conditions.

---

## Scope of Work

1. **Methodological Replication**: 
   - Deploy rigorous statistical measures derived from multivariate and nonparametric methodologies.
   - Validate the theoretical underpinnings of discriminability, fingerprinting, and rank sums in empirical contexts.

2. **Metric Evaluation**:
   - Examine interdependencies between repeatability statistics and their computational convergence properties.
   - Explore boundary cases (e.g., Gaussian assumptions, batch effects).

3. **Augmented Applications**:
   - Extend statistical metrics to accommodate scaling challenges and batch-level data perturbations.
   - Provide actionable insights for repeatability optimization in functional data pipelines.

---

## Technical Implementations

### 1. **Discriminability Analysis**
The discriminability metric encapsulates a probabilistic framework for evaluating intersubject repeatability. Its computation leverages multi-level nested permutations and nonparametric rank assignments across multivariate embeddings. Specific emphasis is placed on resolving computational bottlenecks in \(O(n^3)\) scaling scenarios.

### 2. **Fingerprinting Metrics**
Fingerprint indices operationalize a pairwise comparison methodology, emphasizing probabilistic alignment in subject-specific feature subspaces. Advanced matching heuristics ensure robustness to dimensionality reduction artifacts.

### 3. **Rank-Based Summation Techniques**
Rank sums are computed as ordinal transformations across subspaces defined by pairwise proximity matrices. This method introduces asymptotically unbiased estimators for batch-invariant evaluations.

---


![screenshot](imageFolder/screenshot.png)

## Repository Structure

```plaintext
.
├── data/                             # Raw dataset files
├── src/
│   ├── discriminability.py           # Discriminability implementation
│   ├── fingerprinting.py             # Fingerprinting implementation
│   ├── rank_sums.py                  # Rank sums implementation
├── analysis_results/                 # Numerical results
├── README.md            
