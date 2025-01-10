# Statistical-Repeatability-Analysis

## Overview

This repository implements and analyzes statistical repeatability measures inspired by the methodologies in **"Statistical Analysis of Data Repeatability Measures"** by Zeyi Wang et al. The project applies advanced repeatability measures, including discriminability, fingerprinting, and rank sums, to a novel dataset. The analysis provides actionable insights for data repeatability in high-dimensional and multivariate settings, forming the basis for a forthcoming research paper.

---

## Objectives

1. **Replication and Validation**:
   - Reproduce the statistical methods described in the referenced paper.
   - Validate their effectiveness on a different dataset.

2. **Evaluation**:
   - Compare the robustness of discriminability, rank sums, and fingerprinting under various dataset conditions, including noise and batch effects.

3. **Application**:
   - Leverage repeatability measures to optimize preprocessing pipelines and inform best practices in data acquisition.

---

## Methodologies

### Intraclass Correlation Coefficient (ICC)

The intraclass correlation coefficient (ICC) is defined as:
$$
\text{ICC} = \frac{\sigma^2_\mu}{\sigma^2_\mu + \sigma^2}
$$
where:
- \( \sigma^2_\mu \) represents inter-subject variance.
- \( \sigma^2 \) represents intra-subject variance.

For multivariate data, ICC generalizes to:
$$
\Lambda = \frac{\det(\Sigma_\mu)}{\det(\Sigma_\mu) + \det(\Sigma)}
$$
where \( \Sigma_\mu \) and \( \Sigma \) are inter- and intra-subject covariance matrices, respectively.

---

### Discriminability

Discriminability quantifies the ability to distinguish repeated measurements of the same subject:
$$
D = P\left(\delta_{i,t,t'} < \delta_{i,i',t,t''}; \, \forall i \neq i'\right)
$$
where:
- \( \delta \) is a distance metric.
- \( t, t' \) are measurement indices.

The sample discriminability estimator is:
$$
\hat{D} = \frac{1}{n \cdot s \cdot (s - 1) \cdot (n - 1) \cdot s} \sum_{i=1}^n \sum_{t=1}^s \sum_{t' \neq t} \sum_{i' \neq i} \sum_{t''=1}^s I(\delta_{i,t,t'} < \delta_{i,i',t,t''})
$$

---

### Fingerprinting

Fingerprinting identifies the proportion of correct subject matches in repeated measurements:
$$
\text{Fingerprint Index} = \frac{1}{n} \sum_{i=1}^n I\left(\delta_{i,1,2} < \min_{i' \neq i} \delta_{i,i',1,2}\right)
$$

---

### Rank Sums

Rank sums extend fingerprinting by incorporating the ranks of distances:
$$
R_n = \sum_{i=1}^n \text{rank}\left(\delta_{i,1,2}\right)
$$
This measure is robust to batch effects and non-Gaussian noise.

---

### Batch Effects

The analysis accounts for two types of batch effects:
1. **Mean shifts**:
   $$
   \mu_{i,t} = \mu_{i,1} + t, \quad t > 1
   $$
2. **Scaling effects**:
   $$
   e_{i,t} \sim \mathcal{N}(0, t \cdot \sigma^2)
   $$

---




## Repository Structure

```plaintext
.
├── data/       # Raw dataset files
├── src/
│   ├── discriminability.py           # Discriminability implementation
│   ├── fingerprinting.py             # Fingerprinting implementation
│   ├── rank_sums.py                  # Rank sums implementation
├── analysis_results/                 # Numerical results
├── README.md            # This file
