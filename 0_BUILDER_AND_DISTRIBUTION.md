# Bayesian Tensor Network (BTN) Workflow

This document explains the initialization and structure of the BTN as demonstrated in `showcase_btn.py`.

## 1. Network Initialization (Mu)

The process starts with a standard `quimb.TensorNetwork` representing the mean (μ) of the weights.
Example: a 3-node chain

* **T1:** (x1, k1)
* **T2:** (k1, x2, y1, k2)
* **T3:** (k2, y2)

Where:

* *k1, k2*: internal bond indices
* *x1, x2*: input indices (have priors)
* *y1, y2*: output indices (no priors)

## 2. The Builder Process

Creating `BTN(mu, ...)` triggers `BTNBuilder`, which constructs three components:

### A. Edge Distributions (Gamma)

Created for every index not in `output_dimensions` and not equal to `batch_dim`.

* Parameters: concentration α, rate β
* Stored as `quimb.Tensor`
* Tags: `{bond}_alpha`, `{bond}_beta`, plus `"prior"` or `"posterior"`
* Each parameter tensor carries the bond index itself, enabling contraction.

**Example:**
Bond `k1` prior → α has index `(k1,)` and tags `{k1_alpha, prior}`.

### B. Sigma Topology

Represents covariance structure for each node.

* Duplicates all non-output indices with `_prime` suffix
* Output indices stay shared

**Example (T2):**

* Original: `(k1, x2, y1, k2)`
* Sigma: `(k1, x2, k2, k1_prime, x2_prime, k2_prime, y1)`
* Tags: `{T2_sigma}`

### C. Node Distributions (Multivariate Gaussian)

Each node becomes a Gaussian distribution:

* **Posterior Q:**

  * Mean: original μ tensor
  * Covariance: σ tensor
* **Prior P:**

  * Mean: zero tensor
  * Precision: determined by Gamma edge priors

Nodes are keyed by sorted tags, e.g., `('LAYER_2', 'T2')`.

## 3. Data Flow

* **Input:** user supplies μ network
* **Builder:** scans indices → builds Gamma priors/posteriors → builds σ topology → binds distributions
* **Forward pass:** uses μ for contraction
* **Optimization:** expectations (mean, entropy, etc.) return `quimb.Tensor` objects retaining index labels for contraction
