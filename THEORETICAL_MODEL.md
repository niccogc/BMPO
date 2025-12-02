# System Context: Generic Bayesian Tensor Network (GBTN)

**Role:** You are an expert in Bayesian Tensor Networks, Variational Inference, and Probabilistic Graphical Models.
**Objective:** Implement or derive the Variational Inference (VI) algorithm for a generic Tensor Network using Moment Matching.

## 1. Model Specification

### 1.1 Components
*   **Structure:** A Tensor Network defined by a set of **Nodes** and **Bonds** (Edges).
*   **Inference Method:** Bayesian Variational Inference (VI) via Moment Matching.
*   **Distributions:**
    *   **Likelihood ($L$):** Gaussian with precision $\tau$.
    *   **Node Prior ($P$) / Posterior ($Q$):** Gaussian.
    *   **Bond Prior ($p$) / Posterior ($q$):** Gamma.

### 1.2 Variational Approximation
The joint probability $p$ and variational family $q$ factorize as:
$$
\begin{aligned}
    p &= L(\bm{y}|\bm{A}, \bm{x}, \tau) \prod_{i \in \text{nodes}} P_i(A^i) \prod_{j \in \text{bonds}} p_j(\lambda_j) \\
    q &= \prod_{i \in \text{nodes}} Q_i(A^i) \prod_{j \in \text{bonds}} q_j(\lambda_j)
\end{aligned}
$$

---

## 2. Data Structures & State Variables

Often we will use polynomial models, so there should be support for repeated input nodes. Meaning that 3 repeated nodes shouldnt occupy three times the memory.

### 2.1 Node Parameters (Gaussian)
For every node $i$, the distribution $Q_i$ is parameterized by:
*   **$\bm{\mu}^i$ (Mean Tensor):** $\mathbb{E}[A^i]$. Maintains the original topology/dimensions of the tensor network.
*   **$\bm{\Sigma}^i$ (Covariance Tensor):** $\mathbb{E}[A^i \otimes A^i] - \bm{\mu}^i \otimes \bm{\mu}^i$. Has the same topology as $\bm{\mu}$ but **squared bond dimensions**.

### 2.2 Bond Parameters (Gamma)
For every bond (edge) $i$, the distribution $q_i$ is a Gamma distribution over dimension $j$:
*   $q_i(j) = \text{Gamma}(\lambda_{ij} | \alpha_{ij}, \beta_{ij})$
*   **State:** $\vec{\alpha}_i, \vec{\beta}_i$ (Vectorized parameters for all dimensions of bond $i$).

### 2.3 Likelihood Parameter
*   **$\tau$ (Precision):** Gamma distributed with parameters $\alpha_\tau, \beta_\tau$.

---

## 3. Helper Definitions

*   **$B(i)$:** Set of bonds connected to Node $i$.
*   **$A(k)$:** Set of learnable nodes (excludes inputs and fixed nodes) connected to Bond $k$.
*   **$T_i \bm{A}$ (Projection):** The derivative $\frac{\partial \bm{A}}{\partial A^i}$. represents the contraction of the entire network *excluding* node $A^i$.
*   **$\bm{\theta}^{B(i)}$ (Bond Expectations):** A tensor representing the expectation of the bond variables connected to node $i$.
    $$ \theta^{B(i)} = \bigotimes_{b \in B(i)} \mathbb{E}[\vec{\lambda}_b] \quad \text{where } \mathbb{E}[\lambda] = \alpha / \beta $$
    *Note: When used in linear algebra operations, $\bm{\theta}$ acts as a diagonal matrix.*

---

## 4. Inference Algorithm (Update Rules)

Iterate the following updates until convergence.

### Step 1: Update Node Distributions ($Q_i$)
Update the mean ($\mu$) and covariance ($\Sigma$) for a specific node $i$ given inputs $x_n$ and targets $y_n$.

$$
\begin{aligned}
    \hat{\Sigma}^{i} &= \left[ \mathbb{E}[\tau] \left( \sum_n (T_i \bm{\Sigma}) \cdot (x_n \otimes x_n) + \sum_n (T_i \bm{\mu} x_n) \otimes (T_i \bm{\mu} x_n) \right) + \bm{\theta}^{B(i)} \right]^{-1} \\
    \hat{\mu}^{i} &= \mathbb{E}[\tau] \hat{\Sigma}^{i} \sum_n y_n (T_i \bm{\mu} x_n)
\end{aligned}
$$

### Step 2: Update Bond Distributions ($q_{bond}$)
Update the Gamma parameters for bond $i$.
*   $\text{dim}(i)$: Dimension of bond $i$.
*   $|A(i)|$: Number of learnable nodes sharing bond $i$.
*   $\text{tr}_{B(a)/i}$: Partial trace over all bonds of node $a$ *except* bond $i$.
*   Trace of a diagonal matrix by a full matrix has a simple form to be computed.

$$
\begin{aligned}
    \vec{\alpha}_{i} &= \vec{\alpha}_{i}^0 + \frac{|A(i)|\text{dim}(i)}{2} \\
    \vec{\beta}_{i} &= \vec{\beta}_{i}^0 + \frac{1}{2} \sum_{a \in A(i)} \text{tr}_{B(a)/i} \left[ (\Sigma^a + \mu^a \otimes \mu^a) \bm{\theta}^{B(a)/i} \right]
\end{aligned}
$$

### Step 3: Update Global Noise ($\tau$)
Update the precision of the likelihood.
*   $S$: Total number of samples.
*   $\bm{y}$: Target vector.
*   $\bm{\mu x}$: Prediction of the mean network.

$$
\begin{aligned}
    \alpha_\tau &= \alpha_\tau^0 + \frac{S}{2} \\
    \beta_\tau &= \beta_\tau^0 - \|\bm{y}\|^2 + \bm{y} \cdot (\bm{\mu x}) + \frac{1}{2} \sum_n \left( \bm{\Sigma}(x_n \otimes x_n) + (\bm{\mu} x_n) \otimes (\bm{\mu} x_n) \right)
\end{aligned}
$$

---

## 5. Edge Cases & Notes

1.  **Multiple Outputs:** If the network has multiple output modes (e.g., multi-class classification), sum the update terms over the class outputs in addition to the sample summation.
2.  **Expectations:**
    *   $\mathbb{E}[\tau] = \alpha_\tau / \beta_\tau$
    *   $\mathbb{E}[\lambda] = \alpha_{bond} / \beta_{bond}$
3.  **Topology:** $\bm{\mu}$ network preserves original topology. $\bm{\Sigma}$ network duplicates bonds (squared dimensions).
