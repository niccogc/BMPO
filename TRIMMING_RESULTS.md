# Bayesian MPO Trimming Results

## Summary

Successfully implemented and validated automatic rank reduction (trimming) for Bayesian MPO models. The trim functionality correctly reduces model complexity while maintaining or improving performance.

---

## Key Findings

### 1. Bond Evolution Without Trimming (rank=8, 50 iterations)

**Configuration:**
- Initial rank: 8 for both r1 and r2 bonds
- Random prior initialization
- No automatic trimming

**Results:**
- **Final MSE:** 0.1953
- **MSE Improvement:** 78.3% (0.8992 → 0.1953)
- **Final E[τ]:** 3.93
- **Sparsity:** NO natural sparsification observed
  - All 8 dimensions remain active (none < 1.0)
  - r1: min=3.37, max=13.85, mean=6.85
  - r2: min=3.50, max=24.47, mean=9.36

**Key Insight:** Without explicit trimming, the model does NOT naturally push any rank dimensions to zero. All dimensions remain active with expectations between 3-25.

**Visualization:** `bond_evolution.png`

---

### 2. Manual Trimming Test (rank 8→2)

**Configuration:**
- Initial rank: 8
- Trained for 10 iterations
- Manual trim with thresholds: r1≥12.0, r2≥16.0
- Continued training for 10 more iterations

**Results:**

| Stage | r1 rank | r2 rank | MSE | Notes |
|-------|---------|---------|-----|-------|
| Initial | 8 | 8 | 0.8992 | Untrained |
| Pre-trim (iter 10) | 8 | 8 | 0.2920 | After training |
| Post-trim | 2 | 2 | 0.8991 | Immediately after trim |
| Final (iter 20) | 2 | 2 | **0.0117** | After continued training |

**Key Results:**
- ✅ `prior_bond_params` correctly updated after trim
- ✅ Model successfully trained after trimming
- ✅ Final performance **better** than pre-trim (0.2920 → 0.0117)
- Rank reduced by 75% with improved accuracy!

---

### 3. Automatic Trimming During Training (threshold=10)

**Configuration:**
- Initial rank: 8
- Trim threshold: 10.0 (applied after each iteration)
- Training: 30 iterations

**Results:**

| Iteration | r1 rank | r2 rank | MSE | Event |
|-----------|---------|---------|-----|-------|
| 1 | 8 | 8 | 0.8992 | Initial |
| 5 | 8 | 8 | 0.3325 | Before trim |
| 6 | 1 | 2 | 0.8085 | **Trim event** |
| 30 | 1 | 2 | **0.0801** | Final |

**Key Results:**
- **MSE Improvement:** 91.1% (0.8992 → 0.0801)
- **Rank Reduction:** 
  - r1: 8 → 1 (87.5% reduction)
  - r2: 8 → 2 (75.0% reduction)
- **Trim Event:** Occurred at iteration 6
- **Final E[τ]:** 10.59 (excellent noise precision)

**Observations:**
- After trimming at iteration 6, MSE jumped from 0.33 → 0.81 (expected)
- Model recovered quickly: by iteration 12, MSE back to 0.080
- Compact model (rank 1-2) performs better than full rank 8 model!

**Visualization:** `auto_trim_evolution.png`

---

## Technical Details

### Trim Implementation

The `trim()` method in `tensor/bayesian_mpo.py`:

1. **Determines indices to keep** based on μ-expectations and thresholds
2. **Trims μ-MPO** using `BMPONetwork.trim()`
3. **Trims Σ-MPO** by trimming both 'o' and 'i' dimensions
4. **Updates `prior_bond_params`** to match new dimensions ✓ (Fixed!)
   ```python
   self.prior_bond_params[label]['concentration0'] = torch.index_select(old_conc, 0, indices)
   self.prior_bond_params[label]['rate0'] = torch.index_select(old_rate, 0, indices)
   ```

### Why Manual Trimming is Necessary

The Bayesian MPO with standard Gamma priors does **not** naturally induce sparsity:

- **Gamma priors** encourage positive values but don't push to zero
- All rank dimensions remain active (expectations 3-25)
- Explicit trimming required to reduce model complexity

**Potential Future Work:**
- Implement sparsity-inducing priors (e.g., spike-and-slab)
- Add regularization to encourage dimension suppression
- Investigate ARD (Automatic Relevance Determination) priors

---

## Prior Initialization

✅ Random prior initialization implemented:

```python
bmpo = create_bayesian_tensor_train(
    ...,
    random_priors=True,
    prior_seed=42
)
```

**Example random priors (rank 8):**
- r1 concentration0: [4.80, 4.68, 0.88, 1.10, 1.21, 2.19, 4.29, 4.42]
- r1 rate0: [2.20, 3.26, 0.90, 3.66, 3.31, 2.47, 0.84, 3.58]
- r2 concentration0: [4.69, 1.18, 1.69, 1.09, 1.63, 1.55, 1.52, 1.93]
- r2 rate0: [3.26, 3.19, 4.32, 0.65, 1.89, 2.45, 4.70, 1.05]

---

## Recommendations

### For Best Performance:

1. **Start with higher rank** (e.g., 8) for expressiveness
2. **Use automatic trimming** with moderate threshold (e.g., 10)
3. **Enable random priors** for better exploration
4. **Train longer** after trimming to recover performance

### Example Usage:

```python
bmpo = create_bayesian_tensor_train(
    num_blocks=3,
    bond_dim=8,
    input_features=2,
    output_shape=1,
    random_priors=True,
    prior_seed=42
)

# Train with automatic trimming
bmpo.fit(X_train, y_train, max_iter=30, trim_threshold=10.0, verbose=True)
```

---

## Conclusion

✅ **All tests passed successfully!**

The trimming functionality:
- Correctly reduces model rank
- Maintains `prior_bond_params` consistency
- Enables continued training post-trim
- Often **improves** final performance (smaller, better models)

The final compact models achieve:
- **91% MSE reduction** with automatic trimming
- **75-87% rank reduction**
- Excellent noise precision (E[τ] ≈ 10.6)
