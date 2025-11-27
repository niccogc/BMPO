# Block Variational Update Implementation

## Summary

Successfully implemented `update_block_variational` method for Bayesian MPO that updates both μ (mean) and Σ (covariance) parameters for individual blocks using variational inference.

## Implementation Details

### Update Equations

For block i with parameters (μ^i, Σ^i):

```
μ^i = -E[τ] Σ^i Σₙ yₙ · J_μ(xₙ)
Σ^i⁻¹ = -E[τ] Σₙ [J_Σ(xₙ) + J_μ(xₙ) ⊗ J_μ(xₙ)] - diag(Θ)
```

Where:
- `J_μ(xₙ) = ∂μ(xₙ)/∂block_i` - gradient of μ-MPO output w.r.t. block i
- `J_Σ(xₙ)` - gradient of Σ-MPO output w.r.t. Σ-block i (currently approximated as zero)
- `Θ = outer_product(E[bond_1], E[bond_2], ...)` - theta tensor from Gamma expectations
- `diag(Θ)` - diagonal matrix constructed from flattened Θ

### Key Implementation Choices

1. **Gradient Computation**: Uses PyTorch autograd to compute `∂μ(x)/∂block`
   - Enables gradients on block tensor
   - Forward pass for each sample
   - Backward pass to get gradients
   - More reliable than manual Jacobian manipulation

2. **Batched Processing**: Loops over samples to accumulate:
   - `Σₙ yₙ · ∇_block μ(xₙ)` - weighted gradient sum
   - `Σₙ ∇_block μ(xₙ) ⊗ ∇_block μ(xₙ)` - gradient outer product (Gramian)

3. **Σ-Jacobian Approximation**: Currently `J_Σ ≈ 0`
   - Reasonable when variance is small relative to mean
   - TODO: Implement full Σ-Jacobian computation for more accurate updates

4. **Numerical Stability**: 
   - Adds small regularization (ε=1e-6) to precision matrix before inversion
   - Uses label-based Σ reshaping (robust to node construction order)

### Method Signature

```python
def update_block_variational(
    self, 
    block_idx: int, 
    X: torch.Tensor,  # Shape: (S, feature_dims)
    y: torch.Tensor   # Shape: (S,) for scalar output
) -> None
```

### Testing

Test file: `test_block_update.py`

Creates a 2-block Bayesian TT and successfully updates both blocks:
- Block 0: shape (1, 4, 3) ✓
- Block 1: shape (3, 1, 4) ✓

All updates complete without errors and maintain correct tensor shapes.

## Integration with Existing Infrastructure

The implementation uses:
- `compute_theta_tensor()` - for computing Θ from bond expectations
- `_sigma_to_matrix()` - for converting Σ-block to matrix form  
- `_matrix_to_sigma()` - for converting matrix back to Σ-block
- `forward_mu()` - for network forward passes
- PyTorch autograd - for gradient computation

## Next Steps

1. **Implement Σ-Jacobian computation** for more accurate covariance updates
2. **Implement full coordinate ascent loop** combining:
   - τ updates (`update_tau_variational`) ✓
   - Bond updates (`update_bond_variational`) ✓
   - Block updates (`update_block_variational`) ✓
3. **Test on real datasets** to verify convergence
4. **Optimize performance** - potentially batch gradient computation

## Files Modified

- `tensor/bayesian_mpo.py`: Added `update_block_variational` method
- `test_block_update.py`: Test file for block updates

## Related Documentation

- `BOND_UPDATE_SUMMARY.md` - Bond parameter updates
- `PARTIAL_TRACE_SUMMARY.md` - Partial trace computation
- `LABEL_BASED_RESHAPING.md` - Σ-block reshaping
- `UNIFIED_BONDS_SUMMARY.md` - Unified bond parameterization
