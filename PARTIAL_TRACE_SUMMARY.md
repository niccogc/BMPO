# Partial Trace Update Method

## Purpose

The `partial_trace_update` method computes the partial trace needed for variational updates of bond parameters. It contracts over all dimensions except one "focus" dimension.

## Formula

```
result_i = Σ_{j,k,...} [diag(Σ) + μ²]_{ijk...} × Θ_{jk... (excluding i)}
```

Where:
- `i` is the focus dimension (the one we keep)
- `j, k, ...` are all other dimensions (summed over)
- `diag(Σ)` is the diagonal of the Σ-block
- `μ²` is element-wise squared μ-block
- `Θ` is the theta tensor excluding the focus label

## Method Signature

```python
bmpo.partial_trace_update(block_idx, focus_label)
```

**Parameters:**
- `block_idx`: Index of the block (0 to N-1)
- `focus_label`: The bond label to keep (not sum over)

**Returns:**
- Vector of length equal to the dimension of `focus_label`

## Algorithm Steps

1. **Extract diagonal of Σ-block**
   - Σ has shape `(d1_o, d1_i, d2_o, d2_i, ...)` with paired outer/inner indices
   - Flatten and permute to matrix form `(d, d)`
   - Extract diagonal and reshape back to block shape

2. **Compute v = diag(Σ) + μ²**
   - Element-wise addition

3. **Get Θ excluding focus_label**
   - Uses `compute_theta_tensor(block_idx, exclude_labels=[focus_label])`
   - Results in tensor where focus dimension has size 1

4. **Multiply v × Θ**
   - Element-wise multiplication (broadcasting handles size-1 dimension)

5. **Sum over all dimensions except focus**
   - Returns vector of size matching focus_label dimension

## Example

```python
# Block 1: labels=['r1', 'c2', 'p2', 'r2'], shape=(4, 1, 5, 4)

# Focus on 'p2' (dimension 2, size 5)
result = bmpo.partial_trace_update(1, 'p2')
# Returns: tensor of shape (5,)

# This computes:
# result[l] = Σ_{i,j,k} [diag(Σ) + μ²]_{ijlk} × Θ_{ijk (without p2)}
# where sum is over r1(i), c2(j), r2(k), keeping only p2(l)
```

## Test Results

```
Block 0: ['c1', 'p1', 'r1'], shape=(1, 5, 4)
  'p1' (size 5): output shape (5,) ✓
  'r1' (size 4): output shape (4,) ✓

Block 1: ['r1', 'c2', 'p2', 'r2'], shape=(4, 1, 5, 4)
  'r1' (size 4): output shape (4,) ✓
  'p2' (size 5): output shape (5,) ✓
  'r2' (size 4): output shape (4,) ✓

Block 2: ['r2', 'c3', 'p3'], shape=(4, 1, 5)
  'r2' (size 4): output shape (4,) ✓
  'p3' (size 5): output shape (5,) ✓
```

**Manual verification:** ✓ Matches exact computation
**All tests pass:** ✓

## Use Case

This method is used in variational coordinate ascent updates for bond parameters (Gamma distributions). For each bond, we need to compute updates that involve contracting over all other dimensions while keeping the bond dimension we're updating.

## Files Modified

- `tensor/bayesian_mpo.py`: Added `partial_trace_update()` method
- `test_partial_trace.py`: Comprehensive tests including manual verification
