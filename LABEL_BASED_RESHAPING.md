# Label-Based Σ Reshaping

## Problem

Previously, Σ-block reshaping assumed a specific order of outer/inner indices:
```python
# OLD: Assumes alternating outer/inner pattern
outer_indices = list(range(0, 2*n_dims, 2))  # [0, 2, 4, ...]
inner_indices = list(range(1, 2*n_dims, 2))  # [1, 3, 5, ...]
```

This is fragile and breaks if the node construction order changes.

## Solution

Use **label-based mapping** to identify outer and inner dimensions:

### New Utility Methods

**1. `_get_sigma_to_mu_permutation(block_idx)`**

Maps μ-node labels to corresponding Σ-node outer/inner indices.

```python
# μ labels: ['r1', 'c2', 'p2', 'r2']
# Σ labels: ['r1o', 'r1i', 'c2o', 'c2i', 'p2o', 'p2i', 'r2o', 'r2i']

outer_idx, inner_idx = bmpo._get_sigma_to_mu_permutation(1)
# Returns: ([0, 2, 4, 6], [1, 3, 5, 7])
```

**Algorithm:**
- For each μ label, find `label + 'o'` and `label + 'i'` in Σ labels
- Return their indices

**2. `_sigma_to_matrix(block_idx)`**

Converts Σ-block tensor to (d, d) matrix using label-based permutation.

```python
sigma_matrix = bmpo._sigma_to_matrix(block_idx)
# Returns: (d, d) matrix where d = μ.numel()
```

**Algorithm:**
1. Get outer/inner indices using labels
2. Permute Σ tensor: [outer dims, then inner dims]
3. Reshape to (d, d) matrix

## Refactored Methods

### `get_block_q_distribution(block_idx)`

**Before:**
```python
# Assumed order
outer_indices = list(range(0, 2*n_dims, 2))
inner_indices = list(range(1, 2*n_dims, 2))
perm = outer_indices + inner_indices
```

**After:**
```python
# Label-based
sigma_matrix = self._sigma_to_matrix(block_idx)
```

Much cleaner! ~10 lines → 1 line

### `partial_trace_update(block_idx, focus_label)`

**Before:**
```python
# Manual permutation
outer_indices = list(range(0, 2*n_dims, 2))
inner_indices = list(range(1, 2*n_dims, 2))
perm = outer_indices + inner_indices
sigma_permuted = sigma_tensor.permute(*perm)
sigma_matrix = sigma_permuted.reshape(d, d)
diag_sigma_flat = torch.diagonal(sigma_matrix)
```

**After:**
```python
# Label-based
sigma_matrix = self._sigma_to_matrix(block_idx)
diag_sigma_flat = torch.diagonal(sigma_matrix)
```

Simpler and more robust!

## Benefits

1. **Robust**: Works regardless of node construction order
2. **Clear**: Labels explicitly show the mapping
3. **Maintainable**: Less magic numbers, more semantic
4. **Verified**: All existing tests pass ✓

## Test Results

```
Block 0:
  μ labels: ['c1', 'p1', 'r1']
  Σ labels: ['c1o', 'c1i', 'p1o', 'p1i', 'r1o', 'r1i']
  Outer indices: [0, 2, 4]
  Inner indices: [1, 3, 5]
    c1 -> c1o, c1i ✓
    p1 -> p1o, p1i ✓
    r1 -> r1o, r1i ✓

Block 1:
  μ labels: ['r1', 'c2', 'p2', 'r2']
  Σ labels: ['r1o', 'r1i', 'c2o', 'c2i', 'p2o', 'p2i', 'r2o', 'r2i']
  Outer indices: [0, 2, 4, 6]
  Inner indices: [1, 3, 5, 7]
    r1 -> r1o, r1i ✓
    c2 -> c2o, c2i ✓
    p2 -> p2o, p2i ✓
    r2 -> r2o, r2i ✓
```

All tests pass! ✓

## Files Modified

- `tensor/bayesian_mpo.py`:
  - Added `_get_sigma_to_mu_permutation()`
  - Added `_sigma_to_matrix()`
  - Refactored `get_block_q_distribution()` 
  - Refactored `partial_trace_update()`
  - Removed TODO comment about label-based indexing ✓

## Related TODOs Addressed

- Line 527: "TODO: FIND OUTER AND INNER through label, not order." ✓ DONE
