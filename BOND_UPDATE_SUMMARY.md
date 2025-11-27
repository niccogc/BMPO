# Bond Variational Update

## Method

`update_bond_variational(label)`

Updates the Gamma distribution parameters for a bond using coordinate ascent.

## Update Formulas

For bond with label (e.g., 'r1', 'p2'):

```
concentration_q = concentration_p + N_b × dim(bond)

rate_q = rate_p + Σ_{blocks with bond} partial_trace_update(block, label)
```

**Where:**
- `N_b` = number of learnable blocks associated with the bond (1 or 2)
- `dim(bond)` = dimension size of the bond
- `concentration_p`, `rate_p` = prior hyperparameters
- `partial_trace_update(block, label)` = Σ_{others} [diag(Σ) + μ²] × Θ_{without label}

## Examples

### Vertical Bond (1 block)

```python
# Bond 'p1': connects block [0], size 5
bmpo.update_bond_variational('p1')

# Updates:
# concentration_q = 1.0 + 1 × 5 = 6.0
# rate_q = 1.0 + partial_trace(block=0, 'p1')
```

### Horizontal Bond (2 blocks)

```python
# Bond 'r1': connects blocks [0, 1], size 4
bmpo.update_bond_variational('r1')

# Updates:
# concentration_q = 1.0 + 2 × 4 = 9.0
# rate_q = 1.0 + partial_trace(block=0, 'r1') 
#              + partial_trace(block=1, 'r1')
```

## Test Results

```
Bond 'p1' (vertical, 1 block):
  Before: concentration = 2.00, rate = 1.00
  After:  concentration = 6.00, rate = 5.06
  Formula: 1.0 + 1×5 = 6.0 ✓

Bond 'p2' (vertical, 1 block):
  Before: concentration = 2.00, rate = 1.00
  After:  concentration = 6.00, rate = 33.77
  Formula: 1.0 + 1×5 = 6.0 ✓

Bond 'r1' (horizontal, 2 blocks):
  Before: concentration = 2.00, rate = 1.00
  After:  concentration = 9.00, rate = 7.45
  Formula: 1.0 + 2×4 = 9.0 ✓

Bond 'r2' (horizontal, 2 blocks):
  Before: concentration = 2.00, rate = 1.00
  After:  concentration = 9.00, rate = 6.07
  Formula: 1.0 + 2×4 = 9.0 ✓
```

**All parameters valid for Gamma distributions (> 0)** ✓

## Multiple Updates

Running multiple updates shows convergence behavior:

```
Bond 'r1': 
  Iteration 0: concentration = 9.00, rate = 7.45
  Iteration 1: concentration = 9.00, rate = 6.46  (converging)
  Iteration 2: concentration = 9.00, rate = 6.46  (converged)
```

Note: Concentration stays constant (only depends on prior and structure),
while rate converges as the model updates.

## Implementation Details

1. **Gets bond info**: size, associated blocks
2. **Computes concentration**: trivial formula
3. **Computes rate**: sums partial traces from all associated blocks
4. **Updates parameters**: via `update_mu_params()`

## Files Modified

- `tensor/bayesian_mpo.py`: Added `update_bond_variational()` method
- `test_bond_update.py`: Comprehensive tests

## Next Steps

- Implement full coordinate ascent loop
- Update all bonds iteratively
- Integrate with τ updates
- Add convergence checking

