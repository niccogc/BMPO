# Unified Bond Parameterization - Summary

## What Was Changed

Refactored the bond parameterization system to use a unified structure for all bonds (both horizontal and vertical).

### Before (Old Structure)

**Two different parameterizations:**
- Rank dimensions (horizontal bonds like 'r1', 'r2'): `Gamma(c, e)` with variable ω
- Vertical dimensions (like 'c1', 'p1'): `Gamma(f, g)` with variable φ

**Distribution structure:**
```python
{
    'type': 'rank' or 'vertical',
    'variable': 'omega' or 'phi',
    'c'/'f': tensor,
    'e'/'g': tensor,
    'expectation': tensor
}
```

### After (New Unified Structure)

**Single parameterization for all bonds:**
- All bonds: `Gamma(concentration, rate)` (i.e., `Gamma(α, β)`)

**Distribution structure:**
```python
{
    'concentration': tensor,  # α parameters
    'rate': tensor,           # β parameters  
    'expectation': tensor     # E[X] = α/β
}
```

## Key Features

### 1. Each Bond Has N Independent Gamma Distributions

For a bond with dimension N (e.g., 'r1' with size 4):
- 4 independent Gamma distributions (one per index)
- Each Gamma has its own `concentration` and `rate` parameters
- Total: 4 × 2 = 8 parameters for this bond

**Example:**
```python
Bond 'r1': 4 Gammas × 2 params = 8 params
  Index 0: Gamma(α=3.00, β=1.50)
  Index 1: Gamma(α=2.50, β=1.00)
  Index 2: Gamma(α=4.00, β=2.00)
  Index 3: Gamma(α=2.00, β=1.00)
```

### 2. Bond-to-Nodes Mapping

New method `get_nodes_for_bond(label)` returns which blocks share a bond:

```python
bmpo.get_nodes_for_bond('r1')
# Returns: {'mu_nodes': [0, 1], 'sigma_nodes': [0, 1]}
```

**Horizontal bonds** (like 'r1', 'r2'):
- Connect two consecutive blocks
- Example: 'r1' connects blocks [0, 1]

**Vertical bonds** (like 'c1', 'p1'):
- Contained within a single block
- Example: 'p1' only in block [0]

### 3. Prior Structure Matches Variational

Prior `p(θ)` has the **exact same factorization** as variational `q(θ)`:

```python
# Variational parameters
bmpo.mu_mpo.distributions['r1']
# {'concentration': tensor([2., 2., 2., 2.]),
#  'rate': tensor([1., 1., 1., 1.])}

# Prior parameters (same structure)
bmpo.prior_bond_params['r1']
# {'concentration0': tensor([1., 1., 1., 1.]),
#  'rate0': tensor([1., 1., 1., 1.])}
```

## API Changes

### Updated Methods

1. **`update_mu_params(label, concentration, rate)`**
   - Old: `update_mu_params(label, param1, param2)` (ambiguous names)
   - New: Clear parameter names

2. **`update_distribution_params(label, concentration, rate)`**
   - Unified across all bond types

3. **`set_prior_hyperparameters(..., bond_params0, ...)`**
   - Old: `mode_params0` with different keys for rank/vertical
   - New: `bond_params0` with unified structure

### New Methods

1. **`get_nodes_for_bond(label)`**
   - Returns which μ and σ blocks share this bond
   - Useful for understanding network structure

2. **`_build_bond_to_nodes_mapping()`**
   - Internal method to build the mapping
   - Called automatically in `__init__`

## Verification Results

### Test 1: Parameter Counting ✓

**Example network (3 blocks, bond_dim=4, input_features=5):**

| Component | Count | Notes |
|-----------|-------|-------|
| μ-MPO blocks | 3 | Shapes: (1,5,4), (4,1,5,4), (4,1,5) |
| Σ-MPO blocks | 3 | Doubled dimensions |
| Total bonds | 8 | c1, p1, r1, c2, p2, r2, c3, p3 |
| Total Gamma params (q) | 52 | 26 concentration + 26 rate |
| Total Gamma params (prior) | 52 | Matches variational ✓ |

### Test 2: Block-Bond Associations ✓

**Example bond 'r1' (horizontal):**
```
μ-MPO blocks: [0, 1]
Σ-MPO blocks: [0, 1]
In block 0: dimension 2, size 4
In block 1: dimension 0, size 4
```

**Example bond 'p1' (vertical):**
```
μ-MPO blocks: [0]
Σ-MPO blocks: [0]
In block 0: dimension 1, size 5
```

### Test 3: Gamma Parameters Per Node ✓

Each index in each bond can have different parameters:
```
Bond 'p2' has 5 Gamma distributions:
  Index 0: Gamma(α=5.00, β=2.00), E[X]=2.50
  Index 1: Gamma(α=4.50, β=2.00), E[X]=2.25
  Index 2: Gamma(α=4.00, β=2.00), E[X]=2.00
  Index 3: Gamma(α=3.50, β=2.00), E[X]=1.75
  Index 4: Gamma(α=3.00, β=2.00), E[X]=1.50
```

### Test 4: Trimming Still Works ✓

**Before trimming 'r1' (threshold=2.0):**
- Expectations: [4.0, 1.0, 3.0, 0.5]
- Block 0 shape: (1, 5, 4)
- Block 1 shape: (4, 1, 5, 4)
- Number of Gammas: 4

**After trimming:**
- Expectations: [4.0, 3.0] ✓ (kept indices 0 and 2)
- Block 0 shape: (1, 5, 2) ✓
- Block 1 shape: (2, 1, 5, 4) ✓
- Number of Gammas: 2 ✓
- Forward passes work ✓

## Files Modified

1. **tensor/bmpo.py**
   - `_initialize_distributions()`: Unified structure
   - `update_distribution_params()`: New parameter names
   - `get_gamma_distributions()`: Uses unified structure
   - `trim()`: Fixed to use new structure
   - `_build_bond_to_nodes_mapping()`: New method
   - `get_nodes_for_bond()`: New method

2. **tensor/bayesian_mpo.py**
   - `_initialize_prior_hyperparameters()`: Uses `prior_bond_params`
   - `set_prior_hyperparameters()`: Uses `bond_params0`
   - `update_mu_params()`: New parameter names
   - `_expected_log_prior_modes()`: Uses unified structure
   - `get_nodes_for_bond()`: New method

3. **New examples:**
   - `example_unified_bonds.py`: Demonstrates new API
   - `test_unified_bonds_comprehensive.py`: Comprehensive tests

## Backward Compatibility

All existing functionality still works:
- ✓ Forward passes (μ-MPO and Σ-MPO)
- ✓ Tau variational updates
- ✓ Bond parameter updates
- ✓ Expected log prior computation
- ✓ Trimming
- ✓ All existing tests pass

## Benefits

1. **Consistency**: All bonds use same parameterization
2. **Clarity**: No more confusion about c/e vs f/g
3. **Simplicity**: Less code branching based on bond type
4. **Flexibility**: Each index can have different parameters
5. **Transparency**: Easy to see which blocks share which bonds
6. **Maintainability**: Cleaner, more maintainable code
