# BTN Forward Operations Guide

Comprehensive guide to forward passes, environment calculations, and batched operations in the Bayesian Tensor Network (BTN) class.

---

## Table of Contents
1. [Input Preparation](#input-preparation)
2. [Forward Operations](#forward-operations)
3. [Environment Calculations](#environment-calculations)
4. [Forward with Target](#forward-with-target)
5. [Batched Operations](#batched-operations)
6. [Backend Compatibility](#backend-compatibility)
7. [Examples](#examples)

---

## Input Preparation

### `prepare_inputs(input_data, for_sigma=False)`

Prepares input tensors for forward pass through the network.

**Parameters:**
- `input_data`: Dict mapping input index names to data arrays
- `for_sigma`: If True, doubles inputs with `_prime` suffix for sigma network

**Two Scenarios:**

#### Scenario 1: Single Input for All Nodes
```python
# Same data replicated to all input nodes (x1, x2, ...)
input_data = {'x1': np.random.randn(batch_size, features)}
inputs = btn.prepare_inputs(input_data, for_sigma=False)
# Creates: x1(s, x1, data), x2(s, x2, data), ... 
# All pointing to the SAME data object (no copying)
```

#### Scenario 2: Separate Inputs Per Node
```python
# Different data for each input node
input_data = {'x1': data_1, 'x2': data_2}
inputs = btn.prepare_inputs(input_data, for_sigma=False)
# Creates: x1(s, x1, data_1), x2(s, x2, data_2)
```

**For Sigma Network:**
```python
sigma_inputs = btn.prepare_inputs(input_data, for_sigma=True)
# Doubles each input with _prime version:
# x1(s, x1, data), x1_prime(s, x1_prime, data),
# x2(s, x2, data), x2_prime(s, x2_prime, data)
```

**Returns:** List of `quimb.Tensor` objects

---

## Forward Operations

### `forward(tn, inputs, sum_over_batch=False, sum_over_output=False)`

Performs batched forward pass through a tensor network.

**Parameters:**
- `tn`: TensorNetwork (mu or sigma)
- `inputs`: List of batches, each batch is a list of input tensors
- `sum_over_batch`: If True, sums over batch dimension
- `sum_over_output`: If True, sums over output dimensions

**Flag Combinations:**

| `sum_over_batch` | `sum_over_output` | Output Indices | Shape |
|------------------|-------------------|----------------|-------|
| False | False | `(s, y1, y2, ...)` | `(batch_size, out_dims...)` |
| True | False | `(y1, y2, ...)` | `(out_dims...)` |
| False | True | `(s,)` | `(batch_size,)` |
| True | True | `()` | scalar |

**Example:**
```python
# Multiple batches
batches = [batch1_inputs, batch2_inputs, batch3_inputs]

# Default: Keep batch and output dimensions
result = btn.forward(mu_tn, batches)
# Shape: (total_samples, y_dim1, y_dim2, ...)

# Sum over batch dimension (on-the-fly, memory efficient!)
result_sum = btn.forward(mu_tn, batches, sum_over_batch=True)
# Shape: (y_dim1, y_dim2, ...)
```

**Implementation Details:**
- When `sum_over_batch=True`: Results are summed **on-the-fly** using `result = result + batch_result`
- When `sum_over_batch=False`: Batches are collected and concatenated along batch dimension
- Uses pure quimb operations throughout

---

## Environment Calculations

### `get_environment(tn, target_tag, copy=True, sum_over_batch=False, sum_over_output=False)`

Calculates environment for a single batch by removing target tensor and contracting the rest.

**Parameters:**
- `tn`: TensorNetwork with inputs attached (single batch)
- `target_tag`: Tag identifying tensor to remove
- `copy`: Whether to copy network before modification
- `sum_over_batch`: If True, removes batch_dim from output
- `sum_over_output`: If True, removes output_dimensions from output

**What the Environment Contains:**
- **Batch dimension `s`**: Always preserved (unless `sum_over_batch=True`)
- **"Hole" indices**: Bonds, inputs where removed tensor connects
- **Other output dimensions**: From remaining tensors (NOT from removed tensor)

**Example:**
```python
# Network: T1(x1, y1, k1) <-> T2(k1, x2, y1, k2) <-> T3(k2, y2)
full_tn = mu_tn & inputs  # Add inputs to network

# Remove T2, get environment
env = btn.get_environment(full_tn, 'T2')
# Indices: ('s', 'y2', 'k1', 'x2', 'k2')
#   - 's': batch dimension
#   - 'y2': output from T3 (T2's y1 is gone)
#   - 'k1', 'k2', 'x2': holes where T2 connected

# Sum over batch
env_sum = btn.get_environment(full_tn, 'T2', sum_over_batch=True)
# Indices: ('y2', 'k1', 'x2', 'k2')  # No 's'

# Sum over output dimensions (keep holes!)
env_no_out = btn.get_environment(full_tn, 'T2', sum_over_output=True)
# Indices: ('s', 'k1', 'x2', 'k2')  # No 'y2'
```

### `get_environment_batched(tn_base, target_tag, input_batches, ...)`

Calculates environment over multiple batches.

**Parameters:**
- `tn_base`: Base TensorNetwork WITHOUT inputs
- `target_tag`: Tag identifying tensor to remove
- `input_batches`: List of batches
- `copy`, `sum_over_batch`, `sum_over_output`: Same as `get_environment`

**Example:**
```python
# Concatenate batches
env = btn.get_environment_batched(mu_tn, 'T1', batches, sum_over_batch=False)
# Shape: (total_samples, ...)

# Sum on-the-fly (memory efficient!)
env_sum = btn.get_environment_batched(mu_tn, 'T1', batches, sum_over_batch=True)
# Shape: (...) - no batch dimension
```

**Environment with Target y:**
```python
# Add target y to inputs
inputs_with_y = inputs + [y_tensor]

# When y is added, output indices contract!
env_with_y = btn.get_environment(mu_tn & inputs_with_y, 'T1')
# y1, y2 are contracted with network outputs
# Only holes and unconsumed outputs remain
```

---

## Forward with Target

### `forward_with_target(tn, inputs, y, mode='dot', sum_over_batch=False)`

Forward pass coupled with target output y (single batch).

**Parameters:**
- `tn`: TensorNetwork
- `inputs`: List of input tensors (single batch)
- `y`: Target tensor with indices `(batch_dim, output_dims...)`
- `mode`: `'dot'` or `'squared_error'`
- `sum_over_batch`: If True, sums over batch dimension

**Modes:**

#### Dot Product
Computes scalar product: `forward · y`

```python
y = qt.Tensor(y_data, inds=('s', 'y1', 'y2'))

# Keep batch dimension
dot_result = btn.forward_with_target(mu_tn, inputs, y, mode='dot', sum_over_batch=False)
# Indices: ('s',)
# Shape: (batch_size,)
# Each element is the dot product for that sample

# Sum over batch
dot_sum = btn.forward_with_target(mu_tn, inputs, y, mode='dot', sum_over_batch=True)
# Scalar: total dot product across all samples
```

**Implementation:** Adds y to network and contracts: `tn & inputs & y`

#### Squared Error
Computes: `(forward - y)²`

```python
# Keep batch dimension
se_result = btn.forward_with_target(mu_tn, inputs, y, mode='squared_error', sum_over_batch=False)
# Indices: ('s',)
# Shape: (batch_size,)
# Each element is the squared error for that sample

# Sum over batch (total loss)
se_sum = btn.forward_with_target(mu_tn, inputs, y, mode='squared_error', sum_over_batch=True)
# Scalar: total squared error
```

**Implementation:** 
1. Compute forward using quimb
2. Compute difference: `diff = forward - y` (quimb subtraction)
3. Square: `squared_diff = diff ** 2` (quimb element-wise power)
4. Contract over output dimensions

---

## Batched Operations

### Memory-Efficient On-the-Fly Summing

When `sum_over_batch=True`, results are **never stored** - they're summed immediately:

```python
# BAD (old approach): Store all batches then sum
batch_results = []
for batch in batches:
    batch_results.append(compute(batch))  # Memory grows!
result = sum(batch_results)

# GOOD (current approach): Sum on-the-fly
result = None
for batch in batches:
    batch_result = compute(batch)
    if result is None:
        result = batch_result
    else:
        result = result + batch_result  # Quimb addition
```

This is crucial for large datasets where storing all batch results would exceed memory!

### Concatenation

When `sum_over_batch=False`, batches are collected and concatenated:

```python
batch_results = [compute(batch) for batch in batches]
result = concatenate(batch_results, axis=0)  # Along batch dimension
```

Uses backend-appropriate concatenation (numpy, torch, jax).

---

## Backend Compatibility

The implementation is **fully backend-agnostic**! Tested and works with:

✅ **NumPy** 
✅ **PyTorch**
✅ **JAX**

**Key Points:**
- No data copying in `prepare_inputs` - all tensors point to the same data object
- Quimb operations (`+`, `-`, `**`, `contract`) work with all backends
- Concatenation auto-detects backend via `np.concatenate`
- Summation uses quimb's `+` operator (backend-agnostic)

**Example with PyTorch:**
```python
import torch

# Create network with torch tensors
t1_data = torch.rand(3, 2, 4, dtype=torch.float64)
t1 = qt.Tensor(t1_data, inds=('x1', 'y1', 'k1'), tags={'T1'})
# ... create network ...

# Everything works the same!
inputs = btn.prepare_inputs({'x1': torch_input_data})
result = btn.forward(mu_tn, batches, sum_over_batch=True)
# result.data is a torch.Tensor
```

---

## Examples

### Complete Workflow: MU Network

```python
import numpy as np
import quimb.tensor as qt
from tensor.btn import BTN

# 1. Create network
t1 = qt.Tensor(np.random.rand(4, 2, 5), inds=('x1', 'y1', 'k1'), tags={'T1'})
t2 = qt.Tensor(np.random.rand(5, 4, 2), inds=('k1', 'x2', 'y2'), tags={'T2'})
mu_tn = t1 & t2

btn = BTN(mu_tn, output_dimensions=['y1', 'y2'], batch_dim='s')

# 2. Prepare batches
batches = []
for i in range(3):
    input_data = np.random.randn(10, 4)  # 10 samples, 4 features
    inputs = btn.prepare_inputs({'x1': input_data}, for_sigma=False)
    batches.append(inputs)

# 3. Forward pass
result = btn.forward(mu_tn, batches, sum_over_batch=False)
# Shape: (30, 2, 2) - all batches concatenated

# 4. Forward with sum
result_sum = btn.forward(mu_tn, batches, sum_over_batch=True)
# Shape: (2, 2) - summed over all 30 samples

# 5. Environment
env = btn.get_environment_batched(mu_tn, 'T1', batches, sum_over_batch=False)
# Shape: (30, ...) - environment for all samples

env_sum = btn.get_environment_batched(mu_tn, 'T1', batches, sum_over_batch=True)
# Shape: (...) - summed environment
```

### Complete Workflow: SIGMA Network

```python
# 1. Prepare SIGMA inputs (doubled with primes)
sigma_batches = []
for i in range(3):
    input_data = np.random.randn(10, 4)
    sigma_inputs = btn.prepare_inputs({'x1': input_data}, for_sigma=True)
    # Creates: x1, x1_prime, x2, x2_prime
    sigma_batches.append(sigma_inputs)

# 2. Forward pass (SIGMA network)
sigma_result = btn.forward(btn.sigma, sigma_batches, sum_over_batch=True)
# Uses sigma network (btn.sigma) instead of mu

# 3. Environment (SIGMA)
sigma_env = btn.get_environment_batched(btn.sigma, 'T1_sigma', sigma_batches, sum_over_batch=True)
# Prime indices preserved in environment
```

### Training Loop Example

```python
# Training with squared error loss
total_loss = 0

for epoch in range(num_epochs):
    epoch_loss = 0
    
    for batch_inputs in data_loader:
        # Prepare inputs
        inputs = btn.prepare_inputs({'x1': batch_inputs}, for_sigma=False)
        
        # Create target
        y = qt.Tensor(batch_targets, inds=('s', 'y1', 'y2'))
        
        # Compute loss (sum over batch)
        loss = btn.forward_with_target(
            mu_tn, inputs, y, 
            mode='squared_error', 
            sum_over_batch=True
        )
        
        epoch_loss += loss
        
        # ... gradient updates ...
    
    print(f"Epoch {epoch}: Loss = {epoch_loss}")
```

---

## Summary

### Core Principles
1. **No unnecessary copying**: Tensors point to same data when appropriate
2. **On-the-fly operations**: Sum during iteration, not after
3. **Pure quimb operations**: Backend-agnostic throughout
4. **Flexible flags**: Control batch/output summing independently

### Key Methods
- `prepare_inputs`: Create input tensors (with/without primes)
- `forward`: Batched forward pass
- `get_environment`: Environment for single batch
- `get_environment_batched`: Environment for multiple batches
- `forward_with_target`: Coupled with target y (dot product or squared error)

### Performance
- ✅ Memory efficient (on-the-fly summing)
- ✅ Backend agnostic (NumPy, PyTorch, JAX)
- ✅ Mathematically verified
- ✅ Handles arbitrary batch sizes

---

**For more examples, see:**
- `test_prepare_inputs.py` - Input preparation
- `test_single_batch_flags.py` - Single batch operations
- `test_batched_with_target.py` - Batched operations with target
- `test_backends.py` - Backend compatibility
