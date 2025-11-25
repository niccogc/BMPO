# BMPO (Bayesian Matrix Product Operator) Module

## Overview

The BMPO module (`tensor/bmpo.py`) implements Bayesian tensor networks where each node stores prior distribution parameters for each of its modes/dimensions.

## Key Components

### 1. BMPONode

A `TensorNode` that additionally stores prior parameters for each dimension.

**Structure:**
- For a node with shape `(r1, r2, f)` and dimension labels `['r1', 'r2', 'f']`:
  - `prior_params['r1']`: array of shape `(r1_size,)` or `(r1_size, n_params)`
  - `prior_params['r2']`: array of shape `(r2_size,)` or `(r2_size, n_params)`
  - `prior_params['f']`: array of shape `(f_size,)` or `(f_size, n_params)`

**Key Methods:**
- `get_prior_params(label)`: Get parameters for a specific dimension
- `update_prior_params(label, params)`: Update parameters for a dimension
- `copy()`: Deep copy including prior parameters
- `to(device, dtype)`: Move node and parameters to device/dtype

**Example:**
```python
import torch
from tensor.bmpo import BMPONode

# Create node with Gamma prior params (concentration, rate)
prior_params = {
    'r1': torch.tensor([[2.0, 1.0], [3.0, 1.5], [2.5, 1.2]]),  # 3 indices, 2 params each
    'r2': torch.tensor([[1.5, 0.5], [2.0, 1.0]]),              # 2 indices, 2 params each
    'f': torch.tensor([[1.0, 1.0]])                             # 1 index, 2 params
}

node = BMPONode(
    tensor_or_shape=(3, 2, 1),
    dim_labels=['r1', 'r2', 'f'],
    prior_params=prior_params
)
```

### 2. BMPONetwork

A `TensorNetwork` for Bayesian tensor networks.

**Key Methods:**
- `forward(x)`: Inherited from `TensorNetwork`, performs full network contraction
- `get_jacobian(node)`: Compute the Jacobian (J) for a node
  - This is the network contracted **without** the specified block
  - Leaves the node's indices free
  - This is the gradient computation you need

**What `get_jacobian` does:**
Given a network with nodes `[node1, node2, node3]` and input nodes, calling `network.get_jacobian(node2)` returns a `TensorNode` representing the contraction of:
- All input nodes
- node1
- node3
- Any other connected nodes

But **NOT** node2. The resulting tensor has free indices corresponding to node2's dimensions and the output dimensions.

## How to Use

### Basic Node Creation

```python
from tensor.bmpo import BMPONode

# Simple node with default prior params (zeros)
node = BMPONode(
    tensor_or_shape=(3, 4, 2),
    dim_labels=['r1', 'r2', 'f'],
    name='my_node'
)

# Access prior params
print(node.get_prior_params('r1'))  # Shape: (3,)
```

### Network with Gradient Computation

```python
from tensor.bmpo import BMPONode, BMPONetwork

# Create nodes
input_node = BMPONode((2, 5), ['f', 's'], name='input')
node1 = BMPONode((1, 3, 2), ['r0', 'r1', 'f'], l='r0', r='r1', name='node1')
node2 = BMPONode((3, 1, 2), ['r1', 'r2', 'f'], l='r1', r='r2', name='node2')

# Connect nodes
input_node.connect(node1, 'f')
input_node.connect(node2, 'f')
node1.connect(node2, 'r1')

# Create network
network = BMPONetwork(
    input_nodes=[input_node],
    main_nodes=[node1, node2]
)

# Forward pass
network.set_input(data)
output = network.forward(data, to_tensor=True)

# Compute Jacobian for gradient
jacobian = network.get_jacobian(node1)
# jacobian.tensor contains the contraction without node1
# Use this for gradient-based updates
```

## What's NOT Included (By Design)

- **No `get_A_b` computation**: We don't compute the Hessian (A) or the full gradient vector (b)
- **No node update method**: The optimization/update algorithm is left for you to implement
- **No prior logic**: The nodes just store the parameters, they don't enforce any distribution

## Use Cases

Store prior parameters for different distributions:

**Gamma Distribution** (concentration, rate):
```python
prior_params = {
    'r1': torch.tensor([[2.0, 1.0], [3.0, 1.5], [2.5, 1.2]])
}
```

**Gaussian Distribution** (mean, log_std):
```python
prior_params = {
    'r1': torch.tensor([[0.0, -1.0], [0.5, -0.5], [-0.5, -0.8]])
}
```

**Custom Parameters** (any shape):
```python
prior_params = {
    'r1': torch.randn(5, 4)  # 5 indices, 4 params each
}
```

## Example Files

- `example_bmpo.py`: Basic node creation, parameter updates, copying
- `tensor/bmpo.py`: Full implementation
