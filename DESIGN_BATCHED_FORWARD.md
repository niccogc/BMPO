# Design: Batched Forward with Input Nodes

## Current Problem
- Inputs are added dynamically during forward
- We loop over samples and contract each one individually  
- This is slow and inefficient

## New Design

### 1. Network Initialization
```python
# mu_tn includes BOTH parameters AND input nodes
mu_tn = qtn.TensorNetwork([
    param_A,    # Parameter tensors
    param_B,
    input_node  # Input node with DUMMY data, shape (batch, features)
])

# Input node has batch dimension built-in
input_node = qtn.Tensor(
    data=np.ones((1, features)),  # Dummy data
    inds=('batch', 'p1'),
    tags='input_features'
)
```

### 2. Forward Pass (Batched)
```python
def forward_mu(self, inputs: Dict[str, torch.Tensor]):
    # inputs['features'] has shape (batch_size, features)
    batch_size = inputs['features'].shape[0]
    
    # Step 1: Set input node data to actual batch
    for input_name, input_data in inputs.items():
        input_tag = f'input_{input_name}'
        # Update the input node's data in-place
        self.mu_network.set_node_data(input_tag, input_data)
    
    # Step 2: Contract the ENTIRE network ONCE
    # The batch dimension stays throughout
    result = self.mu_network.contract()
    
    # Result has shape (batch_size, output_dims)
    return result
```

### 3. Benefits
- ✅ No loop over samples
- ✅ Single quimb contraction
- ✅ Batch dimension preserved throughout
- ✅ Input nodes are part of the network structure
- ✅ Simple and efficient

### 4. For Sigma Network
Same approach:
```python
sigma_tn = qtn.TensorNetwork([
    param_A_sigma,  # Doubled parameter tensors
    param_B_sigma,
    input_node_o,   # Input outer: (batch, p1o)
    input_node_i    # Input inner: (batch, p1i), SAME data
])
```

## Implementation Plan

1. Update `BayesianTensorNetwork.__init__` to accept mu_tn WITH input nodes
2. Add method to set input node data: `set_input_data()`
3. Simplify `forward_mu()` and `forward_sigma()` - just set data and contract
4. Update builder to create input nodes with dummy data
