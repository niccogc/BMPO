# Design: Batched Forward with Einsum (No Quimb Contraction)

## Key Insight
The input nodes in mu_tn are just **structural placeholders** (no batch dimension).
Forward pass uses **numpy/torch einsum** for batching, NOT quimb contraction.

## Design

### 1. Network Structure (No Batch)
```python
# Input node: just structure, dummy data
input_node = qtn.Tensor(
    data=np.ones(features),  # No batch dimension!
    inds=('p1',),
    tags='input_features'
)

# Parameters
param_A = qtn.Tensor(data=A_data, inds=('p1', 'r1'), tags='A')
param_B = qtn.Tensor(data=B_data, inds=('r1', 'out'), tags='B')

mu_tn = qtn.TensorNetwork([input_node, param_A, param_B])
```

### 2. Forward Pass (Batched with Einsum)
```python
def forward_mu(self, inputs: Dict[str, torch.Tensor]):
    # inputs['features']: (batch, features)
    
    # Extract parameter tensors
    A = self.mu_tn['A'].data  # numpy array
    B = self.mu_tn['B'].data
    
    # Get input data
    X = inputs['features'].numpy()  # (batch, features)
    
    # Batched einsum contractions
    # Step 1: X @ A -> (batch, r1)
    result = np.einsum('bf,fr->br', X, A)
    
    # Step 2: result @ B -> (batch, out)  
    result = np.einsum('br,ro->bo', result, B)
    
    # Convert to torch
    return torch.from_numpy(result)
```

### 3. Benefits
- ✅ Input nodes are structural (no batch dimension in TN)
- ✅ Forward pass is batched with numpy/torch
- ✅ No per-sample loop
- ✅ Fast and simple
- ✅ Works with any network topology (just need to build einsum string)

### 4. For General Networks
For arbitrary topology, we need to:
1. Determine contraction order from the quimb network
2. Build einsum strings dynamically
3. Handle batch dimension correctly

OR simpler:
- Use quimb to get the **contraction path** once
- Apply that path with batched tensors

## Question
Should we:
A) Build einsum strings manually based on network structure
B) Use quimb's optimizer to get contraction path, then apply with batched data
C) Something else?
