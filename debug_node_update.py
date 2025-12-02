"""
Debug script to compare BayesianMPO and BayesianTensorNetwork node updates.
"""

import torch
import numpy as np
import quimb.tensor as qtn

from tensor.bayesian_tn import BayesianTensorNetwork

def test_projection_shape():
    """Test what shape the projection actually returns."""
    print("\n" + "="*70)
    print("DEBUG: Projection Shape Analysis")
    print("="*70)
    
    # Create simple network
    np.random.seed(42)
    torch.manual_seed(42)
    
    d_in = 2
    r1 = 3
    d_out = 1
    batch_size = 5
    
    # Create tensors
    input_data = np.ones((batch_size, d_in))
    input_tensor = qtn.Tensor(data=input_data, inds=('batch', 'x'), tags='input')  # type: ignore
    
    A_data = np.random.randn(d_in, r1, d_out) * 0.1
    A_tensor = qtn.Tensor(data=A_data, inds=('x', 'r1', 'batch_out'), tags='A')  # type: ignore
    
    B_data = np.random.randn(r1, d_out) * 0.1
    B_tensor = qtn.Tensor(data=B_data, inds=('r1', 'batch_out2'), tags='B')  # type: ignore
    
    tn = qtn.TensorNetwork([input_tensor, A_tensor, B_tensor])
    
    btn = BayesianTensorNetwork(
        mu_tn=tn.copy(),
        learnable_tags=['A', 'B'],
        input_tags=['input'],
        device=torch.device('cpu'),
        dtype=torch.float64
    )
    
    # Create input data
    x = torch.linspace(-1, 1, batch_size, dtype=torch.float64)
    X = torch.stack([torch.ones_like(x), x], dim=1)
    y = 2.0 * x + 1.0
    
    inputs_dict = {'input': X}
    
    print("\n1. Forward pass to set network state:")
    y_pred = []
    for i in range(batch_size):
        sample_input = {'input': X[i:i+1]}
        y_i = btn.forward_mu(sample_input)
        y_pred.append(y_i.item())
        print(f"   Sample {i}: y_pred={y_i.item():.4f}")
    
    print("\n2. Projection for node A with ALL samples:")
    # Try with all samples at once
    J_mu_all = btn.compute_projection('A', network_type='mu', inputs=inputs_dict)
    print(f"   Projection shape (all samples): {J_mu_all.shape}")
    print(f"   Node A shape: {btn.mu_network.get_node_shape('A')}")
    
    print("\n3. Projection for node A SINGLE sample:")
    sample_inputs = {'input': X[0:1]}
    J_mu_single = btn.compute_projection('A', network_type='mu', inputs=sample_inputs)
    print(f"   Projection shape (single): {J_mu_single.shape}")
    
    print("\n4. Loop over samples (current implementation):")
    for i in range(min(3, batch_size)):
        sample_inputs = {'input': X[i:i+1]}
        J_mu = btn.compute_projection('A', network_type='mu', inputs=sample_inputs)
        print(f"   Sample {i}: J_mu shape = {J_mu.shape}, numel = {J_mu.numel()}")
        print(f"              J_mu flat: {J_mu.flatten()[:5]}...")
    
    print("\n5. What we SHOULD do (batch einsum like BayesianMPO):")
    print("   BayesianMPO uses:")
    print("   - Forward pass with ALL samples")
    print("   - get_b() to contract J with y over samples")
    print("   - Returns tensor with node shape ONLY")
    print("\n   The key: get_b contracts OVER the batch dimension")
    print("   So if J has shape (batch, x, r1, batch_out)")
    print("   and y has shape (batch,)")
    print("   The einsum does: 'bxrB,b->xrB' (sum over batch)")
    
    print("\n6. Testing manual einsum contraction:")
    # Get projection for all samples
    J_mu_all = btn.compute_projection('A', network_type='mu', inputs=inputs_dict)
    print(f"   J_mu_all shape: {J_mu_all.shape}")
    
    # Try to identify which dimension is batch
    # In quimb, the batch dimension should be the first one
    if J_mu_all.dim() == 4:  # (batch, x, r1, batch_out)
        print("   Assuming shape is (batch, x, r1, batch_out)")
        # Contract with y over batch dimension
        sum_y_J_mu = torch.einsum('bxrB,b->xrB', J_mu_all, y)
        print(f"   sum_y_J_mu shape: {sum_y_J_mu.shape}")
        print(f"   Expected node A shape: {btn.mu_network.get_node_shape('A')}")
        
        # Outer product - need to use SAME index for repeated dims
        # Shape is (batch, x, r1, batch_out)
        # We want: sum over batch, outer product over (x,r1,batch_out)
        sum_J_outer = torch.einsum('bxrB,bXRO->xrBXRO', J_mu_all, J_mu_all)
        print(f"   sum_J_outer shape: {sum_J_outer.shape}")
        d = np.prod(btn.mu_network.get_node_shape('A'))
        sum_J_outer_flat = sum_J_outer.reshape(d, d)
        print(f"   After flattening to (d,d): {sum_J_outer_flat.shape}")

if __name__ == "__main__":
    test_projection_shape()
