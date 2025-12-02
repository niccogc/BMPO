"""
Compare projections between BayesianMPO and BayesianTensorNetwork.

Goal: Verify that compute_projection() gives the same Jacobian as _compute_forward_without_node().
"""

import torch
import numpy as np
import quimb.tensor as qtn

from tensor.bayesian_mpo import BayesianMPO
from tensor.bayesian_tn import BayesianTensorNetwork
from tensor.node import TensorNode


def create_simple_mpo():
    """Create a simple 2-block MPO for BayesianMPO."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Block shapes: A: (x, r1, batch_out), B: (r1, batch_out)
    d_in = 2
    r1 = 3
    d_out = 1
    
    # Create mu nodes
    A_tensor = torch.randn(d_in, r1, d_out, dtype=torch.float64) * 0.1
    B_tensor = torch.randn(r1, d_out, dtype=torch.float64) * 0.1
    
    node_A = TensorNode(A_tensor, dim_labels=['x', 'r1', 'batch_out'])
    node_B = TensorNode(B_tensor, dim_labels=['r1', 'batch_out'])
    
    # Connect nodes
    node_A.connect_to(node_B, 'r1')
    
    mu_nodes = [node_A, node_B]
    
    # Create input node placeholder
    input_node = TensorNode(
        torch.ones(1, d_in, dtype=torch.float64),
        dim_labels=['s', 'x']
    )
    input_node.connect_to(node_A, 'x')
    
    bmpo = BayesianMPO(
        mu_nodes=mu_nodes,
        input_nodes=[input_node],
        rank_labels={'r1'},
        tau_alpha=torch.tensor(2.0, dtype=torch.float64),
        tau_beta=torch.tensor(1.0, dtype=torch.float64)
    )
    
    return bmpo


def create_simple_quimb_tn():
    """Create equivalent quimb TensorNetwork."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    d_in = 2
    r1 = 3
    d_out = 1
    
    # Create tensors (same values as MPO)
    input_data = np.ones((1, d_in))
    A_data = np.random.randn(d_in, r1, d_out) * 0.1
    B_data = np.random.randn(r1, d_out) * 0.1
    
    input_tensor = qtn.Tensor(data=input_data, inds=('batch', 'x'), tags='input')  # type: ignore
    A_tensor = qtn.Tensor(data=A_data, inds=('x', 'r1', 'batch_out'), tags='A')  # type: ignore
    B_tensor = qtn.Tensor(data=B_data, inds=('r1', 'batch_out2'), tags='B')  # type: ignore
    
    tn = qtn.TensorNetwork([input_tensor, A_tensor, B_tensor])
    
    btn = BayesianTensorNetwork(
        mu_tn=tn.copy(),
        learnable_tags=['A', 'B'],
        input_tags=['input'],
        tau_alpha=torch.tensor(2.0, dtype=torch.float64),
        tau_beta=torch.tensor(1.0, dtype=torch.float64),
        device=torch.device('cpu'),
        dtype=torch.float64
    )
    
    return btn


def test_projection_comparison():
    """Test that projections match between implementations."""
    print("\n" + "="*70)
    print("TEST: Projection Comparison")
    print("="*70)
    
    # Create both models
    print("\n1. Creating models...")
    bmpo = create_simple_mpo()
    btn = create_simple_quimb_tn()
    
    # Create test data
    batch_size = 5
    x = torch.linspace(-1, 1, batch_size, dtype=torch.float64)
    X = torch.stack([torch.ones_like(x), x], dim=1)
    y = 2.0 * x + 1.0
    
    print(f"   Data shape: X={X.shape}, y={y.shape}")
    
    # Test BayesianMPO projection
    print("\n2. BayesianMPO projection (Jacobian):")
    # Forward pass to set up network state
    bmpo.forward_mu(X, to_tensor=False)
    # Get Jacobian for block 0 (node A)
    J_bmpo = bmpo._compute_forward_without_node(bmpo.mu_nodes[0], bmpo.mu_mpo)
    print(f"   J_bmpo shape: {J_bmpo.shape}")
    print(f"   J_bmpo dim_labels: {J_bmpo.dim_labels}")
    print(f"   J_bmpo tensor shape: {J_bmpo.tensor.shape}")
    
    # Test BayesianTensorNetwork projection
    print("\n3. BayesianTensorNetwork projection:")
    inputs_dict = {'input': X}
    J_btn = btn.compute_projection('A', network_type='mu', inputs=inputs_dict)
    print(f"   J_btn shape: {J_btn.shape}")
    
    # Compare
    print("\n4. Comparison:")
    print(f"   Node A shape: {bmpo.mu_nodes[0].shape}")
    print(f"   Expected: (batch={batch_size}, node_shape)")
    
    # Try to match shapes
    if J_bmpo.tensor.shape == J_btn.shape:
        print("   ✓ Shapes match!")
        diff = (J_bmpo.tensor - J_btn).abs().max().item()
        print(f"   Max absolute difference: {diff}")
        if diff < 1e-10:
            print("   ✓ Values match!")
        else:
            print("   ✗ Values differ!")
    else:
        print(f"   ✗ Shape mismatch!")
        print(f"      BayesianMPO:  {J_bmpo.tensor.shape}")
        print(f"      BayesianTN:   {J_btn.shape}")
        
        # Try to understand the difference
        print("\n5. Debugging shape difference:")
        print(f"   BayesianMPO dim_labels: {J_bmpo.dim_labels}")
        print(f"   Expected: should have 's' (sample) dimension")
        
        # Check if we need to reshape/permute
        J_bmpo_flat = J_bmpo.tensor.reshape(batch_size, -1)
        J_btn_flat = J_btn.reshape(batch_size, -1)
        print(f"   After flattening node dims:")
        print(f"      BayesianMPO:  {J_bmpo_flat.shape}")
        print(f"      BayesianTN:   {J_btn_flat.shape}")
        
        if J_bmpo_flat.shape == J_btn_flat.shape:
            diff = (J_bmpo_flat - J_btn_flat).abs().max().item()
            print(f"   Max absolute difference: {diff}")


if __name__ == "__main__":
    test_projection_comparison()
