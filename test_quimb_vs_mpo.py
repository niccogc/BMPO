"""
Test to verify BayesianTensorNetwork matches BayesianMPO behavior.

Tests:
1. forward_mu - forward pass through mu network
2. forward_sigma - forward pass through sigma network  
3. compute_projection - Jacobian computation

Goal: Ensure quimb implementation gives SAME results as working BayesianMPO.
"""

import torch
import numpy as np
import quimb.tensor as qtn

from tensor.bayesian_mpo import BayesianMPO
from tensor.bayesian_tn import BayesianTensorNetwork
from tensor.node import TensorNode


def create_bayesian_mpo():
    """Create a simple BayesianMPO for testing."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Simple 2-block MPO using constructor connections
    d_in = 2
    r1 = 3
    d_out = 1
    
    # Block A: (r0, x, r1, batch_out) with connections
    A_tensor = torch.randn(1, d_in, r1, d_out, dtype=torch.float64) * 0.1
    node_A = TensorNode(
        A_tensor, 
        dim_labels=['r0', 'x', 'r1', 'batch_out'],
        l='r0', r='r1',
        name='A'
    )
    
    # Block B: (r1, r2, batch_out) with connections
    B_tensor = torch.randn(r1, 1, d_out, dtype=torch.float64) * 0.1
    node_B = TensorNode(
        B_tensor, 
        dim_labels=['r1', 'r2', 'batch_out'],
        l='r1', r='r2',
        name='B'
    )
    
    mu_nodes = [node_A, node_B]
    
    bmpo = BayesianMPO(
        mu_nodes=mu_nodes,
        rank_labels={'r0', 'r1', 'r2'},
        tau_alpha=torch.tensor(2.0, dtype=torch.float64),
        tau_beta=torch.tensor(1.0, dtype=torch.float64)
    )
    
    return bmpo


def create_bayesian_tn():
    """Create equivalent BayesianTensorNetwork (NEW structure).
    
    NOTE: This should match the BayesianMPO structure as closely as possible.
    BayesianMPO has:
    - Block A: (r0=1, x=2, r1=3, batch_out=1)
    - Block B: (r1=3, r2=1, batch_out=1)
    
    Quimb structure:
    - Block A contracts with input on 'x', with B on 'r1', outputs to 'out'
    - Block B contracts with A on 'r1', outputs to 'out'
    - Edge ranks (r0, r2) become free indices in quimb (don't exist)
    """
    torch.manual_seed(42)
    np.random.seed(42)
    
    r0 = 1  # Left edge rank
    d_in = 2
    r1 = 3
    r2 = 1  # Right edge rank
    d_out = 1
    
    # PARAMETERS ONLY (no inputs in TN!)
    # Match BayesianMPO structure: (r0, x, r1, batch_out) -> but r0=1 can be squeezed
    # Actually for quimb, we don't need the edge ranks as separate indices
    # They'll just be part of the shape
    
    # A: has shape that includes r0, x, r1, out
    # But since r0=1, we can treat it as (x, r1, out)
    A_data = np.random.randn(r0, d_in, r1, d_out) * 0.1
    A_tensor = qtn.Tensor(data=A_data, inds=('r0', 'x', 'r1', 'out'), tags='A')  # type: ignore
    
    # B: (r1, r2, out)
    B_data = np.random.randn(r1, r2, d_out) * 0.1
    B_tensor = qtn.Tensor(data=B_data, inds=('r1', 'r2', 'out'), tags='B')  # type: ignore
    
    param_tn = qtn.TensorNetwork([A_tensor, B_tensor])
    
    # Input contracts on 'x' index
    input_indices = {'features': ['x']}
    
    btn = BayesianTensorNetwork(
        param_tn=param_tn,
        input_indices=input_indices,
        learnable_tags=['A', 'B'],
        tau_alpha=torch.tensor(2.0, dtype=torch.float64),
        tau_beta=torch.tensor(1.0, dtype=torch.float64),
        device=torch.device('cpu'),
        dtype=torch.float64
    )
    
    return btn


def test_forward_mu():
    """Test 1: forward_mu matches between implementations."""
    print("\n" + "="*70)
    print("TEST 1: forward_mu")
    print("="*70)
    
    bmpo = create_bayesian_mpo()
    btn = create_bayesian_tn()
    
    # Test data
    batch_size = 5
    x = torch.linspace(-1, 1, batch_size, dtype=torch.float64)
    X = torch.stack([torch.ones_like(x), x], dim=1)  # (5, 2)
    
    print(f"\nInput shape: {X.shape}")
    
    # BayesianMPO forward
    print("\n1. BayesianMPO forward_mu:")
    y_mpo = bmpo.forward_mu(X, to_tensor=True)
    print(f"   Output shape: {y_mpo.shape}")
    print(f"   Output values: {y_mpo.squeeze()}")
    
    # BayesianTensorNetwork forward
    print("\n2. BayesianTensorNetwork forward_mu:")
    inputs_dict = {'features': X}
    y_tn = btn.forward_mu(inputs_dict)
    print(f"   Output shape: {y_tn.shape}")
    print(f"   Output values: {y_tn.squeeze()}")
    
    # Compare
    print("\n3. Comparison:")
    if y_mpo.shape == y_tn.shape:
        diff = (y_mpo - y_tn).abs().max().item()
        print(f"   Max absolute difference: {diff}")
        if diff < 1e-10:
            print("   ✓ PASS: forward_mu matches!")
        else:
            print("   ✗ FAIL: Values differ!")
            print(f"   MPO: {y_mpo.squeeze()}")
            print(f"   TN:  {y_tn.squeeze()}")
    else:
        print(f"   ✗ FAIL: Shape mismatch!")
        print(f"   MPO: {y_mpo.shape}")
        print(f"   TN:  {y_tn.shape}")


def test_projection():
    """Test 2: compute_projection (Jacobian) matches."""
    print("\n" + "="*70)
    print("TEST 2: compute_projection (Jacobian)")
    print("="*70)
    
    bmpo = create_bayesian_mpo()
    btn = create_bayesian_tn()
    
    # Test data
    batch_size = 5
    x = torch.linspace(-1, 1, batch_size, dtype=torch.float64)
    X = torch.stack([torch.ones_like(x), x], dim=1)
    
    print(f"\nInput shape: {X.shape}")
    print(f"Node A shape: {bmpo.mu_nodes[0].shape}")
    
    # BayesianMPO projection (Jacobian)
    print("\n1. BayesianMPO projection for node A:")
    bmpo.forward_mu(X, to_tensor=False)  # Set up network state
    J_mpo = bmpo._compute_forward_without_node(bmpo.mu_nodes[0], bmpo.mu_mpo)
    print(f"   Jacobian shape: {J_mpo.shape}")
    print(f"   Jacobian dim_labels: {J_mpo.dim_labels}")
    print(f"   Jacobian tensor shape: {J_mpo.tensor.shape}")
    
    # BayesianTensorNetwork projection
    print("\n2. BayesianTensorNetwork projection for node A:")
    inputs_dict = {'features': X}
    J_tn = btn.compute_projection('A', network_type='mu', inputs=inputs_dict)
    print(f"   Projection shape: {J_tn.shape}")
    
    # Compare
    print("\n3. Comparison:")
    print(f"   Expected: (batch={batch_size}, *node_shape)")
    print(f"   Node A shape: {bmpo.mu_nodes[0].shape}")
    
    # The Jacobian from BayesianMPO has permuted dimensions
    # We need to understand the dimension order
    print(f"\n   BayesianMPO Jacobian dims: {J_mpo.dim_labels}")
    print(f"   Should contain 's' (sample) and node dims: {bmpo.mu_nodes[0].dim_labels}")
    
    # Try to match by reshaping
    # BayesianMPO: shape is permuted, has 's' dimension
    # BayesianTensorNetwork: (batch, *node_shape)
    
    # For node A with shape (2, 3, 1) = (x, r1, batch_out)
    # BayesianMPO gives: (x, s, batch_out, r1) = (2, 5, 1, 3)
    # BayesianTN should give: (batch, x, r1, batch_out) = (5, 2, 3, 1)
    
    # Let's flatten and compare
    d = np.prod(bmpo.mu_nodes[0].shape)
    J_mpo_batch = J_mpo.tensor.reshape(-1, batch_size, d).permute(1, 0, 2).reshape(batch_size, d)
    J_tn_batch = J_tn.reshape(batch_size, d)
    
    print(f"\n   After flattening to (batch, d):")
    print(f"   MPO shape: {J_mpo_batch.shape}")
    print(f"   TN shape:  {J_tn_batch.shape}")
    
    if J_mpo_batch.shape == J_tn_batch.shape:
        diff = (J_mpo_batch - J_tn_batch).abs().max().item()
        print(f"   Max absolute difference: {diff}")
        if diff < 1e-10:
            print("   ✓ PASS: Projection matches!")
        else:
            print("   ✗ FAIL: Values differ!")
            print(f"   Sample 0 MPO: {J_mpo_batch[0, :3]}")
            print(f"   Sample 0 TN:  {J_tn_batch[0, :3]}")
    else:
        print(f"   ✗ FAIL: Shape mismatch even after flattening!")


def test_forward_sigma():
    """Test 3: forward_sigma matches."""
    print("\n" + "="*70)
    print("TEST 3: forward_sigma")
    print("="*70)
    
    bmpo = create_bayesian_mpo()
    btn = create_bayesian_tn()
    
    # Test data
    batch_size = 3
    x = torch.linspace(-1, 1, batch_size, dtype=torch.float64)
    X = torch.stack([torch.ones_like(x), x], dim=1)
    
    print(f"\nInput shape: {X.shape}")
    
    # BayesianMPO forward_sigma
    print("\n1. BayesianMPO forward_sigma:")
    try:
        y_sigma_mpo = bmpo.forward_sigma(X, to_tensor=True)
        print(f"   Output shape: {y_sigma_mpo.shape}")
        print(f"   Output sample: {y_sigma_mpo[0, :3]}")
    except Exception as e:
        print(f"   Error: {e}")
        return
    
    # BayesianTensorNetwork forward_sigma
    print("\n2. BayesianTensorNetwork forward_sigma:")
    inputs_dict = {'features': X}
    try:
        y_sigma_tn = btn.forward_sigma(inputs_dict)
        print(f"   Output shape: {y_sigma_tn.shape}")
        print(f"   Output sample: {y_sigma_tn[0, :3]}")
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Compare
    print("\n3. Comparison:")
    if y_sigma_mpo.shape == y_sigma_tn.shape:
        diff = (y_sigma_mpo - y_sigma_tn).abs().max().item()
        print(f"   Max absolute difference: {diff}")
        if diff < 1e-10:
            print("   ✓ PASS: forward_sigma matches!")
        else:
            print("   ✗ FAIL: Values differ!")
    else:
        print(f"   ✗ FAIL: Shape mismatch!")
        print(f"   MPO: {y_sigma_mpo.shape}")
        print(f"   TN:  {y_sigma_tn.shape}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("COMPARING BayesianMPO vs BayesianTensorNetwork")
    print("="*70)
    
    test_forward_mu()
    test_projection()
    test_forward_sigma()
    
    print("\n" + "="*70)
    print("TESTS COMPLETE")
    print("="*70)
