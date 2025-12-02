"""
Simple test: Compare BayesianMPO vs BayesianTensorNetwork

Tests only 3 methods:
1. forward_mu
2. forward_sigma  
3. compute_projection

Use existing working BayesianMPO from builder, create matching quimb network.
"""

import torch
import numpy as np
import quimb.tensor as qtn

from tensor.bayesian_mpo_builder import create_bayesian_tensor_train
from tensor.bayesian_tn import BayesianTensorNetwork


def create_matching_quimb_network(bmpo):
    """
    Create a quimb BayesianTensorNetwork that matches the structure of bmpo.
    
    The bmpo has blocks with structure:
    - Block 0: (r0, f, r1) where r0=1 (edge)
    - Block 1: (r1, f, r2) 
    - Block 2: (r2, f, r3) where r3=1 (edge)
    
    For quimb, we create tensors with same shape but different index structure.
    """
    print("Creating matching quimb network...")
    print(f"  BMPOhas {len(bmpo.mu_nodes)} blocks")
    
    # Get dimensions from bmpo
    param_tensors = []
    for i, node in enumerate(bmpo.mu_nodes):
        print(f"  Block {i}: shape={node.shape}, labels={node.dim_labels}")
        
        # Create quimb tensor with same data
        data = node.tensor.detach().cpu().numpy()
        
        # Indices: rank indices contract between blocks, 'f' contracts with input, 'out' for output
        # For 3-block MPS: 
        #   Block 0: (r0, f, r1) -> but r0=1 so we use ('r0', 'f', 'r1', 'out')
        #   Block 1: (r1, f, r2) -> ('r1', 'f', 'r2', 'out')
        #   Block 2: (r2, f, r3) -> but r3=1 so we use ('r2', 'f', 'r3', 'out')
        
        # Actually, need to match BayesianMPO structure which has 'batch_out' not 'out'
        # And rank indices are 'r0', 'r1', 'r2', 'r3'
        
        # Each block has UNIQUE physical index: p1, p2, p3
        # Structure: rank indices contract between blocks, physical indices contract with inputs
        # NO batch_out dimension needed in quimb!
        if i == 0:
            inds = ('r0', 'p1', 'r1')
        elif i == 1:
            inds = ('r1', 'p2', 'r2')
        else:  # i == 2
            inds = ('r2', 'p3', 'r3')
        
        # BayesianMPO has shape (r_left, physical, r_right, batch_out)
        # But for quimb we only need (r_left, physical, r_right)
        # So drop the last dimension
        if len(data.shape) == 4:
            data = data[..., 0]  # Drop batch_out dimension
        
        tensor = qtn.Tensor(data=data, inds=inds, tags=f'block{i}')  # type: ignore
        param_tensors.append(tensor)
    
    mu_tn = qtn.TensorNetwork(param_tensors)
    
    # Input contracts on MULTIPLE indices: p1, p2, p3
    # Each block contracts with a different index, but they all use the SAME input data!
    # This is a polynomial feature expansion - same input repeated
    input_indices = {'features': ['p1', 'p2', 'p3']}
    
    # Create sigma network using builder
    from tensor.bayesian_tn_builder import create_sigma_network
    learnable_tags = [f'block{i}' for i in range(len(bmpo.mu_nodes))]
    sigma_tn = create_sigma_network(mu_tn, learnable_tags)
    
    btn = BayesianTensorNetwork(
        mu_tn=mu_tn,
        sigma_tn=sigma_tn,
        input_indices=input_indices,
        learnable_tags=learnable_tags,
        tau_alpha=bmpo.tau_alpha.clone(),
        tau_beta=bmpo.tau_beta.clone(),
        device=torch.device('cpu'),
        dtype=torch.float64
    )
    
    print("  Quimb network created!")
    return btn


def test_forward_mu(bmpo, btn, X):
    """Test 1: forward_mu"""
    print("\n" + "="*70)
    print("TEST 1: forward_mu")
    print("="*70)
    
    print(f"\nInput shape: {X.shape}")
    
    # BayesianMPO
    print("\n1. BayesianMPO forward_mu:")
    y_mpo = bmpo.forward_mu(X, to_tensor=True)
    print(f"   Output shape: {y_mpo.shape}")
    print(f"   Output (first 5): {y_mpo.squeeze()[:5]}")
    
    # BayesianTensorNetwork
    print("\n2. BayesianTensorNetwork forward_mu:")
    inputs_dict = {'features': X}
    try:
        y_btn = btn.forward_mu(inputs_dict)
        print(f"   Output shape: {y_btn.shape}")
        print(f"   Output (first 5): {y_btn.squeeze()[:5]}")
        
        # Compare
        print("\n3. Comparison:")
        y_mpo_flat = y_mpo.squeeze()
        y_btn_flat = y_btn.squeeze()
        
        if y_mpo_flat.shape == y_btn_flat.shape:
            diff = (y_mpo_flat - y_btn_flat).abs().max().item()
            print(f"   Max difference: {diff:.2e}")
            if diff < 1e-8:
                print("   ✓ PASS: forward_mu matches!")
                return True
            else:
                print("   ✗ FAIL: Values differ!")
                print(f"   MPO: {y_mpo_flat[:3]}")
                print(f"   BTN: {y_btn_flat[:3]}")
                return False
        else:
            print(f"   ✗ FAIL: Shape mismatch!")
            print(f"   MPO: {y_mpo_flat.shape}")
            print(f"   BTN: {y_btn_flat.shape}")
            return False
            
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_sigma(bmpo, btn, X):
    """Test 2: forward_sigma"""
    print("\n" + "="*70)
    print("TEST 2: forward_sigma")
    print("="*70)
    
    print(f"\nInput shape: {X.shape}")
    
    # BayesianMPO
    print("\n1. BayesianMPO forward_sigma:")
    try:
        y_sigma_mpo = bmpo.forward_sigma(X, to_tensor=True)
        print(f"   Output shape: {y_sigma_mpo.shape}")
        print(f"   Output sample: {y_sigma_mpo.squeeze()[0]:.6f}")
    except Exception as e:
        print(f"   ERROR: {e}")
        return False
    
    # BayesianTensorNetwork
    print("\n2. BayesianTensorNetwork forward_sigma:")
    inputs_dict = {'features': X}
    try:
        y_sigma_btn = btn.forward_sigma(inputs_dict)
        print(f"   Output shape: {y_sigma_btn.shape}")
        print(f"   Output sample: {y_sigma_btn.squeeze()[0]:.6f}")
        
        # Compare
        print("\n3. Comparison:")
        y_mpo_flat = y_sigma_mpo.squeeze()
        y_btn_flat = y_sigma_btn.squeeze()
        
        if y_mpo_flat.shape == y_btn_flat.shape:
            diff = (y_mpo_flat - y_btn_flat).abs().max().item()
            print(f"   Max difference: {diff:.2e}")
            if diff < 1e-8:
                print("   ✓ PASS: forward_sigma matches!")
                return True
            else:
                print("   ✗ FAIL: Values differ!")
                return False
        else:
            print(f"   ✗ FAIL: Shape mismatch!")
            return False
            
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_projection(bmpo, btn, X):
    """Test 3: compute_projection"""
    print("\n" + "="*70)
    print("TEST 3: compute_projection (Jacobian)")
    print("="*70)
    
    print(f"\nInput shape: {X.shape}")
    print(f"Testing projection for block 0")
    print(f"  Block 0 shape: {bmpo.mu_nodes[0].shape}")
    
    # BayesianMPO - need to run forward first
    print("\n1. BayesianMPO projection:")
    bmpo.forward_mu(X, to_tensor=False)  # Set up network state
    try:
        J_mpo = bmpo._compute_forward_without_node(bmpo.mu_nodes[0], bmpo.mu_mpo)
        print(f"   Jacobian shape: {J_mpo.shape}")
        print(f"   Jacobian dim_labels: {J_mpo.dim_labels}")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # BayesianTensorNetwork
    print("\n2. BayesianTensorNetwork projection:")
    inputs_dict = {'features': X}
    try:
        J_btn = btn.compute_projection('block0', network_type='mu', inputs=inputs_dict)
        print(f"   Projection shape: {J_btn.shape}")
        
        # Compare
        print("\n3. Comparison:")
        print(f"   Expected: (batch_size, *node_shape)")
        
        # Need to reshape/permute to match
        batch_size = X.shape[0]
        node_shape = bmpo.mu_nodes[0].shape
        d = int(np.prod(node_shape))
        
        # Flatten both to (batch, d) for comparison
        J_mpo_flat = J_mpo.tensor.reshape(-1, batch_size).T  # Reshape and transpose
        J_btn_flat = J_btn.reshape(batch_size, -1)
        
        print(f"   After flattening:")
        print(f"   MPO: {J_mpo_flat.shape}")
        print(f"   BTN: {J_btn_flat.shape}")
        
        if J_mpo_flat.shape == J_btn_flat.shape:
            diff = (J_mpo_flat - J_btn_flat).abs().max().item()
            print(f"   Max difference: {diff:.2e}")
            if diff < 1e-8:
                print("   ✓ PASS: Projection matches!")
                return True
            else:
                print("   ✗ FAIL: Values differ!")
                print(f"   Sample 0 MPO: {J_mpo_flat[0, :3]}")
                print(f"   Sample 0 BTN: {J_btn_flat[0, :3]}")
                return False
        else:
            print(f"   ✗ FAIL: Shape mismatch!")
            return False
            
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING: BayesianMPO vs BayesianTensorNetwork")
    print("="*70)
    
    # Create test data
    torch.manual_seed(42)
    np.random.seed(42)
    
    batch_size = 10
    input_features = 2
    
    x = torch.linspace(-1, 1, batch_size, dtype=torch.float64)
    X = torch.stack([torch.ones_like(x), x], dim=1)
    
    print(f"\nTest data: {batch_size} samples, {input_features} features")
    print(f"X shape: {X.shape}")
    
    # Create BayesianMPO (3 blocks)
    print("\nCreating BayesianMPO (3 blocks, bond_dim=3)...")
    bmpo = create_bayesian_tensor_train(
        num_blocks=3,
        bond_dim=3,
        input_features=input_features,
        output_shape=1,
        constrict_bond=False,
        tau_alpha=torch.tensor(2.0, dtype=torch.float64),
        tau_beta=torch.tensor(1.0, dtype=torch.float64),
        dtype=torch.float64,
        seed=42
    )
    print("  ✓ BayesianMPO created")
    
    # Create matching quimb network
    btn = create_matching_quimb_network(bmpo)
    
    # Run tests
    results = []
    results.append(("forward_mu", test_forward_mu(bmpo, btn, X)))
    results.append(("forward_sigma", test_forward_sigma(bmpo, btn, X)))
    results.append(("projection", test_projection(bmpo, btn, X)))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:20s}: {status}")
    
    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED")
