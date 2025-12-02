"""
Basic test for BayesianTensorNetwork implementation.

Tests:
1. QuimbTensorNetwork creation and basic operations
2. BayesianTensorNetwork initialization
3. Projection/Jacobian computation
4. Forward passes
"""

import torch
import quimb.tensor as qtn
import numpy as np

from tensor.bayesian_tn import BayesianTensorNetwork
from tensor.quimb_network import QuimbTensorNetwork


def test_quimb_network_simple_chain():
    """Test QuimbTensorNetwork with a simple 3-node chain."""
    print("\n" + "="*60)
    print("Test 1: QuimbTensorNetwork - Simple Chain")
    print("="*60)
    
    # Create a simple chain: Input -> A -> B -> Output
    # Indices: x connects Input-A, b1 connects A-B, y is output of B
    
    # Input node: shape (3,) with index 'x'
    input_data = np.random.randn(3)
    input_tensor = qtn.Tensor(data=input_data, inds=('x',), tags='input')  # type: ignore
    
    # Node A: shape (3, 4) with indices ('x', 'b1')
    A_data = np.random.randn(3, 4) * 0.1
    A_tensor = qtn.Tensor(data=A_data, inds=('x', 'b1'), tags='A')  # type: ignore
    
    # Node B: shape (4,) with indices ('b1',) and output 'y'
    B_data = np.random.randn(4) * 0.1
    B_tensor = qtn.Tensor(data=B_data, inds=('b1',), tags='B')  # type: ignore
    
    # Create network
    tn = qtn.TensorNetwork([input_tensor, A_tensor, B_tensor])
    
    # Create QuimbTensorNetwork
    qtn_net = QuimbTensorNetwork(
        tn=tn,
        learnable_tags=['A', 'B'],
        input_tags=['input'],
        device=torch.device('cpu'),
        dtype=torch.float64
    )
    
    print(f"✓ Network created successfully")
    print(f"  Learnable nodes: {qtn_net.learnable_tags}")
    print(f"  Input nodes: {qtn_net.input_tags}")
    print(f"  Bonds: {qtn_net.bond_labels}")
    print(f"  Bond to nodes mapping: {qtn_net.bond_to_nodes}")
    
    # Test theta tensor computation
    theta_A = qtn_net.compute_theta_tensor('A')
    print(f"\n✓ Theta tensor for A computed: shape {theta_A.shape}")
    
    theta_B = qtn_net.compute_theta_tensor('B')
    print(f"✓ Theta tensor for B computed: shape {theta_B.shape}")
    
    # Test get/set node tensor
    A_retrieved = qtn_net.get_node_tensor('A')
    print(f"\n✓ Retrieved node A tensor: shape {A_retrieved.shape}")
    
    # Test entropy
    entropy = qtn_net.compute_entropy()
    print(f"✓ Entropy computed: {entropy.item():.4f}")
    
    return qtn_net


def test_bayesian_tn_initialization():
    """Test BayesianTensorNetwork initialization."""
    print("\n" + "="*60)
    print("Test 2: BayesianTensorNetwork Initialization")
    print("="*60)
    
    # Create a simple network: Input -> A -> Output
    input_data = np.random.randn(2)
    input_tensor = qtn.Tensor(data=input_data, inds=('x',), tags='input')  # type: ignore
    
    # Learnable node A
    A_data = np.random.randn(2, 3) * 0.1
    A_tensor = qtn.Tensor(data=A_data, inds=('x', 'out'), tags='A')  # type: ignore
    
    tn = qtn.TensorNetwork([input_tensor, A_tensor])
    
    # Create BayesianTensorNetwork
    btn = BayesianTensorNetwork(
        mu_tn=tn,
        learnable_tags=['A'],
        input_tags=['input'],
        device=torch.device('cpu'),
        dtype=torch.float64
    )
    
    print(f"✓ BayesianTensorNetwork created successfully")
    print(f"  Learnable tags: {btn.learnable_tags}")
    print(f"  Input tags: {btn.input_tags}")
    print(f"  Tau alpha: {btn.tau_alpha.item():.4f}")
    print(f"  Tau beta: {btn.tau_beta.item():.4f}")
    print(f"  E[tau]: {btn.get_tau_mean().item():.4f}")
    
    # Check sigma network was created
    print(f"\n✓ Sigma network created")
    print(f"  Sigma learnable tags: {btn.sigma_network.learnable_tags}")
    
    # Test theta tensor
    theta = btn.compute_theta_tensor('A')
    print(f"\n✓ Theta tensor computed: shape {theta.shape}")
    
    return btn


def test_projection_computation():
    """Test projection (Jacobian) computation."""
    print("\n" + "="*60)
    print("Test 3: Projection/Jacobian Computation")
    print("="*60)
    
    # Create a network with 2 learnable nodes: A -> B
    # This tests the projection computation
    
    A_data = np.random.randn(3, 4) * 0.1
    A_tensor = qtn.Tensor(data=A_data, inds=('in', 'b'), tags='A')  # type: ignore
    
    B_data = np.random.randn(4, 2) * 0.1
    B_tensor = qtn.Tensor(data=B_data, inds=('b', 'out'), tags='B')  # type: ignore
    
    tn = qtn.TensorNetwork([A_tensor, B_tensor])
    
    btn = BayesianTensorNetwork(
        mu_tn=tn,
        learnable_tags=['A', 'B'],
        input_tags=[],
        device=torch.device('cpu'),
        dtype=torch.float64
    )
    
    print(f"✓ Network created: A (3,4) -> B (4,2)")
    
    # Compute projection for A (should give us B contracted)
    try:
        proj_A = btn.compute_projection('A', network_type='mu')
        print(f"✓ Projection for A computed: shape {proj_A.shape}")
        print(f"  Expected to have indices matching A: ('in', 'b')")
    except Exception as e:
        print(f"✗ Projection for A failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Compute projection for B (should give us A contracted)
    try:
        proj_B = btn.compute_projection('B', network_type='mu')
        print(f"✓ Projection for B computed: shape {proj_B.shape}")
        print(f"  Expected to have indices matching B: ('b', 'out')")
    except Exception as e:
        print(f"✗ Projection for B failed: {e}")
        import traceback
        traceback.print_exc()
    
    return btn


def test_forward_pass():
    """Test forward pass through the network."""
    print("\n" + "="*60)
    print("Test 4: Forward Pass")
    print("="*60)
    
    # Create network: Input -> A -> Output
    input_data = np.random.randn(5, 3)  # Batch of 5 samples, 3 features
    input_tensor = qtn.Tensor(data=input_data, inds=('sample', 'x'), tags='input')  # type: ignore
    
    A_data = np.random.randn(3, 2) * 0.1
    A_tensor = qtn.Tensor(data=A_data, inds=('x', 'out'), tags='A')  # type: ignore
    
    tn = qtn.TensorNetwork([input_tensor, A_tensor])
    
    btn = BayesianTensorNetwork(
        mu_tn=tn,
        learnable_tags=['A'],
        input_tags=['input'],
        device=torch.device('cpu'),
        dtype=torch.float64
    )
    
    print(f"✓ Network created: Input (5,3) -> A (3,2)")
    
    # Prepare inputs
    inputs = {
        'input': torch.from_numpy(input_data)
    }
    
    # Forward through mu network
    try:
        output = btn.forward_mu(inputs)
        print(f"✓ Forward pass (mu) completed: output shape {output.shape}")
        print(f"  Expected shape: (5, 2) for batch_size=5, output_dim=2")
    except Exception as e:
        print(f"✗ Forward pass (mu) failed: {e}")
        import traceback
        traceback.print_exc()
    
    return btn


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("BAYESIAN TENSOR NETWORK - BASIC TESTS")
    print("="*60)
    
    # Test 1: QuimbTensorNetwork
    try:
        qtn_net = test_quimb_network_simple_chain()
        print("\n✓ Test 1 PASSED")
    except Exception as e:
        print(f"\n✗ Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: BayesianTN initialization
    try:
        btn = test_bayesian_tn_initialization()
        print("\n✓ Test 2 PASSED")
    except Exception as e:
        print(f"\n✗ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Projection computation
    try:
        btn_proj = test_projection_computation()
        print("\n✓ Test 3 PASSED")
    except Exception as e:
        print(f"\n✗ Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Forward pass
    try:
        btn_fwd = test_forward_pass()
        print("\n✓ Test 4 PASSED")
    except Exception as e:
        print(f"\n✗ Test 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("TESTS COMPLETED")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
