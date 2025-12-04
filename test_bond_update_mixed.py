"""
Test bond updates with mixed learnable and fixed nodes.

Verifies that bond updates only count learnable nodes, not fixed nodes.
From THEORETICAL_MODEL.md: A(k) = Set of LEARNABLE nodes connected to Bond k.
"""

import torch
import quimb.tensor as qtn
import numpy as np

from tensor.bayesian_tn import BayesianTensorNetwork
from tensor.bayesian_tn_builder import create_sigma_network


def test_bond_update_with_fixed_node():
    """
    Test bond update when one node is fixed (non-learnable).
    
    Network: INPUT(x, r1) -- A(r1, r2) -- B(r2, y)
    - INPUT: Fixed node (not learnable)
    - A, B: Learnable nodes
    
    Bond r1 connects INPUT and A:
    - A(r1) should only count A (1 learnable node)
    - NOT count INPUT (fixed)
    
    Bond r2 connects A and B:
    - A(r2) should count both A and B (2 learnable nodes)
    """
    print("\n" + "="*70)
    print("TEST: Bond Update with Fixed Node")
    print("="*70)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create network: INPUT(x, r1) -- A(r1, r2) -- B(r2, y)
    INPUT_data = torch.ones(3, 2, dtype=torch.float64)  # Fixed input node
    A_data = torch.randn(2, 3, dtype=torch.float64)
    B_data = torch.randn(3, 1, dtype=torch.float64)
    
    INPUT = qtn.Tensor(data=INPUT_data, inds=('x', 'r1'), tags='INPUT')
    A = qtn.Tensor(data=A_data, inds=('r1', 'r2'), tags='A')
    B = qtn.Tensor(data=B_data, inds=('r2', 'y'), tags='B')
    mu_tn = qtn.TensorNetwork([INPUT, A, B])
    
    # Only A and B are learnable (INPUT is fixed)
    learnable_tags = ['A', 'B']
    
    # Create sigma network
    sigma_tn = create_sigma_network(mu_tn, learnable_tags, output_indices=['y'])
    
    print(f"Network structure:")
    print(f"  INPUT(x, r1) [FIXED] -- r1 -- A(r1, r2) [LEARNABLE] -- r2 -- B(r2, y) [LEARNABLE]")
    print(f"  Learnable nodes: {learnable_tags}")
    
    # Create model
    model = BayesianTensorNetwork(
        mu_tn=mu_tn,
        sigma_tn=sigma_tn,
        input_indices={'data': ['x']},
        output_indices=['y'],
        learnable_tags=learnable_tags,
        dtype=torch.float64
    )
    
    print(f"\nBond labels: {model.mu_network.bond_labels}")
    
    # Check bond_to_nodes mapping
    print(f"\nBond to nodes mapping:")
    for bond in model.mu_network.bond_labels:
        nodes = model.mu_network.get_nodes_for_bond(bond)
        print(f"  {bond}: {nodes}")
    
    # ========================================================================
    # VERIFY r1 bond
    # ========================================================================
    print("\n" + "-"*70)
    print("Bond r1: Should only count A (not INPUT)")
    print("-"*70)
    
    r1_nodes = model.mu_network.get_nodes_for_bond('r1')
    print(f"Nodes connected to r1: {r1_nodes}")
    
    assert 'INPUT' not in r1_nodes, "INPUT should NOT be in r1 nodes (it's fixed)"
    assert 'A' in r1_nodes, "A should be in r1 nodes"
    assert len(r1_nodes) == 1, f"r1 should have 1 learnable node, got {len(r1_nodes)}"
    
    print(f"✓ Correct: r1 has {len(r1_nodes)} learnable node(s)")
    
    # Check alpha update for r1
    bond_params_r1 = model.mu_network.distributions['r1']
    alpha_0_r1 = bond_params_r1['alpha_0']
    bond_dim_r1 = len(bond_params_r1['alpha'])
    
    # Expected: α = α_0 + |A(r1)| * dim(r1) / 2
    # |A(r1)| = 1 (only A is learnable)
    expected_alpha_contrib_r1 = len(r1_nodes) * bond_dim_r1 / 2.0
    print(f"Expected alpha contribution: |A(r1)|={len(r1_nodes)} × dim={bond_dim_r1} / 2 = {expected_alpha_contrib_r1}")
    
    # ========================================================================
    # VERIFY r2 bond
    # ========================================================================
    print("\n" + "-"*70)
    print("Bond r2: Should count both A and B")
    print("-"*70)
    
    r2_nodes = model.mu_network.get_nodes_for_bond('r2')
    print(f"Nodes connected to r2: {r2_nodes}")
    
    assert 'A' in r2_nodes, "A should be in r2 nodes"
    assert 'B' in r2_nodes, "B should be in r2 nodes"
    assert len(r2_nodes) == 2, f"r2 should have 2 learnable nodes, got {len(r2_nodes)}"
    
    print(f"✓ Correct: r2 has {len(r2_nodes)} learnable node(s)")
    
    # Check alpha update for r2
    bond_params_r2 = model.mu_network.distributions['r2']
    bond_dim_r2 = len(bond_params_r2['alpha'])
    
    # Expected: α = α_0 + |A(r2)| * dim(r2) / 2
    # |A(r2)| = 2 (both A and B are learnable)
    expected_alpha_contrib_r2 = len(r2_nodes) * bond_dim_r2 / 2.0
    print(f"Expected alpha contribution: |A(r2)|={len(r2_nodes)} × dim={bond_dim_r2} / 2 = {expected_alpha_contrib_r2}")
    
    # ========================================================================
    # PERFORM BOND UPDATES
    # ========================================================================
    print("\n" + "-"*70)
    print("Performing Bond Updates")
    print("-"*70)
    
    # Store initial parameters
    alpha_before_r1 = model.mu_network.distributions['r1']['alpha'].clone()
    alpha_before_r2 = model.mu_network.distributions['r2']['alpha'].clone()
    
    print(f"\nBefore updates:")
    print(f"  r1 alpha: {alpha_before_r1[0].item():.4f}")
    print(f"  r2 alpha: {alpha_before_r2[0].item():.4f}")
    
    # Update bonds
    model.update_bond_variational('r1')
    model.update_bond_variational('r2')
    
    alpha_after_r1 = model.mu_network.distributions['r1']['alpha']
    alpha_after_r2 = model.mu_network.distributions['r2']['alpha']
    
    print(f"\nAfter updates:")
    print(f"  r1 alpha: {alpha_after_r1[0].item():.4f}")
    print(f"  r2 alpha: {alpha_after_r2[0].item():.4f}")
    
    # Verify alpha updates
    actual_alpha_r1 = alpha_after_r1[0].item()
    expected_alpha_r1 = (alpha_0_r1[0] + expected_alpha_contrib_r1).item()
    
    print(f"\nr1 alpha verification:")
    print(f"  Expected: {expected_alpha_r1:.4f}")
    print(f"  Actual:   {actual_alpha_r1:.4f}")
    print(f"  Match: {np.isclose(actual_alpha_r1, expected_alpha_r1)}")
    
    assert np.isclose(actual_alpha_r1, expected_alpha_r1), \
        f"r1 alpha mismatch: expected {expected_alpha_r1}, got {actual_alpha_r1}"
    
    print("\n✓ SUCCESS: Bond updates correctly handle fixed nodes!")


def test_bond_update_all_nodes_fixed():
    """
    Test that bonds connected only to fixed nodes are not updated.
    
    Network: INPUT1(x, r1) -- INPUT2(r1, y)
    - Both nodes are fixed
    - Bond r1 should have NO learnable nodes
    - Bond update should be skipped or handle gracefully
    """
    print("\n" + "="*70)
    print("TEST: Bond Update with All Fixed Nodes")
    print("="*70)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create network with all fixed nodes
    INPUT1_data = torch.ones(3, 2, dtype=torch.float64)
    INPUT2_data = torch.ones(2, 1, dtype=torch.float64)
    
    INPUT1 = qtn.Tensor(data=INPUT1_data, inds=('x', 'r1'), tags='INPUT1')
    INPUT2 = qtn.Tensor(data=INPUT2_data, inds=('r1', 'y'), tags='INPUT2')
    mu_tn = qtn.TensorNetwork([INPUT1, INPUT2])
    
    # No learnable nodes!
    learnable_tags = []
    
    # Create sigma network (empty)
    sigma_tn = create_sigma_network(mu_tn, learnable_tags, output_indices=['y'])
    
    print(f"Network structure:")
    print(f"  INPUT1(x, r1) [FIXED] -- r1 -- INPUT2(r1, y) [FIXED]")
    print(f"  Learnable nodes: {learnable_tags if learnable_tags else 'NONE'}")
    
    # Create model
    model = BayesianTensorNetwork(
        mu_tn=mu_tn,
        sigma_tn=sigma_tn,
        input_indices={'data': ['x']},
        output_indices=['y'],
        learnable_tags=learnable_tags,
        dtype=torch.float64
    )
    
    print(f"\nBond labels: {model.mu_network.bond_labels}")
    
    # Check that r1 has no learnable nodes
    if 'r1' in model.mu_network.bond_labels:
        r1_nodes = model.mu_network.get_nodes_for_bond('r1')
        print(f"Nodes connected to r1: {r1_nodes if r1_nodes else 'NONE'}")
        
        assert len(r1_nodes) == 0, f"r1 should have 0 learnable nodes, got {len(r1_nodes)}"
        print("✓ Correct: r1 has no learnable nodes")
        
        # Try to update - should be skipped
        print("\nAttempting bond update (should be skipped)...")
        model.update_bond_variational('r1')
        print("✓ Bond update handled gracefully")
    else:
        print("✓ Bond r1 not in learnable bonds (as expected)")
    
    print("\n✓ SUCCESS: All-fixed network handled correctly!")


def test_bond_update_partial_learning():
    """
    Test more complex case with multiple fixed and learnable nodes.
    
    Network: 
        INPUT(x, r1) -- A(r1, r2) -- B(r2, r3) -- FIXED(r3, y)
    
    Learnable: A, B
    Fixed: INPUT, FIXED
    
    Bonds:
    - r1: A(r1) = {A} → |A(r1)| = 1
    - r2: A(r2) = {A, B} → |A(r2)| = 2
    - r3: A(r3) = {B} → |A(r3)| = 1
    """
    print("\n" + "="*70)
    print("TEST: Bond Update with Partial Learning")
    print("="*70)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    INPUT_data = torch.ones(4, 2, dtype=torch.float64)
    A_data = torch.randn(2, 3, dtype=torch.float64)
    B_data = torch.randn(3, 2, dtype=torch.float64)
    FIXED_data = torch.ones(2, 1, dtype=torch.float64)
    
    INPUT = qtn.Tensor(data=INPUT_data, inds=('x', 'r1'), tags='INPUT')
    A = qtn.Tensor(data=A_data, inds=('r1', 'r2'), tags='A')
    B = qtn.Tensor(data=B_data, inds=('r2', 'r3'), tags='B')
    FIXED = qtn.Tensor(data=FIXED_data, inds=('r3', 'y'), tags='FIXED')
    mu_tn = qtn.TensorNetwork([INPUT, A, B, FIXED])
    
    learnable_tags = ['A', 'B']
    
    sigma_tn = create_sigma_network(mu_tn, learnable_tags, output_indices=['y'])
    
    print(f"Network structure:")
    print(f"  INPUT(x,r1) [FIXED] -- A(r1,r2) [LEARN] -- B(r2,r3) [LEARN] -- FIXED(r3,y) [FIXED]")
    print(f"  Learnable: {learnable_tags}")
    
    model = BayesianTensorNetwork(
        mu_tn=mu_tn,
        sigma_tn=sigma_tn,
        input_indices={'data': ['x']},
        output_indices=['y'],
        learnable_tags=learnable_tags,
        dtype=torch.float64
    )
    
    print(f"\nBond to nodes mapping:")
    for bond in ['r1', 'r2', 'r3']:
        if bond in model.mu_network.bond_labels:
            nodes = model.mu_network.get_nodes_for_bond(bond)
            print(f"  {bond}: {nodes} → |A({bond})| = {len(nodes)}")
    
    # Verify expectations
    r1_nodes = model.mu_network.get_nodes_for_bond('r1')
    r2_nodes = model.mu_network.get_nodes_for_bond('r2')
    r3_nodes = model.mu_network.get_nodes_for_bond('r3')
    
    assert len(r1_nodes) == 1 and 'A' in r1_nodes, "r1 should only have A"
    assert len(r2_nodes) == 2 and 'A' in r2_nodes and 'B' in r2_nodes, "r2 should have A and B"
    assert len(r3_nodes) == 1 and 'B' in r3_nodes, "r3 should only have B"
    
    print("\n✓ All bond counts correct!")
    print("✓ SUCCESS: Partial learning case handled correctly!")


if __name__ == '__main__':
    test_bond_update_with_fixed_node()
    test_bond_update_all_nodes_fixed()
    test_bond_update_partial_learning()
    
    print("\n" + "="*70)
    print("ALL BOND UPDATE TESTS PASSED!")
    print("="*70)
