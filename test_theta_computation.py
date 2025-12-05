"""
Test theta_block_computation method for BTN class.

Tests verify:
1. Shape consistency with node shape minus excluded dimensions
2. Correct outer product computation
3. Exclusion of output dimensions and batch dim
4. Custom bond exclusions
"""

import numpy as np
import quimb.tensor as qt
from tensor.btn import BTN


def create_simple_btn():
    """Create a simple BTN for testing with known structure."""
    # Create a simple tensor network: one node with multiple bonds
    # Node shape: (batch=10, bond_a=2, bond_b=3, bond_c=4, output=5)
    data = np.random.randn(10, 2, 3, 4, 5)
    node = qt.Tensor(data, inds=('s', 'a', 'b', 'c', 'out'), tags={'node1'})
    
    # Create mu network
    mu = qt.TensorNetwork([node])
    
    # Create BTN
    btn = BTN(
        mu=mu,
        output_dimensions=['out'],
        batch_dim='s',
        input_indices=[]
    )
    
    return btn


def test_theta_shape_basic():
    """Test that theta has correct shape for a simple node."""
    print("\n=== Test 1: Basic Shape Check ===")
    btn = create_simple_btn()
    
    # Get node shape
    node = btn.mu['node1']
    print(f"Node indices: {node.inds}")
    print(f"Node shape: {node.shape}")
    
    # Compute theta
    theta = btn.theta_block_computation('node1')
    
    print(f"Theta indices: {theta.inds}")
    print(f"Theta shape: {theta.shape}")
    
    # Expected shape: (2, 3, 4) - all bonds except output and batch
    expected_shape = (2, 3, 4)
    expected_inds = ('a', 'b', 'c')
    
    assert theta.shape == expected_shape, f"Expected shape {expected_shape}, got {theta.shape}"
    assert set(theta.inds) == set(expected_inds), f"Expected indices {expected_inds}, got {theta.inds}"
    
    print("✓ Shape test passed!")


def test_theta_excludes_output_and_batch():
    """Test that output dimensions and batch dim are excluded."""
    print("\n=== Test 2: Output and Batch Exclusion ===")
    btn = create_simple_btn()
    
    theta = btn.theta_block_computation('node1')
    
    # Verify output and batch not in theta
    assert 'out' not in theta.inds, "Output dimension should be excluded"
    assert 's' not in theta.inds, "Batch dimension should be excluded"
    
    print(f"Theta indices: {theta.inds}")
    print("✓ Output and batch exclusion test passed!")


def test_theta_custom_exclusion():
    """Test custom bond exclusion."""
    print("\n=== Test 3: Custom Bond Exclusion ===")
    btn = create_simple_btn()
    
    # Exclude bond 'b'
    theta = btn.theta_block_computation('node1', exclude_bonds=['b'])
    
    print(f"Theta indices (excluding 'b'): {theta.inds}")
    print(f"Theta shape: {theta.shape}")
    
    # Expected shape: (2, 4) - bonds a and c only
    expected_shape = (2, 4)
    expected_inds = ('a', 'c')
    
    assert theta.shape == expected_shape, f"Expected shape {expected_shape}, got {theta.shape}"
    assert set(theta.inds) == set(expected_inds), f"Expected indices {expected_inds}, got {theta.inds}"
    assert 'b' not in theta.inds, "Bond 'b' should be excluded"
    
    print("✓ Custom exclusion test passed!")


def test_theta_computation_values():
    """Test that theta values are correct outer products of bond expectations."""
    print("\n=== Test 4: Theta Values Correctness ===")
    btn = create_simple_btn()
    
    # Get individual bond expectations
    e_a = btn.q_bonds['a'].mean()
    e_b = btn.q_bonds['b'].mean()
    e_c = btn.q_bonds['c'].mean()
    
    print(f"E[λ_a] shape: {e_a.shape}, data: {e_a.data}")
    print(f"E[λ_b] shape: {e_b.shape}, data: {e_b.data}")
    print(f"E[λ_c] shape: {e_c.shape}, data: {e_c.data}")
    
    # Compute theta
    theta = btn.theta_block_computation('node1')
    
    # Manually compute expected outer product
    expected = np.einsum('i,j,k->ijk', e_a.data, e_b.data, e_c.data)
    
    print(f"Theta shape: {theta.shape}")
    print(f"Expected outer product shape: {expected.shape}")
    
    # Check values match
    np.testing.assert_allclose(theta.data, expected, rtol=1e-10)
    
    print("✓ Value correctness test passed!")


def test_theta_with_multiple_nodes():
    """Test theta computation for network with multiple nodes."""
    print("\n=== Test 5: Multiple Nodes ===")
    
    # Create network with two nodes sharing a bond
    node1_data = np.random.randn(10, 2, 3, 5)  # (s, a, shared, out)
    node2_data = np.random.randn(10, 3, 4, 5)  # (s, shared, d, out)
    
    node1 = qt.Tensor(node1_data, inds=('s', 'a', 'shared', 'out'), tags={'node1'})
    node2 = qt.Tensor(node2_data, inds=('s', 'shared', 'd', 'out'), tags={'node2'})
    
    mu = qt.TensorNetwork([node1, node2])
    
    btn = BTN(
        mu=mu,
        output_dimensions=['out'],
        batch_dim='s',
        input_indices=[]
    )
    
    # Compute theta for node1
    theta1 = btn.theta_block_computation('node1')
    print(f"Theta1 indices: {theta1.inds}, shape: {theta1.shape}")
    
    # Expected: bonds 'a' and 'shared' (exclude 's' and 'out')
    assert theta1.shape == (2, 3), f"Expected shape (2, 3), got {theta1.shape}"
    assert set(theta1.inds) == {'a', 'shared'}, f"Expected indices {{'a', 'shared'}}, got {theta1.inds}"
    
    # Compute theta for node2
    theta2 = btn.theta_block_computation('node2')
    print(f"Theta2 indices: {theta2.inds}, shape: {theta2.shape}")
    
    # Expected: bonds 'shared' and 'd' (exclude 's' and 'out')
    assert theta2.shape == (3, 4), f"Expected shape (3, 4), got {theta2.shape}"
    assert set(theta2.inds) == {'shared', 'd'}, f"Expected indices {{'shared', 'd'}}, got {theta2.inds}"
    
    print("✓ Multiple nodes test passed!")


def test_theta_single_bond():
    """Test theta for node with only one bond."""
    print("\n=== Test 6: Single Bond Node ===")
    
    # Create node with just one bond (plus batch and output)
    data = np.random.randn(10, 7, 5)  # (s, a, out)
    node = qt.Tensor(data, inds=('s', 'a', 'out'), tags={'node1'})
    
    mu = qt.TensorNetwork([node])
    btn = BTN(mu=mu, output_dimensions=['out'], batch_dim='s', input_indices=[])
    
    theta = btn.theta_block_computation('node1')
    
    print(f"Theta indices: {theta.inds}, shape: {theta.shape}")
    
    # Should just be the bond expectation itself
    assert theta.shape == (7,), f"Expected shape (7,), got {theta.shape}"
    assert theta.inds == ('a',), f"Expected indices ('a',), got {theta.inds}"
    
    # Verify it equals the bond expectation
    e_a = btn.q_bonds['a'].mean()
    np.testing.assert_allclose(theta.data, e_a.data, rtol=1e-10)
    
    print("✓ Single bond test passed!")


if __name__ == '__main__':
    test_theta_shape_basic()
    test_theta_excludes_output_and_batch()
    test_theta_custom_exclusion()
    test_theta_computation_values()
    test_theta_with_multiple_nodes()
    test_theta_single_bond()
    
    print("\n" + "="*50)
    print("ALL TESTS PASSED! ✓")
    print("="*50)
