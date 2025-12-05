"""
Test count_trainable_nodes_on_bond method.

Tests verify:
1. Counting trainable vs not trainable nodes on a bond
2. Bonds shared by multiple nodes
3. Bonds with all trainable nodes
4. Bonds with all not trainable nodes
5. Bonds with no nodes (should not happen in valid TN)
"""

import numpy as np
import quimb.tensor as qt
from tensor.btn import BTN, NOT_TRAINABLE_TAG


def test_mixed_trainable_nodes_on_bond():
    """Test counting when bond has both trainable and not trainable nodes."""
    print("\n=== Test 1: Mixed Trainable Nodes on Bond ===")
    
    # Create network where bond 'a' is shared by 3 nodes
    # node1 (trainable), node2 (trainable), input (not trainable)
    node1 = qt.Tensor(np.random.randn(2, 3, 5), inds=('a', 'b', 'out'), tags={'node1'})
    node2 = qt.Tensor(np.random.randn(2, 4, 5), inds=('a', 'c', 'out'), tags={'node2'})
    input_node = qt.Tensor(np.random.randn(2, 6, 5), inds=('a', 'd', 'out'), tags={'input'})
    
    mu = qt.TensorNetwork([node1, node2, input_node])
    
    btn = BTN(
        mu=mu,
        output_dimensions=['out'],
        not_trainable_nodes=['input']
    )
    
    # Count trainable nodes on bond 'a'
    count = btn.count_trainable_nodes_on_bond('a')
    print(f"Bond 'a' has {count} trainable nodes")
    
    assert count == 2, f"Expected 2 trainable nodes, got {count}"
    
    print("✓ Mixed trainable nodes test passed!")


def test_all_trainable_nodes_on_bond():
    """Test when all nodes sharing a bond are trainable."""
    print("\n=== Test 2: All Trainable Nodes on Bond ===")
    
    # Create network where bond 'b' is shared by 2 trainable nodes
    node1 = qt.Tensor(np.random.randn(2, 3, 5), inds=('a', 'b', 'out'), tags={'node1'})
    node2 = qt.Tensor(np.random.randn(3, 4, 5), inds=('b', 'c', 'out'), tags={'node2'})
    input_node = qt.Tensor(np.random.randn(6, 5), inds=('d', 'out'), tags={'input'})
    
    mu = qt.TensorNetwork([node1, node2, input_node])
    
    btn = BTN(
        mu=mu,
        output_dimensions=['out'],
        not_trainable_nodes=['input']
    )
    
    # Count trainable nodes on bond 'b'
    count = btn.count_trainable_nodes_on_bond('b')
    print(f"Bond 'b' has {count} trainable nodes")
    
    assert count == 2, f"Expected 2 trainable nodes, got {count}"
    
    print("✓ All trainable nodes test passed!")


def test_no_trainable_nodes_on_bond():
    """Test when all nodes sharing a bond are not trainable."""
    print("\n=== Test 3: No Trainable Nodes on Bond ===")
    
    # Create network where bond 'shared' connects two input nodes
    input1 = qt.Tensor(np.random.randn(2, 3, 5), inds=('x1', 'shared', 'out'), tags={'input1'})
    input2 = qt.Tensor(np.random.randn(3, 4, 5), inds=('shared', 'x2', 'out'), tags={'input2'})
    hidden = qt.Tensor(np.random.randn(6, 5), inds=('d', 'out'), tags={'hidden'})
    
    mu = qt.TensorNetwork([input1, input2, hidden])
    
    btn = BTN(
        mu=mu,
        output_dimensions=['out'],
        not_trainable_nodes=['input1', 'input2']
    )
    
    # Count trainable nodes on bond 'shared'
    count = btn.count_trainable_nodes_on_bond('shared')
    print(f"Bond 'shared' has {count} trainable nodes")
    
    assert count == 0, f"Expected 0 trainable nodes, got {count}"
    
    print("✓ No trainable nodes test passed!")


def test_single_node_on_bond():
    """Test counting for a bond that only appears in one node (leaf index)."""
    print("\n=== Test 4: Single Node on Bond ===")
    
    # Create network where 'a' is a leaf index (only in node1)
    node1 = qt.Tensor(np.random.randn(2, 3, 5), inds=('a', 'b', 'out'), tags={'node1'})
    node2 = qt.Tensor(np.random.randn(3, 4, 5), inds=('b', 'c', 'out'), tags={'node2'})
    
    mu = qt.TensorNetwork([node1, node2])
    
    btn = BTN(
        mu=mu,
        output_dimensions=['out']
    )
    
    # Count trainable nodes on leaf bond 'a'
    count = btn.count_trainable_nodes_on_bond('a')
    print(f"Leaf bond 'a' has {count} trainable nodes")
    
    assert count == 1, f"Expected 1 trainable node, got {count}"
    
    print("✓ Single node on bond test passed!")


def test_count_across_multiple_bonds():
    """Test counting across different bonds in the same network."""
    print("\n=== Test 5: Count Across Multiple Bonds ===")
    
    # Create more complex network
    input_node = qt.Tensor(np.random.randn(2, 3, 5), inds=('x', 'a', 'out'), tags={'input'})
    hidden1 = qt.Tensor(np.random.randn(3, 4, 5), inds=('a', 'b', 'out'), tags={'h1'})
    hidden2 = qt.Tensor(np.random.randn(4, 6, 5), inds=('b', 'c', 'out'), tags={'h2'})
    output_node = qt.Tensor(np.random.randn(6, 5), inds=('c', 'out'), tags={'output'})
    
    mu = qt.TensorNetwork([input_node, hidden1, hidden2, output_node])
    
    btn = BTN(
        mu=mu,
        output_dimensions=['out'],
        not_trainable_nodes=['input']
    )
    
    # Count trainable nodes on different bonds
    count_a = btn.count_trainable_nodes_on_bond('a')
    count_b = btn.count_trainable_nodes_on_bond('b')
    count_c = btn.count_trainable_nodes_on_bond('c')
    count_x = btn.count_trainable_nodes_on_bond('x')
    
    print(f"Bond 'a': {count_a} trainable nodes (input-h1)")
    print(f"Bond 'b': {count_b} trainable nodes (h1-h2)")
    print(f"Bond 'c': {count_c} trainable nodes (h2-output)")
    print(f"Bond 'x': {count_x} trainable nodes (input leaf)")
    
    assert count_a == 1, f"Bond 'a' should have 1 trainable node (h1), got {count_a}"
    assert count_b == 2, f"Bond 'b' should have 2 trainable nodes (h1, h2), got {count_b}"
    assert count_c == 2, f"Bond 'c' should have 2 trainable nodes (h2, output), got {count_c}"
    assert count_x == 0, f"Bond 'x' should have 0 trainable nodes (input leaf), got {count_x}"
    
    print("✓ Multiple bonds test passed!")


def test_nonexistent_bond():
    """Test counting for a bond that doesn't exist."""
    print("\n=== Test 6: Nonexistent Bond ===")
    
    node1 = qt.Tensor(np.random.randn(2, 3, 5), inds=('a', 'b', 'out'), tags={'node1'})
    node2 = qt.Tensor(np.random.randn(3, 4, 5), inds=('b', 'c', 'out'), tags={'node2'})
    
    mu = qt.TensorNetwork([node1, node2])
    btn = BTN(mu=mu, output_dimensions=['out'])
    
    # Count on nonexistent bond
    count = btn.count_trainable_nodes_on_bond('nonexistent')
    print(f"Nonexistent bond has {count} trainable nodes")
    
    assert count == 0, f"Expected 0 for nonexistent bond, got {count}"
    
    print("✓ Nonexistent bond test passed!")


if __name__ == '__main__':
    test_mixed_trainable_nodes_on_bond()
    test_all_trainable_nodes_on_bond()
    test_no_trainable_nodes_on_bond()
    test_single_node_on_bond()
    test_count_across_multiple_bonds()
    test_nonexistent_bond()
    
    print("\n" + "="*50)
    print("ALL TESTS PASSED! ✓")
    print("="*50)
