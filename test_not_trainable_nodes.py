"""
Test not_trainable_nodes tagging in BTN class.

Tests verify:
1. NT tag is added to specified nodes
2. Multiple nodes can be tagged
3. Nodes maintain their original tags
4. Can query trainable vs not trainable nodes
"""

import numpy as np
import quimb.tensor as qt
from tensor.btn import BTN, NOT_TRAINABLE_TAG


def test_single_not_trainable_node():
    """Test that a single node gets tagged with NT."""
    print("\n=== Test 1: Single Not Trainable Node ===")
    
    # Create network with 3 nodes
    node1 = qt.Tensor(np.random.randn(2, 3, 5), inds=('a', 'b', 'out'), tags={'node1'})
    node2 = qt.Tensor(np.random.randn(3, 4, 5), inds=('b', 'c', 'out'), tags={'node2'})
    node3 = qt.Tensor(np.random.randn(4, 2, 5), inds=('c', 'a', 'out'), tags={'node3', 'input'})
    
    mu = qt.TensorNetwork([node1, node2, node3])
    
    # Create BTN with node3 as not trainable
    btn = BTN(
        mu=mu,
        output_dimensions=['out'],
        not_trainable_nodes=['node3']
    )
    
    # Check that node3 has NT tag
    node3_tensor = btn.mu['node3']
    print(f"Node3 tags: {node3_tensor.tags}")
    
    assert NOT_TRAINABLE_TAG in node3_tensor.tags, "NT tag should be in node3"
    assert 'node3' in node3_tensor.tags, "Original tag should be preserved"
    assert 'input' in node3_tensor.tags, "Additional tags should be preserved"
    
    # Check that other nodes don't have NT tag
    node1_tensor = btn.mu['node1']
    node2_tensor = btn.mu['node2']
    
    assert NOT_TRAINABLE_TAG not in node1_tensor.tags, "NT tag should not be in node1"
    assert NOT_TRAINABLE_TAG not in node2_tensor.tags, "NT tag should not be in node2"
    
    print("✓ Single not trainable node test passed!")


def test_multiple_not_trainable_nodes():
    """Test that multiple nodes can be tagged as not trainable."""
    print("\n=== Test 2: Multiple Not Trainable Nodes ===")
    
    # Create network with 4 nodes
    input1 = qt.Tensor(np.random.randn(2, 3, 5), inds=('x1', 'b', 'out'), tags={'input1'})
    input2 = qt.Tensor(np.random.randn(4, 3, 5), inds=('x2', 'b', 'out'), tags={'input2'})
    hidden1 = qt.Tensor(np.random.randn(3, 6, 5), inds=('b', 'd', 'out'), tags={'hidden1'})
    output = qt.Tensor(np.random.randn(6, 5), inds=('d', 'out'), tags={'output'})
    
    mu = qt.TensorNetwork([input1, input2, hidden1, output])
    
    # Create BTN with both inputs as not trainable
    btn = BTN(
        mu=mu,
        output_dimensions=['out'],
        not_trainable_nodes=['input1', 'input2']
    )
    
    # Check that both inputs have NT tag
    input1_tensor = btn.mu['input1']
    input2_tensor = btn.mu['input2']
    
    print(f"Input1 tags: {input1_tensor.tags}")
    print(f"Input2 tags: {input2_tensor.tags}")
    
    assert NOT_TRAINABLE_TAG in input1_tensor.tags, "NT tag should be in input1"
    assert NOT_TRAINABLE_TAG in input2_tensor.tags, "NT tag should be in input2"
    
    # Check that hidden and output don't have NT tag
    hidden1_tensor = btn.mu['hidden1']
    output_tensor = btn.mu['output']
    
    assert NOT_TRAINABLE_TAG not in hidden1_tensor.tags, "NT tag should not be in hidden1"
    assert NOT_TRAINABLE_TAG not in output_tensor.tags, "NT tag should not be in output"
    
    print("✓ Multiple not trainable nodes test passed!")


def test_query_not_trainable_nodes():
    """Test that we can query for trainable vs not trainable nodes."""
    print("\n=== Test 3: Query Not Trainable Nodes ===")
    
    # Create network
    node1 = qt.Tensor(np.random.randn(2, 3, 5), inds=('a', 'b', 'out'), tags={'node1'})
    node2 = qt.Tensor(np.random.randn(3, 4, 5), inds=('b', 'c', 'out'), tags={'node2'})
    node3 = qt.Tensor(np.random.randn(4, 2, 5), inds=('c', 'a', 'out'), tags={'node3'})
    
    mu = qt.TensorNetwork([node1, node2, node3])
    
    btn = BTN(
        mu=mu,
        output_dimensions=['out'],
        not_trainable_nodes=['node1', 'node3']
    )
    
    # Query not trainable nodes using NT tag
    not_trainable = btn.mu.select_tensors(NOT_TRAINABLE_TAG, which='any')
    print(f"Not trainable nodes count: {len(not_trainable)}")
    
    assert len(not_trainable) == 2, "Should have 2 not trainable nodes"
    
    # Check tags
    not_trainable_tags = {tag for t in not_trainable for tag in t.tags if tag != NOT_TRAINABLE_TAG}
    print(f"Not trainable node tags: {not_trainable_tags}")
    
    assert 'node1' in not_trainable_tags, "node1 should be not trainable"
    assert 'node3' in not_trainable_tags, "node3 should be not trainable"
    assert 'node2' not in not_trainable_tags, "node2 should be trainable"
    
    print("✓ Query not trainable nodes test passed!")


def test_no_not_trainable_nodes():
    """Test BTN when no nodes are marked as not trainable."""
    print("\n=== Test 4: No Not Trainable Nodes ===")
    
    # Create network
    node1 = qt.Tensor(np.random.randn(2, 3, 5), inds=('a', 'b', 'out'), tags={'node1'})
    node2 = qt.Tensor(np.random.randn(3, 4, 5), inds=('b', 'c', 'out'), tags={'node2'})
    
    mu = qt.TensorNetwork([node1, node2])
    
    # Create BTN without specifying not_trainable_nodes
    btn = BTN(
        mu=mu,
        output_dimensions=['out']
    )
    
    # Check that no nodes have NT tag
    for tensor in btn.mu:
        assert NOT_TRAINABLE_TAG not in tensor.tags, f"No tensor should have NT tag"
    
    print(f"not_trainable_nodes list: {btn.not_trainable_nodes}")
    assert btn.not_trainable_nodes == [], "Should be empty list"
    
    print("✓ No not trainable nodes test passed!")


def test_tag_based_selection():
    """Test that tag-based selection works for trainable nodes."""
    print("\n=== Test 5: Tag-Based Selection for Training ===")
    
    # Create network
    input_node = qt.Tensor(np.random.randn(2, 3, 5), inds=('x', 'b', 'out'), tags={'input'})
    hidden1 = qt.Tensor(np.random.randn(3, 4, 5), inds=('b', 'c', 'out'), tags={'hidden1'})
    hidden2 = qt.Tensor(np.random.randn(4, 6, 5), inds=('c', 'd', 'out'), tags={'hidden2'})
    output_node = qt.Tensor(np.random.randn(6, 5), inds=('d', 'out'), tags={'output'})
    
    mu = qt.TensorNetwork([input_node, hidden1, hidden2, output_node])
    
    btn = BTN(
        mu=mu,
        output_dimensions=['out'],
        not_trainable_nodes=['input']
    )
    
    # Get all tensors
    all_tensors = list(btn.mu)
    print(f"Total tensors: {len(all_tensors)}")
    
    # Get trainable tensors (those without NT tag)
    trainable_tensors = [t for t in all_tensors if NOT_TRAINABLE_TAG not in t.tags]
    print(f"Trainable tensors: {len(trainable_tensors)}")
    
    assert len(trainable_tensors) == 3, "Should have 3 trainable tensors"
    
    # Get not trainable tensors
    not_trainable_tensors = list(btn.mu.select_tensors(NOT_TRAINABLE_TAG, which='any'))
    print(f"Not trainable tensors: {len(not_trainable_tensors)}")
    
    assert len(not_trainable_tensors) == 1, "Should have 1 not trainable tensor"
    
    print("✓ Tag-based selection test passed!")


if __name__ == '__main__':
    test_single_not_trainable_node()
    test_multiple_not_trainable_nodes()
    test_query_not_trainable_nodes()
    test_no_not_trainable_nodes()
    test_tag_based_selection()
    
    print("\n" + "="*50)
    print("ALL TESTS PASSED! ✓")
    print("="*50)
