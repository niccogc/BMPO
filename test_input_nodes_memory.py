"""
Test that input nodes share memory correctly.

For μ-MPO: All input nodes should point to the same tensor (share memory)
For Σ-MPO: All input nodes should point to the same tensor (share memory)
"""

import torch
from tensor.bayesian_mpo_builder import create_bayesian_tensor_train
from tensor.node import TensorNode
from tensor.bayesian_mpo import BayesianMPO


def test_mu_input_nodes():
    """Test μ-MPO input nodes."""
    print("=" * 70)
    print("Testing μ-MPO Input Nodes")
    print("=" * 70)
    
    # Create a Bayesian TT
    bmpo = create_bayesian_tensor_train(
        num_blocks=3,
        bond_dim=4,
        input_features=5,
        output_shape=1,
        seed=42
    )
    
    print(f"\nNumber of μ-MPO input nodes: {len(bmpo.input_nodes)}")
    print(f"Number of μ-MPO main nodes: {len(bmpo.mu_nodes)}")
    
    # Check if all input nodes share memory
    if len(bmpo.input_nodes) > 0:
        print(f"\nμ-MPO input node shapes:")
        for i, node in enumerate(bmpo.input_nodes):
            print(f"  Input node {i}: shape {node.tensor.shape}, labels {node.dim_labels}")
        
        # Check memory sharing
        print(f"\nChecking memory sharing for μ-MPO inputs:")
        first_data_ptr = bmpo.input_nodes[0].tensor.data_ptr()
        all_share_memory = True
        
        for i, node in enumerate(bmpo.input_nodes):
            shares = node.tensor.data_ptr() == first_data_ptr
            print(f"  Node {i} data_ptr: {node.tensor.data_ptr()} - {'✓ SHARES' if shares else '✗ DIFFERENT'}")
            if not shares:
                all_share_memory = False
        
        if all_share_memory:
            print(f"\n✓ All {len(bmpo.input_nodes)} μ-MPO input nodes share the same memory!")
        else:
            print(f"\n✗ μ-MPO input nodes DO NOT share memory (PROBLEM!)")
        
        # Test: Set data on first node, verify it appears on all
        print(f"\nTest: Setting data on first input node...")
        test_data = torch.randn(10, 5)
        bmpo.input_nodes[0].tensor = test_data
        
        print(f"  First node data_ptr after set: {bmpo.input_nodes[0].tensor.data_ptr()}")
        for i in range(1, len(bmpo.input_nodes)):
            same = torch.equal(bmpo.input_nodes[i].tensor, test_data)
            print(f"  Node {i} has same data: {'✓ YES' if same else '✗ NO'}")
    else:
        print("\nNo input nodes found for μ-MPO!")


def test_sigma_input_nodes():
    """Test Σ-MPO input nodes."""
    print("\n" + "=" * 70)
    print("Testing Σ-MPO Input Nodes")
    print("=" * 70)
    
    # Create a Bayesian TT
    bmpo = create_bayesian_tensor_train(
        num_blocks=3,
        bond_dim=4,
        input_features=5,
        output_shape=1,
        seed=42
    )
    
    # Check if Σ input nodes exist
    if hasattr(bmpo, 'sigma_input_nodes'):
        print(f"\nNumber of Σ-MPO input nodes: {len(bmpo.sigma_input_nodes)}")
        print(f"Number of μ-MPO input nodes: {len(bmpo.input_nodes)}")
        print(f"Ratio: {len(bmpo.sigma_input_nodes)} / {len(bmpo.input_nodes)} = {len(bmpo.sigma_input_nodes) / len(bmpo.input_nodes):.1f}")
        print(f"  (Should be 2:1 since Σ has 'o' and 'i' for each μ input)")
        
        print(f"\nΣ-MPO input node shapes:")
        for i, node in enumerate(bmpo.sigma_input_nodes):
            print(f"  Σ-Input node {i}: shape {node.tensor.shape}, labels {node.dim_labels}")
        
        # Check memory sharing patterns
        print(f"\nChecking memory sharing for Σ-MPO inputs:")
        print(f"  (Note: 'o' and 'i' pairs should share memory)")
        
        # Check pairs (0,1), (2,3), (4,5), etc.
        num_mu_inputs = len(bmpo.input_nodes)
        for mu_idx in range(num_mu_inputs):
            o_idx = 2 * mu_idx
            i_idx = 2 * mu_idx + 1
            
            if i_idx < len(bmpo.sigma_input_nodes):
                o_node = bmpo.sigma_input_nodes[o_idx]
                i_node = bmpo.sigma_input_nodes[i_idx]
                
                o_ptr = o_node.tensor.data_ptr()
                i_ptr = i_node.tensor.data_ptr()
                shares = (o_ptr == i_ptr)
                
                print(f"  Pair {mu_idx}: o[{o_idx}] ptr={o_ptr}, i[{i_idx}] ptr={i_ptr} - {'✓ SHARE' if shares else '✗ DIFFERENT'}")
        
        # Test: Set data and check propagation
        print(f"\nTest: Setting data on Σ input pairs...")
        test_data = torch.randn(10, 5)
        
        # Set on first 'o' node
        bmpo.sigma_input_nodes[0].tensor = test_data
        print(f"  Set data on Σ-input[0] (outer)")
        
        # Check if pair shares it
        same_as_inner = torch.equal(bmpo.sigma_input_nodes[1].tensor, test_data)
        print(f"  Σ-input[1] (inner) has same data: {'✓ YES' if same_as_inner else '✗ NO'}")
        
    else:
        print("\nNo sigma_input_nodes attribute found!")
        print("This might be okay if using manual construction instead of builder.")


def test_manual_construction():
    """Test manual construction without builder."""
    print("\n" + "=" * 70)
    print("Testing Manual Construction")
    print("=" * 70)
    
    # Create μ-MPO node
    mu_node = TensorNode(
        tensor_or_shape=(2, 3, 2),
        dim_labels=['r0', 'f', 'r1'],
        l='r0',
        r='r1',
        name='mu1'
    )
    
    # Create input node
    input_node = TensorNode(
        tensor_or_shape=(1, 3),
        dim_labels=['s', 'f'],
        name='X1'
    )
    
    input_node.connect(mu_node, 'f')
    
    # Create BayesianMPO
    bmpo = BayesianMPO(
        mu_nodes=[mu_node],
        input_nodes=[input_node],
        rank_labels={'r0', 'r1'}
    )
    
    print(f"\nManual construction:")
    print(f"  Number of μ-MPO input nodes: {len(bmpo.input_nodes)}")
    print(f"  Number of Σ-MPO input nodes: {len(bmpo.sigma_mpo.input_nodes)}")
    
    print(f"\nμ-MPO input nodes:")
    for i, node in enumerate(bmpo.input_nodes):
        print(f"  Input {i}: {node.name}, shape {node.tensor.shape}")
    
    print(f"\nΣ-MPO input nodes:")
    for i, node in enumerate(bmpo.sigma_mpo.input_nodes):
        print(f"  Input {i}: {node.name}, shape {node.tensor.shape}")
    
    if len(bmpo.sigma_mpo.input_nodes) == 0:
        print(f"\n  Note: Σ-MPO has no input nodes in manual construction")
        print(f"  This is expected - Σ inputs are set separately in forward pass")


def main():
    """Run all tests."""
    test_mu_input_nodes()
    test_sigma_input_nodes()
    test_manual_construction()
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\nKey points:")
    print("  1. Check if μ-MPO input nodes share memory")
    print("  2. Check if Σ-MPO has 2x input nodes (outer + inner)")
    print("  3. Check if Σ-MPO input pairs share memory")
    print("  4. Verify data propagation when setting inputs")
    print("=" * 70)


if __name__ == "__main__":
    main()
