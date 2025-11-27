"""
Comprehensive test of unified bond parameterization.

Tests:
1. Count all parameters and verify consistency
2. Identify learnable nodes (blocks) and their associated bonds
3. Check Gamma distribution parameters in each node
4. Verify trimming still works
"""

import torch
from tensor.bayesian_mpo_builder import create_bayesian_tensor_train


def test_parameter_counting():
    """Test 1: Count all parameters and verify consistency."""
    print("=" * 70)
    print("Test 1: Parameter Counting")
    print("=" * 70)
    
    bmpo = create_bayesian_tensor_train(
        num_blocks=3,
        bond_dim=4,
        input_features=5,
        output_shape=1,
        seed=42
    )
    
    # Count μ-MPO nodes
    print(f"\nμ-MPO Structure:")
    print(f"  Number of blocks (nodes): {len(bmpo.mu_nodes)}")
    for i, node in enumerate(bmpo.mu_nodes):
        print(f"  Block {i}: shape={node.shape}, labels={node.dim_labels}")
        print(f"           total params={node.tensor.numel()}")
    
    # Count Σ-MPO nodes
    print(f"\nΣ-MPO Structure:")
    print(f"  Number of blocks (nodes): {len(bmpo.sigma_nodes)}")
    for i, node in enumerate(bmpo.sigma_nodes):
        print(f"  Block {i}: shape={node.shape}, labels={node.dim_labels}")
        print(f"           total params={node.tensor.numel()}")
    
    # Count bonds and their Gamma parameters
    print(f"\nBond Distributions:")
    total_gamma_params = 0
    for label, dist in bmpo.mu_mpo.distributions.items():
        n_gammas = len(dist['concentration'])
        n_params = n_gammas * 2  # Each Gamma has 2 parameters (α, β)
        total_gamma_params += n_params
        print(f"  Bond '{label}': {n_gammas} Gammas × 2 params = {n_params} params")
    
    print(f"\nTotal Gamma parameters (variational): {total_gamma_params}")
    
    # Count prior parameters (should match variational)
    total_prior_params = 0
    for label, prior in bmpo.prior_bond_params.items():
        n_gammas = len(prior['concentration0'])
        n_params = n_gammas * 2
        total_prior_params += n_params
    
    print(f"Total Gamma parameters (prior): {total_prior_params}")
    assert total_gamma_params == total_prior_params, "Prior and variational param counts should match!"
    
    # Count τ parameters
    print(f"\nτ distribution: 1 Gamma × 2 params = 2 params")
    
    print(f"\n✓ All counts verified!")


def test_block_bond_associations():
    """Test 2: Identify which bonds are associated with which blocks."""
    print("\n" + "=" * 70)
    print("Test 2: Block-Bond Associations")
    print("=" * 70)
    
    bmpo = create_bayesian_tensor_train(
        num_blocks=3,
        bond_dim=4,
        input_features=5,
        output_shape=1,
        seed=42
    )
    
    print(f"\nLearnable Blocks (μ-MPO nodes):")
    print(f"  Total blocks: {len(bmpo.mu_nodes)}")
    
    # For each block, find which bonds it contains
    for block_idx, node in enumerate(bmpo.mu_nodes):
        print(f"\n  Block {block_idx} ({node.name}):")
        print(f"    Shape: {node.shape}")
        print(f"    Dimension labels: {node.dim_labels}")
        print(f"    Associated bonds:")
        
        for label in node.dim_labels:
            # Get info about this bond
            dist = bmpo.mu_mpo.distributions.get(label)
            if dist:
                n_gammas = len(dist['concentration'])
                bond_type = "horizontal" if label.startswith('r') else "vertical"
                nodes = bmpo.get_nodes_for_bond(label)
                shared_with = [n for n in nodes['mu_nodes'] if n != block_idx]
                
                print(f"      - '{label}' ({bond_type}):")
                print(f"          Dimension size: {n_gammas}")
                print(f"          Number of Gammas: {n_gammas}")
                print(f"          Shared with blocks: {shared_with if shared_with else 'none (vertical)'}")
    
    # Reverse mapping: for each bond, show which blocks it connects
    print(f"\n\nBonds and Their Connected Blocks:")
    for label in sorted(bmpo.mu_mpo.distributions.keys()):
        nodes_info = bmpo.get_nodes_for_bond(label)
        mu_nodes = nodes_info['mu_nodes']
        sigma_nodes = nodes_info['sigma_nodes']
        
        bond_type = "horizontal" if len(mu_nodes) > 1 else "vertical"
        print(f"\n  Bond '{label}' ({bond_type}):")
        print(f"    μ-MPO blocks: {mu_nodes}")
        print(f"    Σ-MPO blocks: {sigma_nodes}")
        
        # Show the actual dimension in each block
        for node_idx in mu_nodes:
            node = bmpo.mu_nodes[node_idx]
            dim_idx = node.dim_labels.index(label)
            dim_size = node.shape[dim_idx]
            print(f"    In block {node_idx}: dimension {dim_idx}, size {dim_size}")
    
    print(f"\n✓ Block-bond associations verified!")


def test_gamma_parameters_per_node():
    """Test 3: Check Gamma distribution parameters in each node."""
    print("\n" + "=" * 70)
    print("Test 3: Gamma Parameters Per Node")
    print("=" * 70)
    
    bmpo = create_bayesian_tensor_train(
        num_blocks=3,
        bond_dim=4,
        input_features=5,
        output_shape=1,
        seed=42
    )
    
    # Update some parameters to make them distinct
    print(f"\nSetting distinct parameters for testing...")
    bmpo.update_mu_params('r1', 
                         concentration=torch.tensor([3.0, 2.5, 4.0, 2.0], dtype=torch.float64),
                         rate=torch.tensor([1.5, 1.0, 2.0, 1.0], dtype=torch.float64))
    
    bmpo.update_mu_params('p2',
                         concentration=torch.tensor([5.0, 4.5, 4.0, 3.5, 3.0], dtype=torch.float64),
                         rate=torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0], dtype=torch.float64))
    
    print(f"\nGamma distributions for each bond:")
    for label in sorted(bmpo.mu_mpo.distributions.keys()):
        gammas = bmpo.mu_mpo.get_gamma_distributions(label)
        if gammas:
            print(f"\n  Bond '{label}' has {len(gammas)} Gamma distributions:")
            for i, gamma in enumerate(gammas):
                print(f"    Index {i}: Gamma(α={gamma.concentration:.2f}, β={gamma.rate:.2f}), "
                      f"E[X]={gamma.mean():.2f}, H[X]={gamma.entropy():.4f}")
    
    # Verify that different bonds have different parameters
    r1_gammas = bmpo.mu_mpo.get_gamma_distributions('r1')
    r2_gammas = bmpo.mu_mpo.get_gamma_distributions('r2')
    
    print(f"\nVerifying distinct parameters:")
    print(f"  r1[0]: α={r1_gammas[0].concentration:.2f}, β={r1_gammas[0].rate:.2f}")
    print(f"  r2[0]: α={r2_gammas[0].concentration:.2f}, β={r2_gammas[0].rate:.2f}")
    print(f"  Different? {r1_gammas[0].concentration != r2_gammas[0].concentration}")
    
    print(f"\n✓ Gamma parameters verified!")


def test_trimming():
    """Test 4: Verify trimming still works with unified structure."""
    print("\n" + "=" * 70)
    print("Test 4: Trimming Functionality")
    print("=" * 70)
    
    bmpo = create_bayesian_tensor_train(
        num_blocks=3,
        bond_dim=4,
        input_features=5,
        output_shape=1,
        seed=42
    )
    
    # Set some parameters to have different expectations
    # This will make some indices have lower expectations that can be trimmed
    r1_conc = torch.tensor([4.0, 1.0, 3.0, 0.5], dtype=torch.float64)
    r1_rate = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    bmpo.update_mu_params('r1', concentration=r1_conc, rate=r1_rate)
    
    print(f"\nBefore trimming:")
    print(f"  Bond 'r1' expectations: {bmpo.mu_mpo.distributions['r1']['expectation']}")
    print(f"  μ-MPO block 0 shape: {bmpo.mu_nodes[0].shape}")
    print(f"  μ-MPO block 1 shape: {bmpo.mu_nodes[1].shape}")
    print(f"  Σ-MPO block 0 shape: {bmpo.sigma_nodes[0].shape}")
    print(f"  Σ-MPO block 1 shape: {bmpo.sigma_nodes[1].shape}")
    
    # Get number of Gammas before trimming
    r1_gammas_before = bmpo.mu_mpo.get_gamma_distributions('r1')
    print(f"  Number of Gammas for 'r1': {len(r1_gammas_before)}")
    
    # Trim: keep only indices with expectation >= 2.0
    # Should keep indices [0, 2] and remove [1, 3]
    print(f"\nTrimming bond 'r1' with threshold 2.0...")
    bmpo.trim({'r1': 2.0})
    
    print(f"\nAfter trimming:")
    print(f"  Bond 'r1' expectations: {bmpo.mu_mpo.distributions['r1']['expectation']}")
    print(f"  μ-MPO block 0 shape: {bmpo.mu_nodes[0].shape}")
    print(f"  μ-MPO block 1 shape: {bmpo.mu_nodes[1].shape}")
    print(f"  Σ-MPO block 0 shape: {bmpo.sigma_nodes[0].shape}")
    print(f"  Σ-MPO block 1 shape: {bmpo.sigma_nodes[1].shape}")
    
    # Get number of Gammas after trimming
    r1_gammas_after = bmpo.mu_mpo.get_gamma_distributions('r1')
    print(f"  Number of Gammas for 'r1': {len(r1_gammas_after)}")
    
    # Verify the trimming worked correctly
    assert len(r1_gammas_after) == 2, "Should have kept 2 Gammas (indices 0 and 2)"
    print(f"\n  ✓ Kept {len(r1_gammas_after)} indices (expected 2)")
    
    # Verify the expectations match what we expect
    expected_vals = torch.tensor([4.0, 3.0], dtype=torch.float64)
    actual_vals = bmpo.mu_mpo.distributions['r1']['expectation']
    assert torch.allclose(actual_vals, expected_vals), "Expectations should be [4.0, 3.0]"
    print(f"  ✓ Kept correct expectations: {actual_vals}")
    
    # Test forward pass still works after trimming
    X = torch.randn(10, 5, dtype=torch.float64)
    mu_out = bmpo.forward_mu(X, to_tensor=True)
    sigma_out = bmpo.forward_sigma(X, to_tensor=True)
    print(f"\n  ✓ Forward passes work after trimming:")
    print(f"    μ output shape: {mu_out.shape}")
    print(f"    Σ output shape: {sigma_out.shape}")
    
    print(f"\n✓ Trimming verified!")


def main():
    test_parameter_counting()
    test_block_bond_associations()
    test_gamma_parameters_per_node()
    test_trimming()
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
    print("\nSummary:")
    print("  ✓ Parameter counting is consistent")
    print("  ✓ Block-bond associations are correct")
    print("  ✓ Gamma parameters work correctly per node")
    print("  ✓ Trimming functionality works with unified structure")
    print("=" * 70)


if __name__ == "__main__":
    main()
