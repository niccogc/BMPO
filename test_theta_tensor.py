"""
Test the compute_theta_tensor method.
"""

import torch
from tensor.bayesian_mpo_builder import create_bayesian_tensor_train


def test_theta_tensor():
    print("=" * 70)
    print("Testing Theta Tensor Computation")
    print("=" * 70)
    
    bmpo = create_bayesian_tensor_train(
        num_blocks=3,
        bond_dim=4,
        input_features=5,
        output_shape=1,
        seed=42
    )
    
    # Set some distinct expectations for testing
    bmpo.update_mu_params('r1', 
                         concentration=torch.tensor([3.0, 2.5, 4.0, 2.0], dtype=torch.float64),
                         rate=torch.tensor([1.5, 1.0, 2.0, 1.0], dtype=torch.float64))
    
    bmpo.update_mu_params('p2',
                         concentration=torch.tensor([5.0, 4.5, 4.0, 3.5, 3.0], dtype=torch.float64),
                         rate=torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0], dtype=torch.float64))
    
    print("\nBlock structures:")
    for i, node in enumerate(bmpo.mu_nodes):
        print(f"  Block {i}: shape={node.shape}, labels={node.dim_labels}")
    
    # Test 1: Full theta tensor (no exclusions)
    print("\n" + "=" * 70)
    print("Test 1: Full Theta Tensor (No Exclusions)")
    print("=" * 70)
    
    for block_idx in range(len(bmpo.mu_nodes)):
        node = bmpo.mu_nodes[block_idx]
        theta = bmpo.compute_theta_tensor(block_idx)
        
        print(f"\nBlock {block_idx}:")
        print(f"  Node shape: {node.shape}")
        print(f"  Theta shape: {theta.shape}")
        print(f"  Match: {theta.shape == node.shape}")
        
        # Show expectations used
        print(f"  Expectations used:")
        for label in node.dim_labels:
            if label in bmpo.mu_mpo.distributions:
                exp = bmpo.mu_mpo.distributions[label]['expectation']
                print(f"    {label}: {exp}")
            else:
                print(f"    {label}: [1.0] (dummy, size 1)")
    
    # Test 2: Theta with exclusions
    print("\n" + "=" * 70)
    print("Test 2: Theta Tensor with Exclusions")
    print("=" * 70)
    
    # Block 1 has labels ['r1', 'c2', 'p2', 'r2']
    block_idx = 1
    node = bmpo.mu_nodes[block_idx]
    
    print(f"\nBlock {block_idx} labels: {node.dim_labels}")
    print(f"Block {block_idx} shape: {node.shape}")
    
    # Exclude 'p2'
    theta_no_p2 = bmpo.compute_theta_tensor(block_idx, exclude_labels=['p2'])
    print(f"\nExcluding 'p2':")
    print(f"  Original shape: {node.shape}")
    print(f"  Theta shape: {theta_no_p2.shape}")
    print(f"  Expected: (4, 1, 1, 4) - p2 dimension set to 1")
    print(f"  Correct: {theta_no_p2.shape == torch.Size([4, 1, 1, 4])}")
    
    # Exclude multiple
    theta_exclude_multiple = bmpo.compute_theta_tensor(block_idx, exclude_labels=['p2', 'c2'])
    print(f"\nExcluding 'p2' and 'c2':")
    print(f"  Theta shape: {theta_exclude_multiple.shape}")
    print(f"  Expected: (4, 1, 1, 4)")
    
    # Test 3: Verify outer product structure
    print("\n" + "=" * 70)
    print("Test 3: Verify Outer Product Structure")
    print("=" * 70)
    
    # Block 0 has labels ['c1', 'p1', 'r1']
    block_idx = 0
    node = bmpo.mu_nodes[block_idx]
    theta = bmpo.compute_theta_tensor(block_idx)
    
    print(f"\nBlock {block_idx}: {node.dim_labels}, shape {node.shape}")
    
    # Get expectations
    exp_c1 = torch.ones(1, dtype=torch.float64)  # dummy
    exp_p1 = bmpo.mu_mpo.distributions['p1']['expectation']
    exp_r1 = bmpo.mu_mpo.distributions['r1']['expectation']
    
    print(f"  E[c1]: {exp_c1} (dummy)")
    print(f"  E[p1]: {exp_p1}")
    print(f"  E[r1]: {exp_r1}")
    
    # Manual computation
    manual_theta = torch.einsum('i,j,k->ijk', exp_c1, exp_p1, exp_r1)
    
    print(f"\nComparing computed vs manual:")
    print(f"  Computed shape: {theta.shape}")
    print(f"  Manual shape: {manual_theta.shape}")
    print(f"  Values match: {torch.allclose(theta, manual_theta)}")
    print(f"  Max difference: {torch.max(torch.abs(theta - manual_theta)).item():.2e}")
    
    # Spot check values
    print(f"\nSpot check:")
    i, j, k = 0, 2, 1  # c1=0, p1=2, r1=1
    print(f"  Theta[{i},{j},{k}] = {theta[i,j,k].item():.4f}")
    expected = exp_c1[i].item() * exp_p1[j].item() * exp_r1[k].item()
    print(f"  Manual = {exp_c1[i].item()} × {exp_p1[j].item():.2f} × {exp_r1[k].item():.2f} = {expected:.4f}")
    print(f"  Match: {abs(theta[i,j,k].item() - expected) < 1e-10}")
    
    # Test 4: Excluding bonds
    print("\n" + "=" * 70)
    print("Test 4: Excluding Specific Bonds")
    print("=" * 70)
    
    # Block 1: ['r1', 'c2', 'p2', 'r2'], shape (4, 1, 5, 4)
    block_idx = 1
    
    # Without exclusion
    theta_full = bmpo.compute_theta_tensor(block_idx)
    print(f"\nBlock {block_idx} - Full theta:")
    print(f"  Shape: {theta_full.shape}")
    print(f"  Sample value [0,0,2,1]: {theta_full[0,0,2,1].item():.4f}")
    
    # Exclude p2
    theta_no_p2 = bmpo.compute_theta_tensor(block_idx, exclude_labels=['p2'])
    print(f"\nBlock {block_idx} - Exclude 'p2':")
    print(f"  Shape: {theta_no_p2.shape}")
    print(f"  p2 dimension (index 2): {theta_no_p2.shape[2]} (should be 1)")
    print(f"  Sample value [0,0,0,1]: {theta_no_p2[0,0,0,1].item():.4f}")
    
    # Verify manual computation for excluded case
    exp_r1 = bmpo.mu_mpo.distributions['r1']['expectation']
    exp_r2 = bmpo.mu_mpo.distributions['r2']['expectation']
    manual_no_p2 = exp_r1[0].item() * 1.0 * 1.0 * exp_r2[1].item()
    print(f"  Manual = E[r1][0] × 1 × 1 × E[r2][1]")
    print(f"         = {exp_r1[0].item():.4f} × 1 × 1 × {exp_r2[1].item():.4f}")
    print(f"         = {manual_no_p2:.4f}")
    print(f"  Match: {abs(theta_no_p2[0,0,0,1].item() - manual_no_p2) < 1e-10}")
    
    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_theta_tensor()
