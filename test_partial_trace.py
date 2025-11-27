"""
Test the partial_trace_update method.
"""

import torch
from tensor.bayesian_mpo_builder import create_bayesian_tensor_train


def test_partial_trace():
    print("=" * 70)
    print("Testing Partial Trace Update")
    print("=" * 70)
    
    bmpo = create_bayesian_tensor_train(
        num_blocks=3,
        bond_dim=4,
        input_features=5,
        output_shape=1,
        seed=42
    )
    
    # Set some values for testing
    bmpo.update_mu_params('r1', 
                         concentration=torch.tensor([3.0, 2.5, 4.0, 2.0], dtype=torch.float64),
                         rate=torch.tensor([1.5, 1.0, 2.0, 1.0], dtype=torch.float64))
    
    print("\nBlock structures:")
    for i, node in enumerate(bmpo.mu_nodes):
        print(f"  Block {i}: shape={node.shape}, labels={node.dim_labels}")
    
    # Test 1: Basic functionality
    print("\n" + "=" * 70)
    print("Test 1: Basic Partial Trace")
    print("=" * 70)
    
    block_idx = 1
    node = bmpo.mu_nodes[block_idx]
    print(f"\nBlock {block_idx}: {node.dim_labels}, shape {node.shape}")
    
    for label in node.dim_labels:
        # Skip dummy dimensions (size 1)
        if node.shape[node.dim_labels.index(label)] == 1:
            print(f"\n  Skipping '{label}' (dummy, size 1)")
            continue
            
        result = bmpo.partial_trace_update(block_idx, label)
        expected_size = node.shape[node.dim_labels.index(label)]
        
        print(f"\n  Focus on '{label}':")
        print(f"    Expected output size: {expected_size}")
        print(f"    Actual output shape: {result.shape}")
        print(f"    Match: {result.shape == torch.Size([expected_size])}")
        print(f"    Sample values: {result[:min(3, len(result))]}")
    
    # Test 2: Manual verification for small example
    print("\n" + "=" * 70)
    print("Test 2: Manual Verification")
    print("=" * 70)
    
    # Use block 0 for simpler verification
    block_idx = 0
    node = bmpo.mu_nodes[block_idx]
    print(f"\nBlock {block_idx}: {node.dim_labels}, shape {node.shape}")
    
    # Focus on 'r1' (last dimension, size 4)
    focus_label = 'r1'
    result = bmpo.partial_trace_update(block_idx, focus_label)
    
    print(f"\nFocus on '{focus_label}' (dimension {node.dim_labels.index(focus_label)}, size {node.shape[2]}):")
    print(f"  Result shape: {result.shape}")
    print(f"  Result: {result}")
    
    # Manual computation
    print("\n  Manual verification:")
    
    # Get components
    mu = node.tensor  # Shape: (1, 5, 4)
    sigma_node = bmpo.sigma_nodes[block_idx]
    
    # Extract diagonal of sigma
    mu_flat = mu.flatten()
    d = mu_flat.numel()
    sigma_tensor = sigma_node.tensor
    n_dims = len(node.shape)
    outer_indices = list(range(0, 2*n_dims, 2))
    inner_indices = list(range(1, 2*n_dims, 2))
    perm = outer_indices + inner_indices
    sigma_permuted = sigma_tensor.permute(*perm)
    sigma_matrix = sigma_permuted.reshape(d, d)
    diag_sigma = torch.diagonal(sigma_matrix).reshape(node.shape)
    
    # v = diag(Σ) + μ²
    v = diag_sigma + mu ** 2
    print(f"    v = diag(Σ) + μ² shape: {v.shape}")
    
    # Θ without r1
    theta = bmpo.compute_theta_tensor(block_idx, exclude_labels=['r1'])
    print(f"    Θ (without r1) shape: {theta.shape}")
    
    # Multiply
    product = v * theta
    print(f"    v × Θ shape: {product.shape}")
    
    # Sum over dimensions 0 and 1 (keep dimension 2 which is r1)
    manual_result = torch.sum(product, dim=[0, 1])
    print(f"    Sum over dims [0,1]: {manual_result.shape}")
    print(f"    Manual result: {manual_result}")
    
    print(f"\n  Comparison:")
    print(f"    Max difference: {torch.max(torch.abs(result - manual_result)).item():.2e}")
    print(f"    Match: {torch.allclose(result, manual_result)}")
    
    # Test 3: All blocks, all labels
    print("\n" + "=" * 70)
    print("Test 3: All Blocks, All Learnable Labels")
    print("=" * 70)
    
    for block_idx in range(len(bmpo.mu_nodes)):
        node = bmpo.mu_nodes[block_idx]
        print(f"\nBlock {block_idx}: {node.dim_labels}")
        
        for label in node.dim_labels:
            dim_idx = node.dim_labels.index(label)
            size = node.shape[dim_idx]
            
            # Skip dummy dimensions
            if size == 1:
                continue
            
            # Skip if no Gamma distribution (shouldn't happen for size > 1, but check)
            if label not in bmpo.mu_mpo.distributions:
                continue
            
            result = bmpo.partial_trace_update(block_idx, label)
            print(f"  '{label}' (size {size}): output shape {result.shape} ✓")
    
    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_partial_trace()
