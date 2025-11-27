"""
Example demonstrating the compute_theta_tensor method.

Theta tensor: Θ_{ijk...} = E_q[bond1]_i × E_q[bond2]_j × E_q[bond3]_k × ...

This is the structured prior/regularization tensor computed from the 
expectations of Gamma distributions on each bond.
"""

import torch
from tensor.bayesian_mpo_builder import create_bayesian_tensor_train


def main():
    print("=" * 70)
    print("Theta Tensor Computation Example")
    print("=" * 70)
    
    bmpo = create_bayesian_tensor_train(
        num_blocks=3,
        bond_dim=4,
        input_features=5,
        output_shape=1,
        seed=42
    )
    
    # Example 1: Compute full Theta tensor for each block
    print("\n1. Full Theta Tensors")
    print("-" * 70)
    
    for block_idx, node in enumerate(bmpo.mu_nodes):
        theta = bmpo.compute_theta_tensor(block_idx)
        print(f"\nBlock {block_idx}:")
        print(f"  Labels: {node.dim_labels}")
        print(f"  Shape: {node.shape}")
        print(f"  Theta shape: {theta.shape} (matches block)")
        print(f"  Theta = outer product of bond expectations")
    
    # Example 2: Exclude specific bonds
    print("\n\n2. Excluding Specific Bonds")
    print("-" * 70)
    
    block_idx = 1
    node = bmpo.mu_nodes[block_idx]
    print(f"\nBlock {block_idx}: {node.dim_labels}, shape {node.shape}")
    
    # Full
    theta_full = bmpo.compute_theta_tensor(block_idx)
    print(f"\n  Full Theta: shape {theta_full.shape}")
    
    # Exclude p2 (physical bond)
    theta_no_p2 = bmpo.compute_theta_tensor(block_idx, exclude_labels=['p2'])
    print(f"  Exclude 'p2': shape {theta_no_p2.shape}")
    print(f"    → p2 dimension set to 1: {node.shape} → {theta_no_p2.shape}")
    
    # Exclude r1 (horizontal bond)
    theta_no_r1 = bmpo.compute_theta_tensor(block_idx, exclude_labels=['r1'])
    print(f"  Exclude 'r1': shape {theta_no_r1.shape}")
    print(f"    → r1 dimension set to 1: {node.shape} → {theta_no_r1.shape}")
    
    # Example 3: Use case - different regularization per bond type
    print("\n\n3. Use Case: Selective Regularization")
    print("-" * 70)
    print("\nYou can compute Theta tensors with different bond contributions:")
    print("  - Full Theta: all bonds contribute")
    print("  - Theta without physical bonds: exclude p1, p2, p3")
    print("  - Theta with only rank bonds: exclude all non-rank bonds")
    
    block_idx = 1
    node = bmpo.mu_nodes[block_idx]
    
    # Only rank bonds
    physical_bonds = [label for label in node.dim_labels if label.startswith('p')]
    theta_rank_only = bmpo.compute_theta_tensor(block_idx, exclude_labels=physical_bonds)
    
    print(f"\nBlock {block_idx}:")
    print(f"  All labels: {node.dim_labels}")
    print(f"  Physical bonds to exclude: {physical_bonds}")
    print(f"  Theta (rank only): shape {theta_rank_only.shape}")
    print(f"  Original shape: {node.shape}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
