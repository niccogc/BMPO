"""
Example demonstrating Σ-MPO initialization and updates.

The Σ-MPO represents E_q[W ⊗ W^T] and must be initialized such that
the covariance Cov = Σ - μ ⊗ μ^T is positive definite.

This example shows:
1. How Σ-MPO is initialized at creation
2. Verification that covariance is positive definite
3. How Σ-MPO should be updated during variational inference
4. The relationship between Σ, μ, and the covariance
"""

import torch
from tensor.node import TensorNode
from tensor.bayesian_mpo import BayesianMPO


def example_initialization():
    """Show how Σ-MPO is initialized."""
    print("=" * 70)
    print("1. Σ-MPO Initialization")
    print("=" * 70)
    
    # Create a simple BMPO
    mu_node = TensorNode(
        tensor_or_shape=(2, 3, 2),
        dim_labels=['r0', 'f', 'r1'],
        l='r0',
        r='r1',
        name='mu1'
    )
    
    bmpo = BayesianMPO(
        mu_nodes=[mu_node],
        rank_labels={'r0', 'r1'},
        tau_alpha=torch.tensor(2.0),
        tau_beta=torch.tensor(1.0)
    )
    
    print("\nAt initialization:")
    print(f"  μ-node shape: {mu_node.shape}")
    print(f"  Σ-node shape: {bmpo.sigma_nodes[0].shape}")
    print(f"  Block dimension d = {mu_node.tensor.numel()}")
    
    print("\nInitialization strategy:")
    print("  Σ is initialized such that:")
    print("    E_q[W ⊗ W^T] = μ ⊗ μ^T + σ²I")
    print(f"  where σ² = 1.0 (initial variance)")
    print()
    print("  This ensures:")
    print("    Cov = E_q[W ⊗ W^T] - E_q[W] ⊗ E_q[W]^T")
    print("        = (μ ⊗ μ^T + σ²I) - μ ⊗ μ^T")
    print("        = σ²I")
    print("  which is positive definite!")
    
    # Verify
    q_block = bmpo.get_block_q_distribution(0)
    cov = q_block.covariance_matrix
    
    print(f"\nVerification:")
    print(f"  Covariance diagonal: all equal to 1.0")
    print(f"    {torch.diag(cov)[:6]}")
    print(f"  Off-diagonal: all zero")
    print(f"    Max off-diag: {(cov - torch.diag(torch.diag(cov))).abs().max().item():.6e}")
    
    eigvals = torch.linalg.eigvalsh(cov)
    print(f"  Eigenvalues: all equal to 1.0")
    print(f"    Min: {eigvals[0].item():.6f}")
    print(f"    Max: {eigvals[-1].item():.6f}")


def example_covariance_structure():
    """Show the relationship between Σ, μ, and covariance."""
    print("\n" + "=" * 70)
    print("2. Relationship: Σ, μ, and Covariance")
    print("=" * 70)
    
    # Create BMPO with specific μ values
    mu_node = TensorNode(
        tensor_or_shape=(2, 3),
        dim_labels=['r0', 'f'],
        name='mu1'
    )
    
    # Set specific values for μ
    mu_node.tensor = torch.tensor([
        [0.5, 0.3, 0.2],
        [0.4, 0.1, 0.6]
    ])
    
    bmpo = BayesianMPO(mu_nodes=[mu_node], rank_labels={'r0'})
    
    mu_flat = mu_node.tensor.flatten()
    d = mu_flat.numel()
    
    print(f"\nμ-block (flattened):")
    print(f"  {mu_flat}")
    print(f"  Dimension: {d}")
    
    # Get sigma matrix
    sigma_node = bmpo.sigma_nodes[0]
    n_dims = len(mu_node.shape)
    outer_indices = list(range(0, 2*n_dims, 2))
    inner_indices = list(range(1, 2*n_dims, 2))
    perm = outer_indices + inner_indices
    sigma_permuted = sigma_node.tensor.permute(*perm)
    sigma_matrix = sigma_permuted.reshape(d, d)
    
    print(f"\nΣ matrix (E_q[W ⊗ W^T]):")
    print(f"  Shape: {sigma_matrix.shape}")
    print(f"  Diagonal (first 6): {torch.diag(sigma_matrix)}")
    
    # Compute outer product
    mu_outer = torch.outer(mu_flat, mu_flat)
    print(f"\nμ ⊗ μ^T:")
    print(f"  Diagonal (first 6): {torch.diag(mu_outer)}")
    
    # Compute covariance
    cov = sigma_matrix - mu_outer
    print(f"\nCovariance = Σ - μ ⊗ μ^T:")
    print(f"  Diagonal (first 6): {torch.diag(cov)}")
    print(f"  All equal to 1.0: {torch.allclose(torch.diag(cov), torch.ones(d))}")


def example_no_regularization():
    """Show that no regularization is applied during get_block_q_distribution."""
    print("\n" + "=" * 70)
    print("3. No Regularization During Query")
    print("=" * 70)
    
    print("\nImportant: get_block_q_distribution() does NOT add regularization!")
    print()
    print("  - Σ-MPO is initialized at creation to be positive definite")
    print("  - During variational inference, Σ is updated by the algorithm")
    print("  - get_block_q_distribution() simply computes Cov = Σ - μ ⊗ μ^T")
    print("  - No artificial regularization is added")
    print()
    print("This means:")
    print("  ✓ The covariance reflects the true variational distribution")
    print("  ✓ Updates can modify the covariance structure")
    print("  ✓ The algorithm is responsible for maintaining positive definiteness")
    
    # Create BMPO
    mu_node = TensorNode((2, 2), dim_labels=['r0', 'f'], name='mu1')
    bmpo = BayesianMPO(mu_nodes=[mu_node], rank_labels={'r0'})
    
    # Get q-distribution multiple times - should be identical
    q1 = bmpo.get_block_q_distribution(0)
    q2 = bmpo.get_block_q_distribution(0)
    
    cov1 = q1.covariance_matrix
    cov2 = q2.covariance_matrix
    
    print("\nVerification (query twice, should be identical):")
    print(f"  Covariance 1: {torch.diag(cov1)}")
    print(f"  Covariance 2: {torch.diag(cov2)}")
    print(f"  Identical: {torch.allclose(cov1, cov2)}")


def example_sigma_updates():
    """Show how Σ-MPO should be updated during variational inference."""
    print("\n" + "=" * 70)
    print("4. Updating Σ-MPO During Variational Inference")
    print("=" * 70)
    
    print("\nDuring variational inference (e.g., variational Bayes EM):")
    print()
    print("The Σ-MPO should be updated to reflect:")
    print("  E_q[W ⊗ W^T] = updated second moment")
    print()
    print("Example update schemes:")
    print()
    print("1. Closed-form update (if available):")
    print("   Σ_new = computed from E-step")
    print()
    print("2. Gradient-based update:")
    print("   Σ_new = Σ_old + α * ∇_Σ ELBO")
    print()
    print("3. Natural gradient update:")
    print("   Σ_new = updated via natural gradient")
    print()
    
    # Create BMPO
    mu_node = TensorNode((2, 2), dim_labels=['r0', 'f'], name='mu1')
    bmpo = BayesianMPO(mu_nodes=[mu_node], rank_labels={'r0'})
    
    print("Example: Manually updating Σ-node")
    print()
    
    # Initial covariance
    q_initial = bmpo.get_block_q_distribution(0)
    cov_initial = q_initial.covariance_matrix
    print(f"Initial covariance diagonal: {torch.diag(cov_initial)}")
    
    # Simulate an update: increase variance in first dimension
    mu_flat = mu_node.tensor.flatten()
    d = mu_flat.numel()
    
    # Create new Σ = μ ⊗ μ^T + updated_variance
    new_variance = torch.tensor([2.0, 1.5, 1.0, 0.5])  # Different variances
    mu_outer = torch.outer(mu_flat, mu_flat)
    new_sigma = mu_outer + torch.diag(new_variance)
    
    # Update Σ-node (reshape back to doubled structure)
    shape_half = mu_node.shape
    n_dims = len(shape_half)
    expanded_shape = list(shape_half) + list(shape_half)
    sigma_expanded = new_sigma.reshape(*expanded_shape)
    perm = []
    for i in range(n_dims):
        perm.append(i)
        perm.append(i + n_dims)
    sigma_permuted = sigma_expanded.permute(*perm)
    bmpo.sigma_nodes[0].tensor = sigma_permuted
    
    # New covariance
    q_updated = bmpo.get_block_q_distribution(0)
    cov_updated = q_updated.covariance_matrix
    print(f"Updated covariance diagonal: {torch.diag(cov_updated)}")
    print(f"\n✓ Covariance reflects the update (no regularization added)")


def example_positive_definiteness_responsibility():
    """Explain who is responsible for maintaining positive definiteness."""
    print("\n" + "=" * 70)
    print("5. Responsibility for Positive Definiteness")
    print("=" * 70)
    
    print("\nResponsibilities:")
    print()
    print("1. Initialization (BayesianMPO.__init__):")
    print("   ✓ Sets Σ = μ ⊗ μ^T + σ²I")
    print("   ✓ Guarantees positive definite at creation")
    print()
    print("2. Updates (User/Algorithm):")
    print("   ✓ Must ensure updated Σ maintains positive definiteness")
    print("   ✓ Common approaches:")
    print("     - Use natural gradient (automatically maintains)")
    print("     - Project onto positive definite cone")
    print("     - Use parameterization (e.g., Cholesky factor)")
    print()
    print("3. Queries (get_block_q_distribution):")
    print("   ✓ Simply computes Cov = Σ - μ ⊗ μ^T")
    print("   ✓ No regularization or modification")
    print("   ✓ Returns what the algorithm set")
    print()
    print("This design allows:")
    print("  - Clean separation of concerns")
    print("  - Algorithm has full control over updates")
    print("  - No hidden modifications to the variational parameters")


def main():
    """Run all examples."""
    torch.manual_seed(42)
    
    example_initialization()
    example_covariance_structure()
    example_no_regularization()
    example_sigma_updates()
    example_positive_definiteness_responsibility()
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\nΣ-MPO Initialization and Updates:")
    print()
    print("✓ Initialization: Σ = μ ⊗ μ^T + σ²I (σ² = 1.0)")
    print("  - Ensures Cov = σ²I (positive definite)")
    print()
    print("✓ No Regularization: get_block_q_distribution() computes Cov = Σ - μ ⊗ μ^T")
    print("  - Returns true variational covariance")
    print("  - No artificial modifications")
    print()
    print("✓ Updates: Algorithm is responsible")
    print("  - Update Σ-node tensor directly")
    print("  - Must maintain positive definiteness")
    print("  - Common: natural gradient, projection, parameterization")
    print()
    print("✓ Design: Clean separation of concerns")
    print("  - Initialization: ensures valid starting point")
    print("  - Updates: algorithm has full control")
    print("  - Queries: return unmodified values")
    print("=" * 70)


if __name__ == "__main__":
    main()
