"""
Example of full Bayesian MPO structure.

Structure:
- μ-MPO: Mean tensor network
- Σ-MPO: Variation tensor network (doubled structure)
- Prior distributions on μ-MPO
- τ distribution
"""

import torch
from tensor.node import TensorNode
from tensor.bayesian_mpo import BayesianMPO


def main():
    print("=" * 70)
    print("Full Bayesian MPO Example")
    print("=" * 70)
    
    # Example 1: Create Bayesian MPO
    print("\n1. Creating Bayesian MPO")
    print("-" * 70)
    
    # Create μ-MPO nodes (standard MPO)
    mu_node1 = TensorNode(
        tensor_or_shape=(3, 4, 2),
        dim_labels=['r0', 'r1', 'f'],
        l='r0', r='r1',
        name='mu1'
    )
    
    mu_node2 = TensorNode(
        tensor_or_shape=(4, 3, 2),
        dim_labels=['r1', 'r2', 'f'],
        l='r1', r='r2',
        name='mu2'
    )
    
    # Connect μ-nodes
    mu_node1.connect(mu_node2, 'r1')
    mu_node1.connect(mu_node2, 'f')
    
    # Create Bayesian MPO
    bmpo = BayesianMPO(
        mu_nodes=[mu_node1, mu_node2],
        tau_alpha=torch.tensor(3.0),
        tau_beta=torch.tensor(1.5)
    )
    
    print("Bayesian MPO created!")
    bmpo.summary()
    
    # Example 2: μ-MPO structure
    print("\n\n2. μ-MPO Structure")
    print("-" * 70)
    
    print(f"μ-node 1: shape={mu_node1.shape}, labels={mu_node1.dim_labels}")
    print(f"μ-node 2: shape={mu_node2.shape}, labels={mu_node2.dim_labels}")
    
    # Example 3: Σ-MPO structure (doubled)
    print("\n3. Σ-MPO Structure (Doubled)")
    print("-" * 70)
    
    sigma_node1 = bmpo.sigma_nodes[0]
    sigma_node2 = bmpo.sigma_nodes[1]
    
    print(f"Σ-node 1: shape={sigma_node1.shape}, labels={sigma_node1.dim_labels}")
    print(f"  Original μ-node 1 was: {mu_node1.shape}")
    print(f"  Each dimension doubled: r0:(3)->r0o,r0i(3,3), r1:(4)->r1o,r1i(4,4), f:(2)->fo,fi(2,2)")
    
    print(f"\nΣ-node 2: shape={sigma_node2.shape}, labels={sigma_node2.dim_labels}")
    print(f"  Original μ-node 2 was: {mu_node2.shape}")
    
    # Example 4: τ distribution
    print("\n4. τ Distribution")
    print("-" * 70)
    
    print(f"τ ~ Gamma(α={bmpo.tau_alpha.item()}, β={bmpo.tau_beta.item()})")
    print(f"E[τ] = {bmpo.get_tau_mean():.4f}")
    print(f"H[τ] = {bmpo.get_tau_entropy():.4f}")
    
    # Update τ
    bmpo.update_tau(alpha=torch.tensor(5.0), beta=torch.tensor(2.0))
    print(f"\nAfter updating: τ ~ Gamma(α={bmpo.tau_alpha.item()}, β={bmpo.tau_beta.item()})")
    print(f"E[τ] = {bmpo.get_tau_mean():.4f}")
    
    # Example 5: Prior distributions on μ-MPO
    print("\n5. Prior Distributions on μ-MPO")
    print("-" * 70)
    
    print("Distributions for each μ-MPO dimension:")
    for label in ['r0', 'r1', 'r2', 'f']:
        expectations = bmpo.get_mu_expectations(label)
        if expectations is not None:
            dist_info = bmpo.mu_mpo.distributions[label]
            print(f"  {label} ({dist_info['type']}, {dist_info['variable']}): E={expectations}")
    
    # Update μ-MPO parameters
    print("\nUpdating r1 parameters:")
    new_c = torch.tensor([3.0, 1.0, 2.0, 4.0])
    new_e = torch.tensor([1.0, 2.0, 1.0, 1.0])
    bmpo.update_mu_params('r1', param1=new_c, param2=new_e)
    print(f"  New E[ω] for r1: {bmpo.get_mu_expectations('r1')}")
    
    # Example 6: Trimming
    print("\n6. Trimming Both μ-MPO and Σ-MPO")
    print("-" * 70)
    
    print("Before trimming:")
    print(f"  μ-node 1: {mu_node1.shape}")
    print(f"  μ-node 2: {mu_node2.shape}")
    print(f"  Σ-node 1: {sigma_node1.shape}")
    print(f"  Σ-node 2: {sigma_node2.shape}")
    print(f"  r1 expectations: {bmpo.get_mu_expectations('r1')}")
    
    # Trim based on μ-MPO expectations
    bmpo.trim({'r1': 2.0})
    
    print("\nAfter trimming (r1 >= 2.0):")
    print(f"  μ-node 1: {mu_node1.shape}")
    print(f"  μ-node 2: {mu_node2.shape}")
    print(f"  Σ-node 1: {sigma_node1.shape}")
    print(f"  Σ-node 2: {sigma_node2.shape}")
    print(f"  r1 expectations: {bmpo.get_mu_expectations('r1')}")
    print("\nNote: Both μ and Σ were trimmed consistently!")
    print("  r1 dimension in μ: kept 3 indices")
    print("  r1o and r1i dimensions in Σ: both kept 3 indices")
    
    print("\n" + "=" * 70)
    print("Bayesian MPO Example Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
