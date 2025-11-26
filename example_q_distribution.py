"""
Example demonstrating the q-distribution for Bayesian MPO.

The q-distribution is the variational posterior approximation:

q(θ) = q(τ) × ∏_i q(W_i) × ∏_modes ∏_indices q(mode_param)

where:
- q(τ) ~ Gamma(α, β) is the noise/precision distribution
- q(W_i) ~ N(μ_i, Σ_i - μ_i ⊗ μ_i^T) for each block i (Multivariate Normal)
- q(mode_param) ~ Gamma for each mode index (rank ω or vertical φ)

This example shows:
1. Structure of the q-distribution
2. Individual component distributions
3. Sampling from q
4. Computing log q(θ)
5. Use in ELBO computation
"""

import torch
from tensor.node import TensorNode
from tensor.bayesian_mpo import BayesianMPO


def create_simple_bmpo():
    """Create a simple Bayesian MPO for demonstration."""
    # Create μ-MPO nodes
    mu_node1 = TensorNode(
        tensor_or_shape=(2, 3, 3),
        dim_labels=['r0', 'f', 'r1'],
        l='r0',
        r='r1',
        name='mu1'
    )
    
    mu_node2 = TensorNode(
        tensor_or_shape=(3, 3, 2),
        dim_labels=['r1', 'f', 'r2'],
        l='r1',
        r='r2',
        name='mu2'
    )
    
    mu_node1.connect(mu_node2, 'r1', priority=1)
    
    # Create input nodes
    input1 = TensorNode(
        tensor_or_shape=(1, 3),
        dim_labels=['s', 'f'],
        name='X1'
    )
    
    input2 = TensorNode(
        tensor_or_shape=(1, 3),
        dim_labels=['s', 'f'],
        name='X2'
    )
    
    input1.connect(mu_node1, 'f')
    input2.connect(mu_node2, 'f')
    
    # Create BayesianMPO
    bmpo = BayesianMPO(
        mu_nodes=[mu_node1, mu_node2],
        input_nodes=[input1, input2],
        rank_labels={'r0', 'r1', 'r2'},
        tau_alpha=torch.tensor(3.0),
        tau_beta=torch.tensor(1.5)
    )
    
    return bmpo


def example_q_structure():
    """Show the structure of the q-distribution."""
    print("=" * 70)
    print("1. Q-Distribution Structure")
    print("=" * 70)
    
    bmpo = create_simple_bmpo()
    
    print("\nThe q-distribution factorizes as:")
    print("  q(θ) = q(τ) × q(W₁) × q(W₂) × q(ω_modes) × q(φ_modes)")
    print()
    
    print("Components:")
    print(f"  1. q(τ) ~ Gamma(α={bmpo.tau_alpha.item()}, β={bmpo.tau_beta.item()})")
    print(f"     - Noise/precision parameter")
    print()
    
    print(f"  2. Block distributions (Multivariate Normal):")
    for i, node in enumerate(bmpo.mu_nodes):
        d = node.tensor.numel()
        print(f"     - q(W_{i+1}) ~ N(μ_{i+1}, Σ_{i+1}) where:")
        print(f"       • μ_{i+1} = E_q[W_{i+1}] (shape {node.shape} flattened to {d})")
        print(f"       • Σ_{i+1} = E_q[W_{i+1} ⊗ W_{i+1}ᵀ] - μ_{i+1} ⊗ μ_{i+1}ᵀ ({d}×{d})")
    print()
    
    print(f"  3. Mode distributions (Gamma):")
    mode_dists = bmpo.get_mode_q_distributions()
    for label, gamma_list in mode_dists.items():
        dist_info = bmpo.mu_mpo.distributions[label]
        var_name = dist_info['variable']
        dist_type = dist_info['type']
        print(f"     - Mode '{label}' ({dist_type}, {var_name}): {len(gamma_list)} Gamma distributions")


def example_block_distributions():
    """Demonstrate block q-distributions in detail."""
    print("\n" + "=" * 70)
    print("2. Block Q-Distributions (Multivariate Normal)")
    print("=" * 70)
    
    bmpo = create_simple_bmpo()
    
    for i in range(len(bmpo.mu_nodes)):
        print(f"\nBlock {i+1}:")
        print("-" * 40)
        
        mu_node = bmpo.mu_nodes[i]
        sigma_node = bmpo.sigma_nodes[i]
        
        print(f"  μ-node shape: {mu_node.shape}")
        print(f"  Σ-node shape: {sigma_node.shape}")
        
        # Get q-distribution
        q_block = bmpo.get_block_q_distribution(i)
        
        # Show statistics
        mu_flat = mu_node.tensor.flatten()
        d = mu_flat.numel()
        
        print(f"\n  q(W_{i+1}) ~ N(μ, Σ):")
        print(f"    Dimension: {d}")
        print(f"    Mean (μ): shape {mu_flat.shape}")
        print(f"      First 5 values: {mu_flat[:5]}")
        print(f"    Covariance (Σ): shape ({d}, {d})")
        
        # Get covariance matrix
        cov = q_block.covariance_matrix
        if cov is not None:
            print(f"      Diagonal (first 5): {torch.diag(cov)[:5]}")
            print(f"      Min eigenvalue: {torch.linalg.eigvalsh(cov)[0].item():.6f}")
            print(f"      Max eigenvalue: {torch.linalg.eigvalsh(cov)[-1].item():.6f}")
        
        # Entropy
        entropy = q_block.entropy()
        print(f"    Entropy H[q(W_{i+1})]: {entropy.item():.4f}")


def example_mode_distributions():
    """Demonstrate mode q-distributions."""
    print("\n" + "=" * 70)
    print("3. Mode Q-Distributions (Gamma)")
    print("=" * 70)
    
    bmpo = create_simple_bmpo()
    mode_dists = bmpo.get_mode_q_distributions()
    
    for label, gamma_list in mode_dists.items():
        dist_info = bmpo.mu_mpo.distributions[label]
        var_name = dist_info['variable']
        dist_type = dist_info['type']
        
        print(f"\nMode '{label}' ({dist_type}, variable={var_name}):")
        print(f"  Number of indices: {len(gamma_list)}")
        
        if dist_type == 'rank':
            param1_name, param2_name = 'c', 'e'
            param1 = dist_info['c']
            param2 = dist_info['e']
        else:
            param1_name, param2_name = 'f', 'g'
            param1 = dist_info['f']
            param2 = dist_info['g']
        
        print(f"  {param1_name} = {param1}")
        print(f"  {param2_name} = {param2}")
        
        # Show E[X] and E[log X] for each index
        print(f"\n  For each index:")
        for i, gamma in enumerate(gamma_list):
            e_x = gamma.mean()
            e_log_x = gamma.expected_log()
            h = gamma.entropy()
            print(f"    Index {i}: E[{var_name}]={e_x:.4f}, E[log {var_name}]={e_log_x:.4f}, H={h:.4f}")


def example_full_q_distribution():
    """Demonstrate the full q-distribution."""
    print("\n" + "=" * 70)
    print("4. Full Q-Distribution (Product of All Components)")
    print("=" * 70)
    
    bmpo = create_simple_bmpo()
    
    # Get full q-distribution
    q_full = bmpo.get_full_q_distribution()
    
    print("\nFull q-distribution components:")
    print(f"  Total number of distributions: {len(q_full.distributions)}")
    
    # Count by type
    from tensor.probability_distributions import GammaDistribution, MultivariateGaussianDistribution
    
    n_gamma = sum(1 for d in q_full.distributions if isinstance(d, GammaDistribution))
    n_mvn = sum(1 for d in q_full.distributions if isinstance(d, MultivariateGaussianDistribution))
    
    print(f"    - Gamma distributions: {n_gamma}")
    print(f"      • 1 for τ")
    mode_dists = bmpo.get_mode_q_distributions()
    total_mode_gammas = sum(len(gammas) for gammas in mode_dists.values())
    print(f"      • {total_mode_gammas} for mode parameters")
    print(f"    - Multivariate Normal distributions: {n_mvn}")
    print(f"      • {len(bmpo.mu_nodes)} blocks")
    
    # Compute total entropy
    print("\n  Total entropy H[q(θ)]:")
    total_entropy = q_full.entropy()
    print(f"    H[q(θ)] = {total_entropy.item():.4f}")
    
    # Break down by component
    print("\n  Entropy breakdown:")
    print(f"    H[q(τ)] = {bmpo.tau_distribution.entropy().item():.4f}")
    
    block_dists = bmpo.get_all_block_q_distributions()
    for i, dist in enumerate(block_dists):
        print(f"    H[q(W_{i+1})] = {dist.entropy().item():.4f}")
    
    mode_entropy_total = 0.0
    for label, gamma_list in mode_dists.items():
        mode_entropy = sum(g.entropy() for g in gamma_list)
        print(f"    H[q({label})] = {mode_entropy.item():.4f} (sum over {len(gamma_list)} indices)")
        mode_entropy_total += mode_entropy.item()


def example_sampling():
    """Demonstrate sampling from q-distribution."""
    print("\n" + "=" * 70)
    print("5. Sampling from Q-Distribution")
    print("=" * 70)
    
    bmpo = create_simple_bmpo()
    
    # Sample from q
    n_samples = 5
    samples = bmpo.sample_from_q(n_samples=n_samples)
    
    print(f"\nGenerated {n_samples} samples from q(θ):")
    print()
    
    print(f"  τ samples: shape {samples['tau'].shape}")
    print(f"    {samples['tau']}")
    print()
    
    print(f"  Block samples:")
    for i, block_samples in enumerate(samples['blocks']):
        print(f"    W_{i+1}: shape {block_samples.shape}")
        print(f"      Sample 0 (first 5): {block_samples[0, :5]}")
    print()
    
    print(f"  Mode samples:")
    for label, mode_samples in samples['modes'].items():
        print(f"    Mode '{label}':")
        for idx, idx_samples in enumerate(mode_samples):
            print(f"      Index {idx}: {idx_samples}")


def example_log_prob():
    """Demonstrate computing log q(θ)."""
    print("\n" + "=" * 70)
    print("6. Computing log q(θ)")
    print("=" * 70)
    
    bmpo = create_simple_bmpo()
    
    # Sample one set of parameters
    samples = bmpo.sample_from_q(n_samples=1)
    
    # Extract single sample
    theta = {
        'tau': samples['tau'][0],
        'blocks': [block[0] for block in samples['blocks']],
        'modes': {
            label: [idx_samples[0] for idx_samples in mode_samples]
            for label, mode_samples in samples['modes'].items()
        }
    }
    
    # Compute log q(θ)
    log_q_val = bmpo.log_q(theta)
    
    print("\nSampled parameters θ:")
    print(f"  τ = {theta['tau'].item():.4f}")
    print(f"  W_1 (first 5): {theta['blocks'][0][:5]}")
    print(f"  W_2 (first 5): {theta['blocks'][1][:5]}")
    print(f"  Mode parameters (first mode, first 2 indices):")
    first_mode = list(theta['modes'].keys())[0]
    print(f"    {first_mode}: {[v.item() for v in theta['modes'][first_mode][:2]]}")
    
    print(f"\n  log q(θ) = {log_q_val.item():.4f}")
    
    print("\n  Decomposition:")
    print(f"    log q(τ) = {bmpo.tau_distribution.forward().log_prob(theta['tau']).item():.4f}")
    
    block_dists = bmpo.get_all_block_q_distributions()
    for i, (dist, block_val) in enumerate(zip(block_dists, theta['blocks'])):
        log_p = dist.forward().log_prob(block_val).item()
        print(f"    log q(W_{i+1}) = {log_p:.4f}")
    
    mode_dists = bmpo.get_mode_q_distributions()
    for label, gamma_list in mode_dists.items():
        mode_log_prob = sum(
            gamma.forward().log_prob(val) 
            for gamma, val in zip(gamma_list, theta['modes'][label])
        )
        print(f"    log q({label}) = {mode_log_prob.item():.4f} (sum over indices)")


def example_elbo_use_case():
    """Show how q-distribution is used in ELBO computation."""
    print("\n" + "=" * 70)
    print("7. ELBO Computation Use Case")
    print("=" * 70)
    
    bmpo = create_simple_bmpo()
    
    print("\nThe ELBO (Evidence Lower Bound) is:")
    print("  ELBO = E_q[log p(D, θ)] - E_q[log q(θ)]")
    print("       = E_q[log p(D|θ)] + E_q[log p(θ)] - E_q[log q(θ)]")
    print()
    
    print("Components computed using q-distribution:")
    print()
    
    print("1. E_q[log q(θ)] - Entropy term:")
    print("   H[q(θ)] = -E_q[log q(θ)]")
    q_full = bmpo.get_full_q_distribution()
    entropy = q_full.entropy()
    print(f"   H[q(θ)] = {entropy.item():.4f}")
    print()
    
    print("2. E_q[log p(θ)] - Prior term:")
    print("   Requires E_q[log ω], E_q[log φ], E_q[log τ], etc.")
    print()
    print("   Example for τ:")
    e_log_tau = bmpo.tau_distribution.expected_log()
    print(f"     E_q[log τ] = {e_log_tau.item():.4f}")
    print()
    
    print("   Example for mode parameters:")
    mode_dists = bmpo.get_mode_q_distributions()
    for label, gamma_list in list(mode_dists.items())[:2]:  # Show first 2
        e_log_sum = sum(g.expected_log() for g in gamma_list)
        print(f"     E_q[log {label}] = {e_log_sum.item():.4f} (sum over indices)")
    print()
    
    print("3. Sampling for Monte Carlo estimation:")
    print("   For terms like E_q[log p(D|θ)], sample θ ~ q:")
    samples = bmpo.sample_from_q(n_samples=3)
    print(f"     Generated {len(samples['tau'])} samples")
    print(f"     Compute log p(D|θ^(i)) for each sample")
    print(f"     Estimate: (1/N) Σ log p(D|θ^(i))")


def main():
    """Run all examples."""
    torch.manual_seed(42)
    
    example_q_structure()
    example_block_distributions()
    example_mode_distributions()
    example_full_q_distribution()
    example_sampling()
    example_log_prob()
    example_elbo_use_case()
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\nThe q-distribution for Bayesian MPO consists of:")
    print()
    print("1. Block distributions: q(W_i) ~ N(μ_i, Σ_i - μ_i ⊗ μ_i^T)")
    print("   - Mean: μ-MPO block (flattened)")
    print("   - Covariance: Σ-MPO block minus outer product of mean")
    print()
    print("2. Mode distributions: q(mode_param) ~ Gamma(c, e) or Gamma(f, g)")
    print("   - One Gamma per index in each dimension")
    print("   - Rank dimensions (ω) and vertical dimensions (φ)")
    print()
    print("3. Noise distribution: q(τ) ~ Gamma(α, β)")
    print()
    print("Methods available:")
    print("  - get_block_q_distribution(i): Get q(W_i)")
    print("  - get_mode_q_distributions(): Get all Gamma distributions")
    print("  - get_full_q_distribution(): Get product of all components")
    print("  - sample_from_q(n): Sample parameters from q")
    print("  - log_q(θ): Compute log q(θ) for given parameters")
    print()
    print("Use in variational inference:")
    print("  - ELBO computation: E_q[log p(D,θ)] - E_q[log q(θ)]")
    print("  - Parameter updates via gradient ascent on ELBO")
    print("  - Monte Carlo estimation using samples from q")
    print("=" * 70)


if __name__ == "__main__":
    main()
