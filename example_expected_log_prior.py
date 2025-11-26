"""
Example demonstrating E_q[log p(θ)] computation for Bayesian MPO.

The prior p(θ) has the same structure as q(θ) but with separate hyperparameters:
- p(τ) ~ Gamma(α₀, β₀)
- p(Wᵢ) ~ N(0, Σ₀)  (mean 0, structured covariance)
- p(ω), p(φ) ~ Gamma with hyperparameters

This example shows:
1. Default prior initialization
2. Setting custom prior hyperparameters
3. Computing E_q[log p(θ)] and its components
4. How this fits into ELBO computation
"""

import torch
from tensor.node import TensorNode
from tensor.bayesian_mpo import BayesianMPO


def create_simple_bmpo():
    """Create a simple Bayesian MPO for demonstration."""
    # Create μ-MPO nodes
    mu_node1 = TensorNode(
        tensor_or_shape=(2, 3, 2),
        dim_labels=['r0', 'f', 'r1'],
        l='r0',
        r='r1',
        name='mu1'
    )
    
    mu_node2 = TensorNode(
        tensor_or_shape=(2, 3, 2),
        dim_labels=['r1', 'f', 'r2'],
        l='r1',
        r='r2',
        name='mu2'
    )
    
    mu_node1.connect(mu_node2, 'r1', priority=1)
    
    # Create BayesianMPO
    bmpo = BayesianMPO(
        mu_nodes=[mu_node1, mu_node2],
        rank_labels={'r0', 'r1', 'r2'},
        tau_alpha=torch.tensor(3.0),
        tau_beta=torch.tensor(1.5)
    )
    
    return bmpo


def example_default_priors():
    """Show default prior initialization."""
    print("=" * 70)
    print("1. Default Prior Initialization")
    print("=" * 70)
    
    bmpo = create_simple_bmpo()
    
    print("\nBy default, priors are uninformative/weakly informative:")
    print("(In practice, you should set informative priors before inference)")
    print()
    
    print("Prior for τ:")
    print(f"  p(τ) ~ Gamma(α₀={bmpo.prior_tau_alpha.item()}, β₀={bmpo.prior_tau_beta.item()}) - uninformative")
    print(f"  Variational: q(τ) ~ Gamma(α={bmpo.tau_alpha.item()}, β={bmpo.tau_beta.item()})")
    print()
    
    print("Prior for mode parameters:")
    for label, params in bmpo.prior_mode_params.items():
        dist_info = bmpo.mu_mpo.distributions[label]
        if 'c0' in params:
            print(f"  {label} (rank): p(ω) ~ Gamma(c₀={params['c0'][0].item()}, e₀={params['e0'][0].item()})")
        else:
            print(f"  {label} (vertical): p(φ) ~ Gamma(f₀={params['f0'][0].item()}, g₀={params['g0'][0].item()})")
    print()
    
    print("Prior for blocks:")
    for i, sigma0 in enumerate(bmpo.prior_block_sigma0):
        print(f"  Block {i+1}: p(W_{i+1}) ~ N(0, Σ₀) where Σ₀ is {sigma0.shape[0]}×{sigma0.shape[0]}")
        print(f"    Σ₀ initialized as identity: diagonal = {torch.diag(sigma0)[:4]}")


def example_custom_priors():
    """Show setting custom prior hyperparameters."""
    print("\n" + "=" * 70)
    print("2. Setting Custom Prior Hyperparameters")
    print("=" * 70)
    
    bmpo = create_simple_bmpo()
    
    print("\nSetting more informative priors:")
    print()
    
    # Set custom prior for τ
    bmpo.set_prior_hyperparameters(
        tau_alpha0=torch.tensor(5.0),
        tau_beta0=torch.tensor(2.0)
    )
    print(f"Updated p(τ) ~ Gamma(α₀={bmpo.prior_tau_alpha.item()}, β₀={bmpo.prior_tau_beta.item()})")
    print()
    
    # Set custom prior for a specific mode
    custom_mode_params = {
        'r1': {
            'c0': torch.tensor([3.0, 4.0]),  # Different for each index
            'e0': torch.tensor([1.0, 1.5])
        }
    }
    bmpo.set_prior_hyperparameters(mode_params0=custom_mode_params)
    print(f"Updated p(ω) for r1:")
    print(f"  Index 0: Gamma(c₀={bmpo.prior_mode_params['r1']['c0'][0].item()}, e₀={bmpo.prior_mode_params['r1']['e0'][0].item()})")
    print(f"  Index 1: Gamma(c₀={bmpo.prior_mode_params['r1']['c0'][1].item()}, e₀={bmpo.prior_mode_params['r1']['e0'][1].item()})")
    print()
    
    # Set custom prior covariance for a block
    d = bmpo.mu_nodes[0].tensor.numel()
    custom_sigma0 = 0.5 * torch.eye(d)  # Tighter prior (smaller variance)
    bmpo.set_prior_hyperparameters(block_sigma0=[custom_sigma0, bmpo.prior_block_sigma0[1]])
    print(f"Updated Σ₀ for block 1:")
    print(f"  Diagonal (variance): {torch.diag(custom_sigma0)[:4]}")


def example_compute_expected_log_prior():
    """Demonstrate computing E_q[log p(θ)]."""
    print("\n" + "=" * 70)
    print("3. Computing E_q[log p(θ)]")
    print("=" * 70)
    
    bmpo = create_simple_bmpo()
    
    print("\nE_q[log p(θ)] factorizes as:")
    print("  E_q[log p(θ)] = E_q[log p(τ)] + Σᵢ E_q[log p(Wᵢ)] + Σ_modes E_q[log p(mode_params)]")
    print()
    
    # Compute individual components
    print("Individual components:")
    
    # τ component
    log_p_tau = bmpo._expected_log_prior_tau()
    print(f"  E_q[log p(τ)] = {log_p_tau.item():.4f}")
    
    # Block components
    log_p_blocks_total = 0.0
    for i in range(len(bmpo.mu_nodes)):
        log_p_block = bmpo._expected_log_prior_block(i)
        print(f"  E_q[log p(W_{i+1})] = {log_p_block.item():.4f}")
        log_p_blocks_total += log_p_block.item()
    
    # Mode components
    log_p_modes = bmpo._expected_log_prior_modes()
    print(f"  E_q[log p(modes)] = {log_p_modes.item():.4f}")
    
    # Total
    log_p_total = bmpo.compute_expected_log_prior()
    print()
    print(f"Total E_q[log p(θ)] = {log_p_total.item():.4f}")
    print(f"  (Sum: {log_p_tau.item():.4f} + {log_p_blocks_total:.4f} + {log_p_modes.item():.4f} = {log_p_tau.item() + log_p_blocks_total + log_p_modes.item():.4f})")


def example_component_breakdown():
    """Show detailed breakdown of each component."""
    print("\n" + "=" * 70)
    print("4. Component Breakdown")
    print("=" * 70)
    
    bmpo = create_simple_bmpo()
    
    print("\nE_q[log p(τ)] computation:")
    print("  For τ ~ Gamma(α, β), log p(τ) = α log β - log Γ(α) + (α-1) log τ - β τ")
    print()
    alpha0 = bmpo.prior_tau_alpha
    beta0 = bmpo.prior_tau_beta
    e_q_log_tau = bmpo.tau_distribution.expected_log()
    e_q_tau = bmpo.tau_distribution.mean()
    
    print(f"  Prior: p(τ) ~ Gamma(α₀={alpha0.item()}, β₀={beta0.item()})")
    print(f"  Variational: q(τ) ~ Gamma(α={bmpo.tau_alpha.item()}, β={bmpo.tau_beta.item()})")
    print(f"  E_q[log τ] = {e_q_log_tau.item():.4f}")
    print(f"  E_q[τ] = {e_q_tau.item():.4f}")
    print()
    
    term1 = alpha0 * torch.log(beta0)
    term2 = -torch.lgamma(alpha0)
    term3 = (alpha0 - 1) * e_q_log_tau
    term4 = -beta0 * e_q_tau
    
    print(f"  E_q[log p(τ)] = α₀ log β₀ - log Γ(α₀) + (α₀-1) E_q[log τ] - β₀ E_q[τ]")
    print(f"                = {term1.item():.4f} + {term2.item():.4f} + {term3.item():.4f} + {term4.item():.4f}")
    print(f"                = {(term1 + term2 + term3 + term4).item():.4f}")
    print()
    
    print("\nE_q[log p(W₁)] computation:")
    print("  For W ~ N(0, Σ₀), log p(W) = -d/2 log(2π) - 1/2 log|Σ₀| - 1/2 tr(Σ₀⁻¹ E_q[W ⊗ Wᵀ])")
    print()
    
    mu_node = bmpo.mu_nodes[0]
    d = mu_node.tensor.numel()
    sigma0 = bmpo.prior_block_sigma0[0]
    
    print(f"  Dimension d = {d}")
    print(f"  Prior covariance Σ₀: diagonal = {torch.diag(sigma0)[:4]}")
    print(f"  E_q[W ⊗ Wᵀ] from Σ-MPO")
    
    log_p_w = bmpo._expected_log_prior_block(0)
    print(f"  E_q[log p(W₁)] = {log_p_w.item():.4f}")
    print()
    
    print("\nE_q[log p(mode_params)] computation:")
    print("  Sum over all modes and indices")
    print("  Each: α₀ log β₀ - log Γ(α₀) + (α₀-1) E_q[log θ] - β₀ E_q[θ]")
    
    for label in list(bmpo.prior_mode_params.keys())[:2]:  # Show first 2
        gammas = bmpo.mu_mpo.get_gamma_distributions(label)
        prior_params = bmpo.prior_mode_params[label]
        
        print(f"\n  Mode '{label}':")
        for i, gamma in enumerate(gammas[:2]):  # Show first 2 indices
            e_q_log = gamma.expected_log()
            e_q = gamma.mean()
            
            if 'c0' in prior_params:
                alpha0 = prior_params['c0'][i]
                beta0 = prior_params['e0'][i]
            else:
                alpha0 = prior_params['f0'][i]
                beta0 = prior_params['g0'][i]
            
            log_p_theta = (alpha0 * torch.log(beta0) - torch.lgamma(alpha0) 
                          + (alpha0 - 1) * e_q_log - beta0 * e_q)
            
            print(f"    Index {i}: E_q[log p(θ)] = {log_p_theta.item():.4f}")


def example_elbo_context():
    """Show how E_q[log p(θ)] fits into ELBO."""
    print("\n" + "=" * 70)
    print("5. ELBO Context")
    print("=" * 70)
    
    bmpo = create_simple_bmpo()
    
    print("\nThe ELBO (Evidence Lower Bound) is:")
    print("  ELBO = E_q[log p(D|θ)] + E_q[log p(θ)] - E_q[log q(θ)]")
    print()
    print("where:")
    print("  • E_q[log p(D|θ)] = likelihood term (requires data)")
    print("  • E_q[log p(θ)] = prior term (computed here)")
    print("  • E_q[log q(θ)] = -H[q(θ)] (negative entropy)")
    print()
    
    # Compute terms we have
    e_q_log_p = bmpo.compute_expected_log_prior()
    q_full = bmpo.get_full_q_distribution()
    entropy_q = q_full.entropy()
    
    print("Components we can compute:")
    print(f"  E_q[log p(θ)] = {e_q_log_p.item():.4f}")
    print(f"  H[q(θ)] = {entropy_q.item():.4f}")
    print(f"  -E_q[log q(θ)] = H[q(θ)] = {entropy_q.item():.4f}")
    print()
    
    print("For complete ELBO, need:")
    print("  E_q[log p(D|θ)] = E_q[log p(y|forward_model(x, W), τ)]")
    print("  This requires data (x, y) and will be computed separately")
    print()
    
    print(f"Partial ELBO (without likelihood):")
    partial_elbo = e_q_log_p + entropy_q
    print(f"  {e_q_log_p.item():.4f} + {entropy_q.item():.4f} = {partial_elbo.item():.4f}")


def example_modular_design():
    """Show the modular design of E_q[log p] computation."""
    print("\n" + "=" * 70)
    print("6. Modular Design")
    print("=" * 70)
    
    print("\nThe E_q[log p(θ)] computation is modular:")
    print()
    print("Main method:")
    print("  compute_expected_log_prior()")
    print("    └─ Calls component methods:")
    print()
    print("Component methods:")
    print("  _expected_log_prior_tau()")
    print("    └─ Computes E_q[log p(τ)] for Gamma prior")
    print()
    print("  _expected_log_prior_block(i)")
    print("    └─ Computes E_q[log p(Wᵢ)] for multivariate Normal prior")
    print()
    print("  _expected_log_prior_modes()")
    print("    └─ Computes E_q[log p(mode_params)] for all Gamma priors")
    print()
    print("Benefits:")
    print("  ✓ Easy to modify individual components")
    print("  ✓ Can compute partial terms separately")
    print("  ✓ Easy to debug (check each component)")
    print("  ✓ Easy to remove if needed")


def main():
    """Run all examples."""
    torch.manual_seed(42)
    
    example_default_priors()
    example_custom_priors()
    example_compute_expected_log_prior()
    example_component_breakdown()
    example_elbo_context()
    example_modular_design()
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\nE_q[log p(θ)] Computation:")
    print()
    print("✓ Prior Structure: Same as q(θ) with separate hyperparameters")
    print("  - p(τ) ~ Gamma(α₀, β₀)")
    print("  - p(Wᵢ) ~ N(0, Σ₀)")
    print("  - p(ω), p(φ) ~ Gamma(c₀, e₀) or Gamma(f₀, g₀)")
    print()
    print("✓ Methods:")
    print("  - set_prior_hyperparameters(): Set custom priors")
    print("  - compute_expected_log_prior(): Compute E_q[log p(θ)]")
    print("  - Component methods for each term")
    print()
    print("✓ ELBO Usage:")
    print("  ELBO = E_q[log p(D|θ)] + E_q[log p(θ)] - E_q[log q(θ)]")
    print("  E_q[log p(θ)] is computed here")
    print()
    print("✓ Modular Design:")
    print("  - Easy to modify/remove components")
    print("  - Separate methods for debugging")
    print("  - Clean separation of concerns")
    print("=" * 70)


if __name__ == "__main__":
    main()
