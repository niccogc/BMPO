"""
Example demonstrating variational update for τ parameters.

Shows:
1. The update formulas for α_q and β_q
2. How to perform a single update step
3. Verification that parameters change appropriately
4. The role of data in updating precision
"""

import torch
from tensor.node import TensorNode
from tensor.bayesian_mpo import BayesianMPO


def create_simple_bmpo():
    """Create a simple Bayesian MPO for demonstration."""
    # Create μ-MPO node
    mu_node = TensorNode(
        tensor_or_shape=(1, 3, 1),
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
    
    # Create BayesianMPO with specific prior
    bmpo = BayesianMPO(
        mu_nodes=[mu_node],
        input_nodes=[input_node],
        rank_labels={'r0', 'r1'},
        tau_alpha=torch.tensor(2.0),  # Initial variational
        tau_beta=torch.tensor(1.0)
    )
    
    # Set informative prior
    bmpo.set_prior_hyperparameters(
        tau_alpha0=torch.tensor(1.0),  # α_p
        tau_beta0=torch.tensor(1.0)    # β_p
    )
    
    return bmpo


def example_update_formula():
    """Show the update formula for τ parameters."""
    print("=" * 70)
    print("1. τ Variational Update Formula")
    print("=" * 70)
    
    print("\nFor q(τ) ~ Gamma(α_q, β_q), the coordinate ascent update is:")
    print()
    print("  α_q = α_p + S/2")
    print()
    print("  β_q = β_p + Σ_s [y_s · μ-MPO(x_s)] + 1/2 Σ_s [Σ-MPO(x_s)]")
    print()
    print("where:")
    print("  - S is the number of samples")
    print("  - α_p, β_p are prior hyperparameters")
    print("  - μ-MPO(x_s) is the mean prediction for sample s")
    print("  - Σ-MPO(x_s) is the variance contraction for sample s")
    print("  - y_s · μ-MPO(x_s) is the scalar product (data · prediction)")
    print()
    print("Intuition:")
    print("  - α_q increases with more data (concentration)")
    print("  - β_q increases with prediction error (rate)")
    print("  - Precision τ ~ Gamma(α_q, β_q) becomes more certain with data")


def example_single_update():
    """Demonstrate a single update step."""
    print("\n" + "=" * 70)
    print("2. Single Update Step")
    print("=" * 70)
    
    bmpo = create_simple_bmpo()
    
    # Generate toy data
    S = 10  # Number of samples
    feature_dim = 3
    X = torch.randn(S, feature_dim)
    y = torch.randn(S, 1)  # Scalar output
    
    print(f"\nData:")
    print(f"  Samples: S = {S}")
    print(f"  Input shape: X = {X.shape}")
    print(f"  Output shape: y = {y.shape}")
    
    print(f"\nBefore update:")
    print(f"  Prior: p(τ) ~ Gamma(α_p={bmpo.prior_tau_alpha.item()}, β_p={bmpo.prior_tau_beta.item()})")
    print(f"  Variational: q(τ) ~ Gamma(α_q={bmpo.tau_alpha.item()}, β_q={bmpo.tau_beta.item()})")
    print(f"  E_q[τ] = α_q/β_q = {bmpo.get_tau_mean().item():.4f}")
    
    # Perform update
    print(f"\nPerforming variational update...")
    bmpo.update_tau_variational(X, y)
    
    print(f"\nAfter update:")
    print(f"  Variational: q(τ) ~ Gamma(α_q={bmpo.tau_alpha.item():.4f}, β_q={bmpo.tau_beta.item():.4f})")
    print(f"  E_q[τ] = α_q/β_q = {bmpo.get_tau_mean().item():.4f}")
    
    # Show the change
    expected_alpha = bmpo.prior_tau_alpha + S / 2.0
    print(f"\nVerification:")
    print(f"  Expected α_q = α_p + S/2 = {bmpo.prior_tau_alpha.item()} + {S}/2 = {expected_alpha.item()}")
    print(f"  Actual α_q = {bmpo.tau_alpha.item()}")
    print(f"  Match: {torch.allclose(bmpo.tau_alpha, expected_alpha)}")


def example_effect_of_data_fit():
    """Show how data fit affects τ updates."""
    print("\n" + "=" * 70)
    print("3. Effect of Data Fit on τ")
    print("=" * 70)
    
    print("\nComparing two scenarios:")
    print("  A) Good fit: predictions close to targets")
    print("  B) Poor fit: predictions far from targets")
    
    # Scenario A: Good fit
    bmpo_good = create_simple_bmpo()
    X = torch.randn(10, 3)
    
    # Generate targets close to predictions
    y_pred = bmpo_good.forward_mu(X, to_tensor=True)
    y_good = y_pred + 0.1 * torch.randn_like(y_pred)  # Small noise
    
    print(f"\nScenario A (good fit):")
    print(f"  Before: q(τ) ~ Gamma({bmpo_good.tau_alpha.item()}, {bmpo_good.tau_beta.item()})")
    bmpo_good.update_tau_variational(X, y_good)
    print(f"  After:  q(τ) ~ Gamma({bmpo_good.tau_alpha.item():.4f}, {bmpo_good.tau_beta.item():.4f})")
    print(f"  E_q[τ] = {bmpo_good.get_tau_mean().item():.4f} (precision)")
    
    # Scenario B: Poor fit
    bmpo_poor = create_simple_bmpo()
    y_poor = y_pred + 10.0 * torch.randn_like(y_pred)  # Large noise
    
    print(f"\nScenario B (poor fit):")
    print(f"  Before: q(τ) ~ Gamma({bmpo_poor.tau_alpha.item()}, {bmpo_poor.tau_beta.item()})")
    bmpo_poor.update_tau_variational(X, y_poor)
    print(f"  After:  q(τ) ~ Gamma({bmpo_poor.tau_alpha.item():.4f}, {bmpo_poor.tau_beta.item():.4f})")
    print(f"  E_q[τ] = {bmpo_poor.get_tau_mean().item():.4f} (precision)")
    
    print(f"\nObservation:")
    print(f"  Good fit → higher β_q → lower precision (more certain)")
    print(f"  Poor fit → higher β_q → higher precision reflects uncertainty")
    print(f"  (Note: interpretation depends on model structure)")


def example_multiple_updates():
    """Show multiple update iterations."""
    print("\n" + "=" * 70)
    print("4. Multiple Update Iterations")
    print("=" * 70)
    
    bmpo = create_simple_bmpo()
    X = torch.randn(10, 3)
    y = torch.randn(10, 1)
    
    print("\nPerforming 5 update iterations:")
    print(f"{'Iter':<6} {'α_q':<10} {'β_q':<10} {'E[τ]':<10}")
    print("-" * 40)
    
    print(f"{'0':<6} {bmpo.tau_alpha.item():<10.4f} {bmpo.tau_beta.item():<10.4f} {bmpo.get_tau_mean().item():<10.4f}")
    
    for i in range(1, 6):
        bmpo.update_tau_variational(X, y)
        print(f"{i:<6} {bmpo.tau_alpha.item():<10.4f} {bmpo.tau_beta.item():<10.4f} {bmpo.get_tau_mean().item():<10.4f}")
    
    print("\nNote: In practice, τ updates are part of coordinate ascent")
    print("      where all parameters (W, ω, φ, τ) are updated iteratively")


def example_interpretation():
    """Explain the interpretation of τ updates."""
    print("\n" + "=" * 70)
    print("5. Interpretation")
    print("=" * 70)
    
    print("\nτ represents the precision (inverse variance) of the likelihood:")
    print("  p(y|x, W, τ) ~ N(y | f(x, W), τ⁻¹)")
    print()
    print("The variational update:")
    print("  - Increases α_q with more samples (more confident)")
    print("  - Increases β_q with larger residuals (adapt to noise)")
    print()
    print("After convergence:")
    print("  - E_q[τ] estimates the precision of observations")
    print("  - Var_q[τ] = α_q/β_q² quantifies uncertainty in precision")
    print()
    print("In ELBO:")
    print("  - Higher τ → likelihood prefers closer predictions")
    print("  - Lower τ → likelihood is more tolerant of errors")
    print("  - Variational update balances fit vs. model complexity")


def main():
    """Run all examples."""
    torch.manual_seed(42)
    
    example_update_formula()
    example_single_update()
    example_effect_of_data_fit()
    example_multiple_updates()
    example_interpretation()
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\nτ Variational Update:")
    print()
    print("✓ Update Formula:")
    print("  α_q = α_p + S/2")
    print("  β_q = β_p + Σ_s [y_s · μ-MPO(x_s)] + 1/2 Σ_s [Σ-MPO(x_s)]")
    print()
    print("✓ Implementation:")
    print("  - update_tau_variational(X, y) performs the update")
    print("  - Automatically computes μ-MPO and Σ-MPO contractions")
    print("  - Updates internal tau_distribution")
    print()
    print("✓ Use in Coordinate Ascent:")
    print("  - Part of iterative variational inference")
    print("  - Updates τ given current W, ω, φ")
    print("  - Alternates with other parameter updates")
    print("=" * 70)


if __name__ == "__main__":
    main()
