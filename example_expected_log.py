"""
Example demonstrating E[log X] computations for probability distributions.

Shows:
1. E[log X] for Gamma distributions
2. E[log X] with alternative parameters
3. E[log X] for Product distributions
4. Verification with Monte Carlo sampling
"""

import torch
from tensor.probability_distributions import (
    GammaDistribution, 
    MultivariateGaussianDistribution,
    ProductDistribution
)


def example_gamma_expected_log():
    """Demonstrate E[log X] for Gamma distribution."""
    print("=" * 70)
    print("1. Gamma Distribution E[log X]")
    print("=" * 70)
    
    # Create Gamma distribution: Gamma(concentration=3, rate=2)
    concentration = torch.tensor(3.0)
    rate = torch.tensor(2.0)
    gamma = GammaDistribution(concentration=concentration, rate=rate)
    
    print(f"\nGamma(α={concentration.item()}, β={rate.item()})")
    print(f"E[X] = α/β = {gamma.mean().item():.4f}")
    print(f"E[log X] = ψ(α) - log(β) = {gamma.expected_log().item():.4f}")
    print(f"  where ψ(α) = digamma({concentration.item()}) = {torch.digamma(concentration).item():.4f}")
    
    # Verify with Monte Carlo sampling
    print("\nVerification with Monte Carlo:")
    samples = gamma.forward().sample((100000,))
    log_samples = torch.log(samples)
    empirical_mean_log = log_samples.mean()
    print(f"  Empirical E[log X] (100k samples): {empirical_mean_log.item():.4f}")
    print(f"  Analytical E[log X]:                {gamma.expected_log().item():.4f}")
    print(f"  Error: {abs(empirical_mean_log - gamma.expected_log()).item():.4f}")
    
    print("\n" + "-" * 70)
    print("2. E[log X] with Alternative Parameters")
    print("-" * 70)
    
    # Compute E[log X] using different parameters
    print("\nOriginal: Gamma(α=3, β=2)")
    print(f"  E[log X] = {gamma.expected_log().item():.4f}")
    
    alt_concentration = torch.tensor(5.0)
    alt_rate = torch.tensor(3.0)
    print(f"\nUsing alternative parameters: Gamma(α=5, β=3)")
    print(f"  E[log X] = {gamma.expected_log(concentration=alt_concentration, rate=alt_rate).item():.4f}")
    
    # This is useful for variational inference where we compute expectations
    # with respect to q(X) while parameterized by different hyperparameters
    print("\nUse case: Computing E_q[log X] with q's parameters")
    print("  (useful in variational Bayes for computing ELBO)")


def example_gamma_vector():
    """Demonstrate E[log X] for vector of Gamma distributions."""
    print("\n" + "=" * 70)
    print("3. Vector of Gamma Distributions")
    print("=" * 70)
    
    # Create vector of Gamma distributions (e.g., for a dimension with 4 indices)
    concentrations = torch.tensor([2.0, 3.0, 1.5, 4.0])
    rates = torch.tensor([1.0, 2.0, 0.5, 2.5])
    
    gamma = GammaDistribution(concentration=concentrations, rate=rates)
    
    print(f"\nVector of 4 Gamma distributions:")
    print(f"  α = {concentrations}")
    print(f"  β = {rates}")
    print(f"\nE[X] for each:")
    print(f"  {gamma.mean()}")
    print(f"\nE[log X] for each:")
    print(f"  {gamma.expected_log()}")
    
    # Component-wise formula
    print(f"\nComponent-wise: ψ(α) - log(β)")
    for i in range(len(concentrations)):
        print(f"  Index {i}: ψ({concentrations[i].item():.1f}) - log({rates[i].item():.1f}) = {gamma.expected_log()[i].item():.4f}")


def example_product_distribution():
    """Demonstrate E[log X] for Product distributions."""
    print("\n" + "=" * 70)
    print("4. Product Distribution E[log X]")
    print("=" * 70)
    
    # Create multiple Gamma distributions
    gamma1 = GammaDistribution(concentration=torch.tensor(3.0), rate=torch.tensor(1.0))
    gamma2 = GammaDistribution(concentration=torch.tensor(2.0), rate=torch.tensor(2.0))
    gamma3 = GammaDistribution(concentration=torch.tensor(4.0), rate=torch.tensor(1.5))
    
    product = ProductDistribution([gamma1, gamma2, gamma3])
    
    print("\nProduct of 3 independent Gamma distributions:")
    print(f"  Gamma1(α=3.0, β=1.0): E[log X1] = {gamma1.expected_log().item():.4f}")
    print(f"  Gamma2(α=2.0, β=2.0): E[log X2] = {gamma2.expected_log().item():.4f}")
    print(f"  Gamma3(α=4.0, β=1.5): E[log X3] = {gamma3.expected_log().item():.4f}")
    
    total_expected_log = product.expected_log()
    print(f"\nFor independent X1, X2, X3:")
    print(f"  E[log(X1 * X2 * X3)] = E[log X1] + E[log X2] + E[log X3]")
    print(f"                       = {gamma1.expected_log().item():.4f} + {gamma2.expected_log().item():.4f} + {gamma3.expected_log().item():.4f}")
    print(f"                       = {total_expected_log.item():.4f}")


def example_variational_bayes_use_case():
    """Show practical use case in variational Bayes."""
    print("\n" + "=" * 70)
    print("5. Variational Bayes Use Case")
    print("=" * 70)
    
    print("\nScenario: Variational inference with Gamma prior/posterior")
    print("\nPrior: π(ω) ~ Gamma(c, e)")
    prior_c = torch.tensor(2.0)
    prior_e = torch.tensor(1.0)
    prior = GammaDistribution(concentration=prior_c, rate=prior_e)
    
    print(f"  c (concentration) = {prior_c.item()}")
    print(f"  e (rate) = {prior_e.item()}")
    
    print("\nVariational posterior: q(ω) ~ Gamma(c', e')")
    posterior_c = torch.tensor(5.0)
    posterior_e = torch.tensor(2.0)
    posterior = GammaDistribution(concentration=posterior_c, rate=posterior_e)
    
    print(f"  c' = {posterior_c.item()}")
    print(f"  e' = {posterior_e.item()}")
    
    print("\nComputing ELBO requires:")
    print("  E_q[log π(ω)] - requires E_q[log ω]")
    
    # E_q[log ω] where q ~ Gamma(c', e')
    e_q_log_omega = posterior.expected_log()
    print(f"\n  E_q[log ω] = ψ(c') - log(e')")
    print(f"             = ψ({posterior_c.item()}) - log({posterior_e.item()})")
    print(f"             = {e_q_log_omega.item():.4f}")
    
    # Can also compute expectation with respect to different parameters
    print("\n  Can compute E[log ω] under prior parameters:")
    print(f"  E_prior[log ω] = {prior.expected_log().item():.4f}")
    
    # Or use posterior's expected_log with prior's parameters
    print("\n  Or use posterior object but with prior's parameters:")
    print(f"  (computed as ψ(c_prior) - log(e_prior))")
    e_prior_log_omega = posterior.expected_log(concentration=prior_c, rate=prior_e)
    print(f"  = {e_prior_log_omega.item():.4f}")


def example_bmpo_dimension():
    """Show how this applies to BMPO network dimensions."""
    print("\n" + "=" * 70)
    print("6. BMPO Network Dimension Example")
    print("=" * 70)
    
    print("\nConsider a rank dimension 'r1' with size 4")
    print("Each index has its own Gamma distribution:")
    
    # Parameters for 4 indices
    c_values = torch.tensor([2.0, 3.0, 1.5, 4.0])
    e_values = torch.tensor([1.0, 1.5, 0.8, 2.0])
    
    print(f"\n  c = {c_values}")
    print(f"  e = {e_values}")
    
    # Create Gamma for this dimension
    gamma_r1 = GammaDistribution(concentration=c_values, rate=e_values)
    
    # Expectations
    expectations = gamma_r1.mean()
    expected_logs = gamma_r1.expected_log()
    
    print(f"\n  E[ω] for each index = {expectations}")
    print(f"  E[log ω] for each index = {expected_logs}")
    
    print("\nFor variational updates, we can compute:")
    print("  - E[ω_i] for trimming decisions")
    print("  - E[log ω_i] for ELBO computation")
    
    # Trimming example
    threshold = 1.5
    print(f"\nTrimming with threshold {threshold}:")
    keep_mask = expectations >= threshold
    keep_indices = torch.where(keep_mask)[0]
    print(f"  Keep indices: {keep_indices.tolist()}")
    print(f"  E[ω] for kept: {expectations[keep_mask]}")
    print(f"  E[log ω] for kept: {expected_logs[keep_mask]}")


def main():
    """Run all examples."""
    torch.manual_seed(42)
    
    example_gamma_expected_log()
    example_gamma_vector()
    example_product_distribution()
    example_variational_bayes_use_case()
    example_bmpo_dimension()
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\nE[log X] methods added to all distributions:")
    print("  1. GammaDistribution: E[log X] = ψ(α) - log(β)")
    print("  2. MultivariateGaussianDistribution: E[log p(X)] (related to entropy)")
    print("  3. ProductDistribution: E[log(X1*X2*...)] = sum of E[log Xi]")
    print("\nAll methods support alternative parameters:")
    print("  dist.expected_log(concentration=alt_c, rate=alt_e)")
    print("\nUse cases:")
    print("  - Variational Bayes ELBO computation")
    print("  - Computing expectations under different distributions")
    print("  - BMPO network parameter updates")
    print("=" * 70)


if __name__ == "__main__":
    main()
