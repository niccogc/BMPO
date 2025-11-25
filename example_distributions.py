"""
Example usage of probability distributions module.

This demonstrates:
1. Creating basic distributions (Gamma, Multivariate Gaussian)
2. Computing entropy
3. Creating product distributions
4. Nesting product distributions
5. Updating distribution parameters
"""

import torch
from tensor.probability_distributions import (
    GammaDistribution,
    MultivariateGaussianDistribution,
    ProductDistribution
)


def main():
    print("=" * 70)
    print("Probability Distributions Usage Examples")
    print("=" * 70)
    
    # Example 1: Basic Gamma Distribution
    print("\n1. Gamma Distribution")
    print("-" * 70)
    gamma = GammaDistribution(
        concentration=torch.tensor(2.0),
        rate=torch.tensor(1.0)
    )
    print(f"Concentration (alpha): {gamma.concentration}")
    print(f"Rate (beta): {gamma.rate}")
    print(f"Entropy: {gamma.entropy():.4f}")
    print(f"Sample: {gamma.forward().sample()}")
    
    # Example 2: Multivariate Gaussian
    print("\n2. Multivariate Gaussian Distribution")
    print("-" * 70)
    mvn = MultivariateGaussianDistribution(
        loc=torch.tensor([0.0, 0.0]),
        covariance_matrix=torch.tensor([[1.0, 0.5], [0.5, 1.0]])
    )
    print(f"Mean: {mvn.loc}")
    print(f"Covariance:\n{mvn.covariance_matrix}")
    print(f"Entropy: {mvn.entropy():.4f}")
    print(f"Sample: {mvn.forward().sample()}")
    
    # Example 3: Product Distribution
    print("\n3. Product Distribution (Independent)")
    print("-" * 70)
    gamma1 = GammaDistribution(torch.tensor(2.0), torch.tensor(1.0))
    gamma2 = GammaDistribution(torch.tensor(3.0), torch.tensor(2.0))
    
    product = ProductDistribution([gamma1, gamma2])
    print(f"Gamma1 entropy: {gamma1.entropy():.4f}")
    print(f"Gamma2 entropy: {gamma2.entropy():.4f}")
    print(f"Product entropy (sum): {product.entropy():.4f}")
    
    samples = product.sample()
    print(f"Samples: [{samples[0]:.4f}, {samples[1]:.4f}]")
    
    # Example 4: Nested Product Distributions
    print("\n4. Nested Product Distributions")
    print("-" * 70)
    gamma3 = GammaDistribution(torch.tensor(5.0), torch.tensor(1.5))
    mvn2 = MultivariateGaussianDistribution(
        loc=torch.tensor([1.0, -1.0]),
        covariance_matrix=torch.eye(2)
    )
    
    # Create first product
    product1 = ProductDistribution([gamma1, gamma2])
    
    # Create nested product
    product2 = ProductDistribution([product1, gamma3, mvn2])
    
    print(f"Product1 entropy (2 Gammas): {product1.entropy():.4f}")
    print(f"Gamma3 entropy: {gamma3.entropy():.4f}")
    print(f"MVN2 entropy: {mvn2.entropy():.4f}")
    print(f"Nested product entropy: {product2.entropy():.4f}")
    print(f"Expected (sum of all): {gamma1.entropy() + gamma2.entropy() + gamma3.entropy() + mvn2.entropy():.4f}")
    
    # Example 5: Updating Parameters
    print("\n5. Updating Distribution Parameters")
    print("-" * 70)
    gamma_update = GammaDistribution(torch.tensor(2.0), torch.tensor(1.0))
    print(f"Initial entropy: {gamma_update.entropy():.4f}")
    print(f"Initial params - concentration: {gamma_update.concentration}, rate: {gamma_update.rate}")
    
    # Update parameters
    gamma_update.update_parameters(
        concentration=torch.tensor(5.0),
        rate=torch.tensor(3.0)
    )
    print(f"\nAfter update entropy: {gamma_update.entropy():.4f}")
    print(f"New params - concentration: {gamma_update.concentration}, rate: {gamma_update.rate}")
    
    # Update MVN
    mvn_update = MultivariateGaussianDistribution(
        loc=torch.tensor([0.0, 0.0]),
        covariance_matrix=torch.eye(2)
    )
    print(f"\nInitial MVN entropy: {mvn_update.entropy():.4f}")
    
    mvn_update.update_parameters(
        covariance_matrix=torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    )
    print(f"After covariance update entropy: {mvn_update.entropy():.4f}")
    
    # Example 6: Log Probability
    print("\n6. Computing Log Probabilities")
    print("-" * 70)
    gamma_a = GammaDistribution(torch.tensor(2.0), torch.tensor(1.0))
    gamma_b = GammaDistribution(torch.tensor(3.0), torch.tensor(2.0))
    product_ab = ProductDistribution([gamma_a, gamma_b])
    
    # Sample values
    val_a = torch.tensor(2.0)
    val_b = torch.tensor(1.5)
    
    log_prob_a = gamma_a.forward().log_prob(val_a)
    log_prob_b = gamma_b.forward().log_prob(val_b)
    log_prob_product = product_ab.log_prob([val_a, val_b])
    
    print(f"log p(x={val_a:.1f} | Gamma_a): {log_prob_a:.4f}")
    print(f"log p(x={val_b:.1f} | Gamma_b): {log_prob_b:.4f}")
    print(f"log p(Product): {log_prob_product:.4f}")
    print(f"Expected (sum): {log_prob_a + log_prob_b:.4f}")
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
