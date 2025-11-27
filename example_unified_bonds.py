"""
Example demonstrating the unified bond parameterization.

All bonds (horizontal and vertical) now use the same Gamma(concentration, rate)
parameterization, making the API more consistent and easier to work with.
"""

import torch
from tensor.bayesian_mpo_builder import create_bayesian_tensor_train


def main():
    print("=" * 70)
    print("Unified Bond Parameterization Example")
    print("=" * 70)
    
    # Create Bayesian MPO
    bmpo = create_bayesian_tensor_train(
        num_blocks=3,
        bond_dim=4,
        input_features=5,
        output_shape=1,
        seed=42
    )
    
    # Example 1: View all bonds and their parameters
    print("\n1. All Bonds and Their Parameters")
    print("-" * 70)
    
    for label, dist in bmpo.mu_mpo.distributions.items():
        size = len(dist['concentration'])
        print(f"\nBond '{label}' (size={size}):")
        print(f"  Concentration (α): {dist['concentration']}")
        print(f"  Rate (β): {dist['rate']}")
        print(f"  E[X] = α/β: {dist['expectation']}")
    
    # Example 2: Get nodes associated with each bond
    print("\n\n2. Bond-to-Nodes Mapping")
    print("-" * 70)
    
    for label in bmpo.mu_mpo.distributions.keys():
        nodes = bmpo.get_nodes_for_bond(label)
        bond_type = "horizontal" if label.startswith('r') else "vertical"
        print(f"\nBond '{label}' ({bond_type}):")
        print(f"  μ-MPO nodes: {nodes['mu_nodes']}")
        print(f"  Σ-MPO nodes: {nodes['sigma_nodes']}")
    
    # Example 3: Update bond parameters
    print("\n\n3. Updating Bond Parameters")
    print("-" * 70)
    
    # Update horizontal bond 'r1'
    label = 'r1'
    print(f"\nUpdating bond '{label}':")
    print(f"  Before: α={bmpo.mu_mpo.distributions[label]['concentration']}")
    
    # Set different parameters for each index in the bond
    new_concentration = torch.tensor([3.0, 2.5, 4.0, 2.0], dtype=torch.float64)
    new_rate = torch.tensor([1.5, 1.0, 2.0, 1.0], dtype=torch.float64)
    
    bmpo.update_mu_params(label, concentration=new_concentration, rate=new_rate)
    
    print(f"  After:  α={bmpo.mu_mpo.distributions[label]['concentration']}")
    print(f"          β={bmpo.mu_mpo.distributions[label]['rate']}")
    print(f"        E[X]={bmpo.mu_mpo.distributions[label]['expectation']}")
    
    # Example 4: Access individual Gamma distributions
    print("\n\n4. Individual Gamma Distributions")
    print("-" * 70)
    
    label = 'r1'
    gammas = bmpo.mu_mpo.get_gamma_distributions(label)
    
    print(f"\nBond '{label}' has {len(gammas)} Gamma distributions:")
    for i, gamma in enumerate(gammas):
        print(f"  Index {i}: Gamma(α={gamma.concentration:.2f}, β={gamma.rate:.2f})")
        print(f"           Mean={gamma.mean():.2f}, Entropy={gamma.entropy():.4f}")
    
    # Example 5: Prior parameters (same structure as q)
    print("\n\n5. Prior Parameters")
    print("-" * 70)
    
    print("\nPrior parameters have the same structure as variational parameters:")
    for label in list(bmpo.prior_bond_params.keys())[:3]:
        prior = bmpo.prior_bond_params[label]
        print(f"\nBond '{label}' prior:")
        print(f"  concentration0: {prior['concentration0']}")
        print(f"  rate0: {prior['rate0']}")
    
    print("\n" + "=" * 70)
    print("Summary:")
    print("  - All bonds use Gamma(concentration, rate) parameterization")
    print("  - Each bond dimension has N independent Gamma distributions")
    print("  - Easy retrieval of associated μ and σ nodes via get_nodes_for_bond()")
    print("  - Prior has same structure as variational distribution")
    print("=" * 70)


if __name__ == "__main__":
    main()
