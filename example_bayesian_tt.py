"""
Example: Building a Bayesian Tensor Train (TT/MPS) structure.
"""

import torch
from tensor.bayesian_mpo_builder import create_bayesian_tensor_train


def main():
    print("=" * 70)
    print("Bayesian Tensor Train Example")
    print("=" * 70)
    
    # Create Bayesian TT
    print("\n1. Creating Bayesian Tensor Train")
    print("-" * 70)
    
    bmpo = create_bayesian_tensor_train(
        num_blocks=3,           # N = 3 blocks
        bond_dim=4,             # r = 4 bond dimension
        input_features=5,       # f = 5 input features
        output_shape=1,         # Single output
        constrict_bond=True,    # Constrict bond dimensions
        tau_alpha=torch.tensor(3.0),
        tau_beta=torch.tensor(1.5),
        dtype=torch.float64,
        seed=42
    )
    
    print("Bayesian TT created!")
    
    # Show μ-MPO structure
    print("\n2. μ-MPO Structure")
    print("-" * 70)
    
    print(f"Number of μ nodes: {len(bmpo.mu_nodes)}")
    for i, node in enumerate(bmpo.mu_nodes):
        print(f"  μ-node {i+1}: shape={node.shape}, labels={node.dim_labels}")
    
    print(f"\nNumber of μ input nodes: {len(bmpo.input_nodes)}")
    for i, node in enumerate(bmpo.input_nodes):
        print(f"  μ-input {i+1}: shape={node.shape}, labels={node.dim_labels}")
    
    # Show Σ-MPO structure
    print("\n3. Σ-MPO Structure (Doubled)")
    print("-" * 70)
    
    print(f"Number of Σ nodes: {len(bmpo.sigma_nodes)}")
    for i, node in enumerate(bmpo.sigma_nodes):
        print(f"  Σ-node {i+1}: shape={node.shape}, labels={node.dim_labels}")
    
    print(f"\nNumber of Σ input nodes: {len(bmpo.sigma_input_nodes)}")
    for i, node in enumerate(bmpo.sigma_input_nodes):
        print(f"  Σ-input {i+1}: shape={node.shape}, labels={node.dim_labels}")
    
    # Show τ distribution
    print("\n4. τ Distribution")
    print("-" * 70)
    
    print(f"τ ~ Gamma(α={bmpo.tau_alpha.item()}, β={bmpo.tau_beta.item()})")
    print(f"E[τ] = {bmpo.get_tau_mean():.4f}")
    
    # Show prior distributions
    print("\n5. Prior Distributions on μ-MPO")
    print("-" * 70)
    
    print("Rank dimensions (horizontal bonds):", bmpo.mu_mpo.rank_labels)
    print("\nDistributions:")
    for label, dist in bmpo.mu_mpo.distributions.items():
        print(f"  {label}: {dist['type']} ({dist['variable']}), size={len(dist['expectation'])}")
        print(f"    E = {dist['expectation']}")
    
    # Test forward pass on μ-MPO
    print("\n6. Forward Pass (μ-MPO)")
    print("-" * 70)
    
    # Create sample input
    batch_size = 10
    x_data = torch.randn(batch_size, 5, dtype=torch.float64)
    print(f"Input shape: {x_data.shape}")
    
    # Set input and forward
    # Note: Need to provide input for each input node
    x_inputs = [x_data] * len(bmpo.input_nodes)
    
    print("\n" + "=" * 70)
    print("Bayesian Tensor Train Example Complete!")
    print("=" * 70)
    
    # Summary
    bmpo.summary()


if __name__ == "__main__":
    main()
