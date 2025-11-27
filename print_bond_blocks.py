"""
Print bond-to-block associations for μ-MPO and Σ-MPO.
Excludes dummy indices (dimensions of size 1).
"""

import torch
from tensor.bayesian_mpo_builder import create_bayesian_tensor_train


def print_bond_blocks():
    bmpo = create_bayesian_tensor_train(
        num_blocks=3,
        bond_dim=4,
        input_features=5,
        output_shape=1,
        seed=42
    )
    
    print("=" * 60)
    print("BOND-TO-BLOCK ASSOCIATIONS")
    print("=" * 60)
    print("\nNote: Dimensions of size 1 are dummy indices (no Gamma dist)")
    
    for label in sorted(bmpo.mu_mpo.distributions.keys()):
        nodes_info = bmpo.get_nodes_for_bond(label)
        mu_nodes = nodes_info['mu_nodes']
        sigma_nodes = nodes_info['sigma_nodes']
        
        dist = bmpo.mu_mpo.distributions[label]
        size = len(dist['concentration'])
        
        print(f"\nBond '{label}' (size={size}):")
        print(f"  μ blocks:  {mu_nodes}")
        print(f"  Σ blocks:  {sigma_nodes}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print_bond_blocks()
