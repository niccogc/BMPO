"""
Example usage of BMPO (Bayesian Matrix Product Operator) module.

Each mode has Gamma distributions (one per index):
- Rank dimensions: Gamma(c, e) with variable ω
- Vertical dimensions: Gamma(f, g) with variable φ
"""

import torch
from tensor.node import TensorNode
from tensor.bmpo import BMPONetwork


def main():
    print("=" * 70)
    print("BMPO Network with Gamma Distributions")
    print("=" * 70)
    
    # Example 1: Create a simple network
    print("\n1. Creating a BMPO Network")
    print("-" * 70)
    
    # Create nodes
    node1 = TensorNode(
        tensor_or_shape=(3, 4, 2),
        dim_labels=['r0', 'r1', 'f'],
        l='r0', r='r1',  # r0 and r1 are rank dimensions
        name='node1'
    )
    
    node2 = TensorNode(
        tensor_or_shape=(4, 3, 2),
        dim_labels=['r1', 'r2', 'f'],
        l='r1', r='r2',  # r1 and r2 are rank dimensions
        name='node2'
    )
    
    # Connect nodes
    node1.connect(node2, 'r1')
    node1.connect(node2, 'f')
    
    # Create network (automatically infers rank dimensions from left_labels/right_labels)
    network = BMPONetwork(
        input_nodes=[],
        main_nodes=[node1, node2]
    )
    
    print(f"Network created with {len(network.main_nodes)} nodes")
    print(f"Rank dimensions (horizontal bonds): {network.rank_labels}")
    
    # Show distribution structure
    print("\nDistribution structure:")
    for label, dist in network.distributions.items():
        print(f"  {label} (size {len(dist['expectation'])}):")
        print(f"    Type: {dist['type']}, Variable: {dist['variable']}")
        if dist['type'] == 'rank':
            print(f"    Gamma(c, e): c={dist['c'][:3]}..., e={dist['e'][:3]}...")
        else:
            print(f"    Gamma(f, g): f={dist['f']}, g={dist['g']}")
        print(f"    Expectations: {dist['expectation']}")
    
    # Example 2: Update distribution parameters
    print("\n2. Updating Distribution Parameters")
    print("-" * 70)
    
    print(f"Before: r1 has {len(network.get_expectations('r1'))} Gamma distributions")
    print(f"  Expectations E[ω]: {network.get_expectations('r1')}")
    
    # Update parameters for r1 dimension
    new_c = torch.tensor([3.0, 1.0, 2.0, 4.0])
    new_e = torch.tensor([1.0, 2.0, 1.0, 1.0])
    network.update_distribution_params('r1', param1=new_c, param2=new_e)
    
    print(f"After updating c and e:")
    print(f"  New c: {network.distributions['r1']['c']}")
    print(f"  New e: {network.distributions['r1']['e']}")
    print(f"  Expectations E[ω] = c/e: {network.get_expectations('r1')}")
    
    # Example 3: Get Gamma distribution objects
    print("\n3. Get Gamma Distribution Objects")
    print("-" * 70)
    
    gammas = network.get_gamma_distributions('r1')
    print(f"r1 has {len(gammas)} Gamma distributions:")
    for i, gamma in enumerate(gammas[:2]):  # Show first 2
        print(f"  Index {i}: Gamma(c={gamma.concentration}, e={gamma.rate})")
        print(f"           Entropy: {gamma.entropy():.4f}")
    
    # Example 4: Compute entropy
    print("\n4. Computing Total Entropy")
    print("-" * 70)
    
    entropy_r1 = network.compute_entropy('r1')
    entropy_f = network.compute_entropy('f')
    
    print(f"Total entropy for r1: {entropy_r1:.4f}")
    print(f"Total entropy for f: {entropy_f:.4f}")
    
    # Example 5: Trim network
    print("\n5. Trimming Network")
    print("-" * 70)
    
    print("Before trimming:")
    print(f"  node1 shape: {node1.shape}")
    print(f"  node2 shape: {node2.shape}")
    print(f"  r1 expectations: {network.get_expectations('r1')}")
    print(f"  f expectations: {network.get_expectations('f')}")
    
    # Trim based on expectation values
    network.trim({
        'r1': 2.0,  # Keep indices where E[ω] >= 2.0
        'f': 1.5    # Keep indices where E[φ] >= 1.5
    })
    
    print("\nAfter trimming (r1>=2.0, f>=1.5):")
    print(f"  node1 shape: {node1.shape}")
    print(f"  node2 shape: {node2.shape}")
    print(f"  r1 expectations: {network.get_expectations('r1')}")
    print(f"  r1 c params: {network.distributions['r1']['c']}")
    print(f"  r1 e params: {network.distributions['r1']['e']}")
    print(f"  f expectations: {network.get_expectations('f')}")
    
    print("\n" + "=" * 70)
    print("BMPO Network Example Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
