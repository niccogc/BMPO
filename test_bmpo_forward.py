"""
Test that BMPO forward pass works correctly with input sharing.
"""
import torch
from tensor.bayesian_mpo_builder import create_bayesian_tensor_train

# Create a Bayesian TT
bmpo = create_bayesian_tensor_train(
    num_blocks=3,
    bond_dim=4,
    input_features=5,
    output_shape=1,
    seed=42
)

print("BEFORE forward pass:")
print(f"  μ-MPO input nodes: {len(bmpo.input_nodes)}")
if bmpo.input_nodes:
    first_ptr = bmpo.input_nodes[0].tensor.data_ptr()
    for i, node in enumerate(bmpo.input_nodes):
        shares = node.tensor.data_ptr() == first_ptr
        status = "✓ SHARES" if shares else "✗ DIFFERENT"
        print(f"    Node {i}: {status}")

# Create input data (use float64 to match default dtype)
x = torch.randn(10, 5, dtype=torch.float64)
print(f"\nInput x data_ptr: {x.data_ptr()}")

# Forward pass μ-MPO
print(f"\nμ-MPO forward pass (using forward_mu):")
mu_output = bmpo.forward_mu(x, to_tensor=True)
print(f"  Output shape: {mu_output.shape}")

print(f"\nAFTER μ-MPO forward, checking μ input nodes:")
if bmpo.input_nodes:
    first_ptr = bmpo.input_nodes[0].tensor.data_ptr()
    x_ptr = x.data_ptr()
    all_share = True
    for i, node in enumerate(bmpo.input_nodes):
        shares_first = node.tensor.data_ptr() == first_ptr
        shares_x = node.tensor.data_ptr() == x_ptr
        print(f"  Node {i}: shares with node[0]: {shares_first}, shares with x: {shares_x}")
        if not (shares_first and shares_x):
            all_share = False
    
    if all_share:
        print(f"  ✓ All μ-MPO input nodes share memory with input x!")
    else:
        print(f"  ✗ μ-MPO input nodes DO NOT all share memory (PROBLEM!)")

# Forward pass Σ-MPO
print(f"\nΣ-MPO forward pass (using forward_sigma):")
sigma_output = bmpo.forward_sigma(to_tensor=True)
print(f"  Output shape: {sigma_output.shape}")
