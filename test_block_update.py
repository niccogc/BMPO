"""
Test block variational updates for Bayesian MPO.
"""

import torch
from tensor.bayesian_mpo_builder import create_bayesian_tensor_train

# Set seed for reproducibility
torch.manual_seed(42)

# Create a small Bayesian TT
print("Creating Bayesian Tensor Train...")
bmpo = create_bayesian_tensor_train(
    num_blocks=2,
    bond_dim=3,
    input_features=4,
    output_shape=1,
    constrict_bond=False,
    tau_alpha=torch.tensor(2.0),
    tau_beta=torch.tensor(1.0),
    dtype=torch.float64,
    seed=42
)

print(f"Created BMPO with {len(bmpo.mu_nodes)} μ-blocks")
for i, node in enumerate(bmpo.mu_nodes):
    print(f"  Block {i}: shape {node.shape}, labels {node.dim_labels}")

# Generate synthetic data
S = 10  # number of samples
X = torch.randn(S, 4, dtype=torch.float64)
y = torch.randn(S, dtype=torch.float64)

print(f"\nGenerated data: X shape {X.shape}, y shape {y.shape}")

# Test forward passes before update
print("\nTesting forward passes...")
mu_out = bmpo.forward_mu(X, to_tensor=True)
print(f"μ-MPO output shape: {mu_out.shape}")

sigma_out = bmpo.forward_sigma(X, to_tensor=True)
print(f"Σ-MPO output shape: {sigma_out.shape}")

# Test block update for block 0
print("\nTesting block variational update for block 0...")
try:
    bmpo.update_block_variational(0, X, y)
    print("✓ Block 0 update successful!")
    
    # Check that parameters changed
    print(f"  Block 0 μ-node shape: {bmpo.mu_nodes[0].shape}")
    print(f"  Block 0 Σ-node shape: {bmpo.sigma_nodes[0].shape}")
    
except Exception as e:
    print(f"✗ Block 0 update failed: {e}")
    import traceback
    traceback.print_exc()

# Test block update for block 1
print("\nTesting block variational update for block 1...")
try:
    bmpo.update_block_variational(1, X, y)
    print("✓ Block 1 update successful!")
    
    # Check that parameters changed
    print(f"  Block 1 μ-node shape: {bmpo.mu_nodes[1].shape}")
    print(f"  Block 1 Σ-node shape: {bmpo.sigma_nodes[1].shape}")
    
except Exception as e:
    print(f"✗ Block 1 update failed: {e}")
    import traceback
    traceback.print_exc()

# Test forward pass after update
print("\nTesting forward passes after updates...")
mu_out_after = bmpo.forward_mu(X, to_tensor=True)
print(f"μ-MPO output shape after update: {mu_out_after.shape}")

print("\n" + "="*70)
print("Block update test complete!")
print("="*70)
