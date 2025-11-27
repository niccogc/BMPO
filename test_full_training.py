"""
Test full coordinate ascent training loop for Bayesian MPO.
"""

import torch
from tensor.bayesian_mpo_builder import create_bayesian_tensor_train

# Set seed for reproducibility
torch.manual_seed(42)

print("Creating Bayesian Tensor Train...")
bmpo = create_bayesian_tensor_train(
    num_blocks=3,
    bond_dim=4,
    input_features=5,
    output_shape=1,
    constrict_bond=False,
    tau_alpha=torch.tensor(2.0),
    tau_beta=torch.tensor(1.0),
    dtype=torch.float64,
    seed=42
)

print(f"Created BMPO with {len(bmpo.mu_nodes)} blocks")
for i, node in enumerate(bmpo.mu_nodes):
    print(f"  Block {i}: shape {node.shape}, labels {node.dim_labels}")
print()

# Generate synthetic data
S = 50  # number of samples
X = torch.randn(S, 5, dtype=torch.float64)
y = torch.randn(S, dtype=torch.float64)

print(f"Generated data: X shape {X.shape}, y shape {y.shape}")
print()

# Test forward passes before training
print("Testing forward passes before training...")
mu_out = bmpo.forward_mu(X, to_tensor=True)
print(f"μ-MPO output shape: {mu_out.shape}")

sigma_out = bmpo.forward_sigma(X, to_tensor=True)
print(f"Σ-MPO output shape: {sigma_out.shape}")
print()

# Run full training loop
print("="*70)
print("RUNNING FULL TRAINING")
print("="*70)
print()

try:
    bmpo.fit(
        X, y,
        max_iter=5,
        verbose=True
    )
    print("\n✓ Training completed successfully!")
    
except Exception as e:
    print(f"\n✗ Training failed: {e}")
    import traceback
    traceback.print_exc()

# Test forward pass after training
print("\nTesting forward passes after training...")
mu_out_after = bmpo.forward_mu(X, to_tensor=True)
print(f"μ-MPO output shape: {mu_out_after.shape}")

print("\n" + "="*70)
print("Full training test complete!")
print("="*70)
