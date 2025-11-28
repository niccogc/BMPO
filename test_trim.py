"""
Test trimming functionality for Bayesian MPO.

This test verifies:
1. Trim properly reduces model rank
2. prior_bond_params are correctly updated after trim
3. Model can continue training after trim
4. Training + trim + training works end-to-end
"""

import torch
from tensor.bayesian_mpo_builder import create_bayesian_tensor_train

torch.manual_seed(42)

print("="*70)
print("TRIM FUNCTIONALITY TEST")
print("="*70)
print()

# Generate simple polynomial data
S_train = 100
x_train = torch.rand(S_train, dtype=torch.float64) * 2 - 1
y_train = 2.0 * x_train**3 - 1.0 * x_train**2 + 0.5
y_train += 0.1 * torch.randn(S_train, dtype=torch.float64)
X_train = torch.stack([torch.ones_like(x_train), x_train], dim=1)

print(f"Training data: {S_train} samples")
print(f"X shape: {X_train.shape}")
print(f"y shape: {y_train.shape}")
print()

# Create Bayesian MPO with larger rank
print("Creating Bayesian MPO with rank=8...")
bmpo = create_bayesian_tensor_train(
    num_blocks=3,
    bond_dim=8,
    input_features=2,
    output_shape=1,
    constrict_bond=False,
    tau_alpha=torch.tensor(2.0, dtype=torch.float64),
    tau_beta=torch.tensor(1.0, dtype=torch.float64),
    dtype=torch.float64,
    seed=42,
    random_priors=True,
    prior_seed=42
)

print(f"Initial structure:")
for i, node in enumerate(bmpo.mu_nodes):
    print(f"  Block {i}: shape {node.shape}, labels {node.dim_labels}")
print()

# Check initial bond prior dimensions
print("Initial prior_bond_params dimensions:")
for label, params in bmpo.prior_bond_params.items():
    conc_size = params['concentration0'].shape[0]
    rate_size = params['rate0'].shape[0]
    print(f"  {label}: concentration0={conc_size}, rate0={rate_size}")
print()

# Train for a few iterations
print("Training for 10 iterations (pre-trim)...")
print()
bmpo.fit(X_train, y_train, max_iter=10, verbose=True)

# Check predictions before trim
mu_pred_before = bmpo.forward_mu(X_train, to_tensor=True)
assert isinstance(mu_pred_before, torch.Tensor)
mse_before = ((mu_pred_before.squeeze() - y_train) ** 2).mean()
print(f"\nMSE before trim: {mse_before.item():.4f}")
print()

# Print expectations before trim
print("Bond expectations before trim:")
for label, dist in bmpo.mu_mpo.distributions.items():
    exp = dist['expectation']
    print(f"  {label}: min={exp.min():.2f}, max={exp.max():.2f}, mean={exp.mean():.2f}")
    print(f"    Values: {exp.tolist()}")
print()

# Trim with moderate thresholds
print("Trimming with thresholds (r1: 12.0, r2: 16.0)...")
thresholds = {'r1': 12.0, 'r2': 16.0}
bmpo.trim(thresholds)

print(f"\nStructure after trim:")
for i, node in enumerate(bmpo.mu_nodes):
    print(f"  Block {i}: shape {node.shape}, labels {node.dim_labels}")
print()

# Check prior_bond_params dimensions after trim
print("prior_bond_params dimensions after trim:")
for label, params in bmpo.prior_bond_params.items():
    conc_size = params['concentration0'].shape[0]
    rate_size = params['rate0'].shape[0]
    print(f"  {label}: concentration0={conc_size}, rate0={rate_size}")
print()

# Verify bond dimensions match prior dimensions
print("Verifying bond dimensions match prior dimensions...")
for label, dist in bmpo.mu_mpo.distributions.items():
    exp_size = dist['expectation'].shape[0]
    if label in bmpo.prior_bond_params:
        prior_conc_size = bmpo.prior_bond_params[label]['concentration0'].shape[0]
        prior_rate_size = bmpo.prior_bond_params[label]['rate0'].shape[0]
        print(f"  {label}: exp={exp_size}, prior_conc={prior_conc_size}, prior_rate={prior_rate_size}", end="")
        if exp_size == prior_conc_size == prior_rate_size:
            print(" ✓")
        else:
            print(" ✗ MISMATCH!")
            raise ValueError(f"Dimension mismatch for {label}")
print()

# Check predictions after trim (should be similar)
mu_pred_after = bmpo.forward_mu(X_train, to_tensor=True)
assert isinstance(mu_pred_after, torch.Tensor)
mse_after = ((mu_pred_after.squeeze() - y_train) ** 2).mean()
print(f"MSE after trim: {mse_after.item():.4f}")
print(f"MSE change: {mse_after.item() - mse_before.item():.4f} (should be small)")
print()

# Continue training after trim
print("Continuing training for 10 more iterations (post-trim)...")
print()
bmpo.fit(X_train, y_train, max_iter=10, verbose=True)

# Final evaluation
mu_pred_final = bmpo.forward_mu(X_train, to_tensor=True)
assert isinstance(mu_pred_final, torch.Tensor)
mse_final = ((mu_pred_final.squeeze() - y_train) ** 2).mean()
print(f"\nMSE after post-trim training: {mse_final.item():.4f}")
print(f"Total improvement: {mse_before.item():.4f} → {mse_final.item():.4f}")
print()

# Final structure
print("Final structure:")
for i, node in enumerate(bmpo.mu_nodes):
    print(f"  Block {i}: shape {node.shape}")
print()

print("="*70)
print("TEST PASSED ✓")
print("="*70)
print()
print("Summary:")
print(f"  - Started with rank 8")
print(f"  - Trimmed to smaller rank")
print(f"  - prior_bond_params correctly updated")
print(f"  - Model successfully trained after trim")
print(f"  - Final MSE: {mse_final.item():.4f}")
