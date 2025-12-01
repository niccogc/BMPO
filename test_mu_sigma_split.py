"""
Test splitting μ and Σ updates.

Strategy:
1. Update ALL μ-blocks (keeping Σ fixed)
2. Update ALL Σ-blocks (using new μ values)

vs original:
- Update μ AND Σ for block 0
- Update μ AND Σ for block 1
- etc.

Environment variables:
- NUM_BLOCKS: Number of blocks (default: 4)
- PROBLEM: '1d' or '2d' (default: '1d')
"""

import torch
import os
from tensor.bayesian_mpo_builder import create_bayesian_tensor_train

torch.manual_seed(42)

NUM_BLOCKS = int(os.environ.get('NUM_BLOCKS', '4'))
PROBLEM = os.environ.get('PROBLEM', '1d')
NUM_SAMPLES = 100

if PROBLEM == '1d':
    INPUT_DIM = 2
    x = torch.rand(NUM_SAMPLES, dtype=torch.float64) * 2 - 1
    y = 2.0 * x**3 - 1.0 * x**2 + 0.5
    y += 0.1 * torch.randn(NUM_SAMPLES, dtype=torch.float64)
    X = torch.stack([torch.ones_like(x), x], dim=1)
else:
    INPUT_DIM = 3
    x = torch.rand(NUM_SAMPLES, dtype=torch.float64) * 2 - 1
    y_coord = torch.rand(NUM_SAMPLES, dtype=torch.float64) * 2 - 1
    y = 0.5 * x**2 + 0.3 * y_coord**2 - 0.4 * x * y_coord + 0.5
    y += 0.05 * torch.randn(NUM_SAMPLES, dtype=torch.float64)
    X = torch.stack([torch.ones_like(x), x, y_coord], dim=1)

print("="*70)
print(f"SPLIT μ/Σ UPDATE TEST: {NUM_BLOCKS} BLOCKS, {PROBLEM.upper()}")
print("="*70)
print()

# Helper to compute MSE
def compute_mse(bmpo, X, y):
    mu_pred = bmpo.forward_mu(X, to_tensor=True)
    assert isinstance(mu_pred, torch.Tensor)
    return ((mu_pred.squeeze() - y)**2).mean().item()

# Method 1: Original (interleaved μ and Σ updates)
print("METHOD 1: Original - update (μ,Σ) block-by-block")
print("-" * 70)

bmpo1 = create_bayesian_tensor_train(
    num_blocks=NUM_BLOCKS, bond_dim=6, input_features=INPUT_DIM,
    output_shape=1, constrict_bond=False,
    tau_alpha=torch.tensor(2.0, dtype=torch.float64),
    tau_beta=torch.tensor(1.0, dtype=torch.float64),
    dtype=torch.float64, seed=42, random_priors=True, prior_seed=42
)

max_iter = 10
print(f"Initial MSE: {compute_mse(bmpo1, X, y):.6f}")

for iteration in range(max_iter):
    # Original: update both μ and Σ for each block sequentially
    for block_idx in range(NUM_BLOCKS):
        bmpo1.update_block_variational(block_idx, X, y)
    
    # Update bonds
    for label in bmpo1.mu_mpo.distributions.keys():
        bmpo1.update_bond_variational(label)
    
    # Update tau
    bmpo1.update_tau_variational(X, y)
    
    mse = compute_mse(bmpo1, X, y)
    tau = bmpo1.get_tau_mean().item()
    mu_pred = bmpo1.forward_mu(X, to_tensor=True)
    assert isinstance(mu_pred, torch.Tensor)
    pred_std = mu_pred.std().item()
    print(f"  Iter {iteration+1:2d}: MSE={mse:.6f}, E[τ]={tau:.2f}, pred_std={pred_std:.6f}")

print(f"Final MSE: {compute_mse(bmpo1, X, y):.6f}")
print()

# Method 2: Split (update all μ first, then all Σ)
print("METHOD 2: Split - update ALL μ-blocks, THEN ALL Σ-blocks")
print("-" * 70)

bmpo2 = create_bayesian_tensor_train(
    num_blocks=NUM_BLOCKS, bond_dim=6, input_features=INPUT_DIM,
    output_shape=1, constrict_bond=False,
    tau_alpha=torch.tensor(2.0, dtype=torch.float64),
    tau_beta=torch.tensor(1.0, dtype=torch.float64),
    dtype=torch.float64, seed=42, random_priors=True, prior_seed=42
)

print(f"Initial MSE: {compute_mse(bmpo2, X, y):.6f}")

# We need to create modified update methods
# For now, let's manually implement the split logic by copying the update code

for iteration in range(max_iter):
    # STAGE 1: Update all μ-blocks with Σ frozen
    # Save current Σ values
    saved_sigmas = [node.tensor.clone() for node in bmpo2.sigma_nodes]
    
    for block_idx in range(NUM_BLOCKS):
        # Update both μ and Σ
        bmpo2.update_block_variational(block_idx, X, y)
        # Immediately restore old Σ (so we only keep μ update)
        bmpo2.sigma_nodes[block_idx].tensor = saved_sigmas[block_idx]
    
    # STAGE 2: Update all Σ-blocks with new μ values
    for block_idx in range(NUM_BLOCKS):
        # Update both μ and Σ
        saved_mu = bmpo2.mu_nodes[block_idx].tensor.clone()
        bmpo2.update_block_variational(block_idx, X, y)
        # Restore μ (so we only keep Σ update)
        bmpo2.mu_nodes[block_idx].tensor = saved_mu
    
    # Update bonds
    for label in bmpo2.mu_mpo.distributions.keys():
        bmpo2.update_bond_variational(label)
    
    # Update tau
    bmpo2.update_tau_variational(X, y)
    
    mse = compute_mse(bmpo2, X, y)
    tau = bmpo2.get_tau_mean().item()
    mu_pred = bmpo2.forward_mu(X, to_tensor=True)
    assert isinstance(mu_pred, torch.Tensor)
    pred_std = mu_pred.std().item()
    print(f"  Iter {iteration+1:2d}: MSE={mse:.6f}, E[τ]={tau:.2f}, pred_std={pred_std:.6f}")

print(f"Final MSE: {compute_mse(bmpo2, X, y):.6f}")
print()

print("="*70)
print("COMPARISON")
print("="*70)
print(f"Method 1 (block-by-block): {compute_mse(bmpo1, X, y):.6f}")
print(f"Method 2 (μ-then-Σ):       {compute_mse(bmpo2, X, y):.6f}")
print()
print("NOTE: Both methods are currently identical because update_block_variational")
print("      updates both μ and Σ together. We need to add methods:")
print("      - update_mu_block_only(block_idx, X, y)")
print("      - update_sigma_block_only(block_idx, X, y)")
print()
print("Let me add these methods to BayesianMPO class...")
