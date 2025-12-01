"""
Debug the block update to find WHERE values become zero.

Print EVERY intermediate value in update_block_variational to pinpoint
the exact step where things go to zero.

Environment variables:
- NUM_BLOCKS: Number of blocks (default: 4)
- BLOCK_IDX: Which block to debug (default: 0)
"""

import torch
import os
from tensor.bayesian_mpo_builder import create_bayesian_tensor_train

torch.manual_seed(42)

NUM_BLOCKS = int(os.environ.get('NUM_BLOCKS', '4'))
BLOCK_IDX = int(os.environ.get('BLOCK_IDX', '0'))
NUM_SAMPLES = 100

# 1D problem
x = torch.rand(NUM_SAMPLES, dtype=torch.float64) * 2 - 1
y = 2.0 * x**3 - 1.0 * x**2 + 0.5
y += 0.1 * torch.randn(NUM_SAMPLES, dtype=torch.float64)
X = torch.stack([torch.ones_like(x), x], dim=1)

print("="*70)
print(f"DEBUG BLOCK {BLOCK_IDX} UPDATE ({NUM_BLOCKS} blocks total)")
print("="*70)
print()

# Create model
bmpo = create_bayesian_tensor_train(
    num_blocks=NUM_BLOCKS,
    bond_dim=6,
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

print(f"Structure:")
for i, node in enumerate(bmpo.mu_nodes):
    print(f"  Block {i}: {node.shape}")
print()

# Now manually execute the update_block_variational code with print statements
print(f"DETAILED UPDATE FOR BLOCK {BLOCK_IDX}")
print("="*70)

mu_node = bmpo.mu_nodes[BLOCK_IDX]
sigma_node = bmpo.sigma_nodes[BLOCK_IDX]

print(f"\n1. Get E[τ]")
E_tau = bmpo.get_tau_mean()
print(f"   E[τ] = {E_tau.item():.6f}")

print(f"\n2. Flatten block shape")
d = mu_node.tensor.numel()
mu_shape = mu_node.shape
print(f"   d = {d}, mu_shape = {mu_shape}")

print(f"\n3. Get Theta tensor (prior precision diagonal)")
theta = bmpo.compute_theta_tensor(BLOCK_IDX)
print(f"   theta.shape = {theta.shape}")
print(f"   theta stats: min={theta.min():.6f}, max={theta.max():.6f}, mean={theta.mean():.6f}")
theta_flat = theta.flatten()
print(f"   theta_flat.shape = {theta_flat.shape}")
print(f"   theta_flat stats: min={theta_flat.min():.6f}, max={theta_flat.max():.6f}, sum={theta_flat.sum():.6f}")

print(f"\n4. Forward pass through μ-MPO")
mu_output = bmpo.forward_mu(X, to_tensor=False)
bmpo.mu_mpo.output_labels = tuple(mu_output.dim_labels)
print(f"   mu_output.shape = {mu_output.shape}")
print(f"   mu_output stats: min={mu_output.tensor.min():.6f}, max={mu_output.tensor.max():.6f}")

print(f"\n5. Prepare y")
y_expanded = y
for _ in range(len(mu_output.shape) - 1):
    y_expanded = y_expanded.unsqueeze(-1)
print(f"   y_expanded.shape = {y_expanded.shape}")
print(f"   y stats: min={y.min():.6f}, max={y.max():.6f}, mean={y.mean():.6f}")

print(f"\n6. Compute J_μ term: Σₙ yₙ · J_μ(xₙ)")
sum_y_dot_J_mu = bmpo.mu_mpo.get_b(mu_node, y_expanded)
print(f"   sum_y_dot_J_mu.shape = {sum_y_dot_J_mu.shape}")
print(f"   sum_y_dot_J_mu stats: min={sum_y_dot_J_mu.min():.6f}, max={sum_y_dot_J_mu.max():.6f}")
print(f"   sum_y_dot_J_mu.abs().sum() = {sum_y_dot_J_mu.abs().sum():.6f}")
sum_y_dot_J_mu_flat = sum_y_dot_J_mu.flatten()
print(f"   sum_y_dot_J_mu_flat.shape = {sum_y_dot_J_mu_flat.shape}")
print(f"   sum_y_dot_J_mu_flat.abs().sum() = {sum_y_dot_J_mu_flat.abs().sum():.6f}")

print(f"\n7. Compute J_μ outer product: Σₙ J_μ(xₙ) ⊗ J_μ(xₙ)")
J_mu_outer = bmpo._compute_mu_jacobian_outer(mu_node, bmpo.mu_mpo, mu_output.shape)
print(f"   J_mu_outer.shape = {J_mu_outer.shape}")
print(f"   J_mu_outer stats: min={J_mu_outer.min():.6f}, max={J_mu_outer.max():.6f}")
n_dims = len(mu_shape)
J_mu_outer_flat = J_mu_outer.flatten(0, n_dims-1).flatten(1, -1)
print(f"   J_mu_outer_flat.shape = {J_mu_outer_flat.shape}")
print(f"   J_mu_outer_flat stats: min={J_mu_outer_flat.min():.6f}, max={J_mu_outer_flat.max():.6f}")
print(f"   J_mu_outer_flat.abs().sum() = {J_mu_outer_flat.abs().sum():.6f}")

print(f"\n8. Compute J_Σ term")
sigma_output = bmpo.forward_sigma(X, to_tensor=False)
bmpo.sigma_mpo.output_labels = tuple(sigma_output.dim_labels)
y_sigma = torch.ones(sigma_output.shape, dtype=mu_node.tensor.dtype, device=mu_node.tensor.device)
J_sigma_sum = bmpo.sigma_mpo.get_b(sigma_node, y_sigma)
print(f"   J_sigma_sum.shape = {J_sigma_sum.shape}")
print(f"   J_sigma_sum stats: min={J_sigma_sum.min():.6f}, max={J_sigma_sum.max():.6f}")
print(f"   J_sigma_sum.abs().sum() = {J_sigma_sum.abs().sum():.6f}")

outer_indices, inner_indices = bmpo._get_sigma_to_mu_permutation(BLOCK_IDX)
perm = outer_indices + inner_indices
J_sigma_permuted = J_sigma_sum.permute(*perm)
sum_J_sigma = J_sigma_permuted.reshape(d, d)
print(f"   sum_J_sigma.shape = {sum_J_sigma.shape}")
print(f"   sum_J_sigma stats: min={sum_J_sigma.min():.6f}, max={sum_J_sigma.max():.6f}")
print(f"   sum_J_sigma.abs().sum() = {sum_J_sigma.abs().sum():.6f}")

print(f"\n9. Compute Σ^(-1) = E[τ] * (sum_J_sigma + J_mu_outer) + diag(theta)")
data_term = sum_J_sigma + J_mu_outer_flat
print(f"   data_term stats: min={data_term.min():.6f}, max={data_term.max():.6f}")
print(f"   data_term.abs().sum() = {data_term.abs().sum():.6f}")

sigma_inv = E_tau * data_term
print(f"   E[τ] * data_term stats: min={sigma_inv.min():.6f}, max={sigma_inv.max():.6f}")
print(f"   E[τ] * data_term.abs().sum() = {sigma_inv.abs().sum():.6f}")

sigma_inv.diagonal().add_(theta_flat)
print(f"   After adding diag(theta):")
print(f"   sigma_inv stats: min={sigma_inv.min():.6f}, max={sigma_inv.max():.6f}")
print(f"   sigma_inv diagonal: min={sigma_inv.diagonal().min():.6f}, max={sigma_inv.diagonal().max():.6f}")
print(f"   sigma_inv.abs().sum() = {sigma_inv.abs().sum():.6f}")

print(f"\n10. Compute Σ = (Σ^(-1))^(-1)")
sigma_cov = torch.inverse(sigma_inv)
print(f"   sigma_cov stats: min={sigma_cov.min():.6f}, max={sigma_cov.max():.6f}")
print(f"   sigma_cov.abs().sum() = {sigma_cov.abs().sum():.6f}")
print(f"   sigma_cov diagonal: min={sigma_cov.diagonal().min():.6f}, max={sigma_cov.diagonal().max():.6f}")

print(f"\n11. Compute μ = E[τ] * Σ * sum_y_dot_J_mu")
product = torch.matmul(sigma_cov, sum_y_dot_J_mu_flat)
print(f"   Σ * sum_y_dot_J_mu stats: min={product.min():.6f}, max={product.max():.6f}")
print(f"   Σ * sum_y_dot_J_mu.abs().sum() = {product.abs().sum():.6f}")

mu_flat = E_tau * product
print(f"   μ_flat = E[τ] * (Σ * sum_y_dot_J_mu)")
print(f"   μ_flat stats: min={mu_flat.min():.6f}, max={mu_flat.max():.6f}")
print(f"   μ_flat.abs().sum() = {mu_flat.abs().sum():.6f}")

print(f"\n12. Reshape and check final result")
mu_new = mu_flat.reshape(mu_shape)
print(f"   μ_new.shape = {mu_new.shape}")
print(f"   μ_new stats: min={mu_new.min():.6f}, max={mu_new.max():.6f}")
print(f"   μ_new.abs().sum() = {mu_new.abs().sum():.6f}")

print(f"\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Block {BLOCK_IDX} tensor BEFORE update: sum={mu_node.tensor.sum():.6f}")
print(f"Block {BLOCK_IDX} tensor AFTER update:  sum={mu_new.sum():.6f}")
print(f"Ratio: {mu_new.sum() / mu_node.tensor.sum():.6e}")
