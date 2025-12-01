"""
Detailed debugging: Find exactly where block values collapse to zero.

Prints every intermediate value in the block update to pinpoint the collapse.
"""
import torch
import sys
from tensor.bayesian_mpo_builder import create_bayesian_tensor_train

# Configuration
NUM_BLOCKS = 4
BOND_DIM = 4
NUM_SAMPLES = 100

torch.manual_seed(42)

# Generate data
x = torch.rand(NUM_SAMPLES, dtype=torch.float64) * 2 - 1
y = 2.0 * x**3 - 1.0 * x**2 + 0.5
y += 0.1 * torch.randn(NUM_SAMPLES, dtype=torch.float64)
X = torch.stack([torch.ones_like(x), x], dim=1)

print("="*80)
print(f"DETAILED BLOCK COLLAPSE DEBUG: {NUM_BLOCKS} blocks, bond_dim={BOND_DIM}")
print("="*80)
print()

# Create model
bmpo = create_bayesian_tensor_train(
    num_blocks=NUM_BLOCKS,
    bond_dim=BOND_DIM,
    input_features=2,
    output_shape=1,
    dtype=torch.float64,
    seed=42,
    random_priors=True,
    prior_seed=42
)

print("Initial block statistics:")
for i, node in enumerate(bmpo.mu_nodes):
    print(f"  Block {i}: mean={node.tensor.mean():.6e}, std={node.tensor.std():.6e}, "
          f"min={node.tensor.min():.6e}, max={node.tensor.max():.6e}")
print()

# Initial forward pass
mu_init = bmpo.forward_mu(X, to_tensor=True)
print(f"Initial predictions: mean={mu_init.mean():.6e}, std={mu_init.std():.6e}")
print()

# Now manually trace through ONE block update to see where it breaks
print("="*80)
print("TRACING FIRST BLOCK UPDATE (Block 0)")
print("="*80)
print()

block_idx = 0
mu_node = bmpo.mu_nodes[block_idx]
sigma_node = bmpo.sigma_nodes[block_idx]

print(f"1. E[τ]:")
E_tau = bmpo.get_tau_mean()
print(f"   E[τ] = {E_tau:.6e}")
print()

print(f"2. Block dimensions:")
d = mu_node.tensor.numel()
mu_shape = mu_node.shape
print(f"   d = {d}, mu_shape = {mu_shape}")
print()

print(f"3. Theta (prior precision):")
theta = bmpo.compute_theta_tensor(block_idx)
theta_flat = theta.flatten()
print(f"   theta: min={theta_flat.min():.6e}, max={theta_flat.max():.6e}, mean={theta_flat.mean():.6e}")
print(f"   theta sum: {theta_flat.sum():.6e}")
print()

print(f"4. Forward μ-MPO:")
mu_output = bmpo.forward_mu(X, to_tensor=False)
bmpo.mu_mpo.output_labels = tuple(mu_output.dim_labels)
print(f"   mu_output: min={mu_output.tensor.min():.6e}, max={mu_output.tensor.max():.6e}")
print()

print(f"5. Prepare y_expanded:")
y_expanded = y
for _ in range(len(mu_output.shape) - 1):
    y_expanded = y_expanded.unsqueeze(-1)
print(f"   y: min={y.min():.6e}, max={y.max():.6e}, mean={y.mean():.6e}")
print()

print(f"6. Compute J_μ term (sum_y_dot_J_mu):")
sum_y_dot_J_mu = bmpo.mu_mpo.get_b(mu_node, y_expanded)
sum_y_dot_J_mu_flat = sum_y_dot_J_mu.flatten()
print(f"   sum_y_dot_J_mu: min={sum_y_dot_J_mu_flat.min():.6e}, max={sum_y_dot_J_mu_flat.max():.6e}")
print(f"   sum_y_dot_J_mu abs_sum: {sum_y_dot_J_mu_flat.abs().sum():.6e}")
print()

print(f"7. Compute J_μ outer product:")
J_mu_outer = bmpo._compute_mu_jacobian_outer(mu_node, bmpo.mu_mpo, mu_output.shape)
n_dims = len(mu_shape)
J_mu_outer_flat = J_mu_outer.flatten(0, n_dims-1).flatten(1, -1)
print(f"   J_mu_outer: shape={J_mu_outer_flat.shape}")
print(f"   J_mu_outer: min={J_mu_outer_flat.min():.6e}, max={J_mu_outer_flat.max():.6e}")
print(f"   J_mu_outer diagonal: min={J_mu_outer_flat.diagonal().min():.6e}, max={J_mu_outer_flat.diagonal().max():.6e}")
print(f"   J_mu_outer trace: {J_mu_outer_flat.diagonal().sum():.6e}")
print()

print(f"8. Compute J_Σ term:")
sigma_output = bmpo.forward_sigma(X, to_tensor=False)
bmpo.sigma_mpo.output_labels = tuple(sigma_output.dim_labels)
print(f"   sigma_output: min={sigma_output.tensor.min():.6e}, max={sigma_output.tensor.max():.6e}")

J_sigma_sum = bmpo.sigma_mpo.get_b(sigma_node, grad=None, is_sigma=True, output_shape=sigma_output.shape)
print(f"   J_sigma_sum: min={J_sigma_sum.min():.6e}, max={J_sigma_sum.max():.6e}")

outer_indices, inner_indices = bmpo._get_sigma_to_mu_permutation(block_idx)
perm = outer_indices + inner_indices
J_sigma_permuted = J_sigma_sum.permute(*perm)
sum_J_sigma = J_sigma_permuted.reshape(d, d)
print(f"   sum_J_sigma: min={sum_J_sigma.min():.6e}, max={sum_J_sigma.max():.6e}")
print(f"   sum_J_sigma diagonal: min={sum_J_sigma.diagonal().min():.6e}, max={sum_J_sigma.diagonal().max():.6e}")
print(f"   sum_J_sigma trace: {sum_J_sigma.diagonal().sum():.6e}")
print()

print(f"9. Compute Σ^(-1) = E[τ] * (sum_J_sigma + J_mu_outer) + diag(theta):")
data_term = sum_J_sigma + J_mu_outer_flat
print(f"   data_term: min={data_term.min():.6e}, max={data_term.max():.6e}")
print(f"   data_term trace: {data_term.diagonal().sum():.6e}")

sigma_inv = E_tau * data_term
print(f"   E[τ] * data_term: min={sigma_inv.min():.6e}, max={sigma_inv.max():.6e}")
print(f"   E[τ] * data_term trace: {sigma_inv.diagonal().sum():.6e}")

sigma_inv.diagonal().add_(theta_flat)
print(f"   After adding diag(theta):")
print(f"     sigma_inv: min={sigma_inv.min():.6e}, max={sigma_inv.max():.6e}")
print(f"     sigma_inv diagonal: min={sigma_inv.diagonal().min():.6e}, max={sigma_inv.diagonal().max():.6e}")
print(f"     sigma_inv trace: {sigma_inv.diagonal().sum():.6e}")
print(f"     sigma_inv condition number: {torch.linalg.cond(sigma_inv):.6e}")
print()

print(f"10. Compute Σ = (Σ^(-1))^(-1):")
rhs = E_tau * sum_y_dot_J_mu_flat
print(f"   rhs: min={rhs.min():.6e}, max={rhs.max():.6e}, abs_sum={rhs.abs().sum():.6e}")

try:
    L = torch.linalg.cholesky(sigma_inv)
    print(f"   ✓ Cholesky decomposition succeeded")
    print(f"   L diagonal: min={L.diagonal().min():.6e}, max={L.diagonal().max():.6e}")
    
    sigma_cov = torch.cholesky_inverse(L)
    print(f"   sigma_cov: min={sigma_cov.min():.6e}, max={sigma_cov.max():.6e}")
    print(f"   sigma_cov diagonal: min={sigma_cov.diagonal().min():.6e}, max={sigma_cov.diagonal().max():.6e}")
    print(f"   sigma_cov trace: {sigma_cov.diagonal().sum():.6e}")
    print(f"   sigma_cov condition number: {torch.linalg.cond(sigma_cov):.6e}")
    
    mu_flat = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
except RuntimeError as e:
    print(f"   ✗ Cholesky failed: {e}")
    print(f"   Falling back to torch.inverse...")
    sigma_cov = torch.inverse(sigma_inv)
    mu_flat = torch.matmul(sigma_cov, rhs)

print()

print(f"11. Final μ update:")
print(f"   μ_flat: min={mu_flat.min():.6e}, max={mu_flat.max():.6e}, abs_sum={mu_flat.abs().sum():.6e}")
mu_new = mu_flat.reshape(mu_shape)
print(f"   μ_new: min={mu_new.min():.6e}, max={mu_new.max():.6e}")
print()

print("="*80)
print("COMPARISON:")
print("="*80)
print(f"BEFORE update: Block 0 abs_sum = {mu_node.tensor.abs().sum():.6e}")
print(f"AFTER update:  Block 0 abs_sum = {mu_new.abs().sum():.6e}")
ratio = mu_new.abs().sum() / mu_node.tensor.abs().sum()
print(f"RATIO: {ratio:.6e}")
print()

if ratio < 0.01:
    print("⚠️  WARNING: Block collapsed by >99%!")
elif ratio < 0.1:
    print("⚠️  WARNING: Block shrank significantly!")
else:
    print("✓ Block update looks reasonable")
