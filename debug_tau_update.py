"""Debug script to understand τ update behavior."""
import torch
from tensor.bayesian_mpo_builder import create_bayesian_tensor_train

# Create simple 1D polynomial: y = 2x^3 - x^2 + 0.5
torch.manual_seed(42)
x = torch.linspace(-1, 1, 20, dtype=torch.float64)
y = 2.0 * x**3 - 1.0 * x**2 + 0.5
X = torch.stack([torch.ones_like(x), x], dim=1)  # Add bias

# Create small BMPO
bmpo = create_bayesian_tensor_train(
    num_blocks=2,
    bond_dim=4,
    input_features=2,
    output_shape=1,
    tau_alpha=torch.tensor(1.0, dtype=torch.float64),
    tau_beta=torch.tensor(1.0, dtype=torch.float64),
    dtype=torch.float64
)

print("=" * 70)
print("DEBUGGING τ UPDATE")
print("=" * 70)
print(f"Data: {len(X)} samples")
print(f"y range: [{y.min():.4f}, {y.max():.4f}]")
print(f"y² sum: {torch.sum(y**2):.4f}")
print()

# Get initial predictions
mu_init = bmpo.forward_mu(X, to_tensor=True)
sigma_init = bmpo.forward_sigma(X, to_tensor=True)

print("INITIAL STATE:")
print(f"  μ range: [{mu_init.min():.4f}, {mu_init.max():.4f}]")
print(f"  μ² sum: {torch.sum(mu_init**2):.4f}")
print(f"  Σ sum: {torch.sum(sigma_init):.4f}")
print(f"  E[τ] = {bmpo.get_tau_mean().item():.4f}")
print()

# One iteration of updates
print("AFTER 1 BLOCK UPDATE:")
bmpo.update_block_variational(0, X, y)
bmpo.update_block_variational(1, X, y)

mu_after = bmpo.forward_mu(X, to_tensor=True)
sigma_after = bmpo.forward_sigma(X, to_tensor=True)

print(f"  μ range: [{mu_after.min():.4f}, {mu_after.max():.4f}]")
print(f"  μ² sum: {torch.sum(mu_after**2):.4f}")
print(f"  Σ sum: {torch.sum(sigma_after):.4f}")
print()

# Manual τ update computation
print("τ UPDATE COMPUTATION:")
S = len(X)
alpha_q = bmpo.prior_tau_alpha + S / 2.0
print(f"  α_q = {bmpo.prior_tau_alpha.item():.4f} + {S}/2 = {alpha_q.item():.4f}")

beta_q = bmpo.prior_tau_beta.clone()
print(f"  β_q starts at β_p = {beta_q.item():.4f}")

term_y_squared = 0.5 * torch.sum(y**2)
beta_q += term_y_squared
print(f"  + 0.5*Σy² = +{term_y_squared.item():.4f} → β_q = {beta_q.item():.4f}")

term_y_mu = torch.sum(y.reshape(-1, 1) * mu_after.reshape(-1, 1))
beta_q -= term_y_mu
print(f"  - Σ(y·μ) = -{term_y_mu.item():.4f} → β_q = {beta_q.item():.4f}")

term_second_moment = 0.5 * torch.sum(sigma_after + mu_after**2)
beta_q += term_second_moment
print(f"  + 0.5*Σ(Σ+μ²) = +{term_second_moment.item():.4f} → β_q = {beta_q.item():.4f}")

print()
print(f"  Final: α_q = {alpha_q.item():.4f}, β_q = {beta_q.item():.4f}")
print(f"  E[τ] = α_q/β_q = {(alpha_q/beta_q).item():.4f}")
print()

# Now actually update
bmpo.update_tau_variational(X, y)
print(f"ACTUAL E[τ] after update: {bmpo.get_tau_mean().item():.4f}")
print("=" * 70)
