"""Debug ELBO likelihood computation."""
import torch
import numpy as np
from tensor.bayesian_mpo_builder import create_bayesian_tensor_train

# Generate simple data
torch.manual_seed(42)
x = torch.rand(20, dtype=torch.float64) * 2 - 1
y = 2.0 * x**3 - 1.0 * x**2 + 0.5
X = torch.stack([torch.ones_like(x), x], dim=1)

print("=" * 70)
print("DEBUGGING E_q[log p(y|θ,τ)]")
print("=" * 70)
print(f"Data: {len(X)} samples")
print()

# Create BMPO
bmpo = create_bayesian_tensor_train(
    num_blocks=2,
    bond_dim=4,
    input_features=2,
    output_shape=1,
    tau_alpha=torch.tensor(2.0, dtype=torch.float64),
    tau_beta=torch.tensor(1.0, dtype=torch.float64),
    dtype=torch.float64,
    seed=42
)

# Train for 1 iteration
for i in range(len(bmpo.mu_nodes)):
    bmpo.update_block_variational(i, X, y)
bmpo.update_tau_variational(X, y)

S = len(X)

# Get predictions
mu_output = bmpo.forward_mu(X, to_tensor=True)
sigma_output = bmpo.forward_sigma(X, to_tensor=True)

# Flatten
mu_pred = mu_output.reshape(S, -1).squeeze(-1)  # (S,)
sigma_pred = sigma_output.reshape(S, -1).squeeze(-1)  # (S,)

# E[τ]
E_tau = bmpo.get_tau_mean()

print("MANUAL CALCULATION:")
print(f"E[τ] = {E_tau.item():.6f}")
print()

# Manual calculation of E_q[log p(y|θ,τ)]
# log p(y|θ,τ) = Σ_s log N(y_s | μ(x_s), τ^{-1})
# = Σ_s [-1/2 log(2π) + 1/2 log(τ) - τ/2 (y_s - μ(x_s))^2]
# Taking expectation over q:
# E_q[log p(y|θ,τ)] = Σ_s [-1/2 log(2π) + 1/2 E[log τ] - E[τ]/2 E_q[(y_s - f(x_s))^2]]
# where E_q[(y_s - f(x_s))^2] = (y_s - μ(x_s))^2 + Σ(x_s)

print("Term by term:")
print()

# Term 1: -S/2 * log(2π)
term1 = -0.5 * S * torch.log(torch.tensor(2 * np.pi, dtype=torch.float64))
print(f"1. -S/2 * log(2π) = -{S}/2 * log(2π) = {term1.item():.4f}")

# Term 2: S/2 * E[log τ]
# For Gamma(α, β), E[log τ] = ψ(α) - log(β)
E_log_tau = torch.digamma(bmpo.tau_alpha) - torch.log(bmpo.tau_beta)
term2 = 0.5 * S * E_log_tau
print(f"2. S/2 * E[log τ] = {S}/2 * {E_log_tau.item():.4f} = {term2.item():.4f}")

# Term 3: -E[τ]/2 * Σ_s [(y_s - μ(x_s))^2 + Σ(x_s)]
residuals_sq = (y - mu_pred) ** 2
sum_residuals = torch.sum(residuals_sq)
sum_sigma = torch.sum(sigma_pred)
term3 = -0.5 * E_tau * (sum_residuals + sum_sigma)
print(f"3. -E[τ]/2 * Σ[(y-μ)² + Σ]:")
print(f"   Σ(y-μ)² = {sum_residuals.item():.4f}")
print(f"   ΣΣ = {sum_sigma.item():.4f}")
print(f"   -E[τ]/2 * {(sum_residuals + sum_sigma).item():.4f} = {term3.item():.4f}")

manual_log_likelihood = term1 + term2 + term3
print()
print(f"MANUAL E_q[log p(y|θ,τ)] = {manual_log_likelihood.item():.4f}")
print()

# Now use the code's calculation
print("CODE CALCULATION:")
residuals_sq_code = (y - mu_pred) ** 2
log_likelihood_code = (
    -0.5 * torch.log(torch.tensor(2 * torch.pi, dtype=torch.float64))
    + 0.5 * torch.log(E_tau)
    - 0.5 * E_tau * (residuals_sq_code + sigma_pred)
).sum()

print(f"Code E_q[log p(y|θ,τ)] = {log_likelihood_code.item():.4f}")
print()

# Breakdown of code calculation
term1_code = -0.5 * S * torch.log(torch.tensor(2 * torch.pi, dtype=torch.float64))
term2_code = 0.5 * S * torch.log(E_tau)  # THIS IS DIFFERENT!
term3_code = -0.5 * E_tau * (residuals_sq_code.sum() + sigma_pred.sum())

print("Code term by term:")
print(f"1. -S/2 * log(2π) = {term1_code.item():.4f}")
print(f"2. S/2 * log(E[τ]) = {term2_code.item():.4f}  <- WRONG! Should use E[log τ]")
print(f"3. -E[τ]/2 * Σ[(y-μ)² + Σ] = {term3_code.item():.4f}")
print()

print("=" * 70)
print("ISSUE FOUND:")
print("Code uses log(E[τ]) but should use E[log τ]")
print(f"log(E[τ]) = log({E_tau.item():.6f}) = {torch.log(E_tau).item():.6f}")
print(f"E[log τ] = {E_log_tau.item():.6f}")
print(f"Difference: {(E_log_tau - torch.log(E_tau)).item():.6f}")
print("=" * 70)
