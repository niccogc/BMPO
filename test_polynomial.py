"""
Test Bayesian MPO on polynomial regression.

Problem: Learn y = a*x^3 + b*x^2 + c from data
Input: [1, x] where x ~ Uniform(-1, 1)
"""

import torch
import matplotlib.pyplot as plt
from tensor.bayesian_mpo_builder import create_bayesian_tensor_train

# Set seed
torch.manual_seed(42)

# True polynomial coefficients
a_true = 2.0
b_true = -1.0
c_true = 0.5

print("="*70)
print("POLYNOMIAL REGRESSION TEST")
print("="*70)
print(f"True function: y = {a_true}*x^3 + {b_true}*x^2 + {c_true}")
print("="*70)
print()

# Generate data
S_train = 100
x_train = torch.rand(S_train, dtype=torch.float64) * 2 - 1  # Uniform(-1, 1)
y_train = a_true * x_train**3 + b_true * x_train**2 + c_true
y_train += 0.1 * torch.randn(S_train, dtype=torch.float64)  # Add noise

# Input format: [1, x]
X_train = torch.stack([torch.ones_like(x_train), x_train], dim=1)  # (S, 2)

print(f"Training data: {S_train} samples")
print(f"X shape: {X_train.shape}  (bias, x)")
print(f"y shape: {y_train.shape}")
print(f"x range: [{x_train.min():.2f}, {x_train.max():.2f}]")
print(f"y range: [{y_train.min():.2f}, {y_train.max():.2f}]")
print()

# Create Bayesian MPO
# We'll use 3 blocks now
print("Creating Bayesian MPO with random priors...")
bmpo = create_bayesian_tensor_train(
    num_blocks=3,
    bond_dim=8,
    input_features=2,  # [1, x]
    output_shape=1,
    constrict_bond=False,
    tau_alpha=torch.tensor(2.0, dtype=torch.float64),
    tau_beta=torch.tensor(1.0, dtype=torch.float64),
    dtype=torch.float64,
    seed=42,
    random_priors=True,
    prior_seed=42
)

print(f"Created BMPO with {len(bmpo.mu_nodes)} blocks:")
for i, node in enumerate(bmpo.mu_nodes):
    print(f"  Block {i}: shape {node.shape}, labels {node.dim_labels}")
print()

# Check initial μ predictions to verify normalization
print("Checking initial μ predictions...")
mu_init = bmpo.forward_mu(X_train, to_tensor=True)
assert isinstance(mu_init, torch.Tensor)
print(f"  Initial μ range: [{mu_init.min():.4f}, {mu_init.max():.4f}]")
print(f"  Initial μ mean: {mu_init.mean():.4f}")
print(f"  Initial μ std: {mu_init.std():.4f}")
print(f"  Target y range: [{y_train.min():.4f}, {y_train.max():.4f}]")
print(f"  Target y mean: {y_train.mean():.4f}")
print(f"  Target y std: {y_train.std():.4f}")
print()

# Train
print("Training...")
print()
bmpo.fit(X_train, y_train, max_iter=20, verbose=True)

print()
print("="*70)
print("EVALUATION")
print("="*70)

# Test on grid
S_test = 50
x_test = torch.linspace(-1, 1, S_test, dtype=torch.float64)
y_test_true = a_true * x_test**3 + b_true * x_test**2 + c_true
X_test = torch.stack([torch.ones_like(x_test), x_test], dim=1)

# Predict
mu_pred_node = bmpo.forward_mu(X_test, to_tensor=False)
sigma_pred_node = bmpo.forward_sigma(X_test, to_tensor=False)

mu_pred = mu_pred_node.tensor.squeeze()
sigma_pred = sigma_pred_node.tensor.squeeze()

# Compute predictive variance (aleatoric + epistemic)
E_tau = bmpo.get_tau_mean()
total_var = sigma_pred + 1.0 / E_tau
total_std = torch.sqrt(total_var)

# Metrics
mse = ((mu_pred - y_test_true) ** 2).mean()
print(f"\nTest MSE: {mse.item():.4f}")
print(f"E[τ]: {E_tau.item():.4f} (noise precision)")
print(f"Mean predictive std: {total_std.mean().item():.4f}")
print()

# Plot
plt.figure(figsize=(12, 5))

# Plot 1: Predictions
plt.subplot(1, 2, 1)
plt.scatter(x_train.numpy(), y_train.numpy(), alpha=0.5, s=20, label='Training data')
plt.plot(x_test.numpy(), y_test_true.numpy(), 'k-', linewidth=2, label='True function')
plt.plot(x_test.numpy(), mu_pred.numpy(), 'r-', linewidth=2, label='Predicted mean')
plt.fill_between(
    x_test.numpy(),
    (mu_pred - 2*total_std).numpy(),
    (mu_pred + 2*total_std).numpy(),
    alpha=0.3,
    color='red',
    label='±2σ confidence'
)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Residuals
plt.subplot(1, 2, 2)
residuals = mu_pred - y_test_true
plt.scatter(x_test.numpy(), residuals.numpy(), alpha=0.7, s=30)
plt.axhline(y=0, color='k', linestyle='--', linewidth=1)
plt.fill_between(
    x_test.numpy(),
    (-2*total_std).numpy(),
    (2*total_std).numpy(),
    alpha=0.3,
    color='gray'
)
plt.xlabel('x')
plt.ylabel('Residual')
plt.title('Prediction Residuals')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('polynomial_regression_test.png', dpi=150, bbox_inches='tight')
print("Plot saved to: polynomial_regression_test.png")
print()
print("="*70)
