"""
Train Bayesian MPO on polynomial with rank=3 and no trimming.

This script:
1. Trains a Bayesian MPO on polynomial data with rank=3
2. No automatic trimming (threshold=0)
3. Plots predicted mean vs true polynomial
4. Plots predicted variance (epistemic uncertainty)
5. Shows training data points
6. Displays confidence intervals
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from tensor.bayesian_mpo_builder import create_bayesian_tensor_train

torch.manual_seed(293847)

print("="*70)
print("BAYESIAN MPO POLYNOMIAL PREDICTION (RANK=3, NO TRIMMING)")
print("="*70)
print()

# True polynomial coefficients
a_true = 2.0
b_true = -1.0
c_true = 0.5

print(f"True polynomial: y = {a_true}*x³ + {b_true}*x² + {c_true}")
print()

# Generate training data
S_train = 1000
x_train = torch.rand(S_train, dtype=torch.float64) * 2 - 1  # Uniform(-1, 1)
y_train = a_true * x_train**3 + b_true * x_train**2 + c_true
noise_std = 0.1
y_train += noise_std * torch.randn(S_train, dtype=torch.float64)  # Add noise
X_train = torch.stack([torch.ones_like(x_train), x_train], dim=1)

print(f"Training data: {S_train} samples")
print(f"Noise std: {noise_std}")
print(f"X shape: {X_train.shape}")
print(f"y range: [{y_train.min():.2f}, {y_train.max():.2f}]")
print()

# Create Bayesian MPO with rank=3, NO trimming
print("Creating Bayesian MPO (rank=3, no trimming)...")
bmpo = create_bayesian_tensor_train(
    num_blocks=3,
    bond_dim=4,
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
    print(f"  Block {i}: shape {node.shape}")
print()

# Train WITHOUT trimming (threshold=0 means no trimming)
print("Training for 50 iterations with NO trimming...")
print()

max_iter = 50
mse_history = []

for iteration in range(max_iter):
    # Update blocks
    for block_idx in range(len(bmpo.mu_nodes)):
        bmpo.update_block_variational(block_idx, X_train, y_train)
    
    # Update bonds
    for label in bmpo.mu_mpo.distributions.keys():
        bmpo.update_bond_variational(label)
    
    # Update tau
    bmpo.update_tau_variational(X_train, y_train)
    
    # Record MSE
    mu_pred = bmpo.forward_mu(X_train, to_tensor=True)
    assert isinstance(mu_pred, torch.Tensor)
    mse = torch.mean((mu_pred.reshape(-1) - y_train.reshape(-1)) ** 2).item()
    mse_history.append(mse)
    
    if (iteration + 1) % 10 == 0:
        tau_mean = bmpo.get_tau_mean().item()
        r1_size = bmpo.mu_mpo.distributions['r1']['expectation'].shape[0]
        r2_size = bmpo.mu_mpo.distributions['r2']['expectation'].shape[0]
        print(f"Iter {iteration+1:2d}: MSE={mse:.4f}, E[τ]={tau_mean:.2f}, ranks=[{r1_size},{r2_size}]")

print()
print("Training complete!")
print()

# Final statistics
print("="*70)
print("FINAL MODEL STATISTICS")
print("="*70)
print()

print("Final structure:")
for i, node in enumerate(bmpo.mu_nodes):
    print(f"  Block {i}: shape {node.shape}")
print()

# Print bond expectations
print("Final bond expectations:")
for label in ['r1', 'r2']:
    exp = bmpo.mu_mpo.distributions[label]['expectation']
    print(f"  {label}: min={exp.min():.2f}, max={exp.max():.2f}, mean={exp.mean():.2f}")
    print(f"       values: {exp.tolist()}")
print()

tau_mean = bmpo.get_tau_mean().item()
alpha = bmpo.tau_alpha.item()
beta = bmpo.tau_beta.item()
tau_var = alpha / (beta ** 2)  # Variance of Gamma(alpha, beta)
print(f"Noise precision τ:")
print(f"  E[τ] = {tau_mean:.4f}")
print(f"  Var[τ] = {tau_var:.4f}")
print(f"  Implied noise std ≈ {1/np.sqrt(tau_mean):.4f} (true: {noise_std})")
print()

# Generate test grid
S_test = 200
x_test = torch.linspace(-1.2, 1.2, S_test, dtype=torch.float64)
y_test_true = a_true * x_test**3 + b_true * x_test**2 + c_true
X_test = torch.stack([torch.ones_like(x_test), x_test], dim=1)

# Predict mean and variance
print("Computing predictions on test grid...")
mu_pred = bmpo.forward_mu(X_test, to_tensor=True)
sigma_pred = bmpo.forward_sigma(X_test, to_tensor=True)

assert isinstance(mu_pred, torch.Tensor)
assert isinstance(sigma_pred, torch.Tensor)
mu_pred = mu_pred.squeeze()
sigma_pred = sigma_pred.squeeze()  # Epistemic uncertainty

# Aleatoric uncertainty (noise)
aleatoric_var = 1.0 / tau_mean

# Total predictive variance = epistemic + aleatoric
total_var = sigma_pred + aleatoric_var
total_std = torch.sqrt(total_var)

# Separate epistemic std
epistemic_std = torch.sqrt(sigma_pred)

print(f"Mean epistemic std: {epistemic_std.mean():.4f}")
print(f"Mean aleatoric std: {np.sqrt(aleatoric_var):.4f}")
print(f"Mean total std: {total_std.mean():.4f}")
print()

# Compute metrics
residuals = mu_pred - y_test_true
rmse = torch.sqrt(torch.mean(residuals**2)).item()
print(f"Test RMSE: {rmse:.4f}")
print()

# Create comprehensive visualization
print("="*70)
print("CREATING VISUALIZATION")
print("="*70)
print()

fig = plt.figure(figsize=(16, 10))

# Plot 1: Predictions with confidence intervals (large)
ax1 = plt.subplot(2, 3, (1, 4))
ax1.scatter(x_train.numpy(), y_train.numpy(), alpha=0.5, s=30, c='gray', 
           label='Training data (noisy)', zorder=5)
ax1.plot(x_test.numpy(), y_test_true.numpy(), 'k-', linewidth=3, 
        label='True polynomial', zorder=10)
ax1.plot(x_test.numpy(), mu_pred.numpy(), 'r-', linewidth=2.5, 
        label='Predicted mean', zorder=8)

# Total uncertainty (±2σ)
ax1.fill_between(
    x_test.numpy(),
    (mu_pred - 2*total_std).numpy(),
    (mu_pred + 2*total_std).numpy(),
    alpha=0.3,
    color='red',
    label='±2σ total (epistemic + aleatoric)',
    zorder=3
)

# Epistemic uncertainty only (±2σ)
ax1.fill_between(
    x_test.numpy(),
    (mu_pred - 2*epistemic_std).numpy(),
    (mu_pred + 2*epistemic_std).numpy(),
    alpha=0.5,
    color='blue',
    label='±2σ epistemic only',
    zorder=4
)

ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('Bayesian MPO Predictions (Rank=3, No Trimming)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-1.2, 1.2)

# Plot 2: Residuals
ax2 = plt.subplot(2, 3, 2)
ax2.scatter(x_test.numpy(), residuals.numpy(), alpha=0.5, s=20, c='purple')
ax2.axhline(y=0, color='k', linestyle='--', linewidth=1.5)
ax2.fill_between(
    x_test.numpy(),
    (-2*total_std).numpy(),
    (2*total_std).numpy(),
    alpha=0.2,
    color='red'
)
ax2.set_xlabel('x', fontsize=11)
ax2.set_ylabel('Residual (predicted - true)', fontsize=11)
ax2.set_title(f'Prediction Residuals (RMSE={rmse:.4f})', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Uncertainty decomposition
ax3 = plt.subplot(2, 3, 3)
ax3.plot(x_test.numpy(), total_std.numpy(), 'r-', linewidth=2, 
        label='Total std', zorder=3)
ax3.plot(x_test.numpy(), epistemic_std.numpy(), 'b-', linewidth=2, 
        label='Epistemic std', zorder=2)
ax3.axhline(y=np.sqrt(aleatoric_var), color='g', linestyle='--', linewidth=2,
           label='Aleatoric std', zorder=1)
ax3.set_xlabel('x', fontsize=11)
ax3.set_ylabel('Standard deviation', fontsize=11)
ax3.set_title('Uncertainty Decomposition', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Plot 4: MSE during training
ax4 = plt.subplot(2, 3, 5)
iterations = np.arange(1, max_iter + 1)
ax4.plot(iterations, mse_history, 'b-', linewidth=2, marker='o', markersize=4)
ax4.set_xlabel('Iteration', fontsize=11)
ax4.set_ylabel('Training MSE', fontsize=11)
ax4.set_title('Training Progress', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_yscale('log')

# Plot 5: Variance comparison
ax5 = plt.subplot(2, 3, 6)
ax5.plot(x_test.numpy(), sigma_pred.numpy(), 'b-', linewidth=2, 
        label='Epistemic variance')
ax5.axhline(y=aleatoric_var, color='g', linestyle='--', linewidth=2,
           label='Aleatoric variance')
ax5.plot(x_test.numpy(), total_var.numpy(), 'r-', linewidth=2, 
        label='Total variance', alpha=0.7)
ax5.set_xlabel('x', fontsize=11)
ax5.set_ylabel('Variance', fontsize=11)
ax5.set_title('Variance Components', fontsize=12, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ciao_bayesian_predictions_rank3.png', dpi=150, bbox_inches='tight')
print("Plot saved to: bayesian_predictions_rank3.png")
print()

# Additional analysis: Check calibration
print("="*70)
print("CALIBRATION ANALYSIS")
print("="*70)
print()

# Compute standardized residuals
std_residuals = residuals / total_std
print(f"Standardized residuals (should be ~ N(0,1) if well-calibrated):")
print(f"  Mean: {std_residuals.mean():.4f} (expected: 0)")
print(f"  Std: {std_residuals.std():.4f} (expected: 1)")
print()

# Check coverage
within_1sigma = (torch.abs(residuals) <= total_std).float().mean().item()
within_2sigma = (torch.abs(residuals) <= 2*total_std).float().mean().item()
within_3sigma = (torch.abs(residuals) <= 3*total_std).float().mean().item()

print(f"Empirical coverage:")
print(f"  Within ±1σ: {within_1sigma*100:.1f}% (expected: 68.3%)")
print(f"  Within ±2σ: {within_2sigma*100:.1f}% (expected: 95.4%)")
print(f"  Within ±3σ: {within_3sigma*100:.1f}% (expected: 99.7%)")
print()

print("="*70)
print("DONE!")
print("="*70)
