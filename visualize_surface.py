"""
Train Bayesian MPO on 2D surface regression.

This script:
1. Generates a 2D surface: z = f(x, y) on [-1, 1] × [-1, 1]
2. Input features: [1, x, y]
3. Uses 5 blocks with rank=6
4. Small trimming threshold=5
5. Visualizes the learned surface and uncertainty
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from tensor.bayesian_mpo_builder import create_bayesian_tensor_train

torch.manual_seed(42)

print("="*70)
print("BAYESIAN MPO 2D SURFACE REGRESSION")
print("="*70)
print()

# Define true surface function
def true_function(x, y):
    """
    Non-singular surface function.
    Combination of polynomial and periodic terms.
    """
    return (0.5 * x**2 + 0.3 * y**2 - 0.4 * x * y + 
            0.2 * torch.sin(3 * x) + 0.2 * torch.cos(3 * y) + 
            0.5)

print("True surface function:")
print("  z = 0.5*x² + 0.3*y² - 0.4*xy + 0.2*sin(3x) + 0.2*cos(3y) + 0.5")
print()

# Generate training data
S_train = 200  # Number of training samples
x_train = torch.rand(S_train, dtype=torch.float64) * 2 - 1  # Uniform(-1, 1)
y_train_input = torch.rand(S_train, dtype=torch.float64) * 2 - 1  # Uniform(-1, 1)

# Compute true surface values
z_train = true_function(x_train, y_train_input)

# Add noise
noise_std = 0.05
z_train += noise_std * torch.randn(S_train, dtype=torch.float64)

# Input format: [1, x, y]
X_train = torch.stack([torch.ones_like(x_train), x_train, y_train_input], dim=1)

print(f"Training data: {S_train} samples")
print(f"Noise std: {noise_std}")
print(f"X shape: {X_train.shape}  (columns: [1, x, y])")
print(f"z range: [{z_train.min():.2f}, {z_train.max():.2f}]")
print()

# Create Bayesian MPO with 5 blocks, rank=6, small trimming
print("Creating Bayesian MPO (4 blocks, rank=6, NO trimming)...")
bmpo = create_bayesian_tensor_train(
    num_blocks=4,
    bond_dim=6,
    input_features=3,  # [1, x, y]
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

# Check initial prediction
print("Testing initial forward pass...")
mu_test = bmpo.forward_mu(X_train[:5], to_tensor=True)
assert isinstance(mu_test, torch.Tensor)
print(f"Initial prediction shape: {mu_test.shape}")
print(f"Initial predictions: {mu_test.squeeze()[:5]}")
print(f"True values: {z_train[:5]}")
print()

# Train with small trimming threshold
print("Training for 40 iterations with NO trimming...")
print()

max_iter = 40
mse_history = []
rank_history = {f'r{i}': [] for i in range(1, 4)}

for iteration in range(max_iter):
    # Update blocks
    for block_idx in range(len(bmpo.mu_nodes)):
        bmpo.update_block_variational(block_idx, X_train, z_train)
    
    # Update bonds
    for label in bmpo.mu_mpo.distributions.keys():
        bmpo.update_bond_variational(label)
    
    # Update tau
    bmpo.update_tau_variational(X_train, z_train)
    
    # NO TRIMMING - just record ranks
    for i in range(1, 4):
        label = f'r{i}'
        rank = bmpo.mu_mpo.distributions[label]['expectation'].shape[0]
        rank_history[label].append(rank)
    
    # Record MSE
    mu_pred = bmpo.forward_mu(X_train, to_tensor=True)
    assert isinstance(mu_pred, torch.Tensor)
    mse = torch.mean((mu_pred.reshape(-1) - z_train.reshape(-1)) ** 2).item()
    mse_history.append(mse)
    
    if (iteration + 1) % 5 == 0:
        tau_mean = bmpo.get_tau_mean().item()
        ranks_str = ','.join([str(rank_history[f'r{i}'][-1]) for i in range(1, 4)])
        print(f"Iter {iteration+1:2d}: MSE={mse:.5f}, E[τ]={tau_mean:.2f}, ranks=[{ranks_str}]")

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

print("Final bond ranks:")
for i in range(1, 3):
    label = f'r{i}'
    rank = rank_history[label][-1]
    initial_rank = rank_history[label][0]
    reduction = (1 - rank / initial_rank) * 100 if initial_rank > 0 else 0
    print(f"  {label}: {initial_rank} → {rank} ({reduction:.1f}% reduction)")
print()

tau_mean = bmpo.get_tau_mean().item()
alpha = bmpo.tau_alpha.item()
beta = bmpo.tau_beta.item()
tau_var = alpha / (beta ** 2)
print(f"Noise precision τ:")
print(f"  E[τ] = {tau_mean:.4f}")
print(f"  Var[τ] = {tau_var:.4f}")
print(f"  Implied noise std ≈ {1/np.sqrt(tau_mean):.4f} (true: {noise_std})")
print()

# Generate test grid for visualization
grid_size = 50
x_grid = torch.linspace(-1.1, 1.1, grid_size, dtype=torch.float64)
y_grid = torch.linspace(-1.1, 1.1, grid_size, dtype=torch.float64)
X_mesh, Y_mesh = torch.meshgrid(x_grid, y_grid, indexing='ij')

# Flatten for prediction
x_flat = X_mesh.flatten()
y_flat = Y_mesh.flatten()
X_test = torch.stack([torch.ones_like(x_flat), x_flat, y_flat], dim=1)

# True surface
Z_true = true_function(X_mesh, Y_mesh)

# Predict
print("Computing predictions on test grid...")
mu_pred = bmpo.forward_mu(X_test, to_tensor=True)
sigma_pred = bmpo.forward_sigma(X_test, to_tensor=True)

assert isinstance(mu_pred, torch.Tensor)
assert isinstance(sigma_pred, torch.Tensor)

# Reshape to grid
Z_pred = mu_pred.reshape(grid_size, grid_size)
Sigma_pred = sigma_pred.reshape(grid_size, grid_size)

# Total uncertainty
aleatoric_var = 1.0 / tau_mean
total_var = Sigma_pred + aleatoric_var
total_std = torch.sqrt(total_var)
epistemic_std = torch.sqrt(Sigma_pred)

print(f"Mean epistemic std: {epistemic_std.mean():.5f}")
print(f"Mean aleatoric std: {np.sqrt(aleatoric_var):.5f}")
print(f"Mean total std: {total_std.mean():.5f}")
print()

# Compute metrics on grid
residuals = Z_pred - Z_true
rmse = torch.sqrt(torch.mean(residuals**2)).item()
# R² score
ss_res = torch.sum(residuals**2).item()
ss_tot = torch.sum((Z_true - Z_true.mean())**2).item()
r2_score = 1 - (ss_res / ss_tot)
print(f"Test RMSE: {rmse:.5f}")
print(f"Test R²: {r2_score:.5f}")
print()

# Create visualization
print("="*70)
print("CREATING VISUALIZATION")
print("="*70)
print()

fig = plt.figure(figsize=(18, 12))

# Convert to numpy for plotting
X_np = X_mesh.numpy()
Y_np = Y_mesh.numpy()
Z_true_np = Z_true.numpy()
Z_pred_np = Z_pred.detach().numpy()
total_std_np = total_std.detach().numpy()
epistemic_std_np = epistemic_std.detach().numpy()

# Plot 1: True surface
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
surf1 = ax1.plot_surface(X_np, Y_np, Z_true_np, cmap='viridis', alpha=0.8)
ax1.scatter(x_train.numpy(), y_train_input.numpy(), z_train.numpy(), 
           c='red', s=10, alpha=0.5)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('True Surface with Training Data', fontweight='bold')
fig.colorbar(surf1, ax=ax1, shrink=0.5)

# Plot 2: Predicted surface
ax2 = fig.add_subplot(2, 3, 2, projection='3d')
surf2 = ax2.plot_surface(X_np, Y_np, Z_pred_np, cmap='plasma', alpha=0.8)
ax2.scatter(x_train.numpy(), y_train_input.numpy(), z_train.numpy(), 
           c='red', s=10, alpha=0.5, label='Training data')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
ax2.set_title('Predicted Surface', fontweight='bold')
fig.colorbar(surf2, ax=ax2, shrink=0.5)

# Plot 3: Total uncertainty (std)
ax3 = fig.add_subplot(2, 3, 3, projection='3d')
surf3 = ax3.plot_surface(X_np, Y_np, total_std_np, cmap='Reds', alpha=0.8)
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('std')
ax3.set_title('Total Uncertainty (Std)', fontweight='bold')
fig.colorbar(surf3, ax=ax3, shrink=0.5)

# Plot 4: Residuals (2D heatmap)
ax4 = fig.add_subplot(2, 3, 4)
residuals_np = residuals.detach().numpy()
im4 = ax4.contourf(X_np, Y_np, residuals_np, levels=20, cmap='RdBu_r')
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_title(f'Residuals (RMSE={rmse:.5f}, R²={r2_score:.5f})', fontweight='bold')
fig.colorbar(im4, ax=ax4)

# Plot 5: Epistemic uncertainty (2D heatmap)
ax5 = fig.add_subplot(2, 3, 5)
im5 = ax5.contourf(X_np, Y_np, epistemic_std_np, levels=20, cmap='YlOrRd')
ax5.scatter(x_train.numpy(), y_train_input.numpy(), c='blue', s=5, alpha=0.3)
ax5.set_xlabel('x')
ax5.set_ylabel('y')
ax5.set_title('Epistemic Uncertainty (Std)', fontweight='bold')
fig.colorbar(im5, ax=ax5)

# Plot 6: Training progress
ax6 = fig.add_subplot(2, 3, 6)
iterations = np.arange(1, max_iter + 1)
ax6.plot(iterations, mse_history, 'b-', linewidth=2, marker='o', markersize=4)
ax6.set_xlabel('Iteration')
ax6.set_ylabel('Training MSE')
ax6.set_title('Training Progress', fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.set_yscale('log')

plt.tight_layout()
plt.savefig('surface_predictions.png', dpi=150, bbox_inches='tight')
print("Plot saved to: surface_predictions.png")
print()

# Additional plot: Rank evolution
fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot rank evolution
ax_ranks = axes[0]
for i in range(1, 3):
    label = f'r{i}'
    ax_ranks.plot(iterations, rank_history[label], marker='o', linewidth=2, 
                 markersize=4, label=label)
ax_ranks.set_xlabel('Iteration')
ax_ranks.set_ylabel('Rank')
ax_ranks.set_title('Bond Rank Evolution', fontweight='bold')
ax_ranks.legend()
ax_ranks.grid(True, alpha=0.3)

# Plot slice comparison at y=0
ax_slice = axes[1]
y_idx = grid_size // 2  # Middle slice
x_slice = X_np[:, y_idx]
z_true_slice = Z_true_np[:, y_idx]
z_pred_slice = Z_pred_np[:, y_idx]
std_slice = total_std_np[:, y_idx]

ax_slice.plot(x_slice, z_true_slice, 'k-', linewidth=2, label='True')
ax_slice.plot(x_slice, z_pred_slice, 'r-', linewidth=2, label='Predicted')
ax_slice.fill_between(x_slice, z_pred_slice - 2*std_slice, 
                      z_pred_slice + 2*std_slice, alpha=0.3, color='red',
                      label='±2σ')
# Add training points near y=0
y_mask = torch.abs(y_train_input) < 0.1
ax_slice.scatter(x_train[y_mask].numpy(), z_train[y_mask].numpy(), 
                c='gray', s=30, alpha=0.5, label='Training data (|y|<0.1)')
ax_slice.set_xlabel('x')
ax_slice.set_ylabel('z')
ax_slice.set_title('Cross-section at y ≈ 0', fontweight='bold')
ax_slice.legend()
ax_slice.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('surface_analysis.png', dpi=150, bbox_inches='tight')
print("Plot saved to: surface_analysis.png")
print()

# Calibration analysis
print("="*70)
print("CALIBRATION ANALYSIS")
print("="*70)
print()

residuals_flat = residuals.flatten()
total_std_flat = total_std.flatten()
std_residuals = residuals_flat / total_std_flat

print(f"Standardized residuals (should be ~ N(0,1) if well-calibrated):")
print(f"  Mean: {std_residuals.mean():.4f} (expected: 0)")
print(f"  Std: {std_residuals.std():.4f} (expected: 1)")
print()

# Check coverage
within_1sigma = (torch.abs(residuals_flat) <= total_std_flat).float().mean().item()
within_2sigma = (torch.abs(residuals_flat) <= 2*total_std_flat).float().mean().item()
within_3sigma = (torch.abs(residuals_flat) <= 3*total_std_flat).float().mean().item()

print(f"Empirical coverage:")
print(f"  Within ±1σ: {within_1sigma*100:.1f}% (expected: 68.3%)")
print(f"  Within ±2σ: {within_2sigma*100:.1f}% (expected: 95.4%)")
print(f"  Within ±3σ: {within_3sigma*100:.1f}% (expected: 99.7%)")
print()

print("="*70)
print("DONE!")
print("="*70)
