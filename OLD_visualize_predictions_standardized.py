"""
Bayesian MPO Prediction Visualization - Standardized with Env Vars

Trains Bayesian MPO and visualizes predictions with uncertainty.

Environment Variables:
- NUM_BLOCKS: Number of blocks (default: 3)
- BOND_DIM: Bond dimension (default: 6)
- MAX_ITER: Training iterations (default: 40)
- NUM_SAMPLES: Training samples (default: 100)
- NOISE_STD: Noise standard deviation (default: 0.1)
- PROBLEM: '1d' or '2d' (default: '1d')
- TRIM_THRESHOLD: Trimming threshold, 0=no trim (default: 0)
- SEED: Random seed (default: 42)
- PRIOR_SEED: Prior random seed (default: 42)
- SAVE_PLOTS: 1 to save plots, 0 to skip (default: 1)
- OUTPUT_PREFIX: Output filename prefix (default: 'predictions')
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from tensor.bayesian_mpo_builder import create_bayesian_tensor_train

# ============================================================================
# HYPERPARAMETERS FROM ENVIRONMENT
# ============================================================================
NUM_BLOCKS = int(os.environ.get('NUM_BLOCKS', '3'))
BOND_DIM = int(os.environ.get('BOND_DIM', '6'))
MAX_ITER = int(os.environ.get('MAX_ITER', '10'))
NUM_SAMPLES = int(os.environ.get('NUM_SAMPLES', '500'))
NOISE_STD = float(os.environ.get('NOISE_STD', '0.1'))
PROBLEM = os.environ.get('PROBLEM', '1d')
TRIM_THRESHOLD = float(os.environ.get('TRIM_THRESHOLD', '0'))
SEED = int(os.environ.get('SEED', '42'))
PRIOR_SEED = int(os.environ.get('PRIOR_SEED', '42'))
SAVE_PLOTS = int(os.environ.get('SAVE_PLOTS', '1'))
OUTPUT_PREFIX = os.environ.get('OUTPUT_PREFIX', 'predictions')

torch.manual_seed(SEED)

print("="*70)
print("BAYESIAN MPO PREDICTION VISUALIZATION")
print("="*70)
print(f"Problem: {PROBLEM.upper()}")
print(f"Hyperparameters:")
print(f"  NUM_BLOCKS={NUM_BLOCKS}, BOND_DIM={BOND_DIM}")
print(f"  MAX_ITER={MAX_ITER}, NUM_SAMPLES={NUM_SAMPLES}")
print(f"  NOISE_STD={NOISE_STD}, TRIM_THRESHOLD={TRIM_THRESHOLD}")
print(f"  SEED={SEED}, PRIOR_SEED={PRIOR_SEED}")
print("="*70)
print()

# ============================================================================
# GENERATE DATA
# ============================================================================
if PROBLEM == '1d':
    # 1D polynomial: y = 2x³ - x² + 0.5
    INPUT_DIM = 2
    x = torch.rand(NUM_SAMPLES, dtype=torch.float64) * 2 - 1
    y = 2.0 * x**3 - 1.0 * x**2 + 0.5
    y += NOISE_STD * torch.randn(NUM_SAMPLES, dtype=torch.float64)
    X = torch.stack([torch.ones_like(x), x], dim=1)
    
    # Test grid
    x_test = torch.linspace(-1.2, 1.2, 200, dtype=torch.float64)
    y_test_true = 2.0 * x_test**3 - 1.0 * x_test**2 + 0.5
    X_test = torch.stack([torch.ones_like(x_test), x_test], dim=1)
    
elif PROBLEM == '2d':
    # 2D surface: z = 0.5x² + 0.3y² - 0.4xy + 0.2sin(3x) + 0.2cos(3y) + 0.5
    INPUT_DIM = 3
    x = torch.rand(NUM_SAMPLES, dtype=torch.float64) * 2 - 1
    y_coord = torch.rand(NUM_SAMPLES, dtype=torch.float64) * 2 - 1
    y = 0.5 * x**2 + 0.3 * y_coord**2 - 0.4 * x * y_coord
    y += 0.2 * torch.sin(3 * x) + 0.2 * torch.cos(3 * y_coord) + 0.5
    y += NOISE_STD * torch.randn(NUM_SAMPLES, dtype=torch.float64)
    X = torch.stack([torch.ones_like(x), x, y_coord], dim=1)
    
    # Test grid
    grid_size = 50
    x_grid = torch.linspace(-1.1, 1.1, grid_size, dtype=torch.float64)
    y_grid = torch.linspace(-1.1, 1.1, grid_size, dtype=torch.float64)
    X_mesh, Y_mesh = torch.meshgrid(x_grid, y_grid, indexing='ij')
    x_flat = X_mesh.flatten()
    y_flat = Y_mesh.flatten()
    X_test = torch.stack([torch.ones_like(x_flat), x_flat, y_flat], dim=1)
    y_test_true = 0.5 * x_flat**2 + 0.3 * y_flat**2 - 0.4 * x_flat * y_flat
    y_test_true += 0.2 * torch.sin(3 * x_flat) + 0.2 * torch.cos(3 * y_flat) + 0.5
else:
    raise ValueError(f"Unknown PROBLEM: {PROBLEM}")

print(f"Data: X.shape={X.shape}, y.shape={y.shape}")
print(f"Target: mean={y.mean():.4f}, std={y.std():.4f}")
print()

# ============================================================================
# CREATE AND TRAIN MODEL
# ============================================================================
print(f"Creating {NUM_BLOCKS}-block Bayesian MPO...")
bmpo = create_bayesian_tensor_train(
    num_blocks=NUM_BLOCKS,
    bond_dim=BOND_DIM,
    input_features=INPUT_DIM,
    output_shape=1,
    constrict_bond=False,
    tau_alpha=torch.tensor(2.0, dtype=torch.float64),
    tau_beta=torch.tensor(1.0, dtype=torch.float64),
    dtype=torch.float64,
    seed=SEED,
    random_priors=True,
    prior_seed=PRIOR_SEED
)

print(f"Structure:")
for i, node in enumerate(bmpo.mu_nodes):
    print(f"  Block {i}: {node.shape}")
print()

print(f"Training for {MAX_ITER} iterations...")
if TRIM_THRESHOLD > 0:
    print(f"  With trimming threshold: {TRIM_THRESHOLD}")
print()
# Manual training loop with optional trimming
# 
mu_pred = bmpo.forward_mu(X, to_tensor=True)
assert isinstance(mu_pred, torch.Tensor)
mse = ((mu_pred.squeeze() - y)**2).mean().item()
print("Initial MSE")
print(mse)
for iteration in range(MAX_ITER):
    for block_idx in range(NUM_BLOCKS):
        bmpo.update_block_variational(block_idx, X, y)
    
    for label in bmpo.mu_mpo.distributions.keys():
        bmpo.update_bond_variational(label)
    
    bmpo.update_tau_variational(X, y)
    
    # Optional trimming
    if TRIM_THRESHOLD > 0:
        thresholds = {label: TRIM_THRESHOLD for label in bmpo.mu_mpo.distributions.keys() 
                     if label.startswith('r')}
        if thresholds:
            bmpo.trim(thresholds)
    
    if (iteration + 1) % 1 == 0:
        mu_pred = bmpo.forward_mu(X, to_tensor=True)
        assert isinstance(mu_pred, torch.Tensor)
        mse = ((mu_pred.squeeze() - y)**2).mean().item()
        tau = bmpo.get_tau_mean().item()
        print(f"  Iter {iteration+1:3d}: MSE={mse:.6f}, E[τ]={tau:.2f}")

print()

# ============================================================================
# EVALUATE
# ============================================================================
mu_pred = bmpo.forward_mu(X_test, to_tensor=True)
sigma_pred = bmpo.forward_sigma(X_test, to_tensor=True)
assert isinstance(mu_pred, torch.Tensor)
assert isinstance(sigma_pred, torch.Tensor)

mu_pred = mu_pred.squeeze()
sigma_pred = sigma_pred.squeeze()

tau_mean = bmpo.get_tau_mean().item()
aleatoric_var = 1.0 / tau_mean
total_var = sigma_pred + aleatoric_var
total_std = torch.sqrt(total_var)
epistemic_std = torch.sqrt(sigma_pred)

residuals = mu_pred - y_test_true
rmse = torch.sqrt((residuals**2).mean()).item()
ss_res = (residuals**2).sum().item()
ss_tot = ((y_test_true - y_test_true.mean())**2).sum().item()
r2 = 1 - (ss_res / ss_tot)

print("RESULTS")
print("="*70)
print(f"Test RMSE: {rmse:.5f}")
print(f"Test R²:   {r2:.5f}")
print(f"E[τ]: {tau_mean:.4f} (implied noise std: {1/np.sqrt(tau_mean):.4f})")
print(f"Mean epistemic std: {epistemic_std.mean():.5f}")
print(f"Mean total std: {total_std.mean():.5f}")
print()

# ============================================================================
# PLOT
# ============================================================================
if SAVE_PLOTS:
    if PROBLEM == '1d':
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Predictions
        ax = axes[0]
        ax.scatter(x.numpy(), y.numpy(), alpha=0.5, s=20, c='gray', label='Training data')
        ax.plot(x_test.numpy(), y_test_true.numpy(), 'k-', linewidth=2, label='True')
        ax.plot(x_test.numpy(), mu_pred.numpy(), 'r-', linewidth=2, label='Predicted mean')
        ax.fill_between(x_test.numpy(),
                        (mu_pred - 2*total_std).numpy(),
                        (mu_pred + 2*total_std).numpy(),
                        alpha=0.3, color='red', label='±2σ total')
        ax.fill_between(x_test.numpy(),
                        (mu_pred - 2*epistemic_std).numpy(),
                        (mu_pred + 2*epistemic_std).numpy(),
                        alpha=0.5, color='blue', label='±2σ epistemic')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Predictions (R²={r2:.3f})', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Residuals
        ax = axes[1]
        ax.scatter(x_test.numpy(), residuals.numpy(), alpha=0.5, s=10, c='purple')
        ax.axhline(y=0, color='k', linestyle='--')
        ax.fill_between(x_test.numpy(),
                        (-2*total_std).numpy(),
                        (2*total_std).numpy(),
                        alpha=0.2, color='red')
        ax.set_xlabel('x')
        ax.set_ylabel('Residual')
        ax.set_title(f'Residuals (RMSE={rmse:.4f})', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
    elif PROBLEM == '2d':
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(15, 5))
        
        Z_pred = mu_pred.reshape(grid_size, grid_size)
        Z_true = y_test_true.reshape(grid_size, grid_size)
        Std_total = total_std.reshape(grid_size, grid_size)
        
        # Plot 1: True surface
        ax = fig.add_subplot(1, 3, 1, projection='3d')
        ax.plot_surface(X_mesh.numpy(), Y_mesh.numpy(), Z_true.numpy(), cmap='viridis', alpha=0.8)
        ax.set_title('True Surface')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        # Plot 2: Predicted surface
        ax = fig.add_subplot(1, 3, 2, projection='3d')
        ax.plot_surface(X_mesh.numpy(), Y_mesh.numpy(), Z_pred.detach().numpy(), cmap='plasma', alpha=0.8)
        ax.set_title(f'Predicted (R²={r2:.3f})')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        # Plot 3: Uncertainty
        ax = fig.add_subplot(1, 3, 3, projection='3d')
        ax.plot_surface(X_mesh.numpy(), Y_mesh.numpy(), Std_total.detach().numpy(), cmap='Reds', alpha=0.8)
        ax.set_title('Uncertainty (std)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    
    plt.tight_layout()
    filename = f'{OUTPUT_PREFIX}_{PROBLEM}_b{NUM_BLOCKS}_d{BOND_DIM}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {filename}")
    print()

print("="*70)
print("DONE")
print("="*70)
