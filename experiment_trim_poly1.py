"""
Trimming Experiment 1: Cubic Polynomial

Tests bond trimming on y = 2x³ - x² + 0.5

Environment variables:
- BOND_DIM: Initial bond dimension (default: 10)
- TRIM_THRESHOLD: Trimming threshold (default: 0.5)  
- MAX_ITER: Training iterations (default: 30)
- SEED: Random seed (default: 42)
"""
import torch
import os
import matplotlib.pyplot as plt

from tensor.bayesian_mpo_builder import create_bayesian_tensor_train

# Configuration
BOND_DIM = int(os.environ.get('BOND_DIM', '10'))
TRIM_THRESHOLD = float(os.environ.get('TRIM_THRESHOLD', '0.5'))
MAX_ITER = int(os.environ.get('MAX_ITER', '30'))
SEED = int(os.environ.get('SEED', '42'))
NUM_BLOCKS = 4
NUM_SAMPLES = 100

torch.manual_seed(SEED)

print("="*80)
print("TRIMMING EXPERIMENT 1: Cubic Polynomial")
print("="*80)
print(f"Polynomial: y = 2x³ - x² + 0.5")
print(f"Config: {NUM_BLOCKS} blocks, initial bond_dim={BOND_DIM}")
print(f"Trim threshold: {TRIM_THRESHOLD}, iterations: {MAX_ITER}")
print("="*80)
print()

# Generate data
x_train = torch.rand(NUM_SAMPLES, dtype=torch.float64) * 2 - 1
y_train = 2.0 * x_train**3 - 1.0 * x_train**2 + 0.5
y_train += 0.1 * torch.randn(NUM_SAMPLES, dtype=torch.float64)
X_train = torch.stack([torch.ones_like(x_train), x_train], dim=1)

# Test data
x_test = torch.linspace(-1.2, 1.2, 200, dtype=torch.float64)
y_test_true = 2.0 * x_test**3 - 1.0 * x_test**2 + 0.5
X_test = torch.stack([torch.ones_like(x_test), x_test], dim=1)

print("INITIAL STATE:")
bmpo = create_bayesian_tensor_train(
    num_blocks=NUM_BLOCKS,
    bond_dim=BOND_DIM,
    input_features=2,
    output_shape=1,
    constrict_bond=False,
    dtype=torch.float64,
    seed=SEED
)

total_params_init = sum(node.tensor.numel() for node in bmpo.mu_nodes)
print(f"  Total parameters: {total_params_init}")
for i, node in enumerate(bmpo.mu_nodes):
    print(f"    Block {i}: shape={node.shape}")
print()

print(f"TRAINING with trimming (threshold={TRIM_THRESHOLD})...")
bmpo.fit(X_train, y_train, max_iter=MAX_ITER, verbose=True, trim_threshold=TRIM_THRESHOLD)
print()

print("FINAL STATE:")
total_params_final = sum(node.tensor.numel() for node in bmpo.mu_nodes)
print(f"  Total parameters: {total_params_final} (reduced by {100*(1-total_params_final/total_params_init):.1f}%)")
for i, node in enumerate(bmpo.mu_nodes):
    print(f"    Block {i}: shape={node.shape}")
print()

# Evaluate
mu_test = bmpo.forward_mu(X_test, to_tensor=True)
mu_train = bmpo.forward_mu(X_train, to_tensor=True)

train_mse = ((mu_train.squeeze() - y_train)**2).mean().item()
test_mse = ((mu_test.squeeze() - y_test_true)**2).mean().item()
test_r2 = 1 - ((mu_test.squeeze() - y_test_true)**2).sum() / ((y_test_true - y_test_true.mean())**2).sum()

print("RESULTS:")
print(f"  Train MSE: {train_mse:.6f}")
print(f"  Test MSE: {test_mse:.6f}")  
print(f"  Test R²: {test_r2:.6f}")
print("="*80)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(x_train.numpy(), y_train.numpy(), alpha=0.5, label='Training data', s=20)
plt.plot(x_test.numpy(), y_test_true.numpy(), 'k--', label='True function', linewidth=2)
plt.plot(x_test.numpy(), mu_test.squeeze().numpy(), 'r-', label='BMPO prediction', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Trim Exp 1: Cubic (Threshold={TRIM_THRESHOLD}, Final Params={total_params_final})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f'trim_exp1_cubic_t{TRIM_THRESHOLD}_b{BOND_DIM}.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved: trim_exp1_cubic_t{TRIM_THRESHOLD}_b{BOND_DIM}.png")
