"""
Adaptive Trimming Experiment: Different thresholds for ranks vs features

Environment variables:
- BOND_DIM: Initial bond dimension (default: 10)
- RANK_THRESHOLD: Threshold for rank dimensions (default: 1.0)
- FEATURE_THRESHOLD: Threshold for feature dimensions (default: 0.5)
- MAX_ITER: Training iterations (default: 30)
- SEED: Random seed (default: 42)
"""
import torch
import os
import matplotlib.pyplot as plt

from tensor.bayesian_mpo_builder import create_bayesian_tensor_train

# Configuration
BOND_DIM = int(os.environ.get('BOND_DIM', '10'))
RANK_THRESHOLD = float(os.environ.get('RANK_THRESHOLD', '1.0'))
FEATURE_THRESHOLD = float(os.environ.get('FEATURE_THRESHOLD', '0.5'))
MAX_ITER = int(os.environ.get('MAX_ITER', '30'))
SEED = int(os.environ.get('SEED', '42'))
NUM_BLOCKS = 4
NUM_SAMPLES = 100

torch.manual_seed(SEED)

print("="*80)
print("ADAPTIVE TRIMMING: Different thresholds for ranks vs features")
print("="*80)
print(f"Polynomial: y = 2x³ - x² + 0.5")
print(f"Config: {NUM_BLOCKS} blocks, initial bond_dim={BOND_DIM}")
print(f"Rank threshold: {RANK_THRESHOLD}, Feature threshold: {FEATURE_THRESHOLD}")
print(f"Iterations: {MAX_ITER}")
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

# Create model
bmpo = create_bayesian_tensor_train(
    num_blocks=NUM_BLOCKS,
    bond_dim=BOND_DIM,
    input_features=2,
    output_shape=1,
    constrict_bond=False,
    dtype=torch.float64,
    seed=SEED
)

print("INITIAL STATE:")
total_params_init = sum(node.tensor.numel() for node in bmpo.mu_nodes)
print(f"  Total parameters: {total_params_init}")
for i, node in enumerate(bmpo.mu_nodes):
    print(f"    Block {i}: shape={node.shape}")
print()

# Build adaptive threshold dict
thresholds = {}
for label in bmpo.mu_mpo.distributions.keys():
    if label.startswith('r'):
        thresholds[label] = RANK_THRESHOLD
    else:  # feature dimensions
        thresholds[label] = FEATURE_THRESHOLD

print(f"Thresholds:")
for label, threshold in thresholds.items():
    print(f"  {label}: {threshold}")
print()

print(f"TRAINING with adaptive trimming...")
bmpo.fit(X_train, y_train, max_iter=MAX_ITER, verbose=True, trim_threshold=thresholds)
print()

print("FINAL STATE:")
total_params_final = sum(node.tensor.numel() for node in bmpo.mu_nodes)
compression_ratio = 100 * (1 - total_params_final / total_params_init)
print(f"  Total parameters: {total_params_final} (reduced by {compression_ratio:.1f}%)")
for i, node in enumerate(bmpo.mu_nodes):
    print(f"    Block {i}: shape={node.shape}")

print()
print("Final distributions:")
for label, dist in bmpo.mu_mpo.distributions.items():
    exp = dist['expectation']
    print(f"  {label}: size={len(exp)}, min={exp.min():.4f}, max={exp.max():.4f}")

# Evaluate
mu_test = bmpo.forward_mu(X_test, to_tensor=True)
mu_train = bmpo.forward_mu(X_train, to_tensor=True)

train_mse = ((mu_train.squeeze() - y_train)**2).mean().item()
test_mse = ((mu_test.squeeze() - y_test_true)**2).mean().item()
test_r2 = 1 - ((mu_test.squeeze() - y_test_true)**2).sum() / ((y_test_true - y_test_true.mean())**2).sum()

print()
print("="*80)
print("RESULTS:")
print(f"  Train MSE: {train_mse:.6f}")
print(f"  Test MSE: {test_mse:.6f}")
print(f"  Test R²: {test_r2:.6f}")
print(f"  Compression: {compression_ratio:.1f}%")
print("="*80)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Predictions
ax1.scatter(x_train.numpy(), y_train.numpy(), alpha=0.5, label='Training data', s=20)
ax1.plot(x_test.numpy(), y_test_true.numpy(), 'k--', label='True function', linewidth=2)
ax1.plot(x_test.numpy(), mu_test.squeeze().numpy(), 'r-', label='BMPO prediction', linewidth=2)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title(f'Adaptive Trim: R²={test_r2:.4f}, Params={total_params_final} ({compression_ratio:.1f}% reduction)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Final expectations
labels_all = list(bmpo.mu_mpo.distributions.keys())
min_expectations = [bmpo.mu_mpo.distributions[l]['expectation'].min().item() for l in labels_all]
colors = ['blue' if l.startswith('r') else 'green' for l in labels_all]
ax2.bar(range(len(labels_all)), min_expectations, color=colors, alpha=0.7)
ax2.set_xticks(range(len(labels_all)))
ax2.set_xticklabels(labels_all, rotation=45)
ax2.set_ylabel('Min Expectation (log scale)')
ax2.set_title('Final Min Expectations\n(Blue=Rank, Green=Feature)')
ax2.axhline(y=RANK_THRESHOLD, color='blue', linestyle='--', alpha=0.5, label=f'Rank threshold={RANK_THRESHOLD}')
ax2.axhline(y=FEATURE_THRESHOLD, color='green', linestyle='--', alpha=0.5, label=f'Feature threshold={FEATURE_THRESHOLD}')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

plt.tight_layout()
filename = f'trim_adaptive_rt{RANK_THRESHOLD}_ft{FEATURE_THRESHOLD}_b{BOND_DIM}.png'
plt.savefig(filename, dpi=150, bbox_inches='tight')
print(f"\nPlot saved: {filename}")
