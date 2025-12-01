"""
Track how expectations evolve during training (no trimming).

This will help us understand what threshold values are sensible.

Environment variables:
- BOND_DIM: Initial bond dimension (default: 10)
- MAX_ITER: Training iterations (default: 30)
- SEED: Random seed (default: 42)
"""
import torch
import os
import matplotlib.pyplot as plt
import numpy as np

from tensor.bayesian_mpo_builder import create_bayesian_tensor_train

# Configuration
BOND_DIM = int(os.environ.get('BOND_DIM', '10'))
MAX_ITER = int(os.environ.get('MAX_ITER', '30'))
SEED = int(os.environ.get('SEED', '42'))
NUM_BLOCKS = 4
NUM_SAMPLES = 100

torch.manual_seed(SEED)

print("="*80)
print("EXPECTATION EVOLUTION TRACKING (NO TRIMMING)")
print("="*80)
print(f"Polynomial: y = 2x³ - x² + 0.5")
print(f"Config: {NUM_BLOCKS} blocks, bond_dim={BOND_DIM}, iterations={MAX_ITER}")
print("="*80)
print()

# Generate data
x_train = torch.rand(NUM_SAMPLES, dtype=torch.float64) * 2 - 1
y_train = 2.0 * x_train**3 - 1.0 * x_train**2 + 0.5
y_train += 0.1 * torch.randn(NUM_SAMPLES, dtype=torch.float64)
X_train = torch.stack([torch.ones_like(x_train), x_train], dim=1)

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

# Track expectations over iterations
expectation_history = {label: [] for label in bmpo.mu_mpo.distributions.keys()}
mse_history = []

print("Initial distributions:")
for label, dist in bmpo.mu_mpo.distributions.items():
    print(f"  {label}: {dist['expectation'].tolist()}")
print()

print("Training...")
for iteration in range(MAX_ITER):
    # Update all blocks
    for block_idx in range(NUM_BLOCKS):
        bmpo.update_block_variational(block_idx, X_train, y_train)
    
    # Update all bonds
    for label in bmpo.mu_mpo.distributions.keys():
        bmpo.update_bond_variational(label)
    
    # Update tau
    bmpo.update_tau_variational(X_train, y_train)
    
    # Track expectations
    for label, dist in bmpo.mu_mpo.distributions.items():
        expectation_history[label].append(dist['expectation'].clone())
    
    # Track MSE
    mu_pred = bmpo.forward_mu(X_train, to_tensor=True)
    mse = ((mu_pred.squeeze() - y_train)**2).mean().item()
    mse_history.append(mse)
    
    if (iteration + 1) % 5 == 0:
        print(f"  Iteration {iteration+1}/{MAX_ITER}: MSE={mse:.6f}")

print()
print("Final distributions:")
for label, dist in bmpo.mu_mpo.distributions.items():
    print(f"  {label}: {dist['expectation'].tolist()}")

# Separate rank and feature labels
rank_labels = [l for l in bmpo.mu_mpo.distributions.keys() if l.startswith('r')]
feature_labels = [l for l in bmpo.mu_mpo.distributions.keys() if l.startswith('p')]

# Plot expectations evolution
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: MSE evolution
ax = axes[0, 0]
ax.plot(mse_history, 'b-', linewidth=2)
ax.set_xlabel('Iteration')
ax.set_ylabel('MSE')
ax.set_title('Training MSE Evolution')
ax.grid(True, alpha=0.3)

# Plot 2: Rank dimensions expectations
ax = axes[0, 1]
for label in rank_labels:
    history = torch.stack(expectation_history[label])  # (iterations, dim_size)
    for i in range(history.shape[1]):
        ax.plot(history[:, i].numpy(), label=f'{label}[{i}]', alpha=0.7)
ax.set_xlabel('Iteration')
ax.set_ylabel('Expectation')
ax.set_title('Rank Dimension Expectations (Bonds)')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# Plot 3: Feature dimensions expectations
ax = axes[1, 0]
for label in feature_labels:
    history = torch.stack(expectation_history[label])
    for i in range(history.shape[1]):
        ax.plot(history[:, i].numpy(), label=f'{label}[{i}]', alpha=0.7)
ax.set_xlabel('Iteration')
ax.set_ylabel('Expectation')
ax.set_title('Feature Dimension Expectations')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# Plot 4: Final expectation values (min per dimension)
ax = axes[1, 1]
labels_all = list(bmpo.mu_mpo.distributions.keys())
min_expectations = [bmpo.mu_mpo.distributions[l]['expectation'].min().item() for l in labels_all]
colors = ['blue' if l.startswith('r') else 'green' for l in labels_all]
ax.bar(range(len(labels_all)), min_expectations, color=colors, alpha=0.7)
ax.set_xticks(range(len(labels_all)))
ax.set_xticklabels(labels_all, rotation=45)
ax.set_ylabel('Min Expectation')
ax.set_title('Final Min Expectations per Dimension\n(Blue=Rank, Green=Feature)')
ax.axhline(y=0.5, color='r', linestyle='--', label='Threshold=0.5')
ax.axhline(y=0.1, color='orange', linestyle='--', label='Threshold=0.1')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
plt.savefig(f'expectation_evolution_b{BOND_DIM}_i{MAX_ITER}.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved: expectation_evolution_b{BOND_DIM}_i{MAX_ITER}.png")

# Print summary statistics
print()
print("="*80)
print("SUMMARY STATISTICS:")
print("="*80)
print("\nFinal min expectations by dimension:")
for label in labels_all:
    exp = bmpo.mu_mpo.distributions[label]['expectation']
    print(f"  {label}: min={exp.min():.4f}, max={exp.max():.4f}, mean={exp.mean():.4f}")

print()
print("Suggested thresholds:")
print("  Conservative (keeps most): 0.1")
print("  Moderate: 0.5")  
print("  Aggressive: 1.0")
print("="*80)
