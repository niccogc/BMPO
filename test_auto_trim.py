"""
Test automatic trimming during training.

This script trains with trim_threshold=10, which means:
- After each iteration, trim bonds with E[X_i] < 10
- Track how the rank evolves over time
- Compare final performance
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from tensor.bayesian_mpo_builder import create_bayesian_tensor_train

torch.manual_seed(42)

print("="*70)
print("AUTOMATIC TRIMMING TEST (threshold=10)")
print("="*70)
print()

# Generate simple polynomial data
S_train = 100
x_train = torch.rand(S_train, dtype=torch.float64) * 2 - 1
y_train = 2.0 * x_train**3 - 1.0 * x_train**2 + 0.5
y_train += 0.1 * torch.randn(S_train, dtype=torch.float64)
X_train = torch.stack([torch.ones_like(x_train), x_train], dim=1)

print(f"Training data: {S_train} samples")
print(f"X shape: {X_train.shape}")
print(f"y shape: {y_train.shape}")
print()

# Create Bayesian MPO with rank=8
print("Creating Bayesian MPO with rank=8 and random priors...")
bmpo = create_bayesian_tensor_train(
    num_blocks=3,
    bond_dim=8,
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
    print(f"  Block {i}: shape {node.shape}, labels {node.dim_labels}")
print()

# Storage for tracking ranks
rank_history = {'r1': [], 'r2': []}
mse_history = []
tau_history = []
max_iter = 30
trim_threshold = 10.0

print(f"Training for {max_iter} iterations with trim_threshold={trim_threshold}...")
print()

# Manual training loop to track ranks at each iteration
for iteration in range(max_iter):
    print(f'Iteration {iteration + 1}/{max_iter}')
    
    # Update blocks
    for block_idx in range(len(bmpo.mu_nodes)):
        bmpo.update_block_variational(block_idx, X_train, y_train)
    
    # Update bonds
    for label in bmpo.mu_mpo.distributions.keys():
        bmpo.update_bond_variational(label)
    
    # Update tau
    bmpo.update_tau_variational(X_train, y_train)
    
    # Record current ranks
    r1_size = bmpo.mu_mpo.distributions['r1']['expectation'].shape[0]
    r2_size = bmpo.mu_mpo.distributions['r2']['expectation'].shape[0]
    rank_history['r1'].append(r1_size)
    rank_history['r2'].append(r2_size)
    
    # Record MSE
    mu_pred = bmpo.forward_mu(X_train, to_tensor=True)
    assert isinstance(mu_pred, torch.Tensor)
    mse = torch.mean((mu_pred.reshape(-1) - y_train.reshape(-1)) ** 2).item()
    mse_history.append(mse)
    
    # Record tau
    tau_mean = bmpo.get_tau_mean().item()
    tau_history.append(tau_mean)
    
    # Print bond expectations before trimming
    print(f'  Before trim: r1_size={r1_size}, r2_size={r2_size}')
    for label in ['r1', 'r2']:
        exp = bmpo.mu_mpo.distributions[label]['expectation']
        print(f'    {label}: min={exp.min():.2f}, max={exp.max():.2f}, mean={exp.mean():.2f}')
    
    # Apply trimming
    thresholds = {}
    for label in ['r1', 'r2']:
        thresholds[label] = trim_threshold
    
    # Check if trimming would remove anything
    n_kept = {'r1': 0, 'r2': 0}
    for label in ['r1', 'r2']:
        exp = bmpo.mu_mpo.distributions[label]['expectation']
        n_kept[label] = (exp >= trim_threshold).sum().item()
    
    if n_kept['r1'] > 0 and n_kept['r2'] > 0:
        # Only trim if we won't remove everything
        if n_kept['r1'] < r1_size or n_kept['r2'] < r2_size:
            print(f'  Trimming: r1 {r1_size}→{n_kept["r1"]}, r2 {r2_size}→{n_kept["r2"]}')
            bmpo.trim(thresholds)
        else:
            print(f'  No trimming needed (all above threshold)')
    else:
        print(f'  Skipping trim (would remove all indices)')
    
    print(f'  MSE: {mse:.4f}, E[τ]: {tau_mean:.4f}')
    print()

print()
print("="*70)
print("PLOTTING RESULTS")
print("="*70)
print()

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

iterations = np.arange(1, max_iter + 1)

# Plot 1: r1 rank evolution
ax = axes[0, 0]
ax.plot(iterations, rank_history['r1'], 'b-', linewidth=2, marker='o', markersize=6)
ax.set_xlabel('Iteration')
ax.set_ylabel('r1 rank')
ax.set_title('Bond r1 Rank Evolution (threshold=10)')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 9])

# Plot 2: r2 rank evolution
ax = axes[0, 1]
ax.plot(iterations, rank_history['r2'], 'r-', linewidth=2, marker='o', markersize=6)
ax.set_xlabel('Iteration')
ax.set_ylabel('r2 rank')
ax.set_title('Bond r2 Rank Evolution (threshold=10)')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 9])

# Plot 3: MSE over time
ax = axes[1, 0]
ax.plot(iterations, mse_history, 'g-', linewidth=2, marker='o')
ax.set_xlabel('Iteration')
ax.set_ylabel('MSE')
ax.set_title('Mean Squared Error Over Time')
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# Plot 4: Tau over time
ax = axes[1, 1]
ax.plot(iterations, tau_history, 'm-', linewidth=2, marker='o')
ax.set_xlabel('Iteration')
ax.set_ylabel('E[τ]')
ax.set_title('Noise Precision Over Time')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('auto_trim_evolution.png', dpi=150, bbox_inches='tight')
print("Plot saved to: auto_trim_evolution.png")
print()

# Print final statistics
print("="*70)
print("FINAL STATISTICS")
print("="*70)
print()

print("Initial ranks: r1=8, r2=8")
print(f"Final ranks: r1={rank_history['r1'][-1]}, r2={rank_history['r2'][-1]}")
print()

print("Final structure:")
for i, node in enumerate(bmpo.mu_nodes):
    print(f"  Block {i}: shape {node.shape}")
print()

print(f"Final MSE: {mse_history[-1]:.4f}")
print(f"Final E[τ]: {tau_history[-1]:.4f}")
print()

print("MSE evolution:")
print(f"  Initial: {mse_history[0]:.4f}")
print(f"  Final: {mse_history[-1]:.4f}")
print(f"  Improvement: {(1 - mse_history[-1]/mse_history[0])*100:.1f}%")
print()

print("Rank reduction:")
r1_reduction = (1 - rank_history['r1'][-1] / rank_history['r1'][0]) * 100
r2_reduction = (1 - rank_history['r2'][-1] / rank_history['r2'][0]) * 100
print(f"  r1: {rank_history['r1'][0]} → {rank_history['r1'][-1]} ({r1_reduction:.1f}% reduction)")
print(f"  r2: {rank_history['r2'][0]} → {rank_history['r2'][-1]} ({r2_reduction:.1f}% reduction)")
print()

# Check when trimming happened
trim_iterations_r1 = []
trim_iterations_r2 = []
for i in range(1, len(rank_history['r1'])):
    if rank_history['r1'][i] < rank_history['r1'][i-1]:
        trim_iterations_r1.append(i+1)
    if rank_history['r2'][i] < rank_history['r2'][i-1]:
        trim_iterations_r2.append(i+1)

print("Trim events:")
print(f"  r1 trimmed at iterations: {trim_iterations_r1}")
print(f"  r2 trimmed at iterations: {trim_iterations_r2}")
print()

print("="*70)
