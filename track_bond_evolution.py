"""
Track bond evolution during training.

This script:
1. Trains a Bayesian MPO with rank=8
2. Records bond expectations at each iteration
3. Plots how each bond dimension evolves over time
4. Uses random prior initialization
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from tensor.bayesian_mpo_builder import create_bayesian_tensor_train

torch.manual_seed(42)

print("="*70)
print("BOND EVOLUTION TRACKING")
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

print(f"Structure:")
for i, node in enumerate(bmpo.mu_nodes):
    print(f"  Block {i}: shape {node.shape}, labels {node.dim_labels}")
print()

# Print initial prior parameters for bonds
print("Initial prior parameters for bonds:")
for label, params in bmpo.prior_bond_params.items():
    if label.startswith('r'):  # Only rank bonds
        conc = params['concentration0']
        rate = params['rate0']
        print(f"  {label}:")
        print(f"    concentration0: {conc.tolist()}")
        print(f"    rate0: {rate.tolist()}")
print()

# Storage for tracking
bond_labels = ['r1', 'r2']
history = {label: [] for label in bond_labels}
mse_history = []
tau_history = []
max_iter = 50

print(f"Training for {max_iter} iterations and tracking bond expectations...")
print()

# Manual training loop to track at each iteration
for iteration in range(max_iter):
    print(f'Iteration {iteration + 1}/{max_iter}', end='')
    
    # Update blocks
    for block_idx in range(len(bmpo.mu_nodes)):
        bmpo.update_block_variational(block_idx, X_train, y_train)
    
    # Update bonds
    for label in bmpo.mu_mpo.distributions.keys():
        bmpo.update_bond_variational(label)
    
    # Update tau
    bmpo.update_tau_variational(X_train, y_train)
    
    # Record bond expectations
    for label in bond_labels:
        exp = bmpo.mu_mpo.distributions[label]['expectation'].clone().detach()
        history[label].append(exp.numpy())
    
    # Record MSE
    mu_pred = bmpo.forward_mu(X_train, to_tensor=True)
    assert isinstance(mu_pred, torch.Tensor)
    mse = torch.mean((mu_pred.reshape(-1) - y_train.reshape(-1)) ** 2).item()
    mse_history.append(mse)
    
    # Record tau
    tau_mean = bmpo.get_tau_mean().item()
    tau_history.append(tau_mean)
    
    print(f' - MSE: {mse:.4f}, E[τ]: {tau_mean:.4f}')

print()
print("="*70)
print("PLOTTING RESULTS")
print("="*70)
print()

# Convert history to numpy arrays
history_arrays = {}
for label in bond_labels:
    history_arrays[label] = np.array(history[label])  # Shape: (iterations, bond_size)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: r1 bond evolution
ax = axes[0, 0]
iterations = np.arange(1, max_iter + 1)
for i in range(history_arrays['r1'].shape[1]):
    ax.plot(iterations, history_arrays['r1'][:, i], label=f'r1[{i}]', marker='o', markersize=3)
ax.set_xlabel('Iteration')
ax.set_ylabel('E[r1_i]')
ax.set_title('Bond r1 Expectations Over Time')
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

# Plot 2: r2 bond evolution
ax = axes[0, 1]
for i in range(history_arrays['r2'].shape[1]):
    ax.plot(iterations, history_arrays['r2'][:, i], label=f'r2[{i}]', marker='o', markersize=3)
ax.set_xlabel('Iteration')
ax.set_ylabel('E[r2_i]')
ax.set_title('Bond r2 Expectations Over Time')
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

# Plot 3: MSE over time
ax = axes[1, 0]
ax.plot(iterations, mse_history, 'b-', linewidth=2, marker='o')
ax.set_xlabel('Iteration')
ax.set_ylabel('MSE')
ax.set_title('Mean Squared Error Over Time')
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# Plot 4: Tau over time
ax = axes[1, 1]
ax.plot(iterations, tau_history, 'r-', linewidth=2, marker='o')
ax.set_xlabel('Iteration')
ax.set_ylabel('E[τ]')
ax.set_title('Noise Precision Over Time')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bond_evolution.png', dpi=150, bbox_inches='tight')
print("Plot saved to: bond_evolution.png")
print()

# Print final statistics
print("="*70)
print("FINAL STATISTICS")
print("="*70)
print()

print("Final bond expectations:")
for label in bond_labels:
    final_exp = history_arrays[label][-1]
    print(f"  {label}:")
    print(f"    Values: {final_exp}")
    print(f"    Min: {final_exp.min():.2f}, Max: {final_exp.max():.2f}, Mean: {final_exp.mean():.2f}")
    print(f"    Std: {final_exp.std():.2f}")
    
    # Check how many are "small" (< 1.0)
    n_small = np.sum(final_exp < 1.0)
    print(f"    Count < 1.0: {n_small}/{len(final_exp)}")
    print()

print(f"Final MSE: {mse_history[-1]:.4f}")
print(f"Final E[τ]: {tau_history[-1]:.4f}")
print()

# Analyze convergence
print("Convergence analysis:")
print(f"  MSE improvement: {mse_history[0]:.4f} → {mse_history[-1]:.4f}")
print(f"  MSE reduction: {(1 - mse_history[-1]/mse_history[0])*100:.1f}%")
print()

# Check if any bonds naturally went to zero
print("Sparsity analysis:")
for label in bond_labels:
    final_exp = history_arrays[label][-1]
    n_near_zero = np.sum(final_exp < 0.1)
    n_very_small = np.sum(final_exp < 0.5)
    n_small = np.sum(final_exp < 1.0)
    print(f"  {label}:")
    print(f"    < 0.1: {n_near_zero}/{len(final_exp)}")
    print(f"    < 0.5: {n_very_small}/{len(final_exp)}")
    print(f"    < 1.0: {n_small}/{len(final_exp)}")
print()

print("="*70)
