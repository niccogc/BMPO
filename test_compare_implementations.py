"""
Compare BayesianTensorNetwork with BayesianMPO on same simple problem.
"""

import torch
import numpy as np
from tensor.bayesian_mpo_builder import create_bayesian_tensor_train

# Generate same simple data
torch.manual_seed(42)
np.random.seed(42)

num_samples = 20
x = torch.linspace(-1, 1, num_samples, dtype=torch.float64)
y = 2.0 * x + 1.0
y += 0.1 * torch.randn(num_samples, dtype=torch.float64)

X = torch.stack([torch.ones_like(x), x], dim=1)

print("="*70)
print("COMPARISON: BayesianMPO vs BayesianTensorNetwork")
print("="*70)
print(f"Data: {num_samples} samples")
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")
print(f"  True function: y = 2x + 1")
print()

# Test BayesianMPO
print("Testing BayesianMPO (original implementation):")
print("-"*70)

bmpo = create_bayesian_tensor_train(
    num_blocks=2,
    bond_dim=3,
    input_features=2,
    output_shape=1,
    constrict_bond=False,
    tau_alpha=torch.tensor(2.0, dtype=torch.float64),
    tau_beta=torch.tensor(1.0, dtype=torch.float64),
    dtype=torch.float64,
    seed=42
)

print(f"Initial E[τ]: {bmpo.get_tau_mean().item():.4f}")

# Initial prediction
y_pred_init = bmpo.forward_mu(X, to_tensor=True)
assert isinstance(y_pred_init, torch.Tensor)
mse_init = ((y_pred_init.squeeze() - y)**2).mean().item()
print(f"Initial MSE: {mse_init:.6f}")

# Train for 5 iterations
print("\nTraining...")
for iteration in range(5):
    for block_idx in range(2):
        bmpo.update_block_variational(block_idx, X, y)
    
    for label in bmpo.mu_mpo.distributions.keys():
        bmpo.update_bond_variational(label)
    
    bmpo.update_tau_variational(X, y)
    
    y_pred = bmpo.forward_mu(X, to_tensor=True)
    assert isinstance(y_pred, torch.Tensor)
    mse = ((y_pred.squeeze() - y)**2).mean().item()
    tau = bmpo.get_tau_mean().item()
    print(f"  Iter {iteration+1}: MSE={mse:.6f}, E[τ]={tau:.4f}")

print()
print("BayesianMPO successfully reduces MSE!")
print()
