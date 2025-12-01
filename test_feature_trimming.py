"""
Test feature dimension trimming.

Environment variables:
- NUM_BLOCKS: Number of blocks (default: 4)
- BOND_DIM: Initial bond dimension (default: 10)
- TRIM_THRESHOLD: Trimming threshold (default: 0.5)
- MAX_ITER: Training iterations (default: 20)
"""
import torch
import os
from tensor.bayesian_mpo_builder import create_bayesian_tensor_train

# Read from environment
NUM_BLOCKS = int(os.environ.get('NUM_BLOCKS', '4'))
BOND_DIM = int(os.environ.get('BOND_DIM', '10'))
TRIM_THRESHOLD = float(os.environ.get('TRIM_THRESHOLD', '0.5'))
MAX_ITER = int(os.environ.get('MAX_ITER', '20'))
NUM_SAMPLES = 100

torch.manual_seed(42)

# Generate polynomial data
x = torch.rand(NUM_SAMPLES, dtype=torch.float64) * 2 - 1
y = 2.0 * x**3 - 1.0 * x**2 + 0.5
y += 0.1 * torch.randn(NUM_SAMPLES, dtype=torch.float64)
X = torch.stack([torch.ones_like(x), x], dim=1)

print("="*80)
print("FEATURE DIMENSION TRIMMING TEST")
print("="*80)
print(f"Config: {NUM_BLOCKS} blocks, bond_dim={BOND_DIM}, threshold={TRIM_THRESHOLD}")
print()

# Create model
bmpo = create_bayesian_tensor_train(
    num_blocks=NUM_BLOCKS,
    bond_dim=BOND_DIM,
    input_features=2,
    output_shape=1,
    dtype=torch.float64,
    seed=42
)

print("BEFORE trimming:")
print(f"  Distributions: {list(bmpo.mu_mpo.distributions.keys())}")
for label, dist in bmpo.mu_mpo.distributions.items():
    print(f"    {label}: size={len(dist['expectation'])}, E={dist['expectation'].tolist()}")

print()
print(f"  Blocks:")
for i, node in enumerate(bmpo.mu_nodes):
    print(f"    Block {i}: shape={node.shape}, labels={node.dim_labels}")

print()
print(f"  Input nodes:")
for i, node in enumerate(bmpo.input_nodes):
    print(f"    Input {i}: shape={node.shape}, labels={node.dim_labels}")

print()
print(f"Training for {MAX_ITER} iterations...")
bmpo.fit(X, y, max_iter=MAX_ITER, verbose=False, trim_threshold=TRIM_THRESHOLD)

print()
print("AFTER trimming:")
print(f"  Distributions: {list(bmpo.mu_mpo.distributions.keys())}")
for label, dist in bmpo.mu_mpo.distributions.items():
    print(f"    {label}: size={len(dist['expectation'])}, E={dist['expectation'].tolist()}")

print()
print(f"  Blocks:")
for i, node in enumerate(bmpo.mu_nodes):
    print(f"    Block {i}: shape={node.shape}, labels={node.dim_labels}")

print()
print(f"  Input nodes:")
for i, node in enumerate(bmpo.input_nodes):
    print(f"    Input {i}: shape={node.shape}, labels={node.dim_labels}")

# Test prediction
mu = bmpo.forward_mu(X, to_tensor=True)
mse = ((mu.squeeze() - y)**2).mean().item()
ss_res = ((mu.squeeze() - y)**2).sum().item()
ss_tot = ((y - y.mean())**2).sum().item()
r2 = 1 - ss_res / ss_tot

print()
print("="*80)
print("RESULTS:")
print(f"  MSE: {mse:.4f}")
print(f"  R²: {r2:.4f}")
print("="*80)
