"""
Debug why 4+ blocks cause model to collapse to zero.

Environment variables:
- NUM_BLOCKS: Number of blocks (default: 4)
- BOND_DIM: Bond dimension (default: 6)
- INPUT_DIM: Input features (default: 2 for 1D, 3 for 2D)
- NUM_SAMPLES: Number of samples (default: 100)
- PROBLEM: '1d' or '2d' (default: '1d')
"""

import torch
import os
from tensor.bayesian_mpo_builder import create_bayesian_tensor_train

torch.manual_seed(42)

# Read hyperparameters from environment
NUM_BLOCKS = int(os.environ.get('NUM_BLOCKS', '4'))
BOND_DIM = int(os.environ.get('BOND_DIM', '6'))
PROBLEM = os.environ.get('PROBLEM', '1d')
NUM_SAMPLES = int(os.environ.get('NUM_SAMPLES', '100'))

if PROBLEM == '1d':
    INPUT_DIM = 2  # [1, x]
else:
    INPUT_DIM = 3  # [1, x, y]

print("="*70)
print(f"DEBUG: {NUM_BLOCKS} BLOCKS, {PROBLEM.upper()} PROBLEM")
print("="*70)
print(f"Config: blocks={NUM_BLOCKS}, bond_dim={BOND_DIM}, input_dim={INPUT_DIM}, samples={NUM_SAMPLES}")
print()

# Generate data
if PROBLEM == '1d':
    x = torch.rand(NUM_SAMPLES, dtype=torch.float64) * 2 - 1
    y = 2.0 * x**3 - 1.0 * x**2 + 0.5
    y += 0.1 * torch.randn(NUM_SAMPLES, dtype=torch.float64)
    X = torch.stack([torch.ones_like(x), x], dim=1)
else:
    x = torch.rand(NUM_SAMPLES, dtype=torch.float64) * 2 - 1
    y_coord = torch.rand(NUM_SAMPLES, dtype=torch.float64) * 2 - 1
    y = 0.5 * x**2 + 0.3 * y_coord**2 - 0.4 * x * y_coord + 0.5
    y += 0.05 * torch.randn(NUM_SAMPLES, dtype=torch.float64)
    X = torch.stack([torch.ones_like(x), x, y_coord], dim=1)

print(f"Data: X.shape={X.shape}, y.shape={y.shape}")
print(f"Target: mean={y.mean():.4f}, std={y.std():.4f}")
print()

# Create model
print(f"Creating {NUM_BLOCKS}-block Bayesian MPO...")
bmpo = create_bayesian_tensor_train(
    num_blocks=NUM_BLOCKS,
    bond_dim=BOND_DIM,
    input_features=INPUT_DIM,
    output_shape=1,
    constrict_bond=False,
    dtype=torch.float64,
    seed=42,
    random_priors=True,
    prior_seed=42
)

print("\n1. CHECK STRUCTURE")
print("-" * 70)
print("Nodes:")
for i, node in enumerate(bmpo.mu_nodes):
    print(f"  Block {i}: shape {node.shape}, labels {node.dim_labels}")
    print(f"    tensor stats: min={node.tensor.min():.6f}, max={node.tensor.max():.6f}, mean={node.tensor.mean():.6f}")

print("\nConnections:")
for i, node in enumerate(bmpo.mu_nodes):
    conn_labels = list(node.connections.keys())
    print(f"  Block {i}: {conn_labels}")

print("\nInput nodes:")
for i, node in enumerate(bmpo.input_nodes):
    print(f"  Input {i}: shape {node.shape}, labels {node.dim_labels}")

print("\n2. CHECK FORWARD PASS")
print("-" * 70)

# Test with small batch
X_test = X[:5]
y_test = y[:5]

print(f"Test input: {X_test.shape}")
mu_init = bmpo.forward_mu(X_test, to_tensor=True)
assert isinstance(mu_init, torch.Tensor)
print(f"Initial prediction shape: {mu_init.shape}")
print(f"Initial predictions: {mu_init.squeeze()}")
print(f"True values: {y_test}")
print(f"Pred stats: min={mu_init.min():.6f}, max={mu_init.max():.6f}, mean={mu_init.mean():.6f}, std={mu_init.std():.6f}")

# Check on full data
mu_full = bmpo.forward_mu(X, to_tensor=True)
assert isinstance(mu_full, torch.Tensor)
mse_init = ((mu_full.squeeze() - y)**2).mean().item()
print(f"\nFull data MSE (initial): {mse_init:.6f}")
print(f"Full pred stats: mean={mu_full.mean():.6f}, std={mu_full.std():.6f}")
print(f"Expected MSE if pred=0: {(y**2).mean():.6f}")
print(f"Expected MSE if pred=mean: {((y - y.mean())**2).mean():.6f}")

print("\n3. CHECK BLOCK UPDATES ONE BY ONE")
print("-" * 70)

for block_idx in range(NUM_BLOCKS):
    print(f"\n--- Updating Block {block_idx} ---")
    
    # Before update
    print(f"Before update:")
    print(f"  Block {block_idx} tensor stats:")
    print(f"    min={bmpo.mu_nodes[block_idx].tensor.min():.6f}")
    print(f"    max={bmpo.mu_nodes[block_idx].tensor.max():.6f}")
    print(f"    mean={bmpo.mu_nodes[block_idx].tensor.mean():.6f}")
    print(f"    sum={bmpo.mu_nodes[block_idx].tensor.sum():.6f}")
    print(f"    abs_sum={bmpo.mu_nodes[block_idx].tensor.abs().sum():.6f}")
    
    mu_before = bmpo.forward_mu(X, to_tensor=True)
    assert isinstance(mu_before, torch.Tensor)
    mse_before = ((mu_before.squeeze() - y)**2).mean().item()
    print(f"  MSE: {mse_before:.6f}")
    print(f"  Pred: mean={mu_before.mean():.6f}, std={mu_before.std():.6f}")
    
    # Perform update
    bmpo.update_block_variational(block_idx, X, y)
    
    # After update
    print(f"After update:")
    print(f"  Block {block_idx} tensor stats:")
    print(f"    min={bmpo.mu_nodes[block_idx].tensor.min():.6f}")
    print(f"    max={bmpo.mu_nodes[block_idx].tensor.max():.6f}")
    print(f"    mean={bmpo.mu_nodes[block_idx].tensor.mean():.6f}")
    print(f"    sum={bmpo.mu_nodes[block_idx].tensor.sum():.6f}")
    print(f"    abs_sum={bmpo.mu_nodes[block_idx].tensor.abs().sum():.6f}")
    
    mu_after = bmpo.forward_mu(X, to_tensor=True)
    assert isinstance(mu_after, torch.Tensor)
    mse_after = ((mu_after.squeeze() - y)**2).mean().item()
    print(f"  MSE: {mse_after:.6f}")
    print(f"  Pred: mean={mu_after.mean():.6f}, std={mu_after.std():.6f}")
    
    # Check if collapsed
    if mu_after.std().item() < 1e-6:
        print(f"  ⚠️  WARNING: Predictions collapsed to near-constant!")
        print(f"  ⚠️  Block {block_idx} update caused collapse!")
        break
    
    if bmpo.mu_nodes[block_idx].tensor.abs().sum() < 1e-6:
        print(f"  ⚠️  WARNING: Block {block_idx} tensor collapsed to zero!")
        break

print("\n4. FINAL STATE")
print("-" * 70)
mu_final = bmpo.forward_mu(X, to_tensor=True)
assert isinstance(mu_final, torch.Tensor)
mse_final = ((mu_final.squeeze() - y)**2).mean().item()
print(f"Final MSE: {mse_final:.6f}")
print(f"Final pred: mean={mu_final.mean():.6f}, std={mu_final.std():.6f}")

print("\n" + "="*70)
print("DONE")
print("="*70)
