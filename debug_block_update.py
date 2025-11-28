"""Debug why μ collapses to zero during block update."""
import torch
from tensor.bayesian_mpo_builder import create_bayesian_tensor_train

# Generate simple data
torch.manual_seed(42)
x_train = torch.rand(20, dtype=torch.float64) * 2 - 1
y_train = 2.0 * x_train**3 - 1.0 * x_train**2 + 0.5
X_train = torch.stack([torch.ones_like(x_train), x_train], dim=1)

print("=" * 70)
print("DEBUGGING BLOCK UPDATE - WHY μ → 0")
print("=" * 70)
print(f"Data: {len(X_train)} samples")
print(f"y: mean={y_train.mean():.4f}, std={y_train.std():.4f}")
print()

# Create BMPO
bmpo = create_bayesian_tensor_train(
    num_blocks=3,
    bond_dim=4,
    input_features=2,
    output_shape=1,
    tau_alpha=torch.tensor(2.0, dtype=torch.float64),
    tau_beta=torch.tensor(1.0, dtype=torch.float64),
    dtype=torch.float64,
    seed=42
)

# Before update
mu_before = bmpo.forward_mu(X_train, to_tensor=True)
print(f"BEFORE UPDATE:")
print(f"  μ: mean={mu_before.mean():.4f}, range=[{mu_before.min():.4f}, {mu_before.max():.4f}]")
print(f"  E[τ] = {bmpo.get_tau_mean().item():.4f}")
print()

# Check block 0 tensor
print(f"Block 0 tensor before:")
print(f"  shape: {bmpo.mu_nodes[0].tensor.shape}")
print(f"  norm: {torch.norm(bmpo.mu_nodes[0].tensor).item():.4f}")
print(f"  mean: {bmpo.mu_nodes[0].tensor.mean().item():.4f}")
print(f"  std: {bmpo.mu_nodes[0].tensor.std().item():.4f}")
print()

# Update block 0
print("UPDATING BLOCK 0...")
bmpo.update_block_variational(0, X_train, y_train)

print(f"Block 0 tensor after:")
print(f"  shape: {bmpo.mu_nodes[0].tensor.shape}")
print(f"  norm: {torch.norm(bmpo.mu_nodes[0].tensor).item():.4f}")
print(f"  mean: {bmpo.mu_nodes[0].tensor.mean().item():.4f}")
print(f"  std: {bmpo.mu_nodes[0].tensor.std().item():.4f}")
print(f"  max abs value: {torch.abs(bmpo.mu_nodes[0].tensor).max().item():.6f}")
print()

# After update
mu_after = bmpo.forward_mu(X_train, to_tensor=True)
print(f"AFTER BLOCK 0 UPDATE:")
print(f"  μ: mean={mu_after.mean():.4f}, range=[{mu_after.min():.4f}, {mu_after.max():.4f}]")
print()

# Update remaining blocks
for block_idx in range(1, len(bmpo.mu_nodes)):
    print(f"UPDATING BLOCK {block_idx}...")
    bmpo.update_block_variational(block_idx, X_train, y_train)
    
    print(f"Block {block_idx} tensor after:")
    print(f"  shape: {bmpo.mu_nodes[block_idx].tensor.shape}")
    print(f"  norm: {torch.norm(bmpo.mu_nodes[block_idx].tensor).item():.4f}")
    print(f"  mean: {bmpo.mu_nodes[block_idx].tensor.mean().item():.4f}")
    print(f"  std: {bmpo.mu_nodes[block_idx].tensor.std().item():.4f}")
    print(f"  max abs value: {torch.abs(bmpo.mu_nodes[block_idx].tensor).max().item():.6f}")
    print()

mu_after_blocks = bmpo.forward_mu(X_train, to_tensor=True)
print(f"AFTER ALL BLOCKS:")
print(f"  μ: mean={mu_after_blocks.mean():.4f}, range=[{mu_after_blocks.min():.4f}, {mu_after_blocks.max():.4f}]")
print()

# Update bonds
print("UPDATING BONDS...")
bond_labels = list(bmpo.mu_mpo.distributions.keys())
for label in bond_labels:
    bmpo.update_bond_variational(label)
print(f"  Updated {len(bond_labels)} bonds")
print()

mu_after_bonds = bmpo.forward_mu(X_train, to_tensor=True)
print(f"AFTER BONDS:")
print(f"  μ: mean={mu_after_bonds.mean():.4f}, range=[{mu_after_bonds.min():.4f}, {mu_after_bonds.max():.4f}]")
print()

# Update τ
print("UPDATING τ...")
print(f"  E[τ] before: {bmpo.get_tau_mean().item():.6f}")
bmpo.update_tau_variational(X_train, y_train)
print(f"  E[τ] after: {bmpo.get_tau_mean().item():.6f}")
print()

mu_iter1 = bmpo.forward_mu(X_train, to_tensor=True)
print(f"END OF ITERATION 1:")
print(f"  μ: mean={mu_iter1.mean():.4f}, range=[{mu_iter1.min():.4f}, {mu_iter1.max():.4f}]")
print(f"  E[τ]: {bmpo.get_tau_mean().item():.6f}")
print()
print("=" * 70)
print("ITERATION 2")
print("=" * 70)
print()

# Iteration 2 - Update blocks
for block_idx in range(len(bmpo.mu_nodes)):
    bmpo.update_block_variational(block_idx, X_train, y_train)

mu_iter2_blocks = bmpo.forward_mu(X_train, to_tensor=True)
print(f"AFTER BLOCK UPDATES:")
print(f"  μ: mean={mu_iter2_blocks.mean():.4f}, range=[{mu_iter2_blocks.min():.4f}, {mu_iter2_blocks.max():.4f}]")

# Check individual block norms
for i, node in enumerate(bmpo.mu_nodes):
    print(f"  Block {i} norm: {torch.norm(node.tensor).item():.6f}, max abs: {torch.abs(node.tensor).max().item():.6f}")

print("=" * 70)
