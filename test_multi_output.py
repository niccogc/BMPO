import torch
import numpy as np
import quimb.tensor as qt
from tensor.btn import BTN
from tensor.builder import Inputs

torch.set_default_dtype(torch.float64)

print("="*70)
print("ELBO TEST WITH MULTIPLE OUTPUT DIMENSIONS")
print("="*70)

# Create dataset with 3 output dimensions
N = 100  # samples
input_dim_per_site = 3  # Features per input site
output_dim = 3  # Number of output classes

# Generate synthetic data
torch.manual_seed(42)
X = torch.randn(N, input_dim_per_site)  # Will be repeated for both x1 and x2
W = torch.randn(input_dim_per_site, output_dim)  # True weights
y = X @ W + 0.1 * torch.randn(N, output_dim)  # Add small noise

print(f"\nDataset: {N} samples, {input_dim_per_site} input dims per site, {output_dim} output dims")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Prepare inputs using torch tensors (Inputs class expects torch tensors)
# Same input tensor repeated for x1 and x2 (like in the polynomial test)
data_stream = Inputs(
    inputs=[X],
    outputs=[y],
    input_labels=['x1', 'x2'],  # Two input sites using same features
    outputs_labels=['y'],
    batch_dim='s',
    batch_size=50
)

# Create proper MPS network with cyclic structure (like nicco_test_poly.py)
# x1 -> b1/b3 -> x2 -> b2 -> y (3 output dims) with b3 connecting back
bond_dim = 16
input_dim_per_site = 3  # Split input features across sites

def init_weights(shape):
    w = torch.randn(*shape, dtype=torch.float64)
    return w/torch.norm(w)

# Node1: First input site with two bonds (b1 and b3, like in the poly test)
node1 = qt.Tensor(
    data=init_weights((input_dim_per_site, bond_dim, bond_dim)),
    inds=['x1', 'b1', 'b3'],
    tags=['Node1']
)

# Node2: Second input site  
node2 = qt.Tensor(
    data=init_weights((bond_dim, input_dim_per_site, bond_dim)),
    inds=['b1', 'x2', 'b2'],
    tags=['Node2']
)

# Node3: Output site with 3 output dimensions and cyclic bond b3
node3 = qt.Tensor(
    data=init_weights((bond_dim, bond_dim, output_dim)),
    inds=['b2', 'b3', 'y'],
    tags=['Node3']
)

mu = qt.TensorNetwork([node1, node2, node3])

print(f"\nNetwork structure:")
print(f"  Node1: {node1.inds} with shape {node1.shape}")
print(f"  Node2: {node2.inds} with shape {node2.shape}")
print(f"  Node3: {node3.inds} with shape {node3.shape}")
print(f"  Output dimension 'y' has size: {mu.ind_size('y')}")
print(f"  Number of output classes: {mu.ind_size('y')}")

# Initialize BTN
print("\n--- Initializing BTN ---")
model = BTN(mu=mu, data_stream=data_stream, not_trainable_nodes=[])

# Test initial KL computation
print("\n--- Initial KL Computation ---")
bond_kl_init = model.compute_bond_kl(verbose=False)
node_kl_init = model.compute_node_kl(verbose=True, debug=False)
print(f"\nTotal Bond KL: {bond_kl_init:.4f}")
print(f"Total Node KL: {node_kl_init:.4f}")

# Test ELBO before training
print("\n--- Initial ELBO ---")
elbo_init = model.compute_elbo(verbose=True, print_components=False)

# Train for a few epochs
print("\n--- Training with ELBO tracking ---")
model.fit(epochs=5, track_elbo=True)

# Test final KL computation
print("\n--- Final KL Computation ---")
bond_kl_final = model.compute_bond_kl(verbose=False)
node_kl_final = model.compute_node_kl(verbose=True, debug=False)
print(f"\nTotal Bond KL: {bond_kl_final:.4f}")
print(f"Total Node KL: {node_kl_final:.4f}")

# Final ELBO
print("\n--- Final ELBO ---")
elbo_final = model.compute_elbo(verbose=True, print_components=False)

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"ELBO change: {elbo_init:.4f} → {elbo_final:.4f} (Δ = {elbo_final - elbo_init:+.4f})")
print(f"Bond KL change: {bond_kl_init:.4f} → {bond_kl_final:.4f}")
print(f"Node KL change: {node_kl_init:.4f} → {node_kl_final:.4f}")

if elbo_final > elbo_init:
    print("\n✓ ELBO INCREASED - Training successful!")
else:
    print("\n✗ ELBO DECREASED - Check implementation")

print("="*70)
