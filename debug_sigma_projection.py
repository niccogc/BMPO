"""
Debug script to verify sigma projection is computed correctly
"""
import torch
from tensor.bayesian_tn import BayesianTensorNetwork

# Simple test: 2D input, 1D output, MPS topology
x_train = torch.randn(10, 2)
y_train = torch.randn(10, 1)

# Create small network
network = BayesianTensorNetwork(
    input_dim=2,
    output_dim=1,
    num_features=2,
    bond_dim=3,
    output_layer=ScalarOutput(),
    topology='mps',
    device='cpu'
)

print("=" * 70)
print("SIGMA PROJECTION DEBUG")
print("=" * 70)

# Get initial state
nodes = list(network.mu_network.learnable_nodes)
print(f"\nNodes: {nodes}")
print(f"Bonds: {list(network.mu_network.bond_labels)}")

# Check node shapes
for node in nodes:
    mu_shape = network.mu_network.get_node_tensor(node).shape
    sigma_tag = node + '_sigma'
    sigma_shape = network.sigma_network.get_node_tensor(sigma_tag).shape
    print(f"\n{node}:")
    print(f"  Mu shape: {mu_shape}")
    print(f"  Sigma shape: {sigma_shape}")

# Perform one update iteration
print("\n" + "=" * 70)
print("UPDATING FIRST NODE")
print("=" * 70)

node_to_update = nodes[0]
print(f"\nUpdating node: {node_to_update}")

# Get node shape and compute d
mu_tensor = network.mu_network.get_node_tensor(node_to_update)
node_shape = mu_tensor.shape
d = torch.prod(torch.tensor(node_shape)).item()
print(f"Node shape: {node_shape}, d={d}")

# Prepare inputs
inputs_dict = {f'x{i}': x_train[:, i] for i in range(2)}

# Compute mu projection
print("\n--- Mu Projection ---")
J_mu = network.compute_projection(node_to_update, network_type='mu', inputs=inputs_dict)
print(f"J_mu shape: {J_mu.shape}")
print(f"J_mu mean: {J_mu.mean().item():.6f}, std: {J_mu.std().item():.6f}")

# Compute sigma projection
print("\n--- Sigma Projection ---")
try:
    J_sigma = network.compute_projection(node_to_update, network_type='sigma', inputs=inputs_dict)
    print(f"J_sigma shape: {J_sigma.shape}")
    print(f"J_sigma mean: {J_sigma.mean().item():.6f}, std: {J_sigma.std().item():.6f}")
    
    # Try to reshape as in the update code
    batch_size = x_train.shape[0]
    num_var_dims = len(node_shape)
    output_dim_start = 1 + 2 * num_var_dims
    output_dims_to_sum = list(range(output_dim_start, J_sigma.ndim))
    
    print(f"\nReshaping logic:")
    print(f"  num_var_dims: {num_var_dims}")
    print(f"  output_dim_start: {output_dim_start}")
    print(f"  output_dims_to_sum: {output_dims_to_sum}")
    
    if output_dims_to_sum:
        J_sigma_summed = J_sigma.sum(dim=output_dims_to_sum)
    else:
        J_sigma_summed = J_sigma
    print(f"  J_sigma_summed shape: {J_sigma_summed.shape}")
    
    # Reshape to (batch, d, d)
    J_sigma_reshaped = J_sigma_summed.reshape(batch_size, d, d)
    print(f"  J_sigma_reshaped shape: {J_sigma_reshaped.shape}")
    
    # Sum over batch
    sum_J_sigma = J_sigma_reshaped.sum(dim=0)
    print(f"  sum_J_sigma shape: {sum_J_sigma.shape}")
    print(f"  sum_J_sigma diagonal mean: {sum_J_sigma.diagonal().mean().item():.6f}")
    print(f"  sum_J_sigma off-diagonal mean: {(sum_J_sigma.sum() - sum_J_sigma.diagonal().sum()).item() / (d*d - d):.6f}")
    
    print("\n✓ Sigma projection computed successfully!")
    
except Exception as e:
    print(f"\n✗ ERROR during sigma projection:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("FULL UPDATE ITERATION")
print("=" * 70)

# Try a full update
try:
    network.update_node_variational(node_to_update, inputs_dict, y_train)
    print("\n✓ Full update completed successfully!")
except Exception as e:
    print(f"\n✗ ERROR during full update:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
