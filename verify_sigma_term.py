"""
Verify that sigma projection term is non-zero in precision matrix calculation.
"""
import torch
import quimb.tensor as qtn
from tensor.bayesian_tn import BayesianTensorNetwork
from tensor.bayesian_tn_builder import create_sigma_network

# Create tiny network: A(x, r1) -- B(r1, y)
torch.manual_seed(42)
A_data = torch.randn(3, 2, dtype=torch.float64)
B_data = torch.randn(2, 1, dtype=torch.float64)

A = qtn.Tensor(data=A_data, inds=('x', 'r1'), tags='A')
B = qtn.Tensor(data=B_data, inds=('r1', 'y'), tags='B')
mu_tn = qtn.TensorNetwork([A, B])

sigma_tn = create_sigma_network(mu_tn, ['A', 'B'], output_indices=['y'])

model = BayesianTensorNetwork(
    mu_tn=mu_tn,
    sigma_tn=sigma_tn,
    input_indices={'data': ['x']},
    output_indices=['y'],
    learnable_tags=['A', 'B'],
    dtype=torch.float64
)

# Generate small batch
batch_size = 5
X = torch.randn(batch_size, 3, dtype=torch.float64)
y = torch.randn(batch_size, 1, dtype=torch.float64)
inputs_dict = {'data': X}

print("=" * 70)
print("VERIFYING SIGMA PROJECTION TERM IN PRECISION MATRIX")
print("=" * 70)

# Patch the update_node_variational to print debug info
original_update = model.update_node_variational

def debug_update(node_tag, inputs_dict, y_data):
    """Wrapper to add debug output."""
    # Get node info
    mu_tensor = model.mu_network.get_node_tensor(node_tag)
    node_shape = mu_tensor.shape
    d = torch.prod(torch.tensor(node_shape)).item()
    
    print(f"\nUpdating node: {node_tag}")
    print(f"  Node shape: {node_shape}, d={d}")
    
    # Compute projections
    J_mu = model.compute_projection(node_tag, network_type='mu', inputs=inputs_dict)
    print(f"  J_mu shape: {J_mu.shape}")
    
    # Flatten and compute outer product term
    J_mu_flat = J_mu.reshape(J_mu.shape[0], -1, J_mu.shape[-1])
    J_mu_batch = J_mu_flat.sum(dim=-1)
    sum_J_mu_outer = torch.einsum('bd,bD->dD', J_mu_batch, J_mu_batch)
    print(f"  sum_J_mu_outer: mean={sum_J_mu_outer.mean().item():.6f}, diagonal_mean={sum_J_mu_outer.diagonal().mean().item():.6f}")
    
    # Compute sigma projection
    J_sigma_all = model.compute_projection(node_tag, network_type='sigma', inputs=inputs_dict)
    print(f"  J_sigma_all shape: {J_sigma_all.shape}")
    
    # Process sigma projection
    num_var_dims = len(node_shape)
    output_dim_start = 1 + 2 * num_var_dims
    output_dims_to_sum = list(range(output_dim_start, J_sigma_all.ndim))
    
    if output_dims_to_sum:
        J_sigma_summed = J_sigma_all.sum(dim=output_dims_to_sum)
    else:
        J_sigma_summed = J_sigma_all
    
    J_sigma_reshaped = J_sigma_summed.reshape(batch_size, d, d)
    sum_J_sigma = J_sigma_reshaped.sum(dim=0)
    
    print(f"  sum_J_sigma: mean={sum_J_sigma.mean().item():.6f}, diagonal_mean={sum_J_sigma.diagonal().mean().item():.6f}")
    
    # Check ratio
    mu_outer_mag = sum_J_mu_outer.abs().mean().item()
    sigma_mag = sum_J_sigma.abs().mean().item()
    ratio = sigma_mag / mu_outer_mag if mu_outer_mag > 0 else 0
    
    print(f"  Sigma/Mu ratio: {ratio:.6f}")
    
    if sigma_mag < 1e-10:
        print("  ⚠️ WARNING: Sigma term is nearly zero!")
    elif ratio < 0.01:
        print("  ⚠️ WARNING: Sigma term is much smaller than mu outer product!")
    else:
        print("  ✓ Sigma term has reasonable magnitude")
    
    # Call original method
    return original_update(node_tag, inputs_dict, y_data)

model.update_node_variational = debug_update

# Run one update iteration
print("\n" + "-" * 70)
print("Running one update iteration...")
print("-" * 70)

try:
    model.variational_update_iteration(y, inputs_dict)
    print("\n" + "=" * 70)
    print("✓ UPDATE SUCCESSFUL - SIGMA TERM IS BEING COMPUTED!")
    print("=" * 70)
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
