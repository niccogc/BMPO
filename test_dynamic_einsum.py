"""
Test dynamic einsum with 3 blocks to verify labels and shapes.
"""
import torch
import quimb.tensor as qtn
from tensor.bayesian_tn import BayesianTensorNetwork
from tensor.bayesian_tn_builder import create_sigma_network

# Create 3-block network: A(x, r1) -- B(r1, r2) -- C(r2, y)
# Small dimensions for clarity
torch.manual_seed(42)

print("=" * 80)
print("TESTING DYNAMIC EINSUM WITH 3 BLOCKS")
print("=" * 80)

# Block dimensions: small for testing
A_data = torch.randn(3, 2, dtype=torch.float64)  # x=3, r1=2
B_data = torch.randn(2, 2, dtype=torch.float64)  # r1=2, r2=2
C_data = torch.randn(2, 1, dtype=torch.float64)  # r2=2, y=1

A = qtn.Tensor(data=A_data, inds=('x', 'r1'), tags='A')
B = qtn.Tensor(data=B_data, inds=('r1', 'r2'), tags='B')
C = qtn.Tensor(data=C_data, inds=('r2', 'y'), tags='C')

mu_tn = qtn.TensorNetwork([A, B, C])
sigma_tn = create_sigma_network(mu_tn, ['A', 'B', 'C'], output_indices=['y'])

model = BayesianTensorNetwork(
    mu_tn=mu_tn,
    sigma_tn=sigma_tn,
    input_indices={'data': ['x']},
    output_indices=['y'],
    learnable_tags=['A', 'B', 'C'],
    dtype=torch.float64
)

print("\nNetwork Structure:")
print("  A(x=3, r1=2) -- B(r1=2, r2=2) -- C(r2=2, y=1)")
print("  Learnable blocks: A, B, C")
print("  Input index: x")
print("  Output index: y")

# Generate small batch of data
batch_size = 5
X = torch.randn(batch_size, 3, dtype=torch.float64)
y = torch.randn(batch_size, 1, dtype=torch.float64)
inputs_dict = {'data': X}

print("\n" + "=" * 80)
print("BLOCK-BY-BLOCK ANALYSIS")
print("=" * 80)

for node_tag in ['A', 'B', 'C']:
    print(f"\n{'='*80}")
    print(f"BLOCK: {node_tag}")
    print(f"{'='*80}")
    
    # Get node information
    node_inds = model.mu_network.get_node_inds(node_tag)
    node_shape = model.mu_network.get_node_shape(node_tag)
    var_shape, out_shape = model.get_node_dimensions(node_tag)
    
    print(f"\n1. Node Structure:")
    print(f"   Indices: {node_inds}")
    print(f"   Full shape: {node_shape}")
    print(f"   Variational shape: {var_shape}")
    print(f"   Output shape: {out_shape}")
    
    # Compute mu projection
    J_mu = model.compute_projection(node_tag, network_type='mu', inputs=inputs_dict)
    print(f"\n2. Mu Projection:")
    print(f"   J_mu shape: {J_mu.shape}")
    print(f"   Expected: (batch={batch_size}, *var_shape={var_shape}, *other_outputs)")
    
    # Compute sigma projection
    J_sigma = model.compute_projection(node_tag, network_type='sigma', inputs=inputs_dict)
    print(f"\n3. Sigma Projection:")
    print(f"   J_sigma shape: {J_sigma.shape}")
    print(f"   Expected: (batch={batch_size}, *var_interleaved, *other_outputs)")
    
    # Simulate what happens in update_node_variational
    n_var_dims = len(var_shape)
    n_other_output_dims = J_mu.ndim - 1 - n_var_dims
    
    # Generate labels
    batch_label = 'b'
    var_labels = ''.join([chr(ord('i') + i) for i in range(n_var_dims)])
    other_out_labels = ''.join([chr(ord('o') + i) for i in range(n_other_output_dims)])
    
    print(f"\n4. Einsum Labels:")
    print(f"   n_var_dims: {n_var_dims}")
    print(f"   n_other_output_dims: {n_other_output_dims}")
    print(f"   batch_label: '{batch_label}'")
    print(f"   var_labels: '{var_labels}'")
    print(f"   other_out_labels: '{other_out_labels}'")
    
    # Mu einsum strings
    J_mu_indices = batch_label + var_labels + other_out_labels
    print(f"\n5. Mu Einsum Strings:")
    print(f"   J_mu indices: '{J_mu_indices}'")
    
    # Build "actual" label representation
    actual_mu_labels = f"(batch, {', '.join(str(ind) for ind in node_inds if ind not in model.output_indices)}"
    if n_other_output_dims > 0:
        actual_mu_labels += f", other_outputs)"
    else:
        actual_mu_labels += ")"
    print(f"   Actual labels: {actual_mu_labels}")
    
    # Term 1: sum_y_J_mu
    einsum_str_1 = f'{J_mu_indices},{batch_label}->{var_labels}'
    actual_einsum_1 = f"J_mu{actual_mu_labels}, y(batch) -> sum_y_J_mu({', '.join(str(ind) for ind in node_inds if ind not in model.output_indices)})"
    print(f"   Term 1 (sum_y_J_mu):")
    print(f"     Einsum: '{einsum_str_1}'")
    print(f"     Actual: {actual_einsum_1}")
    y_vec = y.reshape(batch_size)
    sum_y_J_mu = torch.einsum(einsum_str_1, J_mu, y_vec)
    print(f"     Result shape: {sum_y_J_mu.shape}")
    
    # Term 2: sum_J_mu_outer
    var_labels_upper = var_labels.upper()
    einsum_str_2 = f'{J_mu_indices},{batch_label}{var_labels_upper}{other_out_labels}->{var_labels}{var_labels_upper}'
    var_inds_list = [str(ind) for ind in node_inds if ind not in model.output_indices]
    var_inds_prime = [ind + "'" for ind in var_inds_list]
    actual_einsum_2 = f"J_mu({actual_mu_labels}), J_mu'({actual_mu_labels}) -> sum_J_mu_outer({', '.join(var_inds_list + var_inds_prime)})"
    print(f"   Term 2 (sum_J_mu_outer):")
    print(f"     Einsum: '{einsum_str_2}'")
    print(f"     Actual: {actual_einsum_2}")
    sum_J_mu_outer = torch.einsum(einsum_str_2, J_mu, J_mu)
    print(f"     Result shape: {sum_J_mu_outer.shape}")
    
    # Sigma einsum strings - matching actual implementation
    var_labels_out = var_labels  # Same as mu for 'o'
    var_labels_in = var_labels.upper()  # Uppercase for 'i'
    
    # Build interleaved string
    var_labels_interleaved = ''
    for i in range(n_var_dims):
        var_labels_interleaved += var_labels[i]  # 'o'
        var_labels_interleaved += var_labels_in[i]  # 'i'
    
    J_sigma_indices = batch_label + var_labels_interleaved + other_out_labels
    print(f"\n6. Sigma Einsum Strings:")
    print(f"   var_labels (for 'o'): '{var_labels}'")
    print(f"   var_labels_in (for 'i'): '{var_labels_in}'")
    print(f"   var_labels_interleaved: '{var_labels_interleaved}'")
    print(f"   J_sigma indices: '{J_sigma_indices}'")
    
    # Build actual label representation for sigma
    var_inds_list = [str(ind) for ind in node_inds if ind not in model.output_indices]
    var_inds_out = [ind + "_o" for ind in var_inds_list]
    var_inds_in = [ind + "_i" for ind in var_inds_list]
    # Interleave: ind1_o, ind1_i, ind2_o, ind2_i, ...
    var_inds_interleaved = []
    for i in range(len(var_inds_list)):
        var_inds_interleaved.append(var_inds_out[i])
        var_inds_interleaved.append(var_inds_in[i])
    
    actual_sigma_labels = f"(batch, {', '.join(var_inds_interleaved)}"
    if n_other_output_dims > 0:
        actual_sigma_labels += ", other_outputs)"
    else:
        actual_sigma_labels += ")"
    print(f"   Actual labels: {actual_sigma_labels}")
    
    # Term 3: sum_J_sigma - output should match mu_outer format
    einsum_str_sigma = f'{J_sigma_indices}->{var_labels}{var_labels_upper}'
    actual_einsum_sigma = f"J_sigma{actual_sigma_labels} -> sum_J_sigma({', '.join(var_inds_list + [ind + \"'\" for ind in var_inds_list])})"
    print(f"   Term 3 (sum_J_sigma):")
    print(f"     Einsum: '{einsum_str_sigma}'")
    print(f"     Actual: {actual_einsum_sigma}")
    sum_J_sigma = torch.einsum(einsum_str_sigma, J_sigma)
    print(f"     Result shape: {sum_J_sigma.shape}")
    
    # Verify shapes match
    print(f"\n7. Shape Verification:")
    print(f"   sum_J_mu_outer shape: {sum_J_mu_outer.shape}")
    print(f"   sum_J_sigma shape:    {sum_J_sigma.shape}")
    if sum_J_mu_outer.shape == sum_J_sigma.shape:
        print(f"   ✓ Shapes match!")
    else:
        print(f"   ✗ ERROR: Shapes don't match!")
    
    # Get theta
    theta = model.compute_theta_tensor(node_tag)
    print(f"   theta shape:          {theta.shape}")
    if theta.shape == sum_J_mu_outer.shape:
        print(f"   ✓ Theta shape matches!")
    else:
        print(f"   ⚠ Theta shape different (expected for output nodes)")

print("\n" + "=" * 80)
print("RUNNING FULL UPDATE ITERATION")
print("=" * 80)

# Run actual update
try:
    model.variational_update_iteration(y, inputs_dict)
    print("\n✓ Update iteration completed successfully!")
    
    print("\nUpdated node norms:")
    for tag in ['A', 'B', 'C']:
        mu_node = model.mu_network.get_node_tensor(tag)
        sigma_node = model.sigma_network.get_node_tensor(tag + '_sigma')
        print(f"  {tag}: mu_norm={mu_node.norm().item():.6f}, sigma_norm={sigma_node.norm().item():.6f}")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
