"""
Debug BayesianMPO Jacobian computation to understand exact shapes and values.
"""

import torch
import numpy as np
from tensor.node import TensorNode
from tensor.bayesian_mpo import BayesianMPO


def test_bmpo_jacobian():
    """Test BayesianMPO Jacobian shapes."""
    print("\n" + "="*70)
    print("DEBUG: BayesianMPO Jacobian Shapes")
    print("="*70)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create simple 2-block MPO
    d_in = 2
    r1 = 3
    d_out = 1
    
    # Block A: (x, r1, batch_out)
    A_tensor = torch.randn(d_in, r1, d_out, dtype=torch.float64) * 0.1
    node_A = TensorNode(A_tensor, dim_labels=['x', 'r1', 'batch_out'])
    
    # Block B: (r1, batch_out)
    B_tensor = torch.randn(r1, d_out, dtype=torch.float64) * 0.1
    node_B = TensorNode(B_tensor, dim_labels=['r1', 'batch_out'])
    
    # Connect
    node_A.connect(node_B, 'r1', 'r1')
    
    mu_nodes = [node_A, node_B]
    
    # Input node: (s, x) where s is sample dimension
    input_node = TensorNode(
        torch.ones(1, d_in, dtype=torch.float64),
        dim_labels=['s', 'x']
    )
    input_node.connect(node_A, 'x', 'x')
    
    bmpo = BayesianMPO(
        mu_nodes=mu_nodes,
        input_nodes=[input_node],
        rank_labels={'r1'},
        tau_alpha=torch.tensor(2.0, dtype=torch.float64),
        tau_beta=torch.tensor(1.0, dtype=torch.float64)
    )
    
    # Create test data
    batch_size = 5
    x = torch.linspace(-1, 1, batch_size, dtype=torch.float64)
    X = torch.stack([torch.ones_like(x), x], dim=1)
    y = 2.0 * x + 1.0
    
    print(f"\n1. Data:")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    
    # Forward pass
    print(f"\n2. Forward pass:")
    output = bmpo.forward_mu(X, to_tensor=False)
    print(f"   Output shape: {output.shape}")
    print(f"   Output dim_labels: {output.dim_labels}")
    print(f"   Output tensor shape: {output.tensor.shape}")
    
    # Check what the network state is after forward
    print(f"\n3. Network state after forward:")
    print(f"   Input node shape: {bmpo.mu_mpo.input_nodes[0].shape}")
    print(f"   Input node dim_labels: {bmpo.mu_mpo.input_nodes[0].dim_labels}")
    print(f"   Block A shape: {bmpo.mu_nodes[0].shape}")
    print(f"   Block B shape: {bmpo.mu_nodes[1].shape}")
    
    # Compute Jacobian for block A
    print(f"\n4. Jacobian for block A (without node A):")
    J_A = bmpo._compute_forward_without_node(bmpo.mu_nodes[0], bmpo.mu_mpo)
    print(f"   J_A shape: {J_A.shape}")
    print(f"   J_A dim_labels: {J_A.dim_labels}")
    print(f"   J_A tensor shape: {J_A.tensor.shape}")
    
    # What indices does J_A have?
    print(f"\n5. J_A index analysis:")
    print(f"   Block A had indices: {bmpo.mu_nodes[0].dim_labels}")
    print(f"   J_A has indices: {J_A.dim_labels}")
    print(f"   These should be the indices that connect to A")
    
    # Contract J_A with y
    print(f"\n6. Contracting J_A with y:")
    print(f"   This is what get_b() does")
    
    # Expand y to match output shape
    y_expanded = y
    for _ in range(len(output.shape) - 1):
        y_expanded = y_expanded.unsqueeze(-1)
    print(f"   y_expanded shape: {y_expanded.shape}")
    
    # Try manual contraction
    # J_A has dims like ('s', 'x', 'r1', 'batch_out')
    # We want to contract with y over 's' and sum over non-node dimensions
    # Result should have shape of node A: (x, r1, batch_out)
    
    print(f"\n7. Manual einsum contraction:")
    # Identify sample dimension
    sample_dim = 's'
    node_dims = bmpo.mu_nodes[0].dim_labels
    
    print(f"   Sample dimension: {sample_dim}")
    print(f"   Node dimensions: {node_dims}")
    print(f"   J_A dimensions: {J_A.dim_labels}")
    
    # Build einsum string
    from tensor.utils import EinsumLabeler
    labeler = EinsumLabeler()
    
    J_ein = ''.join([labeler[d] for d in J_A.dim_labels])
    y_ein = labeler[sample_dim]
    
    # Output: only node dimensions
    out_dims = [d for d in J_A.dim_labels if d != sample_dim and d in node_dims]
    out_ein = ''.join([labeler[d] for d in out_dims])
    
    einsum_str = f"{J_ein},{y_ein}->{out_ein}"
    print(f"   Einsum string: {einsum_str}")
    
    result = torch.einsum(einsum_str, J_A.tensor, y.squeeze() if y.dim() > 1 else y)
    print(f"   Result shape: {result.shape}")
    print(f"   Expected node A shape: {bmpo.mu_nodes[0].shape}")
    
    if result.shape == bmpo.mu_nodes[0].shape:
        print("   ✓ Shapes match!")
    else:
        print("   ✗ Shape mismatch!")
    
    # Now test outer product
    print(f"\n8. Jacobian outer product (for Σ^-1):")
    # We want: sum_n J(x_n) ⊗ J(x_n)
    # This should give (node_dims, node_dims)
    
    # Einsum: contract over sample dimension, outer product over node dims
    J_ein1 = ''.join([labeler[d] for d in J_A.dim_labels])
    J_ein2 = ''.join([labeler['_' + d] if d != sample_dim else labeler[d] for d in J_A.dim_labels])
    
    out1 = ''.join([labeler[d] for d in out_dims])
    out2 = ''.join([labeler['_' + d] for d in out_dims])
    
    einsum_str2 = f"{J_ein1},{J_ein2}->{out1}{out2}"
    print(f"   Einsum string: {einsum_str2}")
    
    J_outer = torch.einsum(einsum_str2, J_A.tensor, J_A.tensor)
    print(f"   J_outer shape: {J_outer.shape}")
    
    d = np.prod(bmpo.mu_nodes[0].shape)
    print(f"   After flattening: should be ({d}, {d})")
    J_outer_flat = J_outer.reshape(d, d)
    print(f"   Flattened shape: {J_outer_flat.shape}")
    print("   ✓ This is what goes into Σ^-1!")


if __name__ == "__main__":
    test_bmpo_jacobian()
