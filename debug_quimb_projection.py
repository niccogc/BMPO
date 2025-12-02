"""
Debug the quimb projection to understand what it's actually computing.
"""

import torch
import numpy as np
import quimb.tensor as qtn

from tensor.bayesian_tn import BayesianTensorNetwork


def test_quimb_projection():
    """Test what our quimb projection actually returns."""
    print("\n" + "="*70)
    print("DEBUG: Quimb Projection Analysis")
    print("="*70)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create simple network
    d_in = 2
    r1 = 3
    d_out = 1
    batch_size = 5
    
    # Create test data
    x = torch.linspace(-1, 1, batch_size, dtype=torch.float64)
    X = torch.stack([torch.ones_like(x), x], dim=1)  # (5, 2)
    
    # Create network
    input_data = np.ones((batch_size, d_in))
    A_data = np.random.randn(d_in, r1, d_out) * 0.1
    B_data = np.random.randn(r1, d_out) * 0.1
    
    input_tensor = qtn.Tensor(data=input_data, inds=('batch', 'x'), tags='input')  # type: ignore
    A_tensor = qtn.Tensor(data=A_data, inds=('x', 'r1', 'batch_out'), tags='A')  # type: ignore
    B_tensor = qtn.Tensor(data=B_data, inds=('r1', 'batch_out2'), tags='B')  # type: ignore
    
    tn = qtn.TensorNetwork([input_tensor, A_tensor, B_tensor])
    
    print("\n1. Network structure:")
    print(f"   Input tensor: shape={input_tensor.shape}, inds={input_tensor.inds}")  # type: ignore
    print(f"   A tensor: shape={A_tensor.shape}, inds={A_tensor.inds}")  # type: ignore
    print(f"   B tensor: shape={B_tensor.shape}, inds={B_tensor.inds}")  # type: ignore
    
    # Manual projection: contract everything except A
    print("\n2. Manual projection (contract input + B, not A):")
    
    # Contract input with B (they don't share indices, so just combine)
    # Actually, let's see what indices will be free
    projection_tensors = [input_tensor, B_tensor]
    
    # Collect all indices
    from collections import Counter
    all_inds = []
    for t in projection_tensors:
        all_inds.extend(t.inds)  # type: ignore
    
    ind_counts = Counter(all_inds)
    print(f"   Indices in projection network: {ind_counts}")
    print(f"   Free indices (count=1): {[ind for ind, c in ind_counts.items() if c == 1]}")
    
    # The free indices should be: batch, x, r1, batch_out2
    # These are the indices that would connect to A if A were present
    
    # Create projection network
    proj_tn = qtn.TensorNetwork(projection_tensors)
    
    # What indices should remain after contraction?
    # We want indices that would connect to A
    # A has indices: ('x', 'r1', 'batch_out')
    # But A connects to input via 'x' and to B via 'r1'
    # So the free indices after contracting everything except A should include:
    #   - 'batch' from input (not contracted with anything)
    #   - 'x' from input (would connect to A)
    #   - 'r1' from B (would connect to A)
    #   - 'batch_out2' from B (this is the output)
    
    # Let's contract and see what we get
    result = proj_tn.contract(...)  # type: ignore
    print(f"   Result shape: {result.shape}")  # type: ignore
    print(f"   Result inds: {result.inds}")  # type: ignore
    
    # Now let's use our actual compute_projection method
    print("\n3. BayesianTensorNetwork.compute_projection:")
    
    btn = BayesianTensorNetwork(
        mu_tn=tn.copy(),
        learnable_tags=['A', 'B'],
        input_tags=['input'],
        device=torch.device('cpu'),
        dtype=torch.float64
    )
    
    # Set input data
    inputs_dict = {'input': X}
    
    # Compute projection for A
    J_A = btn.compute_projection('A', network_type='mu', inputs=inputs_dict)
    print(f"   J_A shape: {J_A.shape}")
    print(f"   Node A shape: {btn.mu_network.get_node_shape('A')}")
    
    # The problem: J_A has shape (batch, x, r1, batch_out2)
    # But node A has shape (x, r1, batch_out)
    # The 'batch' dimension comes from the input
    # The 'batch_out2' is the wrong output index
    
    print("\n4. The REAL issue:")
    print("   Node A connects to:")
    print("   - Input via 'x' index")
    print("   - B via 'r1' index")
    print("   - Output via 'batch_out' index")
    print("")
    print("   When we remove A and contract, we get:")
    print("   - 'batch' from input (free)")
    print("   - 'x' from input (would connect to A)")
    print("   - 'r1' from B (would connect to A)")
    print("   - 'batch_out2' from B (free, wrong output)")
    print("")
    print("   But we WANT:")
    print("   - Shape that matches how BayesianMPO does it")
    print("   - BayesianMPO Jacobian: (x, batch, batch_out, r1)")
    print("   - Our projection: (batch, x, r1, batch_out2)")
    print("")
    print("   The dimensions don't even match in NAME!")
    print("   'batch_out' vs 'batch_out2' - these are DIFFERENT indices")
    
    print("\n5. Network connectivity analysis:")
    tn_copy = tn.copy()
    print(f"   All indices in network:")
    all_inds_full = []
    for tid, t in tn_copy.tensor_map.items():
        print(f"     {list(t.tags)[0] if t.tags else tid}: {t.inds}")  # type: ignore
        all_inds_full.extend(t.inds)  # type: ignore
    
    ind_counts_full = Counter(all_inds_full)
    print(f"\n   Index connectivity:")
    for ind, count in sorted(ind_counts_full.items()):
        print(f"     {ind}: appears {count} times", end='')
        if count == 1:
            print(" (FREE - not contracted)")
        elif count == 2:
            print(" (CONTRACTED between 2 tensors)")
        else:
            print(f" (HYPER-INDEX - connects {count} tensors)")
    
    print("\n6. The fundamental problem:")
    print("   Our network has:")
    print("   - A with output index 'batch_out'")
    print("   - B with output index 'batch_out2' (DIFFERENT!)")
    print("   These should be the SAME index if they contract!")
    print("")
    print("   In BayesianMPO, all blocks share the same 'batch_out' index")
    print("   This is how the output is formed")
    print("")
    print("   Our quimb network is INCORRECTLY structured!")


if __name__ == "__main__":
    test_quimb_projection()
