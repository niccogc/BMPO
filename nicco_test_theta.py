import torch
import numpy as np
import quimb.tensor as qt
from tensor.builder import Inputs
from tensor.btn import BTN

# Set precision
torch.set_default_dtype(torch.float64)

def test_theta_and_partial_trace():
    print("\n" + "="*60)
    print("TESTING THETA COMPUTATION & PARTIAL TRACE")
    print("="*60)

    # ---------------------------------------------------------
    # 1. SETUP: 3-Node Network
    # ---------------------------------------------------------
    # Structure: Node1(x, b1) -- Node2(b1, b2) -- Node3(b2, y)
    
    B_sz = 5
    D_b1, D_b2 = 3, 4
    
    # Dummy data just to init BTN
    x = torch.randn(B_sz, 2)
    y = torch.randn(B_sz, 2)
    
    loader = Inputs(
        inputs=[x], outputs=[y],
        outputs_labels=["y"], input_labels=["x"],
        batch_dim="s", batch_size=B_sz
    )
    
    # Create Network Tensors
    t1 = qt.Tensor(torch.randn(2, D_b1), inds=('x', 'b1'), tags={'Node1'})
    t2 = qt.Tensor(torch.randn(D_b1, D_b2), inds=('b1', 'b2'), tags={'Node2'})
    t3 = qt.Tensor(torch.randn(D_b2, 2), inds=('b2', 'y'), tags={'Node3'})
    
    mu_tn = qt.TensorNetwork([t1, t2, t3])
    btn = BTN(mu=mu_tn, data_stream=loader, batch_dim="s")
    
    print("Network: Node1(x,b1) -- Node2(b1,b2) -- Node3(b2,y)")

    # ---------------------------------------------------------
    # 2. MODIFY VARIATIONAL PARAMETERS (q_bonds) & CHECK MEAN
    # ---------------------------------------------------------
    print("\n>>> Setting manual parameters for bonds & checking Mean...")
    
    # Bond b1
    alpha_b1 = torch.rand(D_b1) + 1.0
    beta_b1 = torch.rand(D_b1) + 0.5
    
    btn.q_bonds['b1'].concentration.modify(data=alpha_b1)
    btn.q_bonds['b1'].rate.modify(data=beta_b1)
    
    # Check .mean() implementation
    expected_lambda_b1 = alpha_b1 / beta_b1
    btn_mean_b1 = btn.q_bonds['b1'].mean().data
    
    if torch.allclose(btn_mean_b1, expected_lambda_b1, atol=1e-12):
        print(f"   ✅ PASS: q_bonds['b1'].mean() matches alpha/beta")
    else:
        print(f"   ❌ FAIL: q_bonds['b1'].mean() mismatch")

    # Bond b2
    alpha_b2 = torch.rand(D_b2) + 1.0
    beta_b2 = torch.rand(D_b2) + 0.5
    
    btn.q_bonds['b2'].concentration.modify(data=alpha_b2)
    btn.q_bonds['b2'].rate.modify(data=beta_b2)
    
    expected_lambda_b2 = alpha_b2 / beta_b2
    btn_mean_b2 = btn.q_bonds['b2'].mean().data

    if torch.allclose(btn_mean_b2, expected_lambda_b2, atol=1e-12):
        print(f"   ✅ PASS: q_bonds['b2'].mean() matches alpha/beta")
    else:
        print(f"   ❌ FAIL: q_bonds['b2'].mean() mismatch")

    # ---------------------------------------------------------
    # 3. VERIFY THETA COMPUTATION (Node2)
    # ---------------------------------------------------------
    print("\n>>> CHECKING: theta_block_computation('Node2')")
    
    # Theta ~ E[lambda_b1] (x) E[lambda_b2]
    theta_btn = btn.theta_block_computation(node_tag='Node2')
    theta_data = torch.as_tensor(theta_btn.data)
    theta_inds = theta_btn.inds
    
    print(f"   BTN Theta Inds: {theta_inds}")

    # Manual Outer Product
    if theta_inds == ('b1', 'b2'):
        manual_theta = torch.outer(expected_lambda_b1, expected_lambda_b2)
    elif theta_inds == ('b2', 'b1'):
        manual_theta = torch.outer(expected_lambda_b2, expected_lambda_b1)
    else:
        # Handle cases where indices might be sorted differently
        # Using einsum is safest generic way
        if 'b1' in theta_inds[0]: 
             manual_theta = torch.outer(expected_lambda_b1, expected_lambda_b2)
        else:
             manual_theta = torch.outer(expected_lambda_b2, expected_lambda_b1)
        
    if torch.allclose(theta_data, manual_theta, atol=1e-12):
        print("   ✅ PASS: Theta matches manual outer product.")
    else:
        diff = (theta_data - manual_theta).abs().max()
        print(f"   ❌ FAIL: Theta mismatch. Diff: {diff}")

    # ---------------------------------------------------------
    # 4. VERIFY PARTIAL TRACE (Using Full 4D Sigma structure)
    # ---------------------------------------------------------
    print("\n>>> CHECKING: _get_partial_trace() with 4D Sigma (Repeated Indices)")
    
    # 1. Construct 4D Sigma (b1, b2, b1_prime, b2_prime)
    # It represents a diagonal matrix in the flattened basis.
    # Non-zero only if b1==b1_prime and b2==b2_prime.
    
    sigma_shape = (D_b1, D_b2, D_b1, D_b2)
    sigma_data_4d = torch.zeros(sigma_shape)
    
    # Fill diagonal elements with random values
    diag_values = torch.randn(D_b1, D_b2)
    
    for i in range(D_b1):
        for j in range(D_b2):
            sigma_data_4d[i, j, i, j] = diag_values[i, j]
            
    # Create tensor with primed indices initially
    sigma_node = qt.Tensor(
        data=sigma_data_4d, 
        inds=('b1', 'b2', 'b1_prime', 'b2_prime'), 
        tags={'Node2'}
    )
    
    # 2. Simulate _unprime_indices_tensor behavior
    # This renames *_prime to original. Resulting indices: (b1, b2, b1, b2)
    reindex_map = {'b1_prime': 'b1', 'b2_prime': 'b2'}
    sigma_node_unprimed = sigma_node.reindex(reindex_map)
    
    print(f"   Sigma Inds (Unprimed): {sigma_node_unprimed.inds}") # Expect ('b1', 'b2', 'b1', 'b2')
    
    # 3. Target bond: 'b1'
    # We trace out 'b2' weighted by Theta(b2).
    # Theta comes from block computation excluding 'b1'.
    # Theta indices: ('b2',) corresponding to E[lambda_b2].
    
    # A. BTN Partial Trace
    res_btn = btn._get_partial_trace(
        node=sigma_node_unprimed, 
        node_tag='Node2', 
        bond_tag='b1'
    )
    res_data = torch.as_tensor(res_btn.data)
    
    # B. Manual Partial Trace
    # Operation: sum_{b2} ( sigma_{b1,b2,b1,b2} * theta_{b2} )
    # Since sigma is diagonal, sigma_{b1,b2,b1,b2} = diag_values_{b1,b2}
    # We contract diag_values(b1,b2) with theta(b2) over b2.
    # Result should be a vector(b1) (or diagonal matrix(b1,b1) depending on contraction output)
    
    vec_b1 = torch.einsum('ij, j -> i', diag_values, expected_lambda_b2)
    
    print(f"   BTN Result Shape: {tuple(res_data.shape)}")
    
    # Check if result is vector or diagonal matrix
    if res_data.ndim == 1:
        manual_res = vec_b1
    elif res_data.ndim == 2:
        # If it returns diagonal matrix (b1, b1)
        manual_res = torch.diag(vec_b1)
    else:
        # If it returns (b1, b1) but possibly flattened?
        manual_res = vec_b1
        
    if torch.allclose(res_data, manual_res, atol=1e-12):
        print("   ✅ PASS: Partial Trace matches manual weighted sum.")
    else:
        diff = (res_data - manual_res).abs().max()
        print(f"   ❌ FAIL: Partial Trace mismatch. Diff: {diff}")

if __name__ == "__main__":
    test_theta_and_partial_trace()
