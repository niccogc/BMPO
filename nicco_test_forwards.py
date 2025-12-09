import torch
import numpy as np
import quimb.tensor as qt
from tensor.builder import Inputs
from tensor.btn import BTN

# Set precision for comparison
torch.set_default_dtype(torch.float64)

def test_mps_3block():
    print("\n" + "="*60)
    print("TESTING 3-BLOCK MPS (Input x1, x2, x3)")
    print("="*60)

    # ---------------------------------------------------------
    # 1. SETUP DATA & TENSORS
    # ---------------------------------------------------------
    # Batch size
    B = 10
    
    # Dimensions
    # Inputs: x1, x2, x3
    D_x1, D_x2, D_x3 = 3, 3, 3
    # Bonds: b1, b2
    D_b1, D_b2 = 4, 5
    # Output: y
    D_y = 2
    
    # 1. Generate Inputs (Batch data)
    x1_data = torch.randn(B, D_x1)
    x2_data = torch.randn(B, D_x2)
    x3_data = torch.randn(B, D_x3)
    
    # Target (for Inputs init)
    y_data = torch.randn(B, D_y)

    # Initialize Inputs loader
    # Passing 3 inputs. Labels must match network indices.
    loader = Inputs(
        inputs=[x1_data, x2_data, x3_data],
        outputs=[y_data],
        outputs_labels=["y"],
        input_labels=["x1", "x2", "x3"],
        batch_dim="s",
        batch_size=B
    )

    # 2. Generate Weights (Nodes)
    # Node1: (x1, b1)
    w1 = torch.randn(D_x1, D_b1)
    # Node2: (b1, x2, b2)
    w2 = torch.randn(D_b1, D_x2, D_b2)
    # Node3: (b2, x3, y)
    w3 = torch.randn(D_b2, D_x3, D_y)

    # Create Quimb Tensors
    t1 = qt.Tensor(data=w1, inds=('x1', 'b1'), tags={'Node1'})
    t2 = qt.Tensor(data=w2, inds=('b1', 'x2', 'b2'), tags={'Node2'})
    t3 = qt.Tensor(data=w3, inds=('b2', 'x3', 'y'), tags={'Node3'})
    
    mu_tn = qt.TensorNetwork([t1, t2, t3])

    # Initialize BTN
    model = BTN(mu=mu_tn, data_stream=loader, batch_dim="s")

    print(f"Network Structure: [Node1(x1,b1)] --b1-- [Node2(b1,x2,b2)] --b2-- [Node3(b2,x3,y)]")

    # ---------------------------------------------------------
    # 2. VERIFY FORWARD
    # ---------------------------------------------------------
    print("\n>>> CHECKING: forward()")

    # A. BTN Forward
    btn_out = model.forward(
        tn=model.mu, 
        input_generator=model.data.data_mu, 
        sum_over_batch=False, 
        sum_over_output=False
    )
    btn_out_data = torch.as_tensor(btn_out.data)

    # B. Manual Torch Forward
    # Contract:
    # 1. x1(s,i) * w1(i,a) -> L(s,a)
    # 2. L(s,a) * x2(s,j) * w2(a,j,b) -> M(s,b)
    # 3. M(s,b) * x3(s,k) * w3(b,k,o) -> Out(s,o)
    
    # Single einsum:
    # i=x1, j=x2, k=x3, a=b1, b=b2, o=y, s=batch
    torch_out = torch.einsum(
        'si, sj, sk, ia, ajb, bko -> so',
        x1_data, x2_data, x3_data,
        w1, w2, w3
    )

    # C. Compare
    print(f"BTN Shape:   {tuple(btn_out_data.shape)}")
    print(f"Torch Shape: {tuple(torch_out.shape)}")
    
    if torch.allclose(btn_out_data, torch_out, atol=1e-12):
        print("✅ PASS: Forward matches manual einsum.")
    else:
        diff = (btn_out_data - torch_out).abs().max()
        print(f"❌ FAIL: Forward mismatch. Max diff: {diff}")

    # ---------------------------------------------------------
    # 3. VERIFY GET_ENVIRONMENT (Node2)
    # ---------------------------------------------------------
    print("\n>>> CHECKING: get_environment() for Node2")
    
    # We remove Node2. 
    # Remaining: Node1, Node3, and Inputs.
    # Env for Node2 should expose indices connecting to Node2: b1, x2, b2.
    # And preserve output y.
    
    # A. BTN Env
    env_tn = model.get_environment(
        tn=model.mu,
        target_tag='Node2',
        input_generator=model.data.data_mu,
        copy=True,
        sum_over_batch=False,
        sum_over_output=False
    )
    env_data = torch.as_tensor(env_tn.data)
    env_inds = env_tn.inds

    # B. Manual Torch Env
    # Left Block: x1(s,i) * w1(i,a) -> (s,a)  [a=b1]
    L = torch.einsum('si, ia -> sa', x1_data, w1)
    
    # Right Block: x3(s,k) * w3(b,k,o) -> (s,b,o) [b=b2, o=y]
    R = torch.einsum('sk, bko -> sbo', x3_data, w3)
    
    # Middle Input: x2(s,j) -> (s,j) [j=x2]
    # This input is attached to the network but the node consuming it (Node2) is deleted.
    # So index 'j' becomes an open index of the environment.
    
    # Combine (s,a), (s,j), (s,b,o) -> (s, a, j, b, o)
    # Broadcasting over 's', outer product over others.
    torch_env = torch.einsum('sa, sj, sbo -> sajbo', L, x2_data, R)
    
    print(f"BTN Env Indices: {env_inds}")
    print(f"BTN Shape:       {tuple(env_data.shape)}")
    print(f"Torch Shape:     {tuple(torch_env.shape)}")
    
    # Check permutations
    # Expected indices: s, b1, x2, b2, y
    idx_map = {'s': 0, 'b1': 1, 'x2': 2, 'b2': 3, 'y': 4}
    
    try:
        perm = [idx_map[ix] for ix in env_inds]
        torch_env_aligned = torch_env.permute(*perm)
        
        if torch.allclose(env_data, torch_env_aligned, atol=1e-12):
             print("✅ PASS: get_environment matches manual einsum.")
        else:
            diff = (env_data - torch_env_aligned).abs().max()
            print(f"❌ FAIL: Environment mismatch. Max diff: {diff}")
            
    except KeyError:
         print(f"⚠️ SKIPPED: Indices {env_inds} don't match expected map {list(idx_map.keys())}")


    # ---------------------------------------------------------
    # 4. VERIFY OUTER_OPERATION (Node2)
    # ---------------------------------------------------------
    print("\n>>> CHECKING: outer_operation() for Node2")
    
    # Computes: sum_s (Env_s outer Env_s)
    # Note: BTN's outer_operation internally calls get_environment with sum_over_output=True.
    # So 'y' is summed out before the outer product.
    
    # A. BTN Operation
    outer_btn = model.outer_operation(
        input_generator=model.data.data_mu,
        tn=model.mu,
        node_tag='Node2',
        sum_over_batches=True
    )
    outer_data = torch.as_tensor(outer_btn.data)
    inds_out = outer_btn.inds

    # B. Manual Calculation
    # 1. Env with y summed out
    # Right Block sum y: x3(s,k) * w3(b,k,o) -> (s,b,o) -> sum(o) -> (s,b)
    R_sum_y = torch.einsum('sk, bko -> sb', x3_data, w3)
    
    # Env_no_y = L(s, a) * x2(s, j) * R_sum_y(s, b) -> (s, a, j, b)
    env_no_y = torch.einsum('sa, sj, sb -> sajb', L, x2_data, R_sum_y)
    
    # 2. Outer product summed over batch
    # flatten Env to (s, feature_dim)
    flat_dim = D_b1 * D_x2 * D_b2
    env_flat = env_no_y.reshape(B, flat_dim)
    
    # E.T @ E = sum_s (e_s outer e_s)
    torch_outer_flat = torch.einsum('sf, sg -> fg', env_flat, env_flat)
    
    # Reshape back to tensor indices (a,j,b, a',j',b')
    torch_outer = torch_outer_flat.reshape(D_b1, D_x2, D_b2, D_b1, D_x2, D_b2)
    
    print(f"BTN Outer Shape: {tuple(outer_data.shape)}")
    print(f"BTN Indices:     {inds_out}")
    
    # Align manual to BTN indices
    # We expect indices to be: b1, x2, b2, b1_prime, x2_prime, b2_prime
    # Map our manual dimensions 0..5 to labels
    idx_map_outer = {
        'b1': 0, 'x2': 1, 'b2': 2,
        'b1_prime': 3, 'x2_prime': 4, 'b2_prime': 5
    }
    
    try:
        perm = [idx_map_outer[ix] for ix in inds_out]
        torch_aligned = torch_outer.permute(*perm)
        
        if torch.allclose(outer_data, torch_aligned, atol=1e-12):
            print("✅ PASS: outer_operation matches manual einsum.")
        else:
            diff = (outer_data - torch_aligned).abs().max()
            print(f"❌ FAIL: Outer op mismatch. Max diff: {diff}")
    except KeyError:
        print(f"⚠️ SKIPPED: Outer indices {inds_out} mismatch map.")

    # ---------------------------------------------------------
    # 5. VERIFY FORWARD_WITH_TARGET (DOT & MSE)
    # ---------------------------------------------------------
    print("\n>>> CHECKING: forward_with_target()")

    # A. Mode = 'dot' (Scalar Product)
    # --------------------------------
    print("   -> Mode: 'dot'")
    # Manual: sum( forward_output * target )
    # torch_out (so) * y_data (so) -> scalar sum
    torch_dot = torch.einsum('so, so ->', torch_out, y_data)
    
    # BTN:
    # forward_with_target(..., mode='dot', sum_over_batch=True)
    # Note: Using loader.data_mu_y which yields (inputs, target)
    btn_dot = model.forward_with_target(
        input_generator=model.data.data_mu_y, 
        tn=model.mu,
        mode='dot',
        sum_over_batch=True,
        output_inds=[] # Sum over all indices
    )
    
    # Convert scalar tensor/array to float/tensor
    btn_dot_data = torch.as_tensor(btn_dot.data if hasattr(btn_dot, 'data') else btn_dot)
    
    print(f"   BTN Dot:   {btn_dot_data.item():.5f}")
    print(f"   Torch Dot: {torch_dot.item():.5f}")
    
    if torch.allclose(btn_dot_data, torch_dot, atol=1e-12):
        print("   ✅ PASS: 'dot' mode matches manual calc.")
    else:
        diff = (btn_dot_data - torch_dot).abs()
        print(f"   ❌ FAIL: 'dot' mismatch. Diff: {diff}")

    # B. Mode = 'squared_error' (MSE)
    # --------------------------------
    print("   -> Mode: 'squared_error'")
    # Manual: sum( (forward - target)^2 )
    torch_mse = ((torch_out - y_data)**2).sum()
    
    # BTN:
    btn_mse = model.forward_with_target(
        input_generator=model.data.data_mu_y,
        tn=model.mu,
        mode='squared_error',
        sum_over_batch=True,
        output_inds=[] # Sum over all (scalar loss)
    )
    btn_mse_data = torch.as_tensor(btn_mse.data if hasattr(btn_mse, 'data') else btn_mse)

    print(f"   BTN MSE:   {btn_mse_data.item():.5f}")
    print(f"   Torch MSE: {torch_mse.item():.5f}")

    if torch.allclose(btn_mse_data, torch_mse, atol=1e-12):
        print("   ✅ PASS: 'squared_error' mode matches manual calc.")
    else:
        diff = (btn_mse_data - torch_mse).abs()
        print(f"   ❌ FAIL: 'squared_error' mismatch. Diff: {diff}")

if __name__ == "__main__":
    test_mps_3block()
