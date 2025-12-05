# type: ignore
import numpy as np
import quimb.tensor as qt
from tensor.btn import BTN
from typing import List

def run_test():
    # --- Config ---
    s = 10         
    x_dim = 4      
    bond = 5       
    y1_d, y2_d = 2, 3 
    
    # --- 1. Build Network & Tensors ---
    data_t1 = np.random.rand(x_dim, bond)
    data_t2 = np.random.rand(bond, x_dim, y1_d, bond)
    data_t3 = np.random.rand(bond, y2_d)

    t1 = qt.Tensor(data_t1, inds=('x1', 'k1'), tags={'T1'})
    t2 = qt.Tensor(data_t2, inds=('k1', 'x2', 'y1', 'k2'), tags={'T2'})
    t3 = qt.Tensor(data_t3, inds=('k2', 'y2'), tags={'T3'})
    
    model_tn = t1 & t2 & t3
    
    # --- 2. Prepare Inputs ---
    input_x1 = qt.Tensor(np.random.rand(s, x_dim), inds=('s', 'x1'), tags={'IN_X1'})
    input_x2 = qt.Tensor(np.random.rand(s, x_dim), inds=('s', 'x2'), tags={'IN_X2'})
    inputs = [input_x1, input_x2]

    model = BTN(model_tn, output_dimensions=['y1', 'y2'], batch_dim='s')
    # TN with inputs is used for the "Dynamic" test (inputs already contracted in)
    tn_with_inputs = model_tn & inputs
    
    print(f"\n--- Initial Tensor Index Map ---")
    print(f"T1: {t1.inds} | T2: {t2.inds} | T3: {t3.inds}")
    print(f"Inputs: {input_x1.inds}, {input_x2.inds}")
    print("-" * 35)

    # --- 3. Gold Standard ---
    # The expected result from a full forward pass
    res_forward = model.forward(model_tn, [inputs])
    
    # --- 4. Verification Loop ---
    target_tags = ['T1', 'T2', 'T3']

    for tag in target_tags:
        print(f"\n--- Testing Target: {tag} ---")
        target_t = model_tn[tag]

        # 4A. Test 1: Dynamic Environment (TN + Inputs -> Env)
        # Check if Env * Target reproduces the forward pass
        env_dynamic = model._batch_environment(tn_with_inputs, tag)
        
        # Check if the environment tensor contains the batch dim 's'
        has_s_dim = model.batch_dim in env_dynamic.inds
        print(f"DEBUG [{tag}] Dynamic Env Inds: {env_dynamic.inds} (has 's': {has_s_dim})")

        # Contract manually to verify correctness
        recon_dynamic = (env_dynamic & target_t).contract(output_inds=['s', 'y1', 'y2'])
        recon_dynamic.transpose_('s', 'y1', 'y2')
        
        match_dynamic = np.allclose(res_forward.data, recon_dynamic.data, atol=1e-9)
        max_diff_dyn = np.max(np.abs(res_forward.data - recon_dynamic.data))
        print(f"1. Dynamic Env (Inputs pre-injected): {match_dynamic} | Max Diff: {max_diff_dyn:.2e}")
        assert match_dynamic, f"Dynamic Env failed for {tag}"


        # 4B. Test 2: Static Environment (TN -> Env -> Forward with Inputs)
        # Check if (Env * Target) + Inputs reproduces the forward pass
        
        # 1. Get Environment of the model without inputs (Pure weights)
        env_static = model._batch_environment(model_tn, tag)
        
        # Check if the environment tensor contains the open input index 'x2' (for T1/T3) or 'x1' (for T2/T3)
        # This confirms the fix worked: the input legs are now open.
        print(f"DEBUG [{tag}] Static Env Inds: {env_static.inds} (check if input legs are open: {'x2' in env_static.inds})")

        # 2. Reconstructed Model: (Env_static & Target)
        reconstructed_model = env_static & target_t

        # 3. Pass this reconstruction to forward() to handle input injection
        res_from_static = model.forward(reconstructed_model, [inputs])

        match_static = np.allclose(res_forward.data, res_from_static.data, atol=1e-9)
        max_diff_stat = np.max(np.abs(res_forward.data - res_from_static.data))
        
        print(f"2. Static Env (Forward injects inputs): {match_static} | Max Diff: {max_diff_stat:.2e}")
        assert match_static, f"Static Env -> Forward failed for {tag}"

    print("\n✅ Comprehensive Test Passed: Environment calculation is sound and commutes.")

if __name__ == "__main__":
    run_test()
