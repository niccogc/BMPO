# type: ignore
import numpy as np
import quimb.tensor as qt
from btn import BTN
from typing import List

def run_test():
    # --- Config ---
    s = 10         # Batch size (single batch for simple comparison)
    x_dim = 4      # Input features
    bond = 5       # Internal bond dim
    y1_d, y2_d = 2, 3 # Output dims
    
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

    # Initialize BTN and other variables
    model = BTN(output_dimensions=['y1', 'y2'], batch_dim='s')
    tn_with_inputs = model_tn & inputs
    
    # --- 3. Print All Indices (Per User Request) ---
    print(f"\n--- 3. Initial Tensor Index Map ---")
    print(f"T1 (Block 1):     {t1.inds} | Tags: {t1.tags}")
    print(f"T2 (Target Block):{t2.inds} | Tags: {t2.tags}")
    print(f"T3 (Block 3):     {t3.inds} | Tags: {t3.tags}")
    print(f"Input X1 (Proj):  {input_x1.inds} | Tags: {input_x1.tags}")
    print(f"Input X2 (Proj):  {input_x2.inds} | Tags: {input_x2.tags}")
    print("-" * 35)

    # --- 4. Calculate Gold Standard (Standard Forward Pass) ---
    res_forward = model.forward(model_tn, [inputs])
    
    # --- 5. Environment Reconstruction and Verification Loop ---
    target_tags = ['T1', 'T2', 'T3']

    for tag in target_tags:
        # A. Calculate Environment (E_X)
        env_t = model.get_environment(tn_with_inputs, tag)
        
        # B. Get Target Tensor (T_X)
        target_t = model_tn[tag]
        print(f"env for {tag}", env_t.inds)
        # C. Reconstruct and Contract (R_Recon = E_X @ T_X)
        full_reconstruction = env_t & target_t
        
        R_Recon = full_reconstruction.contract(output_inds=['s', 'y1', 'y2'])
        R_Recon.transpose_('s', 'y1', 'y2')

        # D. Compare R_Recon against R_Expected
        is_close = np.allclose(res_forward.data, R_Recon.data, atol=1e-9)
        max_diff = np.max(np.abs(res_forward.data - R_Recon.data))

        print(f"Test Block {tag} | Max Diff: {max_diff:.2e} | Match: {is_close}")
        assert is_close, f"Verification failed for block {tag}!"

    print("\n✅ Comprehensive Test Passed: Environment method is sound.")

if __name__ == "__main__":
    run_test()
