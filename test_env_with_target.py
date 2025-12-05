# type: ignore
"""
Test get_environment when target y is added to the network.
Compare environment shapes with and without y added.
"""

import numpy as np
import quimb.tensor as qt
from tensor.btn import BTN

def print_result(result, label):
    print(f"\n{label}:")
    if isinstance(result, qt.Tensor):
        print(f"  Indices: {result.inds}")
        print(f"  Shape: {result.data.shape}")
    else:
        print(f"  Scalar value: {result}")

def main():
    print("="*70)
    print(" GET_ENVIRONMENT WITH TARGET Y - SINGLE BATCH TEST")
    print("="*70)
    
    # Setup
    np.random.seed(42)
    x_dim = 3
    bond_dim = 4
    y_dim = 2
    batch_size = 5
    
    # Create network: T1 -[k1]- T2
    data_t1 = np.random.rand(x_dim, y_dim, bond_dim)
    data_t2 = np.random.rand(bond_dim, x_dim, y_dim)

    t1 = qt.Tensor(data_t1, inds=('x1', 'y1', 'k1'), tags={'T1'})
    t2 = qt.Tensor(data_t2, inds=('k1', 'x2', 'y2'), tags={'T2'})
    
    mu_tn = t1 & t2
    btn = BTN(mu_tn, output_dimensions=['y1', 'y2'], batch_dim='s')
    
    print(f"\nNetwork: T1(x1, y1, k1) <-> T2(k1, x2, y2)")
    print(f"Output dimensions: ['y1', 'y2']")
    print(f"Batch size: {batch_size}")
    
    # Prepare inputs
    input_data = np.random.randn(batch_size, x_dim)
    inputs_mu = btn.prepare_inputs({'x1': input_data}, for_sigma=False)
    
    # Create target output y
    y_data = np.random.randn(batch_size, y_dim, y_dim)
    y = qt.Tensor(y_data, inds=('s', 'y1', 'y2'), tags={'target'})
    
    print(f"\nTarget y:")
    print(f"  Indices: {y.inds}")
    print(f"  Shape: {y.data.shape}")
    
    # =========================================================================
    print("\n" + "="*70)
    print(" TEST 1: MU NETWORK - Environment WITHOUT y")
    print("="*70)
    
    # Full network with inputs (no y)
    full_tn_no_y = mu_tn & inputs_mu
    
    print("\n--- T1 Environment (no y) ---")
    env_t1_no_y = btn._batch_environment(full_tn_no_y, 'T1', copy=True, 
                                      sum_over_batch=False, sum_over_output=False)
    print_result(env_t1_no_y, "Environment T1 (no y, keep batch, keep output)")
    
    print("\n--- T1 Environment (no y, sum_over_batch=True) ---")
    env_t1_no_y_sumb = btn._batch_environment(full_tn_no_y, 'T1', copy=True,
                                            sum_over_batch=True, sum_over_output=False)
    print_result(env_t1_no_y_sumb, "Environment T1 (no y, sum batch, keep output)")
    
    print("\n--- T1 Environment (no y, sum_over_output=True) ---")
    env_t1_no_y_sumo = btn._batch_environment(full_tn_no_y, 'T1', copy=True,
                                            sum_over_batch=False, sum_over_output=True)
    print_result(env_t1_no_y_sumo, "Environment T1 (no y, keep batch, sum output)")
    
    # =========================================================================
    print("\n" + "="*70)
    print(" TEST 2: MU NETWORK - Environment WITH y")
    print("="*70)
    
    # Full network with inputs AND y
    full_tn_with_y = mu_tn & inputs_mu & y
    
    print("\nFull network WITH y - outer indices:")
    print(f"  {full_tn_with_y.outer_inds()}")
    
    print("\n--- T1 Environment (with y) ---")
    env_t1_with_y = btn._batch_environment(full_tn_with_y, 'T1', copy=True,
                                        sum_over_batch=False, sum_over_output=False)
    print_result(env_t1_with_y, "Environment T1 (with y, keep batch, keep output)")
    
    print("\nComparison:")
    print(f"  WITHOUT y: {env_t1_no_y.inds}")
    print(f"  WITH y:    {env_t1_with_y.inds}")
    print(f"  --> When y is added, output dims (y1, y2) are contracted!")
    
    print("\n--- T1 Environment (with y, sum_over_batch=True) ---")
    env_t1_with_y_sumb = btn._batch_environment(full_tn_with_y, 'T1', copy=True,
                                              sum_over_batch=True, sum_over_output=False)
    print_result(env_t1_with_y_sumb, "Environment T1 (with y, sum batch, keep output)")
    
    print("\nComparison:")
    print(f"  WITHOUT y (sum batch): {env_t1_no_y_sumb.inds}")
    print(f"  WITH y (sum batch):    {env_t1_with_y_sumb.inds}")
    
    print("\n--- T1 Environment (with y, sum_over_output=True) ---")
    env_t1_with_y_sumo = btn._batch_environment(full_tn_with_y, 'T1', copy=True,
                                              sum_over_batch=False, sum_over_output=True)
    print_result(env_t1_with_y_sumo, "Environment T1 (with y, keep batch, sum output)")
    
    print("\nComparison:")
    print(f"  WITHOUT y (sum output): {env_t1_no_y_sumo.inds}")
    print(f"  WITH y (sum output):    {env_t1_with_y_sumo.inds}")
    
    # =========================================================================
    print("\n" + "="*70)
    print(" TEST 3: SIGMA NETWORK - Environment WITH y")
    print("="*70)
    
    # Prepare sigma inputs
    inputs_sigma = btn.prepare_inputs({'x1': input_data}, for_sigma=True)
    
    print(f"\nSIGMA inputs:")
    for inp in inputs_sigma:
        print(f"  {inp.inds}: shape {inp.data.shape}")
    
    # Full sigma network without y
    full_sigma_no_y = btn.sigma & inputs_sigma
    
    print("\n--- T1_sigma Environment (no y) ---")
    env_sigma_no_y = btn._batch_environment(full_sigma_no_y, 'T1_sigma', copy=True,
                                         sum_over_batch=False, sum_over_output=False)
    print_result(env_sigma_no_y, "SIGMA Environment T1 (no y)")
    
    # Full sigma network with y
    full_sigma_with_y = btn.sigma & inputs_sigma & y
    
    print("\n--- T1_sigma Environment (with y) ---")
    env_sigma_with_y = btn._batch_environment(full_sigma_with_y, 'T1_sigma', copy=True,
                                           sum_over_batch=False, sum_over_output=False)
    print_result(env_sigma_with_y, "SIGMA Environment T1 (with y)")
    
    print("\nComparison:")
    print(f"  WITHOUT y: {env_sigma_no_y.inds}")
    print(f"  WITH y:    {env_sigma_with_y.inds}")
    
    # Check for prime indices
    prime_inds_no_y = [ind for ind in env_sigma_no_y.inds if '_prime' in ind]
    prime_inds_with_y = [ind for ind in env_sigma_with_y.inds if '_prime' in ind]
    print(f"\n  Prime indices WITHOUT y: {prime_inds_no_y}")
    print(f"  Prime indices WITH y: {prime_inds_with_y}")
    
    print("\n--- T1_sigma Environment (with y, sum_over_batch=True) ---")
    env_sigma_with_y_sumb = btn._batch_environment(full_sigma_with_y, 'T1_sigma', copy=True,
                                                 sum_over_batch=True, sum_over_output=False)
    print_result(env_sigma_with_y_sumb, "SIGMA Environment T1 (with y, sum batch)")
    
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    print("\nKey observations:")
    print("  1. WITHOUT y: Environment has output indices (y1, y2)")
    print("  2. WITH y: Output indices are contracted (y connects to network)")
    print("  3. The 'hole' indices (bonds) remain in both cases")
    print("  4. sum_over_batch flag removes 's' from indices")
    print("  5. sum_over_output flag removes remaining output indices")
    print("  6. SIGMA network preserves prime indices in environment")

if __name__ == "__main__":
    main()
