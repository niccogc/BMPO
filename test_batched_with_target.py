# type: ignore
"""
Test batched operations with target y:
- forward_with_target with multiple batches
- get_environment_batched with target y
"""

import numpy as np
import quimb.tensor as qt
from tensor.btn import BTN

def print_result(result, label):
    print(f"\n{label}:")
    if isinstance(result, qt.Tensor):
        print(f"  Indices: {result.inds}")
        print(f"  Shape: {result.data.shape}")
        if result.data.size < 20:
            print(f"  Data: {result.data}")
    else:
        print(f"  Scalar value: {result}")

def main():
    print("="*70)
    print(" BATCHED OPERATIONS WITH TARGET Y")
    print("="*70)
    
    # Setup
    np.random.seed(42)
    x_dim = 3
    bond_dim = 4
    y_dim = 2
    batch_size = 5
    num_batches = 3
    
    # Create network
    t1 = qt.Tensor(np.random.rand(x_dim, y_dim, bond_dim), inds=('x1', 'y1', 'k1'), tags={'T1'})
    t2 = qt.Tensor(np.random.rand(bond_dim, x_dim, y_dim), inds=('k1', 'x2', 'y2'), tags={'T2'})
    
    mu_tn = t1 & t2
    btn = BTN(mu_tn, output_dimensions=['y1', 'y2'], batch_dim='s')
    
    print(f"\nNetwork: T1(x1, y1, k1) <-> T2(k1, x2, y2)")
    print(f"Batch size per batch: {batch_size}")
    print(f"Number of batches: {num_batches}")
    
    # =========================================================================
    print("\n" + "="*70)
    print(" TEST 1: forward_with_target - DOT PRODUCT (Multiple Batches)")
    print("="*70)
    
    # We need to test forward_with_target for each batch individually
    # since it takes a single batch, not multiple batches
    
    print("\n--- Computing dot product per batch and summing manually ---")
    
    dot_results = []
    for i in range(num_batches):
        # Create batch inputs
        input_data = np.random.randn(batch_size, x_dim)
        inputs_mu = btn.prepare_inputs({'x1': input_data}, for_sigma=False)
        
        # Create target y for this batch
        y_data = np.random.randn(batch_size, y_dim, y_dim)
        y = qt.Tensor(y_data, inds=('s', 'y1', 'y2'), tags={'target'})
        
        # Dot product (keep batch)
        dot_result = btn.forward_with_target(mu_tn, inputs_mu, y, mode='dot', sum_over_batch=False)
        print(f"\nBatch {i+1}:")
        print(f"  Dot result indices: {dot_result.inds}")
        print(f"  Dot result shape: {dot_result.data.shape}")
        
        dot_results.append(dot_result)
    
    # Manually sum across batches
    manual_sum = dot_results[0]
    for dr in dot_results[1:]:
        manual_sum = manual_sum + dr
    
    print(f"\nManual sum across batches:")
    print(f"  Indices: {manual_sum.inds}")
    print(f"  Shape: {manual_sum.data.shape}")
    print(f"  Data: {manual_sum.data}")
    
    print("\n--- Now test with sum_over_batch=True per batch ---")
    
    dot_sum_total = 0
    for i in range(num_batches):
        input_data = np.random.randn(batch_size, x_dim)
        inputs_mu = btn.prepare_inputs({'x1': input_data}, for_sigma=False)
        
        y_data = np.random.randn(batch_size, y_dim, y_dim)
        y = qt.Tensor(y_data, inds=('s', 'y1', 'y2'), tags={'target'})
        
        # Dot product (sum over batch)
        dot_sum = btn.forward_with_target(mu_tn, inputs_mu, y, mode='dot', sum_over_batch=True)
        print(f"\nBatch {i+1} dot sum: {dot_sum}")
        
        dot_sum_total += dot_sum
    
    print(f"\nTotal sum across batches: {dot_sum_total}")
    print("✅ Dot product batched operations work!")
    
    # =========================================================================
    print("\n" + "="*70)
    print(" TEST 2: forward_with_target - SQUARED ERROR (Multiple Batches)")
    print("="*70)
    
    print("\n--- Computing squared error per batch ---")
    
    se_results = []
    for i in range(num_batches):
        input_data = np.random.randn(batch_size, x_dim)
        inputs_mu = btn.prepare_inputs({'x1': input_data}, for_sigma=False)
        
        y_data = np.random.randn(batch_size, y_dim, y_dim)
        y = qt.Tensor(y_data, inds=('s', 'y1', 'y2'), tags={'target'})
        
        # Squared error (keep batch)
        se_result = btn.forward_with_target(mu_tn, inputs_mu, y, mode='squared_error', sum_over_batch=False)
        print(f"\nBatch {i+1}:")
        print(f"  SE result indices: {se_result.inds}")
        print(f"  SE result shape: {se_result.data.shape}")
        
        se_results.append(se_result)
    
    # Test summing
    se_sum_total = 0
    for i in range(num_batches):
        input_data = np.random.randn(batch_size, x_dim)
        inputs_mu = btn.prepare_inputs({'x1': input_data}, for_sigma=False)
        
        y_data = np.random.randn(batch_size, y_dim, y_dim)
        y = qt.Tensor(y_data, inds=('s', 'y1', 'y2'), tags={'target'})
        
        se_sum = btn.forward_with_target(mu_tn, inputs_mu, y, mode='squared_error', sum_over_batch=True)
        print(f"\nBatch {i+1} SE sum: {se_sum}")
        
        se_sum_total += se_sum
    
    print(f"\nTotal SE across batches: {se_sum_total}")
    print("✅ Squared error batched operations work!")
    
    # =========================================================================
    print("\n" + "="*70)
    print(" TEST 3: get_environment_batched (Multiple Batches)")
    print("="*70)
    
    # Prepare multiple input batches
    input_batches = []
    for i in range(num_batches):
        input_data = np.random.randn(batch_size, x_dim)
        inputs = btn.prepare_inputs({'x1': input_data}, for_sigma=False)
        input_batches.append(inputs)
    
    print("\n--- Test WITHOUT sum_over_batch (concatenate) ---")
    env_concat = btn.get_environment_batched(mu_tn, 'T1', input_batches,
                                             sum_over_batch=False, sum_over_output=False)
    print_result(env_concat, "Environment (concatenated)")
    
    expected_batch_size = batch_size * num_batches
    batch_idx = env_concat.inds.index('s')
    actual_batch_size = env_concat.data.shape[batch_idx]
    assert actual_batch_size == expected_batch_size, f"Expected batch size {expected_batch_size}, got {actual_batch_size}"
    print(f"✅ Concatenation works! Batch dimension: {actual_batch_size}")
    
    print("\n--- Test WITH sum_over_batch (sum on-the-fly) ---")
    env_sum = btn.get_environment_batched(mu_tn, 'T1', input_batches,
                                          sum_over_batch=True, sum_over_output=False)
    print_result(env_sum, "Environment (summed)")
    
    assert 's' not in env_sum.inds, "Batch dimension should not be present when sum_over_batch=True"
    print("✅ Sum over batch works! No batch dimension in result")
    
    # Manual verification
    print("\n--- Manual verification of sum ---")
    manual_env_sum = None
    for batch_inputs in input_batches:
        full_tn = mu_tn & batch_inputs
        env = btn._batch_environment(full_tn, 'T1', sum_over_batch=True, sum_over_output=False)
        if manual_env_sum is None:
            manual_env_sum = env
        else:
            manual_env_sum = manual_env_sum + env
    
    assert np.allclose(env_sum.data, manual_env_sum.data), "Manual sum doesn't match!"
    print("✅ Manual verification passed!")
    
    # =========================================================================
    print("\n" + "="*70)
    print(" TEST 4: get_environment_batched WITH target y")
    print("="*70)
    
    # Create target y for each batch and add to network
    print("\n--- Environment with y added to network ---")
    
    input_batches_with_y = []
    for i in range(num_batches):
        input_data = np.random.randn(batch_size, x_dim)
        inputs = btn.prepare_inputs({'x1': input_data}, for_sigma=False)
        
        # Create y for this batch
        y_data = np.random.randn(batch_size, y_dim, y_dim)
        y = qt.Tensor(y_data, inds=('s', 'y1', 'y2'), tags={'target'})
        
        # Combine inputs and y
        inputs_with_y = inputs + [y]
        input_batches_with_y.append(inputs_with_y)
    
    # Get environment with y (concatenate)
    env_with_y_concat = btn.get_environment_batched(mu_tn, 'T1', input_batches_with_y,
                                                     sum_over_batch=False, sum_over_output=False)
    print_result(env_with_y_concat, "Environment with y (concatenated)")
    
    # Get environment with y (sum)
    env_with_y_sum = btn.get_environment_batched(mu_tn, 'T1', input_batches_with_y,
                                                  sum_over_batch=True, sum_over_output=False)
    print_result(env_with_y_sum, "Environment with y (summed)")
    
    print("\n✅ Environment with y works for multiple batches!")
    
    # =========================================================================
    print("\n" + "="*70)
    print(" TEST 5: SIGMA Network Batched Environment")
    print("="*70)
    
    # Prepare sigma input batches
    sigma_batches = []
    for i in range(num_batches):
        input_data = np.random.randn(batch_size, x_dim)
        sigma_inputs = btn.prepare_inputs({'x1': input_data}, for_sigma=True)
        sigma_batches.append(sigma_inputs)
    
    print("\n--- SIGMA environment (concatenate) ---")
    env_sigma_concat = btn.get_environment_batched(btn.sigma, 'T1_sigma', sigma_batches,
                                                    sum_over_batch=False, sum_over_output=False)
    print_result(env_sigma_concat, "SIGMA Environment (concatenated)")
    
    # Check for prime indices
    prime_inds = [ind for ind in env_sigma_concat.inds if '_prime' in ind]
    print(f"  Prime indices: {prime_inds}")
    assert len(prime_inds) > 0, "Should have prime indices"
    print("✅ SIGMA environment preserves prime indices!")
    
    print("\n--- SIGMA environment (sum) ---")
    env_sigma_sum = btn.get_environment_batched(btn.sigma, 'T1_sigma', sigma_batches,
                                                 sum_over_batch=True, sum_over_output=False)
    print_result(env_sigma_sum, "SIGMA Environment (summed)")
    
    prime_inds_sum = [ind for ind in env_sigma_sum.inds if '_prime' in ind]
    assert len(prime_inds_sum) > 0, "Should have prime indices"
    print("✅ SIGMA environment sum works!")
    
    print("\n" + "="*70)
    print(" ALL TESTS PASSED! 🎉")
    print("="*70)
    print("\nSummary:")
    print("  ✅ forward_with_target works for multiple batches (dot product)")
    print("  ✅ forward_with_target works for multiple batches (squared error)")
    print("  ✅ get_environment_batched concatenates correctly")
    print("  ✅ get_environment_batched sums on-the-fly correctly")
    print("  ✅ Environment with target y works for batches")
    print("  ✅ SIGMA network batched operations work")
    print("  ✅ Prime indices preserved in SIGMA")

if __name__ == "__main__":
    main()
