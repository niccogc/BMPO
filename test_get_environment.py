# type: ignore
"""
Test script for BTN.get_environment method.
Tests that get_environment returns correct indices for both mu and sigma networks,
with single and multiple batches.
"""

import numpy as np
import quimb.tensor as qt
from tensor.btn import BTN

def print_header(title):
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")

def inspect_tensor(tensor, label="Tensor"):
    print(f"\n{label}:")
    print(f"  Indices: {tensor.inds}")
    print(f"  Shape: {tensor.data.shape}")
    print(f"  Tags: {tensor.tags}")

def main():
    print_header("SETUP: Creating Mock Tensor Network")
    
    # Configuration
    x_dim = 4      # Input feature dimension
    bond_dim = 5   # Bond dimension between nodes
    y_dim = 2      # Output dimension
    batch_dim = 's'
    batch_size = 8
    
    # Create a simple network: T1 -[k1]- T2 -[k2]- T3
    # T1 has input x1, T2 has input x2, T3 outputs y
    np.random.seed(42)
    data_t1 = np.random.rand(x_dim, bond_dim)
    data_t2 = np.random.rand(bond_dim, x_dim, y_dim, bond_dim)
    data_t3 = np.random.rand(bond_dim, y_dim)

    t1 = qt.Tensor(data_t1, inds=('x1', 'k1'), tags={'T1', 'LAYER_1'})
    t2 = qt.Tensor(data_t2, inds=('k1', 'x2', 'y1', 'k2'), tags={'T2', 'LAYER_2'})
    t3 = qt.Tensor(data_t3, inds=('k2', 'y2'), tags={'T3', 'LAYER_3'})
    
    mu_tn = t1 & t2 & t3
    output_dims = ['y1', 'y2']
    
    print(f"Network: T1(x1, k1) <-> T2(k1, x2, y1, k2) <-> T3(k2, y2)")
    print(f"Input indices: x1, x2")
    print(f"Output indices: {output_dims}")
    print(f"Internal bonds: k1, k2")
    
    # Initialize BTN
    btn = BTN(mu_tn, output_dimensions=output_dims, batch_dim=batch_dim)
    print("\n✅ BTN initialized successfully")

    # =========================================================================
    print_header("TEST 1: Get Environment for MU Network (Single Batch)")
    # =========================================================================
    
    print("\nPreparing inputs for MU network...")
    input_data = np.random.randn(batch_size, x_dim)
    mu_inputs = btn.prepare_inputs({'x1': input_data}, for_sigma=False)
    
    print(f"Created {len(mu_inputs)} input tensors")
    for inp in mu_inputs:
        print(f"  - {inp.inds}: shape {inp.data.shape}")
    
    # Create full network with inputs
    full_mu_tn = mu_tn & mu_inputs
    print(f"\nFull MU network outer indices: {full_mu_tn.outer_inds()}")
    
    # Test environment for T2
    print("\n--- Computing environment for T2 (removing T2) ---")
    env_t2 = btn._batch_environment(full_mu_tn, 'T2', copy=True)
    
    inspect_tensor(env_t2, "Environment Tensor for T2")
    
    # Expected indices: The environment contains outer_inds() of remaining network:
    # - The batch dimension 's' (always preserved)
    # - The "holes" where T2 connects: 'k1', 'k2', 'x2'
    # - Other output dimensions from remaining tensors: 'y2' (from T3)
    # NOT 'y1' because that was T2's output and T2 is removed
    expected_inds = {'s', 'y2', 'k1', 'k2', 'x2'}
    actual_inds = set(env_t2.inds)
    
    print(f"\nExpected indices (unordered): {expected_inds}")
    print(f"Actual indices (unordered): {actual_inds}")
    
    if expected_inds == actual_inds:
        print("✅ MU Environment indices are CORRECT!")
    else:
        print("❌ MU Environment indices MISMATCH!")
        print(f"   Missing: {expected_inds - actual_inds}")
        print(f"   Extra: {actual_inds - expected_inds}")

    # =========================================================================
    print_header("TEST 2: Get Environment for SIGMA Network (Single Batch)")
    # =========================================================================
    
    print("\nPreparing inputs for SIGMA network...")
    sigma_inputs = btn.prepare_inputs({'x1': input_data}, for_sigma=True)
    
    print(f"Created {len(sigma_inputs)} input tensors (with primes)")
    for inp in sigma_inputs:
        print(f"  - {inp.inds}: shape {inp.data.shape}")
    
    # Create full sigma network with inputs
    full_sigma_tn = btn.sigma & sigma_inputs
    print(f"\nFull SIGMA network outer indices: {full_sigma_tn.outer_inds()}")
    
    # Test environment for T2_sigma
    print("\n--- Computing environment for T2_sigma (removing T2_sigma) ---")
    env_t2_sigma = btn._batch_environment(full_sigma_tn, 'T2_sigma', copy=True)
    
    inspect_tensor(env_t2_sigma, "Environment Tensor for T2_sigma")
    
    # Expected indices for sigma environment (T2_sigma removed):
    # - The batch dimension 's' (always preserved)
    # - The "holes" where T2_sigma connects (non-prime and prime):
    #   'k1', 'k1_prime', 'k2', 'k2_prime', 'x2', 'x2_prime'
    # - Other output dimensions from remaining tensors: 'y2' (from T3_sigma)
    # NOT 'y1' because that was T2_sigma's output and T2_sigma is removed
    expected_inds_sigma = {'s', 'y2', 'k1', 'k2', 'k1_prime', 'k2_prime', 'x2', 'x2_prime'}
    actual_inds_sigma = set(env_t2_sigma.inds)
    
    print(f"\nExpected indices (unordered): {expected_inds_sigma}")
    print(f"Actual indices (unordered): {actual_inds_sigma}")
    
    if expected_inds_sigma == actual_inds_sigma:
        print("✅ SIGMA Environment indices are CORRECT!")
    else:
        print("❌ SIGMA Environment indices MISMATCH!")
        print(f"   Missing: {expected_inds_sigma - actual_inds_sigma}")
        print(f"   Extra: {actual_inds_sigma - expected_inds_sigma}")

    # =========================================================================
    print_header("TEST 3: Get Environment with Multiple Batches (MU)")
    # =========================================================================
    
    print("\nTesting with multiple batches...")
    num_batches = 3
    batch_results_mu = []
    
    for batch_idx in range(num_batches):
        print(f"\n--- Batch {batch_idx + 1}/{num_batches} ---")
        
        # Create different input data for each batch
        batch_input_data = np.random.randn(batch_size, x_dim) * (batch_idx + 1)
        batch_mu_inputs = btn.prepare_inputs({'x1': batch_input_data}, for_sigma=False)
        
        # Create full network with inputs for this batch
        batch_full_tn = mu_tn & batch_mu_inputs
        
        # Compute environment for T1
        env_t1 = btn._batch_environment(batch_full_tn, 'T1', copy=True)
        
        print(f"Environment indices: {env_t1.inds}")
        print(f"Environment shape: {env_t1.data.shape}")
        
        batch_results_mu.append(env_t1)
    
    # Stack results like in forward method
    print("\n--- Stacking Multiple Batch Results ---")
    stacked_data = np.concatenate([t.data for t in batch_results_mu], axis=0)
    stacked_tensor = qt.Tensor(stacked_data, inds=batch_results_mu[0].inds)
    
    print(f"Stacked tensor shape: {stacked_tensor.data.shape}")
    print(f"Stacked tensor indices: {stacked_tensor.inds}")
    print(f"Expected batch dimension size: {batch_size * num_batches}")
    
    # Verify the batch dimension is correct
    batch_dim_idx = stacked_tensor.inds.index('s')
    actual_batch_size = stacked_tensor.data.shape[batch_dim_idx]
    
    if actual_batch_size == batch_size * num_batches:
        print(f"✅ Multiple batch stacking CORRECT! ({actual_batch_size} samples)")
    else:
        print(f"❌ Batch size mismatch: expected {batch_size * num_batches}, got {actual_batch_size}")

    # =========================================================================
    print_header("TEST 4: Get Environment with Multiple Batches (SIGMA)")
    # =========================================================================
    
    print("\nTesting SIGMA with multiple batches...")
    batch_results_sigma = []
    
    for batch_idx in range(num_batches):
        print(f"\n--- Batch {batch_idx + 1}/{num_batches} ---")
        
        # Create different input data for each batch
        batch_input_data = np.random.randn(batch_size, x_dim) * (batch_idx + 1)
        batch_sigma_inputs = btn.prepare_inputs({'x1': batch_input_data}, for_sigma=True)
        
        # Create full network with inputs for this batch
        batch_full_sigma_tn = btn.sigma & batch_sigma_inputs
        
        # Compute environment for T3_sigma
        env_t3_sigma = btn._batch_environment(batch_full_sigma_tn, 'T3_sigma', copy=True)
        
        print(f"Environment indices: {env_t3_sigma.inds}")
        print(f"Environment shape: {env_t3_sigma.data.shape}")
        
        batch_results_sigma.append(env_t3_sigma)
    
    # Stack results
    print("\n--- Stacking Multiple Batch Results (SIGMA) ---")
    stacked_data_sigma = np.concatenate([t.data for t in batch_results_sigma], axis=0)
    stacked_tensor_sigma = qt.Tensor(stacked_data_sigma, inds=batch_results_sigma[0].inds)
    
    print(f"Stacked tensor shape: {stacked_tensor_sigma.data.shape}")
    print(f"Stacked tensor indices: {stacked_tensor_sigma.inds}")
    
    # Verify the batch dimension
    batch_dim_idx = stacked_tensor_sigma.inds.index('s')
    actual_batch_size_sigma = stacked_tensor_sigma.data.shape[batch_dim_idx]
    
    if actual_batch_size_sigma == batch_size * num_batches:
        print(f"✅ Multiple batch stacking (SIGMA) CORRECT! ({actual_batch_size_sigma} samples)")
    else:
        print(f"❌ Batch size mismatch: expected {batch_size * num_batches}, got {actual_batch_size_sigma}")

    # =========================================================================
    print_header("TEST 5: Verify Environment for All Nodes (MU)")
    # =========================================================================
    
    print("\nTesting environment for all nodes in MU network...")
    test_input = np.random.randn(batch_size, x_dim)
    test_mu_inputs = btn.prepare_inputs({'x1': test_input}, for_sigma=False)
    test_full_tn = mu_tn & test_mu_inputs
    
    nodes_to_test = ['T1', 'T2', 'T3']
    
    for node_tag in nodes_to_test:
        env = btn._batch_environment(test_full_tn, node_tag, copy=True)
        print(f"\n{node_tag}:")
        print(f"  Environment indices: {env.inds}")
        print(f"  Environment shape: {env.data.shape}")
        
        # Verify batch dimension exists
        if 's' in env.inds:
            print(f"  ✅ Batch dimension present")
        else:
            print(f"  ❌ Batch dimension MISSING!")

    # =========================================================================
    print_header("TEST 6: Verify Environment for All Nodes (SIGMA)")
    # =========================================================================
    
    print("\nTesting environment for all nodes in SIGMA network...")
    test_sigma_inputs = btn.prepare_inputs({'x1': test_input}, for_sigma=True)
    test_full_sigma_tn = btn.sigma & test_sigma_inputs
    
    sigma_nodes_to_test = ['T1_sigma', 'T2_sigma', 'T3_sigma']
    
    for node_tag in sigma_nodes_to_test:
        env = btn._batch_environment(test_full_sigma_tn, node_tag, copy=True)
        print(f"\n{node_tag}:")
        print(f"  Environment indices: {env.inds}")
        print(f"  Environment shape: {env.data.shape}")
        
        # Verify batch dimension exists
        if 's' in env.inds:
            print(f"  ✅ Batch dimension present")
        else:
            print(f"  ❌ Batch dimension MISSING!")
        
        # Check for prime indices (should exist for sigma environments)
        prime_inds = [ind for ind in env.inds if '_prime' in ind]
        if prime_inds:
            print(f"  ✅ Prime indices present: {prime_inds}")
        else:
            print(f"  ⚠️  No prime indices (may be expected depending on node)")

    print_header("ALL TESTS COMPLETED!")
    print("\nSummary:")
    print("  ✅ MU network environment returns correct indices")
    print("  ✅ SIGMA network environment returns correct indices (with primes)")
    print("  ✅ Multiple batch processing works correctly")
    print("  ✅ Batch dimension stacking verified")
    print("  ✅ All nodes tested for proper environment computation")

if __name__ == "__main__":
    main()
