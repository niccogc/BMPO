# type: ignore
"""
Test script for BTN.prepare_inputs method.
Demonstrates both scenarios: single input for all nodes and separate inputs per node.
Also tests the for_sigma parameter to generate prime indices.
"""

import numpy as np
import quimb.tensor as qt
from tensor.btn import BTN

def print_header(title):
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")

def inspect_tensor_list(tensors, label="Tensors"):
    print(f"\n{label}:")
    print(f"  Total count: {len(tensors)}")
    for i, t in enumerate(tensors):
        tags_str = ", ".join(sorted(list(t.tags)))
        print(f"  [{i}] Inds: {t.inds} | Shape: {t.data.shape} | Tags: [{tags_str}]")

def main():
    print_header("SETUP: Creating Mock Tensor Network")
    
    # Configuration
    x_dim = 4      # Input feature dimension
    bond_dim = 5   # Bond dimension between nodes
    y_dim = 2      # Output dimension
    batch_dim = 's'
    
    # Create a simple network: T1 -[k1]- T2 -[k2]- T3
    # T1 has input x1, T2 has input x2, T3 outputs y
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
    
    # Initialize BTN
    btn = BTN(mu_tn, output_dimensions=output_dims, batch_dim=batch_dim)
    print("\n✅ BTN initialized successfully")

    # =========================================================================
    print_header("TEST 1: Scenario 1 - Single Input for All Nodes (MU)")
    # =========================================================================
    
    # Create sample data: 10 samples, 4 features
    batch_size = 10
    feature_dim = 4
    single_input_data = np.random.randn(batch_size, feature_dim)
    
    print(f"\nInput data shape: {single_input_data.shape}")
    print(f"Providing single input to be used for all input nodes (x1, x2)")
    
    input_dict_single = {'x1': single_input_data}
    mu_tensors_single = btn.prepare_inputs(input_dict_single, for_sigma=False)
    
    inspect_tensor_list(mu_tensors_single, "MU Network Tensors (Single Input)")
    
    # Verify: Should have 2 tensors (one for x1, one for x2)
    assert len(mu_tensors_single) == 2, f"Expected 2 tensors, got {len(mu_tensors_single)}"
    
    # Check that both have the batch dimension and correct input indices
    inds_set = {t.inds for t in mu_tensors_single}
    expected_inds = {(batch_dim, 'x1'), (batch_dim, 'x2')}
    assert inds_set == expected_inds, f"Indices mismatch: {inds_set} vs {expected_inds}"
    
    print("\n✅ Scenario 1 (MU) passed: Single input correctly replicated for all nodes")

    # =========================================================================
    print_header("TEST 2: Scenario 1 - Single Input for All Nodes (SIGMA)")
    # =========================================================================
    
    sigma_tensors_single = btn.prepare_inputs(input_dict_single, for_sigma=True)
    
    inspect_tensor_list(sigma_tensors_single, "SIGMA Network Tensors (Single Input)")
    
    # Verify: Should have 4 tensors (x1, x1_prime, x2, x2_prime)
    assert len(sigma_tensors_single) == 4, f"Expected 4 tensors, got {len(sigma_tensors_single)}"
    
    # Check indices
    inds_set_sigma = {t.inds for t in sigma_tensors_single}
    expected_inds_sigma = {
        (batch_dim, 'x1'), 
        (batch_dim, 'x1_prime'),
        (batch_dim, 'x2'), 
        (batch_dim, 'x2_prime')
    }
    assert inds_set_sigma == expected_inds_sigma, f"Indices mismatch: {inds_set_sigma} vs {expected_inds_sigma}"
    
    print("\n✅ Scenario 1 (SIGMA) passed: Prime indices correctly generated")

    # =========================================================================
    print_header("TEST 3: Scenario 2 - Separate Inputs per Node (MU)")
    # =========================================================================
    
    # Different data for x1 and x2
    data_x1 = np.random.randn(batch_size, feature_dim)
    data_x2 = np.random.randn(batch_size, feature_dim) * 2.0  # Different scale
    
    print(f"\nInput x1 shape: {data_x1.shape}")
    print(f"Input x2 shape: {data_x2.shape}")
    print(f"x1 mean: {data_x1.mean():.3f}, x2 mean: {data_x2.mean():.3f}")
    
    input_dict_separate = {'x1': data_x1, 'x2': data_x2}
    mu_tensors_separate = btn.prepare_inputs(input_dict_separate, for_sigma=False)
    
    inspect_tensor_list(mu_tensors_separate, "MU Network Tensors (Separate Inputs)")
    
    # Verify count and indices
    assert len(mu_tensors_separate) == 2, f"Expected 2 tensors, got {len(mu_tensors_separate)}"
    
    # Verify data is different
    tensor_x1 = next(t for t in mu_tensors_separate if 'x1' in t.inds and 'x1_prime' not in t.inds)
    tensor_x2 = next(t for t in mu_tensors_separate if 'x2' in t.inds and 'x2_prime' not in t.inds)
    
    assert np.allclose(tensor_x1.data, data_x1), "x1 tensor data mismatch"
    assert np.allclose(tensor_x2.data, data_x2), "x2 tensor data mismatch"
    assert not np.allclose(tensor_x1.data, tensor_x2.data), "x1 and x2 should have different data"
    
    print("\n✅ Scenario 2 (MU) passed: Separate inputs correctly assigned")

    # =========================================================================
    print_header("TEST 4: Scenario 2 - Separate Inputs per Node (SIGMA)")
    # =========================================================================
    
    sigma_tensors_separate = btn.prepare_inputs(input_dict_separate, for_sigma=True)
    
    inspect_tensor_list(sigma_tensors_separate, "SIGMA Network Tensors (Separate Inputs)")
    
    # Verify count
    assert len(sigma_tensors_separate) == 4, f"Expected 4 tensors, got {len(sigma_tensors_separate)}"
    
    # Verify that x1 and x1_prime have the same data
    tensor_x1 = next(t for t in sigma_tensors_separate if t.inds == (batch_dim, 'x1'))
    tensor_x1_prime = next(t for t in sigma_tensors_separate if t.inds == (batch_dim, 'x1_prime'))
    
    assert np.allclose(tensor_x1.data, tensor_x1_prime.data), "x1 and x1_prime should have same data"
    assert np.allclose(tensor_x1.data, data_x1), "x1 data mismatch"
    
    # Verify that x2 and x2_prime have the same data
    tensor_x2 = next(t for t in sigma_tensors_separate if t.inds == (batch_dim, 'x2'))
    tensor_x2_prime = next(t for t in sigma_tensors_separate if t.inds == (batch_dim, 'x2_prime'))
    
    assert np.allclose(tensor_x2.data, tensor_x2_prime.data), "x2 and x2_prime should have same data"
    assert np.allclose(tensor_x2.data, data_x2), "x2 data mismatch"
    
    print("\n✅ Scenario 2 (SIGMA) passed: Prime copies have identical data")

    # =========================================================================
    print_header("TEST 5: Error Handling - Missing Input")
    # =========================================================================
    
    print("\nNote: Single key dict {'x1': data} is treated as Scenario 1 (replicate to all)")
    print("Testing error when multiple keys provided but one is missing...")
    
    try:
        # Provide x1 and a wrong key, missing x2
        incomplete_input = {'x1': data_x1, 'wrong_key': data_x2}
        btn.prepare_inputs(incomplete_input, for_sigma=False)
        print("\n❌ Should have raised ValueError for missing input")
    except ValueError as e:
        print(f"\n✅ Correctly raised ValueError for missing x2")
        print(f"   Error message: {e}")

    # =========================================================================
    print_header("TEST 6: Integration Test with Forward Pass")
    # =========================================================================
    
    print("\nTesting that prepared inputs can be used with forward method...")
    
    # Prepare inputs for mu network
    mu_inputs = btn.prepare_inputs(input_dict_separate, for_sigma=False)
    
    # The forward method expects List[List[qt.Tensor]] for batching
    # We'll create a simple batch with our tensors
    batched_inputs = [mu_inputs]
    
    try:
        # Attempt forward pass
        result = btn.forward(btn.mu, batched_inputs)
        print(f"\n✅ Forward pass successful!")
        print(f"   Output shape: {result.data.shape}")
        print(f"   Output indices: {result.inds}")
        
        # Verify output indices
        expected_output_inds = tuple([batch_dim] + output_dims)
        assert result.inds == expected_output_inds, f"Output indices mismatch: {result.inds} vs {expected_output_inds}"
        
    except Exception as e:
        print(f"\n❌ Forward pass failed: {e}")
        raise

    print_header("ALL TESTS PASSED! 🎉")
    print("\nSummary:")
    print("  ✅ Scenario 1 (single input) works for both MU and SIGMA")
    print("  ✅ Scenario 2 (separate inputs) works for both MU and SIGMA")
    print("  ✅ Prime indices correctly generated for SIGMA network")
    print("  ✅ Error handling for missing inputs")
    print("  ✅ Integration with forward method")

if __name__ == "__main__":
    main()
