# type: ignore
"""
Test flags for SINGLE BATCH to see what indices/labels are returned.
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
            print(f"  Data:\n{result.data}")
    else:
        # Scalar result from quimb when output_inds=[]
        print(f"  Type: {type(result)}")
        print(f"  Scalar value: {result}")

def main():
    print("="*70)
    print(" SINGLE BATCH FLAG TEST")
    print("="*70)
    
    # Simple setup
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
    print(f"Batch dimension: 's'")
    print(f"Batch size: {batch_size}")
    
    # Prepare single batch
    input_data = np.random.randn(batch_size, x_dim)
    inputs = btn.prepare_inputs({'x1': input_data}, for_sigma=False)
    
    print(f"\nInput tensors:")
    for inp in inputs:
        print(f"  {inp.inds}: shape {inp.data.shape}")
    
    # =========================================================================
    print("\n" + "="*70)
    print(" FORWARD - SINGLE BATCH")
    print("="*70)
    
    print("\n--- Case 1: sum_over_batch=False, sum_over_output=False ---")
    result1 = btn._batch_forward(mu_tn, inputs, output_inds=['s', 'y1', 'y2'])
    print_result(result1, "FORWARD (keep batch, keep output)")
    
    print("\n--- Case 2: sum_over_batch=True, sum_over_output=False ---")
    result2 = btn._batch_forward(mu_tn, inputs, output_inds=['y1', 'y2'])
    print_result(result2, "FORWARD (sum batch, keep output)")
    
    print("\n--- Case 3: sum_over_batch=False, sum_over_output=True ---")
    result3 = btn._batch_forward(mu_tn, inputs, output_inds=['s'])
    print_result(result3, "FORWARD (keep batch, sum output)")
    
    print("\n--- Case 4: sum_over_batch=True, sum_over_output=True ---")
    result4 = btn._batch_forward(mu_tn, inputs, output_inds=[])
    print_result(result4, "FORWARD (sum batch, sum output)")
    
    # =========================================================================
    print("\n" + "="*70)
    print(" GET_ENVIRONMENT - SINGLE BATCH")
    print("="*70)
    
    # Create full network with inputs
    full_tn = mu_tn & inputs
    
    print("\n--- Case 1: sum_over_batch=False, sum_over_output=False ---")
    env1 = btn._batch_environment(full_tn, 'T1', copy=True, 
                               sum_over_batch=False, sum_over_output=False)
    print_result(env1, "ENVIRONMENT T1 (keep batch, keep output)")
    
    print("\n--- Case 2: sum_over_batch=True, sum_over_output=False ---")
    env2 = btn._batch_environment(full_tn, 'T1', copy=True,
                               sum_over_batch=True, sum_over_output=False)
    print_result(env2, "ENVIRONMENT T1 (sum batch, keep output)")
    
    print("\n--- Case 3: sum_over_batch=False, sum_over_output=True ---")
    env3 = btn._batch_environment(full_tn, 'T1', copy=True,
                               sum_over_batch=False, sum_over_output=True)
    print_result(env3, "ENVIRONMENT T1 (keep batch, sum output)")
    
    print("\n--- Case 4: sum_over_batch=True, sum_over_output=True ---")
    env4 = btn._batch_environment(full_tn, 'T1', copy=True,
                               sum_over_batch=True, sum_over_output=True)
    print_result(env4, "ENVIRONMENT T1 (sum batch, sum output)")
    
    # Test T2 environment as well
    print("\n" + "="*70)
    print(" GET_ENVIRONMENT T2 - SINGLE BATCH")
    print("="*70)
    
    print("\n--- Case 1: sum_over_batch=False, sum_over_output=False ---")
    env_t2_1 = btn._batch_environment(full_tn, 'T2', copy=True,
                                    sum_over_batch=False, sum_over_output=False)
    print_result(env_t2_1, "ENVIRONMENT T2 (keep batch, keep output)")
    
    print("\n--- Case 2: sum_over_batch=True, sum_over_output=False ---")
    env_t2_2 = btn._batch_environment(full_tn, 'T2', copy=True,
                                    sum_over_batch=True, sum_over_output=False)
    print_result(env_t2_2, "ENVIRONMENT T2 (sum batch, keep output)")
    
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    print("\nAll single batch flag combinations tested!")
    print("Check the indices to see if they match expectations.")

if __name__ == "__main__":
    main()
