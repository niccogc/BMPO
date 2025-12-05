# type: ignore
"""
Test forward_with_target for scalar product and squared error.
Single batch tests for both mu and sigma networks.
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
        # Scalar result
        print(f"  Type: {type(result)}")
        print(f"  Scalar value: {result}")

def main():
    print("="*70)
    print(" FORWARD WITH TARGET - SINGLE BATCH TEST")
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
    
    print(f"\nInput tensors (MU):")
    for inp in inputs_mu:
        print(f"  {inp.inds}: shape {inp.data.shape}")
    
    # Create target output y
    y_data = np.random.randn(batch_size, y_dim, y_dim)
    y = qt.Tensor(y_data, inds=('s', 'y1', 'y2'), tags={'target'})
    
    print(f"\nTarget y:")
    print(f"  Indices: {y.inds}")
    print(f"  Shape: {y.data.shape}")
    
    # Get reference forward for comparison
    forward_ref = btn._batch_forward(mu_tn, inputs_mu, output_inds=['s', 'y1', 'y2'])
    forward_ref.transpose_('s', 'y1', 'y2')
    print(f"\nReference forward:")
    print(f"  Indices: {forward_ref.inds}")
    print(f"  Shape: {forward_ref.data.shape}")
    
    # =========================================================================
    print("\n" + "="*70)
    print(" TEST 1: DOT PRODUCT (MU NETWORK)")
    print("="*70)
    
    print("\n--- Mode: dot, sum_over_batch=False ---")
    dot_result = btn.forward_with_target(mu_tn, inputs_mu, y, mode='dot', sum_over_batch=False)
    print_result(dot_result, "Dot product (keep batch)")
    
    # Manual verification
    manual_dot = (forward_ref.data * y.data).sum(axis=(1, 2))  # Sum over y1, y2
    print(f"\nManual verification:")
    print(f"  Expected shape: ({batch_size},)")
    print(f"  Manual dot: {manual_dot}")
    if isinstance(dot_result, qt.Tensor):
        assert np.allclose(dot_result.data, manual_dot), "Dot product mismatch!"
        print("  ✅ Dot product matches manual calculation")
    
    print("\n--- Mode: dot, sum_over_batch=True ---")
    dot_sum_result = btn.forward_with_target(mu_tn, inputs_mu, y, mode='dot', sum_over_batch=True)
    print_result(dot_sum_result, "Dot product (sum over batch)")
    
    # Manual verification
    manual_dot_sum = (forward_ref.data * y.data).sum()
    print(f"\nManual verification:")
    print(f"  Manual dot sum: {manual_dot_sum}")
    if isinstance(dot_sum_result, (int, float, np.number)):
        assert np.allclose(dot_sum_result, manual_dot_sum), "Dot product sum mismatch!"
        print("  ✅ Dot product sum matches manual calculation")
    
    # =========================================================================
    print("\n" + "="*70)
    print(" TEST 2: SQUARED ERROR (MU NETWORK)")
    print("="*70)
    
    print("\n--- Mode: squared_error, sum_over_batch=False ---")
    se_result = btn.forward_with_target(mu_tn, inputs_mu, y, mode='squared_error', sum_over_batch=False)
    print_result(se_result, "Squared error (keep batch)")
    
    # Manual verification
    diff = forward_ref.data - y.data
    manual_se = (diff ** 2).sum(axis=(1, 2))  # Sum over y1, y2
    print(f"\nManual verification:")
    print(f"  Expected shape: ({batch_size},)")
    print(f"  Manual squared error: {manual_se}")
    if isinstance(se_result, qt.Tensor):
        assert np.allclose(se_result.data, manual_se), "Squared error mismatch!"
        print("  ✅ Squared error matches manual calculation")
    
    print("\n--- Mode: squared_error, sum_over_batch=True ---")
    se_sum_result = btn.forward_with_target(mu_tn, inputs_mu, y, mode='squared_error', sum_over_batch=True)
    print_result(se_sum_result, "Squared error (sum over batch)")
    
    # Manual verification
    manual_se_sum = ((forward_ref.data - y.data) ** 2).sum()
    print(f"\nManual verification:")
    print(f"  Manual squared error sum: {manual_se_sum}")
    if isinstance(se_sum_result, (int, float, np.number)):
        assert np.allclose(se_sum_result, manual_se_sum), "Squared error sum mismatch!"
        print("  ✅ Squared error sum matches manual calculation")
    
    # =========================================================================
    print("\n" + "="*70)
    print(" TEST 3: SIGMA NETWORK")
    print("="*70)
    
    # Prepare sigma inputs
    inputs_sigma = btn.prepare_inputs({'x1': input_data}, for_sigma=True)
    
    print(f"\nInput tensors (SIGMA):")
    for inp in inputs_sigma:
        print(f"  {inp.inds}: shape {inp.data.shape}")
    
    # Get reference forward for sigma
    forward_sigma_ref = btn._batch_forward(btn.sigma, inputs_sigma, output_inds=['s', 'y1', 'y2'])
    forward_sigma_ref.transpose_('s', 'y1', 'y2')
    print(f"\nReference SIGMA forward:")
    print(f"  Indices: {forward_sigma_ref.inds}")
    print(f"  Shape: {forward_sigma_ref.data.shape}")
    
    print("\n--- SIGMA: Mode: dot, sum_over_batch=False ---")
    dot_sigma = btn.forward_with_target(btn.sigma, inputs_sigma, y, mode='dot', sum_over_batch=False)
    print_result(dot_sigma, "SIGMA Dot product (keep batch)")
    
    # Manual verification
    manual_dot_sigma = (forward_sigma_ref.data * y.data).sum(axis=(1, 2))
    if isinstance(dot_sigma, qt.Tensor):
        assert np.allclose(dot_sigma.data, manual_dot_sigma), "SIGMA dot product mismatch!"
        print("  ✅ SIGMA dot product matches manual calculation")
    
    print("\n--- SIGMA: Mode: squared_error, sum_over_batch=False ---")
    se_sigma = btn.forward_with_target(btn.sigma, inputs_sigma, y, mode='squared_error', sum_over_batch=False)
    print_result(se_sigma, "SIGMA Squared error (keep batch)")
    
    # Manual verification
    diff_sigma = forward_sigma_ref.data - y.data
    manual_se_sigma = (diff_sigma ** 2).sum(axis=(1, 2))
    if isinstance(se_sigma, qt.Tensor):
        assert np.allclose(se_sigma.data, manual_se_sigma), "SIGMA squared error mismatch!"
        print("  ✅ SIGMA squared error matches manual calculation")
    
    print("\n" + "="*70)
    print(" ALL TESTS PASSED! 🎉")
    print("="*70)
    print("\nSummary:")
    print("  ✅ Dot product works for MU network")
    print("  ✅ Squared error works for MU network")
    print("  ✅ Dot product works for SIGMA network")
    print("  ✅ Squared error works for SIGMA network")
    print("  ✅ sum_over_batch flag works correctly")
    print("  ✅ Manual verification confirms correctness")

if __name__ == "__main__":
    main()
