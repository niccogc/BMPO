# type: ignore
import numpy as np
import quimb.tensor as qt
from btn import BTN

def run_test():
    # --- Config ---
    s1, s2 = 10, 5 # Batch sizes
    x_dim = 4      # Input features
    bond = 5       # Bond dimension
    y1_d, y2_d = 2, 3 # Output dimensions

    # --- 1. Construct Network (3 Blocks) ---
    # Structure:
    # T1(x1, k1) -> T2(k1, x2, y1, k2) -> T3(k2, y2)
    
    # Generate Data first to reuse for manual check
    data_t1 = np.random.rand(x_dim, bond)
    data_t2 = np.random.rand(bond, x_dim, y1_d, bond)
    data_t3 = np.random.rand(bond, y2_d)

    t1 = qt.Tensor(data_t1, inds=('x1', 'k1'), tags={'T1'})
    t2 = qt.Tensor(data_t2, inds=('k1', 'x2', 'y1', 'k2'), tags={'T2'})
    t3 = qt.Tensor(data_t3, inds=('k2', 'y2'), tags={'T3'})

    model_tn = t1 & t2 & t3
    
    # --- 2. Prepare Batches ---
    # Batch 1
    data_b1_x1 = np.random.rand(s1, x_dim)
    data_b1_x2 = np.random.rand(s1, x_dim)
    b1_x1 = qt.Tensor(data_b1_x1, inds=('s', 'x1'))
    b1_x2 = qt.Tensor(data_b1_x2, inds=('s', 'x2'))
    
    # Batch 2
    data_b2_x1 = np.random.rand(s2, x_dim)
    data_b2_x2 = np.random.rand(s2, x_dim)
    b2_x1 = qt.Tensor(data_b2_x1, inds=('s', 'x1'))
    b2_x2 = qt.Tensor(data_b2_x2, inds=('s', 'x2'))

    batches = [[b1_x1, b1_x2], [b2_x1, b2_x2]]

    # --- 3. Run BTN Forward ---
    print("Running BTN forward pass...")
    
    # FIX: Create dummy values to satisfy the full constructor requirements
    dummy_mu = qt.TensorNetwork([])
    dummy_sigma = qt.TensorNetwork([])
    dummy_fixed = []

    model = BTN(
        mu=dummy_mu,
        sigma=dummy_sigma,
        fixed_nodes=dummy_fixed,
        output_dimensions=['y1', 'y2'], 
        batch_dim='s'
    )
    
    result_tn = model.forward(model_tn, batches)

    # --- 4. Manual Verification (Numpy Einsum) ---
    print("Running manual verification...")

    # Combine all batch inputs for manual calculation
    # Stack inputs along batch dimension: shape (15, x_dim)
    total_in_x1 = np.concatenate([data_b1_x1, data_b2_x1], axis=0)
    total_in_x2 = np.concatenate([data_b1_x2, data_b2_x2], axis=0)

    # Define the contraction equation:
    # Inputs:
    #   total_in_x1: (s, x1) -> 'sa'
    #   T1:          (x1, k1) -> 'ai'
    #   T2:          (k1, x2, y1, k2) -> 'ibcj'
    #   total_in_x2: (s, x2) -> 'sb'
    #   T3:          (k2, y2) -> 'jd'
    # Output:
    #   Result:      (s, y1, y2) -> 'scd'
    
    manual_result = np.einsum(
        'sa, ai, ibcj, sb, jd -> scd',
        total_in_x1, # Input x1
        data_t1,     # T1
        data_t2,     # T2
        total_in_x2, # Input x2
        data_t3      # T3
    )

    # --- 5. Comparisons ---
    expected_s = s1 + s2
    
    print(f"\nTN Shape:     {result_tn.shape}")
    print(f"Manual Shape: {manual_result.shape}")
    
    # Check shapes
    assert result_tn.shape == (expected_s, y1_d, y2_d), "Shape mismatch!"
    
    # Check values (using a small tolerance for float precision)
    is_close = np.allclose(result_tn.data, manual_result, atol=1e-10)
    max_diff = np.max(np.abs(result_tn.data - manual_result))
    
    print(f"Max difference: {max_diff:.2e}")
    assert is_close, "Values do not match manual computation!"
    
    print("\n✅ Test Passed: BTN forward matches manual Einstein summation.")

if __name__ == "__main__":
    run_test()
