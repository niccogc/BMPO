# type: ignore
"""
Example usage of BTN.prepare_inputs method.
Demonstrates the two scenarios for preparing input data.
"""

import numpy as np
import quimb.tensor as qt
from tensor.btn import BTN

def main():
    # Create a simple network topology
    print("Creating a simple BTN network...")
    
    # Network structure: T1(x1, k1) <-> T2(k1, x2, y1, k2) <-> T3(k2, y2)
    t1 = qt.Tensor(np.random.rand(4, 5), inds=('x1', 'k1'), tags={'T1'})
    t2 = qt.Tensor(np.random.rand(5, 4, 2, 5), inds=('k1', 'x2', 'y1', 'k2'), tags={'T2'})
    t3 = qt.Tensor(np.random.rand(5, 2), inds=('k2', 'y2'), tags={'T3'})
    
    mu_tn = t1 & t2 & t3
    btn = BTN(mu_tn, output_dimensions=['y1', 'y2'], batch_dim='s')
    
    print("✅ BTN network created with input indices: x1, x2\n")
    
    # =========================================================================
    # SCENARIO 1: Single input replicated to all nodes
    # =========================================================================
    print("="*70)
    print("SCENARIO 1: Single input data replicated to all input nodes")
    print("="*70)
    
    # Create input data: 8 samples with 4 features each
    input_data_single = np.random.randn(8, 4)
    print(f"Input shape: {input_data_single.shape}")
    
    # For MU network (mean)
    mu_inputs = btn.prepare_inputs({'x1': input_data_single}, for_sigma=False)
    print(f"\nMU network: Created {len(mu_inputs)} input tensors")
    for t in mu_inputs:
        print(f"  - {t.inds}: shape {t.data.shape}")
    
    # For SIGMA network (covariance) - doubles with _prime
    sigma_inputs = btn.prepare_inputs({'x1': input_data_single}, for_sigma=True)
    print(f"\nSIGMA network: Created {len(sigma_inputs)} input tensors (with primes)")
    for t in sigma_inputs:
        print(f"  - {t.inds}: shape {t.data.shape}")
    
    # =========================================================================
    # SCENARIO 2: Separate inputs per node
    # =========================================================================
    print("\n" + "="*70)
    print("SCENARIO 2: Different data for each input node")
    print("="*70)
    
    # Different data for each input
    data_x1 = np.random.randn(8, 4)
    data_x2 = np.random.randn(8, 4) * 2.0  # Different scale
    
    print(f"x1 shape: {data_x1.shape}, mean: {data_x1.mean():.3f}")
    print(f"x2 shape: {data_x2.shape}, mean: {data_x2.mean():.3f}")
    
    # For MU network
    mu_inputs_sep = btn.prepare_inputs({'x1': data_x1, 'x2': data_x2}, for_sigma=False)
    print(f"\nMU network: Created {len(mu_inputs_sep)} input tensors")
    for t in mu_inputs_sep:
        print(f"  - {t.inds}: shape {t.data.shape}, mean: {t.data.mean():.3f}")
    
    # For SIGMA network
    sigma_inputs_sep = btn.prepare_inputs({'x1': data_x1, 'x2': data_x2}, for_sigma=True)
    print(f"\nSIGMA network: Created {len(sigma_inputs_sep)} input tensors (with primes)")
    for t in sigma_inputs_sep:
        print(f"  - {t.inds}: shape {t.data.shape}, mean: {t.data.mean():.3f}")
    
    # =========================================================================
    # Forward pass example
    # =========================================================================
    print("\n" + "="*70)
    print("FORWARD PASS EXAMPLE")
    print("="*70)
    
    # Prepare inputs for forward pass
    batched_inputs = [mu_inputs_sep]  # List of batches
    
    # Perform forward pass
    output = btn.forward(btn.mu, batched_inputs)
    
    print(f"\nForward pass completed!")
    print(f"Output shape: {output.data.shape}")
    print(f"Output indices: {output.inds}")
    print(f"Expected: (batch={8}, y1={2}, y2={2})")
    
    print("\n✅ All examples completed successfully!")

if __name__ == "__main__":
    main()
