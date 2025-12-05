# type: ignore
"""
Test sigma environment and mu environment outer product computations.
"""

import numpy as np
import quimb.tensor as qt
from tensor.btn import BTN


def test_sigma_environment():
    """Test sigma environment computation."""
    print("\n=== Test 1: Sigma Environment ===")
    
    # Create simple 2-node network WITHOUT batch dimension
    # x1, x2 = inputs, h = hidden bond, y = output
    node1 = qt.Tensor(np.random.randn(2, 3, 5), inds=('x1', 'h', 'y'), tags={'node1'})
    node2 = qt.Tensor(np.random.randn(3, 4, 5), inds=('h', 'x2', 'y'), tags={'node2'})
    
    mu = qt.TensorNetwork([node1, node2])
    btn = BTN(mu=mu, output_dimensions=['y'], batch_dim='s')
    
    # Prepare inputs using BTN's prepare_inputs with for_sigma=True
    num_batches = 10
    batch_size = 1
    
    sigma_batches = []
    for i in range(num_batches):
        input_data = {'x1': np.random.randn(batch_size, 2), 'x2': np.random.randn(batch_size, 4)}
        sigma_inputs = btn.prepare_inputs(input_data, for_sigma=True)
        sigma_batches.append(sigma_inputs)
    
    # Compute sigma environment
    sigma_env = btn.get_environment_batched(
        btn.sigma,
        'node1_sigma',
        sigma_batches,
        sum_over_batch=True,
        sum_over_output=True
    )
    
    print(f"Sigma environment: inds={sigma_env.inds}, shape={sigma_env.shape}")
    print(f"Expected: (x1, h, x1_prime, h_prime) - node1's bonds and primes")
    print("✓ Sigma environment test passed!")


def test_mu_environment_outer():
    """Test mu environment outer product computation."""
    print("\n=== Test 2: Mu Environment Outer Product ===")
    
    # Create simple 2-node network WITHOUT batch dimension
    node1 = qt.Tensor(np.random.randn(2, 3, 5), inds=('x1', 'h', 'y'), tags={'node1'})
    node2 = qt.Tensor(np.random.randn(3, 4, 5), inds=('h', 'x2', 'y'), tags={'node2'})
    
    mu = qt.TensorNetwork([node1, node2])
    btn = BTN(mu=mu, output_dimensions=['y'], batch_dim='s')
    
    # Prepare inputs using BTN's prepare_inputs with for_sigma=False (for mu)
    # For compute_environment_outer, pass data with all batches at once
    input_data = {'x1': np.random.randn(10, 2), 'x2': np.random.randn(10, 4)}
    mu_inputs = btn.prepare_inputs(input_data, for_sigma=False)
    
    # Compute mu environment outer
    env_outer = btn.compute_environment_outer('node1', mu_inputs)
    
    print(f"Mu environment outer: inds={env_outer.inds}, shape={env_outer.shape}")
    print(f"Expected: (x1, h, x1_prime, h_prime) - node1's bonds and primes")
    
    expected_inds = {'x1', 'h', 'x1_prime', 'h_prime'}
    actual_inds = set(env_outer.inds)
    assert actual_inds == expected_inds, f"Index mismatch! Expected {expected_inds}, got {actual_inds}"
    
    print("✓ Mu environment outer test passed!")


def test_shapes_consistency():
    """Test that sigma and mu environments have same shape."""
    print("\n=== Test 3: Shape Consistency ===")
    
    # Create network WITHOUT batch dimension
    node1 = qt.Tensor(np.random.randn(2, 3, 5), inds=('x1', 'h', 'y'), tags={'node1'})
    node2 = qt.Tensor(np.random.randn(3, 4, 5), inds=('h', 'x2', 'y'), tags={'node2'})
    
    mu = qt.TensorNetwork([node1, node2])
    btn = BTN(mu=mu, output_dimensions=['y'], batch_dim='s')
    
    num_batches = 10
    batch_size = 1
    
    # Prepare sigma inputs
    sigma_batches = []
    for i in range(num_batches):
        input_data = {'x1': np.random.randn(batch_size, 2), 'x2': np.random.randn(batch_size, 4)}
        sigma_inputs = btn.prepare_inputs(input_data, for_sigma=True)
        sigma_batches.append(sigma_inputs)
    
    # Prepare mu inputs - single call with all batches
    input_data = {'x1': np.random.randn(10, 2), 'x2': np.random.randn(10, 4)}
    mu_inputs = btn.prepare_inputs(input_data, for_sigma=False)
    
    # Get both environments
    sigma_env = btn.get_environment_batched(btn.sigma, 'node1_sigma', sigma_batches,
                                            sum_over_batch=True, sum_over_output=True)
    mu_env_outer = btn.compute_environment_outer('node1', mu_inputs)
    
    print(f"Sigma environment: inds={sigma_env.inds}, shape={sigma_env.shape}")
    print(f"Mu outer product: inds={mu_env_outer.inds}, shape={mu_env_outer.shape}")
    
    # They should have the same indices!
    assert set(sigma_env.inds) == set(mu_env_outer.inds), "Indices should match!"
    assert sigma_env.shape == mu_env_outer.shape, "Shapes should match!"
    
    print("✓ BOTH environments match - can be summed for precision!")


def test_theta_matches():
    """Test that theta has correct shape matching sigma and mu environments."""
    print("\n=== Test 4: Theta Shape Consistency ===")
    
    # Create network WITHOUT batch dimension
    node1 = qt.Tensor(np.random.randn(2, 3, 5), inds=('x1', 'h', 'y'), tags={'node1'})
    node2 = qt.Tensor(np.random.randn(3, 4, 5), inds=('h', 'x2', 'y'), tags={'node2'})
    
    mu = qt.TensorNetwork([node1, node2])
    btn = BTN(mu=mu, output_dimensions=['y'], batch_dim='s')
    
    # Get theta for node1
    theta = btn.theta_block_computation('node1')
    
    print(f"Theta: inds={theta.inds}, shape={theta.shape}")
    print(f"Expected: (x1, h) - node1's bonds WITHOUT primes")
    
    # Prepare inputs
    num_batches = 10
    batch_size = 1
    
    sigma_batches = []
    for i in range(num_batches):
        input_data = {'x1': np.random.randn(batch_size, 2), 'x2': np.random.randn(batch_size, 4)}
        sigma_inputs = btn.prepare_inputs(input_data, for_sigma=True)
        sigma_batches.append(sigma_inputs)
    
    input_data = {'x1': np.random.randn(10, 2), 'x2': np.random.randn(10, 4)}
    mu_inputs = btn.prepare_inputs(input_data, for_sigma=False)
    
    sigma_env = btn.get_environment_batched(btn.sigma, 'node1_sigma', sigma_batches,
                                            sum_over_batch=True, sum_over_output=True)
    mu_env_outer = btn.compute_environment_outer('node1', mu_inputs)
    
    print(f"\nSigma environment: inds={sigma_env.inds}, shape={sigma_env.shape}")
    print(f"Mu outer product: inds={mu_env_outer.inds}, shape={mu_env_outer.shape}")
    print(f"Theta:            inds={theta.inds}, shape={theta.shape}")
    
    # Check theta has right labels (bonds without primes)
    expected_theta_inds = {'x1', 'h'}
    actual_theta_inds = set(theta.inds)
    assert actual_theta_inds == expected_theta_inds, f"Theta indices wrong! Expected {expected_theta_inds}, got {actual_theta_inds}"
    
    # Check sigma/mu have theta labels + primes
    expected_env_inds = {'x1', 'h', 'x1_prime', 'h_prime'}
    assert set(sigma_env.inds) == expected_env_inds, "Sigma env indices wrong!"
    assert set(mu_env_outer.inds) == expected_env_inds, "Mu env indices wrong!"
    
    print("\n✓ Theta has correct shape - bonds without primes!")
    print("✓ Sigma and Mu envs have correct shape - bonds with primes!")
    print("✓ All three are compatible for precision calculation!")


if __name__ == '__main__':
    test_sigma_environment()
    test_mu_environment_outer()
    test_shapes_consistency()
    test_theta_matches()
    
    print("\n" + "="*50)
    print("ALL TESTS PASSED! ✓")
    print("="*50)
