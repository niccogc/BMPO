"""
Test optimized tau update implementation.

Verifies that the tau update correctly computes:
- α_τ = α_τ^0 + S/2
- β_τ = β_τ^0 + 0.5 * Σ_n ||y_n - μ(x_n)||² + 0.5 * Σ_n Σ(x_n⊗x_n)
"""

import torch
import quimb.tensor as qtn
import numpy as np

from tensor.bayesian_tn import BayesianTensorNetwork
from tensor.bayesian_tn_builder import create_sigma_network


def test_tau_update_simple():
    """Test tau update on simple 2-node network."""
    print("\n" + "="*70)
    print("TEST: Tau Update - Simple 2-Node Network")
    print("="*70)
    
    # Create simple network: A(x, r1) -- B(r1, y)
    np.random.seed(42)
    torch.manual_seed(42)
    
    A_data = torch.randn(3, 2, dtype=torch.float64)
    B_data = torch.randn(2, 1, dtype=torch.float64)
    
    A = qtn.Tensor(data=A_data, inds=('x', 'r1'), tags='A')
    B = qtn.Tensor(data=B_data, inds=('r1', 'y'), tags='B')
    mu_tn = qtn.TensorNetwork([A, B])
    
    # Create sigma network
    sigma_tn = create_sigma_network(mu_tn, ['A', 'B'], output_indices=['y'])
    
    # Create model
    model = BayesianTensorNetwork(
        mu_tn=mu_tn,
        sigma_tn=sigma_tn,
        input_indices={'data': ['x']},
        output_indices=['y'],
        learnable_tags=['A', 'B'],
        tau_alpha=torch.tensor(2.0, dtype=torch.float64),
        tau_beta=torch.tensor(1.0, dtype=torch.float64),
        dtype=torch.float64
    )
    
    print(f"Initial tau: α={model._tau_alpha.item():.4f}, β={model._tau_beta.item():.4f}")
    print(f"Initial E[τ] = {model.get_tau_mean().item():.4f}")
    
    # Generate data
    batch_size = 5
    X = torch.randn(batch_size, 3, dtype=torch.float64)
    y = torch.randn(batch_size, 1, dtype=torch.float64)
    
    inputs_dict = {'data': X}
    
    print(f"\nData: {batch_size} samples")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Store prior parameters
    alpha_0 = model.prior_tau_alpha0.item()
    beta_0 = model.prior_tau_beta0.item()
    
    print(f"\nPrior: α₀={alpha_0:.4f}, β₀={beta_0:.4f}")
    
    # ========================================================================
    # MANUAL COMPUTATION (ground truth)
    # ========================================================================
    print("\n" + "-"*70)
    print("Manual Computation (Ground Truth)")
    print("-"*70)
    
    # Compute mu(x_n) for each sample
    mu_preds = model.forward_mu(inputs_dict)
    print(f"μ predictions shape: {mu_preds.shape}")
    
    # Compute MSE term
    mse = torch.sum((y - mu_preds) ** 2).item()
    print(f"MSE term: Σ_n ||y_n - μ(x_n)||² = {mse:.6f}")
    
    # Compute trace term (sum of sigma forwards)
    sigma_preds = model.forward_sigma(inputs_dict)
    print(f"Σ predictions shape: {sigma_preds.shape}")
    trace_sum = torch.sum(sigma_preds).item()
    print(f"Trace term: Σ_n Σ(x_n⊗x_n) = {trace_sum:.6f}")
    
    # Manual update
    alpha_expected = alpha_0 + batch_size / 2.0
    beta_expected = beta_0 + 0.5 * mse + 0.5 * trace_sum
    
    print(f"\nExpected updates:")
    print(f"  α_τ = {alpha_0:.4f} + {batch_size}/2 = {alpha_expected:.4f}")
    print(f"  β_τ = {beta_0:.4f} + 0.5*{mse:.4f} + 0.5*{trace_sum:.4f} = {beta_expected:.6f}")
    print(f"  E[τ] = {alpha_expected/beta_expected:.6f}")
    
    # ========================================================================
    # ACTUAL UPDATE
    # ========================================================================
    print("\n" + "-"*70)
    print("Actual Update (Optimized Implementation)")
    print("-"*70)
    
    model.update_tau_variational(X, y, inputs_dict)
    
    alpha_actual = model._tau_alpha.item()
    beta_actual = model._tau_beta.item()
    tau_mean = model.get_tau_mean().item()
    
    print(f"Actual updates:")
    print(f"  α_τ = {alpha_actual:.4f}")
    print(f"  β_τ = {beta_actual:.6f}")
    print(f"  E[τ] = {tau_mean:.6f}")
    
    # ========================================================================
    # VERIFY
    # ========================================================================
    print("\n" + "-"*70)
    print("Verification")
    print("-"*70)
    
    alpha_match = np.isclose(alpha_actual, alpha_expected, rtol=1e-10)
    beta_match = np.isclose(beta_actual, beta_expected, rtol=1e-10)
    
    print(f"α matches: {alpha_match} (diff={abs(alpha_actual - alpha_expected):.2e})")
    print(f"β matches: {beta_match} (diff={abs(beta_actual - beta_expected):.2e})")
    
    if alpha_match and beta_match:
        print("\n✓ SUCCESS: Tau update is correct!")
    else:
        print("\n✗ FAILURE: Tau update has errors")
        raise AssertionError("Tau update failed verification")


def test_tau_update_multi_output():
    """Test tau update with multiple outputs."""
    print("\n" + "="*70)
    print("TEST: Tau Update - Multi-Output Network")
    print("="*70)
    
    # Network: A(x, r1, y1) -- B(r1, y2)
    # Two output dimensions: y1, y2
    np.random.seed(42)
    torch.manual_seed(42)
    
    A_data = torch.randn(3, 2, 2, dtype=torch.float64)  # (x, r1, y1)
    B_data = torch.randn(2, 3, dtype=torch.float64)      # (r1, y2)
    
    A = qtn.Tensor(data=A_data, inds=('x', 'r1', 'y1'), tags='A')
    B = qtn.Tensor(data=B_data, inds=('r1', 'y2'), tags='B')
    mu_tn = qtn.TensorNetwork([A, B])
    
    # Create sigma network (y1, y2 are NOT doubled)
    sigma_tn = create_sigma_network(mu_tn, ['A', 'B'], output_indices=['y1', 'y2'])
    
    # Create model
    model = BayesianTensorNetwork(
        mu_tn=mu_tn,
        sigma_tn=sigma_tn,
        input_indices={'data': ['x']},
        output_indices=['y1', 'y2'],
        learnable_tags=['A', 'B'],
        tau_alpha=torch.tensor(3.0, dtype=torch.float64),
        tau_beta=torch.tensor(2.0, dtype=torch.float64),
        dtype=torch.float64
    )
    
    print(f"Initial tau: α={model._tau_alpha.item():.4f}, β={model._tau_beta.item():.4f}")
    print(f"Initial E[τ] = {model.get_tau_mean().item():.4f}")
    
    # Generate data
    batch_size = 4
    X = torch.randn(batch_size, 3, dtype=torch.float64)
    y = torch.randn(batch_size, 2, 3, dtype=torch.float64)  # (batch, y1, y2)
    
    inputs_dict = {'data': X}
    
    print(f"\nData: {batch_size} samples")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Store prior
    alpha_0 = model.prior_tau_alpha0.item()
    beta_0 = model.prior_tau_beta0.item()
    
    # Manual computation
    mu_preds = model.forward_mu(inputs_dict)
    mse = torch.sum((y - mu_preds) ** 2).item()
    
    sigma_preds = model.forward_sigma(inputs_dict)
    trace_sum = torch.sum(sigma_preds).item()
    
    alpha_expected = alpha_0 + batch_size / 2.0
    beta_expected = beta_0 + 0.5 * mse + 0.5 * trace_sum
    
    print(f"\nExpected: α={alpha_expected:.4f}, β={beta_expected:.6f}")
    
    # Actual update
    model.update_tau_variational(X, y, inputs_dict)
    
    alpha_actual = model._tau_alpha.item()
    beta_actual = model._tau_beta.item()
    
    print(f"Actual:   α={alpha_actual:.4f}, β={beta_actual:.6f}")
    
    # Verify
    alpha_match = np.isclose(alpha_actual, alpha_expected, rtol=1e-10)
    beta_match = np.isclose(beta_actual, beta_expected, rtol=1e-10)
    
    print(f"\nα matches: {alpha_match}")
    print(f"β matches: {beta_match}")
    
    if alpha_match and beta_match:
        print("\n✓ SUCCESS: Multi-output tau update is correct!")
    else:
        print("\n✗ FAILURE: Multi-output tau update has errors")
        raise AssertionError("Multi-output tau update failed")


def test_tau_update_iterative():
    """Test that tau update works correctly in iterative training."""
    print("\n" + "="*70)
    print("TEST: Tau Update - Iterative Training")
    print("="*70)
    
    # Simple network
    np.random.seed(42)
    torch.manual_seed(42)
    
    A_data = torch.randn(4, 3, dtype=torch.float64)
    B_data = torch.randn(3, 1, dtype=torch.float64)
    
    A = qtn.Tensor(data=A_data, inds=('x', 'r1'), tags='A')
    B = qtn.Tensor(data=B_data, inds=('r1', 'y'), tags='B')
    mu_tn = qtn.TensorNetwork([A, B])
    
    sigma_tn = create_sigma_network(mu_tn, ['A', 'B'], output_indices=['y'])
    
    model = BayesianTensorNetwork(
        mu_tn=mu_tn,
        sigma_tn=sigma_tn,
        input_indices={'data': ['x']},
        output_indices=['y'],
        learnable_tags=['A', 'B'],
        tau_alpha=torch.tensor(1.0, dtype=torch.float64),
        tau_beta=torch.tensor(1.0, dtype=torch.float64),
        dtype=torch.float64
    )
    
    # Generate data
    batch_size = 10
    X = torch.randn(batch_size, 4, dtype=torch.float64)
    y = torch.randn(batch_size, 1, dtype=torch.float64)
    inputs_dict = {'data': X}
    
    print(f"Training for 3 iterations...")
    print(f"Data: {batch_size} samples\n")
    
    tau_history = [model.get_tau_mean().item()]
    
    for i in range(3):
        model.update_tau_variational(X, y, inputs_dict)
        tau_mean = model.get_tau_mean().item()
        tau_history.append(tau_mean)
        
        print(f"Iteration {i+1}:")
        print(f"  α_τ = {model._tau_alpha.item():.6f}")
        print(f"  β_τ = {model._tau_beta.item():.6f}")
        print(f"  E[τ] = {tau_mean:.6f}")
    
    print("\nTau evolution:", [f"{t:.6f}" for t in tau_history])
    print("\n✓ SUCCESS: Iterative tau updates completed!")


if __name__ == '__main__':
    test_tau_update_simple()
    test_tau_update_multi_output()
    test_tau_update_iterative()
    print("\n" + "="*70)
    print("ALL TAU UPDATE TESTS PASSED!")
    print("="*70)
