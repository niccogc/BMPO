"""
Test complete variational update iteration.

Tests the full update cycle:
1. Update all nodes (μ and Σ)
2. Update all bonds (Gamma parameters)
3. Update tau (noise precision)
"""

import torch
import quimb.tensor as qtn
import numpy as np
import matplotlib.pyplot as plt

from tensor.bayesian_tn import BayesianTensorNetwork
from tensor.bayesian_tn_builder import create_sigma_network


def test_single_iteration():
    """Test one complete update iteration."""
    print("\n" + "="*70)
    print("TEST: Single Variational Update Iteration")
    print("="*70)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create simple network: A(x, r1) -- B(r1, y)
    A_data = torch.randn(3, 2, dtype=torch.float64)
    B_data = torch.randn(2, 1, dtype=torch.float64)
    
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
        dtype=torch.float64
    )
    
    print(f"Network: A(x, r1) -- B(r1, y)")
    print(f"Learnable nodes: ['A', 'B']")
    print(f"Bonds: {model.mu_network.bond_labels}")
    
    # Generate data
    batch_size = 10
    X = torch.randn(batch_size, 3, dtype=torch.float64)
    y = torch.randn(batch_size, 1, dtype=torch.float64)
    inputs_dict = {'data': X}
    
    # Store initial state
    A_before = model.mu_network.get_node_tensor('A').clone()
    B_before = model.mu_network.get_node_tensor('B').clone()
    tau_before = model.get_tau_mean().item()
    alpha_r1_before = model.mu_network.distributions['r1']['alpha'][0].item()
    
    print(f"\nBefore iteration:")
    print(f"  A norm: {torch.norm(A_before).item():.6f}")
    print(f"  B norm: {torch.norm(B_before).item():.6f}")
    print(f"  E[τ]: {tau_before:.6f}")
    print(f"  α_r1: {alpha_r1_before:.6f}")
    
    # ========================================================================
    # SINGLE ITERATION
    # ========================================================================
    print("\n" + "-"*70)
    print("Performing one complete iteration...")
    print("-"*70)
    
    model.variational_update_iteration(y, inputs_dict)
    
    # Check final state
    A_after = model.mu_network.get_node_tensor('A')
    B_after = model.mu_network.get_node_tensor('B')
    tau_after = model.get_tau_mean().item()
    alpha_r1_after = model.mu_network.distributions['r1']['alpha'][0].item()
    
    print(f"\nAfter iteration:")
    print(f"  A norm: {torch.norm(A_after).item():.6f}")
    print(f"  B norm: {torch.norm(B_after).item():.6f}")
    print(f"  E[τ]: {tau_after:.6f}")
    print(f"  α_r1: {alpha_r1_after:.6f}")
    
    # Verify parameters changed
    A_changed = not torch.allclose(A_before, A_after)
    B_changed = not torch.allclose(B_before, B_after)
    tau_changed = abs(tau_before - tau_after) > 1e-10
    alpha_changed = abs(alpha_r1_before - alpha_r1_after) > 1e-10
    
    print(f"\nChanges detected:")
    print(f"  A changed: {A_changed}")
    print(f"  B changed: {B_changed}")
    print(f"  τ changed: {tau_changed}")
    print(f"  α_r1 changed: {alpha_changed}")
    
    assert A_changed or B_changed, "At least one node should update"
    assert tau_changed, "Tau should update"
    assert alpha_changed, "Bond parameters should update"
    
    print("\n✓ SUCCESS: Complete iteration executed correctly!")


def test_multi_iteration_convergence():
    """Test convergence over multiple iterations."""
    print("\n" + "="*70)
    print("TEST: Multi-Iteration Convergence")
    print("="*70)
    
    np.random.seed(123)
    torch.manual_seed(123)
    
    # Create network
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
        dtype=torch.float64
    )
    
    # Generate synthetic data with known pattern
    batch_size = 20
    X = torch.randn(batch_size, 4, dtype=torch.float64)
    # True function: simple linear combination
    y_true = (X @ torch.randn(4, 1, dtype=torch.float64)).squeeze()
    y = y_true + 0.1 * torch.randn(batch_size, dtype=torch.float64)  # Add noise
    
    inputs_dict = {'data': X}
    
    print(f"Training on {batch_size} samples")
    print(f"Running 50 iterations...")
    
    # Track history
    tau_history = []
    mse_history = []
    
    for i in range(50):
        # Perform iteration
        model.variational_update_iteration(y, inputs_dict)
        
        # Track metrics
        tau = model.get_tau_mean().item()
        tau_history.append(tau)
        
        # Compute MSE
        y_pred = model.forward_mu(inputs_dict)
        mse = torch.mean((y - y_pred.squeeze()) ** 2).item()
        mse_history.append(mse)
        
        if i % 10 == 0 or i == 49:
            print(f"  Iter {i:2d}: E[τ]={tau:.4f}, MSE={mse:.4f}")
    
    # Check convergence
    print(f"\nConvergence analysis:")
    print(f"  Initial MSE: {mse_history[0]:.4f}")
    print(f"  Final MSE: {mse_history[-1]:.4f}")
    print(f"  MSE reduction: {(1 - mse_history[-1]/mse_history[0])*100:.1f}%")
    
    print(f"  Initial E[τ]: {tau_history[0]:.4f}")
    print(f"  Final E[τ]: {tau_history[-1]:.4f}")
    
    # Verify MSE decreased (or stayed stable)
    assert mse_history[-1] <= mse_history[0] * 1.1, "MSE should not increase significantly"
    
    print("\n✓ SUCCESS: Multi-iteration training completed!")
    
    return tau_history, mse_history


def test_fit_method():
    """Test the fit() method with history tracking."""
    print("\n" + "="*70)
    print("TEST: fit() Method")
    print("="*70)
    
    np.random.seed(456)
    torch.manual_seed(456)
    
    # Create network
    A_data = torch.randn(3, 2, dtype=torch.float64)
    B_data = torch.randn(2, 1, dtype=torch.float64)
    
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
        dtype=torch.float64
    )
    
    # Generate data
    batch_size = 15
    X = torch.randn(batch_size, 3, dtype=torch.float64)
    y = torch.randn(batch_size, 1, dtype=torch.float64)
    inputs_dict = {'data': X}
    
    # Fit model
    print(f"\nFitting model for 30 iterations...")
    history = model.fit(
        X=None,
        y=y,
        inputs_dict=inputs_dict,
        max_iter=30,
        verbose=True
    )
    
    # Check history
    print(f"\nHistory keys: {list(history.keys())}")
    print(f"Number of recorded iterations: {len(history['tau'])}")
    
    assert 'tau' in history, "History should contain tau"
    assert 'tau_alpha' in history, "History should contain tau_alpha"
    assert 'tau_beta' in history, "History should contain tau_beta"
    assert len(history['tau']) == 30, "Should have 30 iterations"
    
    print(f"\nTau evolution (first 5): {history['tau'][:5]}")
    print(f"Tau evolution (last 5): {history['tau'][-5:]}")
    
    print("\n✓ SUCCESS: fit() method works correctly!")
    
    return history


if __name__ == '__main__':
    test_single_iteration()
    tau_hist, mse_hist = test_multi_iteration_convergence()
    fit_hist = test_fit_method()
    
    print("\n" + "="*70)
    print("ALL UPDATE ITERATION TESTS PASSED!")
    print("="*70)
    
    # Optional: Plot results
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(mse_hist)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('MSE')
        ax1.set_title('Training MSE')
        ax1.grid(True)
        
        ax2.plot(tau_hist)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('E[τ]')
        ax2.set_title('Noise Precision')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('update_iteration_convergence.png', dpi=150)
        print("\nPlot saved to: update_iteration_convergence.png")
    except:
        print("\nSkipped plotting (matplotlib may not be available)")
