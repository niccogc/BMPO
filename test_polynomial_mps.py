"""
Test Bayesian Tensor Network (MPS) on 3rd degree polynomial.

Similar to visualize_predictions_standardized.py but using BayesianTensorNetwork.
Tests the complete training loop on: y = 2x³ - x² + 0.5
"""

import torch
import quimb.tensor as qtn
import numpy as np
import matplotlib.pyplot as plt

from tensor.bayesian_tn import BayesianTensorNetwork
from tensor.bayesian_tn_builder import create_sigma_network


def test_polynomial_mps():
    """Test MPS on 3rd degree polynomial: y = 2x³ - x² + 0.5"""
    print("\n" + "="*70)
    print("TEST: Bayesian MPS for 3rd Degree Polynomial")
    print("="*70)
    
    # Hyperparameters
    num_blocks = 3
    bond_dim = 6
    max_iter = 40
    num_samples = 100
    noise_std = 0.1
    seed = 42
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"Hyperparameters:")
    print(f"  Polynomial: y = 2x³ - x² + 0.5")
    print(f"  Blocks: {num_blocks}, Bond dim: {bond_dim}")
    print(f"  Samples: {num_samples}, Noise: {noise_std}")
    print(f"  Iterations: {max_iter}")
    print()
    
    # ========================================================================
    # GENERATE DATA
    # ========================================================================
    # For 3rd degree polynomial: y = 2x³ - x² + 0.5
    # Standard MPS with 3 blocks, each receiving [1, x]
    x = torch.rand(num_samples, dtype=torch.float64) * 2 - 1  # [-1, 1]
    y_true = 2.0 * x**3 - 1.0 * x**2 + 0.5
    y = y_true + noise_std * torch.randn(num_samples, dtype=torch.float64)
    
    # Input features: [1, x] for EACH block
    X = torch.stack([torch.ones_like(x), x], dim=1)
    
    print(f"Training data:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  y range: [{y.min():.3f}, {y.max():.3f}]")
    print()
    
    # Test data
    x_test = torch.linspace(-1.2, 1.2, 200, dtype=torch.float64)
    y_test_true = 2.0 * x_test**3 - 1.0 * x_test**2 + 0.5
    X_test = torch.stack([torch.ones_like(x_test), x_test], dim=1)
    
    # ========================================================================
    # CREATE MPS NETWORK
    # ========================================================================
    print("Creating MPS network...")
    print(f"  Structure: Block0(p0, r1) - Block1(r1, p1, r2) - Block2(r2, p2, y)")
    print(f"  Each block receives [1, x] on its own index")
    print(f"  p0, p1, p2 are separate indices, each dimension 2")
    print()
    
    # Standard MPS: Block0(p0, r1) - Block1(r1, p1, r2) - Block2(r2, p2, y)
    # Each p_i has dimension 2 (for [1, x])
    
    # Initialize blocks with small random values
    torch.manual_seed(seed)
    block0_data = torch.randn(2, bond_dim, dtype=torch.float64) * 0.1  # (p0, r1)
    block1_data = torch.randn(bond_dim, 2, bond_dim, dtype=torch.float64) * 0.1  # (r1, p1, r2)
    block2_data = torch.randn(bond_dim, 2, 1, dtype=torch.float64) * 0.1  # (r2, p2, y)
    
    Block0 = qtn.Tensor(data=block0_data, inds=('p0', 'r1'), tags='Block0')
    Block1 = qtn.Tensor(data=block1_data, inds=('r1', 'p1', 'r2'), tags='Block1')
    Block2 = qtn.Tensor(data=block2_data, inds=('r2', 'p2', 'y'), tags='Block2')
    
    mu_tn = qtn.TensorNetwork([Block0, Block1, Block2])
    
    print("Block shapes:")
    print(f"  Block0: {Block0.shape} -> indices {Block0.inds}")
    print(f"  Block1: {Block1.shape} -> indices {Block1.inds}")
    print(f"  Block2: {Block2.shape} -> indices {Block2.inds}")
    print()
    
    # Create sigma network
    learnable_tags = ['Block0', 'Block1', 'Block2']
    sigma_tn = create_sigma_network(mu_tn, learnable_tags, output_indices=['y'])
    
    # Create model
    # Standard MPS: each index p0, p1, p2 gets the same [1, x] input
    model = BayesianTensorNetwork(
        mu_tn=mu_tn,
        sigma_tn=sigma_tn,
        input_indices={
            'features': ['p0', 'p1', 'p2']  # All 3 indices contract with same [1, x] input
        },
        output_indices=['y'],
        learnable_tags=learnable_tags,
        tau_alpha=torch.tensor(2.0, dtype=torch.float64),
        tau_beta=torch.tensor(1.0, dtype=torch.float64),
        dtype=torch.float64
    )
    
    print(f"Model created successfully!")
    print(f"  Learnable nodes: {model.learnable_tags}")
    print(f"  Bonds: {model.mu_network.bond_labels}")
    print()
    
    # ========================================================================
    # TRAIN MODEL
    # ========================================================================
    print(f"Training for {max_iter} iterations...")
    print()
    
    # All blocks get the same [1, x] input
    inputs_dict = {'features': X}
    
    history = model.fit(
        X=None,
        y=y,
        inputs_dict=inputs_dict,
        max_iter=max_iter,
        verbose=True
    )
    
    print()
    
    # ========================================================================
    # EVALUATE
    # ========================================================================
    print("="*70)
    print("EVALUATION")
    print("="*70)
    
    # Predictions
    inputs_test = {'features': X_test}
    mu_pred = model.forward_mu(inputs_test)
    sigma_pred = model.forward_sigma(inputs_test)
    
    # Uncertainty
    tau_mean = model.get_tau_mean().item()
    aleatoric_var = 1.0 / tau_mean
    total_var = sigma_pred + aleatoric_var
    total_std = torch.sqrt(total_var)
    epistemic_std = torch.sqrt(sigma_pred)
    
    # Metrics
    residuals = mu_pred - y_test_true
    rmse = torch.sqrt((residuals**2).mean()).item()
    ss_res = (residuals**2).sum().item()
    ss_tot = ((y_test_true - y_test_true.mean())**2).sum().item()
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"Test Metrics:")
    print(f"  RMSE: {rmse:.5f}")
    print(f"  R²: {r2:.5f}")
    print()
    print(f"Uncertainty:")
    print(f"  E[τ]: {tau_mean:.4f} (implied noise std: {1/np.sqrt(tau_mean):.4f})")
    print(f"  Mean epistemic std: {epistemic_std.mean():.5f}")
    print(f"  Mean total std: {total_std.mean():.5f}")
    print()
    
    # Check performance
    assert r2 > 0.8, f"R² too low: {r2:.3f} < 0.8"
    assert rmse < 0.5, f"RMSE too high: {rmse:.3f} > 0.5"
    
    print("✓ Performance checks passed!")
    print()
    
    # ========================================================================
    # PLOT
    # ========================================================================
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Predictions
        ax = axes[0]
        ax.scatter(x.numpy(), y.numpy(), alpha=0.5, s=20, c='gray', label='Training data')
        ax.plot(x_test.numpy(), y_test_true.numpy(), 'k-', linewidth=2, label='True')
        ax.plot(x_test.numpy(), mu_pred.numpy(), 'r-', linewidth=2, label='Predicted mean')
        ax.fill_between(x_test.numpy(),
                        (mu_pred - 2*total_std).numpy(),
                        (mu_pred + 2*total_std).numpy(),
                        alpha=0.3, color='red', label='±2σ total')
        ax.fill_between(x_test.numpy(),
                        (mu_pred - 2*epistemic_std).numpy(),
                        (mu_pred + 2*epistemic_std).numpy(),
                        alpha=0.5, color='blue', label='±2σ epistemic')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'MPS: y = 2x³ - x² + 0.5 (R²={r2:.3f})', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Residuals
        ax = axes[1]
        ax.scatter(x_test.numpy(), residuals.numpy(), alpha=0.5, s=10, c='purple')
        ax.axhline(y=0, color='k', linestyle='--')
        ax.fill_between(x_test.numpy(),
                        (-2*total_std).numpy(),
                        (2*total_std).numpy(),
                        alpha=0.2, color='red')
        ax.set_xlabel('x')
        ax.set_ylabel('Residual')
        ax.set_title(f'Residuals (RMSE={rmse:.4f})', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = 'test_polynomial_mps_predictions.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {filename}")
    except Exception as e:
        print(f"Plotting failed (non-critical): {e}")
    
    print()
    print("="*70)
    print("✓ SUCCESS: MPS trained successfully on 3rd degree polynomial!")
    print("="*70)
    
    return history, r2, rmse


if __name__ == '__main__':
    history, r2, rmse = test_polynomial_mps()
    
    print("\nFinal Summary:")
    print(f"  R² = {r2:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  Tau evolution: {history['tau'][0]:.4f} → {history['tau'][-1]:.4f}")
