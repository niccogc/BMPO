"""
Test Bayesian Tensor Network with MULTIPLE OUTPUTS.

Tests MPS topology with 2-dimensional output: y = [y1, y2]
where:
  y1 = 2x³ - x² + 0.5
  y2 = x² + 0.5x
"""

import torch
import quimb.tensor as qtn
import numpy as np
import matplotlib.pyplot as plt

from tensor.bayesian_tn import BayesianTensorNetwork
from tensor.bayesian_tn_builder import create_sigma_network


def test_multioutput_mps():
    """Test MPS with 2-dimensional output."""
    print("\n" + "="*70)
    print("TEST: Multi-Output MPS (y dimension = 2)")
    print("="*70)
    
    # Hyperparameters
    num_blocks = 3
    bond_dim = 6
    max_iter = 50
    num_samples = 100
    noise_std = 0.1
    seed = 42
    output_dim = 2  # TWO outputs!
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"Hyperparameters:")
    print(f"  Blocks: {num_blocks}, Bond dim: {bond_dim}")
    print(f"  Samples: {num_samples}, Noise: {noise_std}")
    print(f"  Iterations: {max_iter}")
    print(f"  OUTPUT DIMENSION: {output_dim}")
    print()
    
    # ========================================================================
    # GENERATE DATA - TWO OUTPUTS
    # ========================================================================
    x = torch.rand(num_samples, dtype=torch.float64) * 2 - 1
    
    # Two different functions
    y1_true = 2.0 * x**3 - 1.0 * x**2 + 0.5
    y2_true = x**2 + 0.5 * x
    
    # Add noise
    y1 = y1_true + noise_std * torch.randn(num_samples, dtype=torch.float64)
    y2 = y2_true + noise_std * torch.randn(num_samples, dtype=torch.float64)
    
    # Stack outputs: shape (batch, output_dim)
    y = torch.stack([y1, y2], dim=1)
    
    X = torch.stack([torch.ones_like(x), x], dim=1)
    
    print(f"Training data:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}  ← TWO outputs!")
    print(f"  y1 range: [{y1.min():.3f}, {y1.max():.3f}]")
    print(f"  y2 range: [{y2.min():.3f}, {y2.max():.3f}]")
    print()
    
    # Test data
    x_test = torch.linspace(-1.2, 1.2, 200, dtype=torch.float64)
    y1_test_true = 2.0 * x_test**3 - 1.0 * x_test**2 + 0.5
    y2_test_true = x_test**2 + 0.5 * x_test
    y_test_true = torch.stack([y1_test_true, y2_test_true], dim=1)
    X_test = torch.stack([torch.ones_like(x_test), x_test], dim=1)
    
    # ========================================================================
    # CREATE MPS WITH 2D OUTPUT
    # ========================================================================
    print("Creating MPS with 2D output...")
    print(f"  Structure: Block0(p0, r1) - Block1(r1, p1, r2) - Block2(r2, p2, y)")
    print(f"  y dimension: {output_dim}")
    print()
    
    # Create MPS: Last block has y with dimension 2
    torch.manual_seed(seed)
    block0_data = torch.randn(2, bond_dim, dtype=torch.float64) * 0.1
    block1_data = torch.randn(bond_dim, 2, bond_dim, dtype=torch.float64) * 0.1
    block2_data = torch.randn(bond_dim, 2, output_dim, dtype=torch.float64) * 0.1  # y dimension = 2
    
    Block0 = qtn.Tensor(data=block0_data, inds=('p0', 'r1'), tags='Block0')
    Block1 = qtn.Tensor(data=block1_data, inds=('r1', 'p1', 'r2'), tags='Block1')
    Block2 = qtn.Tensor(data=block2_data, inds=('r2', 'p2', 'y'), tags='Block2')
    
    mu_tn = qtn.TensorNetwork([Block0, Block1, Block2])
    learnable_tags = ['Block0', 'Block1', 'Block2']
    
    print("Block shapes:")
    for tag in learnable_tags:
        node = mu_tn[tag]
        print(f"  {tag}: {node.shape} -> {node.inds}")
    print()
    
    # Create sigma network
    sigma_tn = create_sigma_network(mu_tn, learnable_tags, output_indices=['y'])
    
    # Create model
    model = BayesianTensorNetwork(
        mu_tn=mu_tn,
        sigma_tn=sigma_tn,
        input_indices={'features': ['p0', 'p1', 'p2']},
        output_indices=['y'],
        learnable_tags=learnable_tags,
        dtype=torch.float64
    )
    
    print(f"Model created!")
    print(f"  Learnable nodes: {model.learnable_tags}")
    print(f"  Bonds: {model.mu_network.bond_labels}")
    print(f"  Output indices: {model.output_indices}")
    print()
    
    # ========================================================================
    # TRAIN MODEL
    # ========================================================================
    print(f"Training for {max_iter} iterations...")
    print()
    
    inputs_dict = {'features': X}
    
    history = model.fit(
        X=None,
        y=y,  # Shape: (batch, 2)
        inputs_dict=inputs_dict,
        max_iter=max_iter,
        verbose=True
    )
    
    print()
    
    # ========================================================================
    # EVALUATE EACH OUTPUT SEPARATELY
    # ========================================================================
    print("="*70)
    print("EVALUATION")
    print("="*70)
    
    inputs_test = {'features': X_test}
    mu_pred = model.forward_mu(inputs_test)
    sigma_pred = model.forward_sigma(inputs_test)
    
    print(f"Predictions shape: {mu_pred.shape}")
    print(f"Expected shape: ({len(x_test)}, {output_dim})")
    print()
    
    # Split outputs
    y1_pred = mu_pred[:, 0]
    y2_pred = mu_pred[:, 1]
    
    sigma1 = sigma_pred[:, 0]
    sigma2 = sigma_pred[:, 1]
    
    # Metrics for each output
    tau_mean = model.get_tau_mean().item()
    aleatoric_var = 1.0 / tau_mean
    
    # Output 1
    residuals1 = y1_pred - y1_test_true
    rmse1 = torch.sqrt((residuals1**2).mean()).item()
    ss_res1 = (residuals1**2).sum().item()
    ss_tot1 = ((y1_test_true - y1_test_true.mean())**2).sum().item()
    r2_1 = 1 - (ss_res1 / ss_tot1)
    
    # Output 2
    residuals2 = y2_pred - y2_test_true
    rmse2 = torch.sqrt((residuals2**2).mean()).item()
    ss_res2 = (residuals2**2).sum().item()
    ss_tot2 = ((y2_test_true - y2_test_true.mean())**2).sum().item()
    r2_2 = 1 - (ss_res2 / ss_tot2)
    
    print(f"OUTPUT 1 (y1 = 2x³ - x² + 0.5):")
    print(f"  RMSE: {rmse1:.5f}")
    print(f"  R²: {r2_1:.5f}")
    print(f"  Epistemic std: {torch.sqrt(sigma1).mean():.5f}")
    print()
    
    print(f"OUTPUT 2 (y2 = x² + 0.5x):")
    print(f"  RMSE: {rmse2:.5f}")
    print(f"  R²: {r2_2:.5f}")
    print(f"  Epistemic std: {torch.sqrt(sigma2).mean():.5f}")
    print()
    
    print(f"OVERALL:")
    print(f"  Average R²: {(r2_1 + r2_2) / 2:.5f}")
    print(f"  Average RMSE: {(rmse1 + rmse2) / 2:.5f}")
    print(f"  E[τ]: {tau_mean:.4f} (shared across outputs)")
    print()
    
    # Check performance
    avg_r2 = (r2_1 + r2_2) / 2
    assert avg_r2 > 0.8, f"Average R² too low: {avg_r2:.3f} < 0.8"
    assert rmse1 < 0.5 and rmse2 < 0.5, f"RMSE too high: {rmse1:.3f}, {rmse2:.3f}"
    
    print("✓ Performance checks passed!")
    print()
    
    # ========================================================================
    # PLOT BOTH OUTPUTS
    # ========================================================================
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Output 1
        ax = axes[0]
        ax.scatter(x.numpy(), y1.numpy(), alpha=0.5, s=20, c='gray', label='Training data')
        ax.plot(x_test.numpy(), y1_test_true.numpy(), 'k-', linewidth=2, label='True')
        ax.plot(x_test.numpy(), y1_pred.numpy(), 'r-', linewidth=2, label='Predicted')
        
        total_std1 = torch.sqrt(sigma1 + aleatoric_var)
        ax.fill_between(x_test.numpy(),
                        (y1_pred - 2*total_std1).numpy(),
                        (y1_pred + 2*total_std1).numpy(),
                        alpha=0.3, color='red', label='±2σ')
        ax.set_xlabel('x')
        ax.set_ylabel('y1')
        ax.set_title(f'Output 1: y1 = 2x³ - x² + 0.5 (R²={r2_1:.3f})', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Output 2
        ax = axes[1]
        ax.scatter(x.numpy(), y2.numpy(), alpha=0.5, s=20, c='gray', label='Training data')
        ax.plot(x_test.numpy(), y2_test_true.numpy(), 'k-', linewidth=2, label='True')
        ax.plot(x_test.numpy(), y2_pred.numpy(), 'b-', linewidth=2, label='Predicted')
        
        total_std2 = torch.sqrt(sigma2 + aleatoric_var)
        ax.fill_between(x_test.numpy(),
                        (y2_pred - 2*total_std2).numpy(),
                        (y2_pred + 2*total_std2).numpy(),
                        alpha=0.3, color='blue', label='±2σ')
        ax.set_xlabel('x')
        ax.set_ylabel('y2')
        ax.set_title(f'Output 2: y2 = x² + 0.5x (R²={r2_2:.3f})', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = 'test_multioutput_mps.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {filename}")
    except Exception as e:
        print(f"Plotting failed: {e}")
    
    print()
    print("="*70)
    print("✓ SUCCESS: Multi-output MPS trained successfully!")
    print("="*70)
    
    return history, r2_1, r2_2, rmse1, rmse2


if __name__ == '__main__':
    history, r2_1, r2_2, rmse1, rmse2 = test_multioutput_mps()
    
    print("\nFinal Summary:")
    print(f"  Output 1: R²={r2_1:.4f}, RMSE={rmse1:.4f}")
    print(f"  Output 2: R²={r2_2:.4f}, RMSE={rmse2:.4f}")
    print(f"  Average R²: {(r2_1 + r2_2)/2:.4f}")
