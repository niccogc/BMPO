"""
Test variational updates for BayesianTensorNetwork.

Creates a simple network and runs a few iterations of variational inference.
"""

import torch
import quimb.tensor as qtn
import numpy as np

from tensor.bayesian_tn import BayesianTensorNetwork


def create_simple_mps():
    """
    Create a simple 2-block MPS: Input -> A -> B -> Output
    
    This mimics the structure in visualize_predictions_standardized.py
    """
    # Dimensions
    d_in = 2      # Input features (1, x) for polynomial
    r1 = 3        # Bond between A and B
    d_out = 1     # Output dimension
    
    # Initialize with small random values
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Input node: (batch, features)
    # We'll set this during training
    input_data = np.ones((1, d_in))  # Dummy, will be replaced
    input_tensor = qtn.Tensor(data=input_data, inds=('batch', 'x'), tags='input')  # type: ignore
    
    # Block A: (x, r1, batch_out) - connects input and outputs to B
    A_data = np.random.randn(d_in, r1, d_out) * 0.1
    A_tensor = qtn.Tensor(data=A_data, inds=('x', 'r1', 'batch_out'), tags='A')  # type: ignore
    
    # Block B: (r1, batch_out) - final contraction
    B_data = np.random.randn(r1, d_out) * 0.1
    B_tensor = qtn.Tensor(data=B_data, inds=('r1', 'batch_out2'), tags='B')  # type: ignore
    
    tn = qtn.TensorNetwork([input_tensor, A_tensor, B_tensor])
    
    return tn


def test_simple_regression():
    """Test on simple 1D polynomial regression."""
    print("\n" + "="*70)
    print("Test: Simple 1D Polynomial Regression")
    print("="*70)
    
    # Generate data: y = 2x + 1 + noise
    torch.manual_seed(42)
    num_samples = 20
    x = torch.linspace(-1, 1, num_samples, dtype=torch.float64)
    y = 2.0 * x + 1.0
    y += 0.1 * torch.randn(num_samples, dtype=torch.float64)
    
    # Create input features: [1, x]
    X = torch.stack([torch.ones_like(x), x], dim=1)
    
    print(f"Data: {num_samples} samples")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  y range: [{y.min():.3f}, {y.max():.3f}]")
    
    # Create network
    print("\nCreating network...")
    tn = create_simple_mps()
    
    btn = BayesianTensorNetwork(
        mu_tn=tn.copy(),
        learnable_tags=['A', 'B'],
        input_tags=['input'],
        tau_alpha=torch.tensor(2.0, dtype=torch.float64),
        tau_beta=torch.tensor(1.0, dtype=torch.float64),
        device=torch.device('cpu'),
        dtype=torch.float64
    )
    
    print(f"  Learnable nodes: {btn.learnable_tags}")
    print(f"  Input nodes: {btn.input_tags}")
    print(f"  Bonds: {btn.mu_network.bond_labels}")
    print(f"  Initial E[τ]: {btn.get_tau_mean().item():.4f}")
    
    # Prepare inputs dict
    inputs_dict = {
        'input': X
    }
    
    # Test initial forward pass
    print("\nTesting initial forward pass...")
    try:
        y_pred_init = []
        for i in range(num_samples):
            sample_input = {'input': X[i:i+1]}
            y_i = btn.forward_mu(sample_input)
            y_pred_init.append(y_i.item() if y_i.numel() == 1 else y_i.squeeze().item())
        
        y_pred_init = torch.tensor(y_pred_init, dtype=torch.float64)
        mse_init = ((y_pred_init - y)**2).mean().item()
        print(f"  Initial MSE: {mse_init:.6f}")
    except Exception as e:
        print(f"  Initial forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run training
    print("\nStarting training...")
    max_iter = 5  # Just a few iterations for testing
    
    try:
        for iteration in range(max_iter):
            print(f"\nIteration {iteration + 1}/{max_iter}")
            
            # Update node A
            print("  Updating node A...")
            btn.update_node_variational('A', X, y, inputs_dict)
            
            # Update node B
            print("  Updating node B...")
            btn.update_node_variational('B', X, y, inputs_dict)
            
            # Update bonds
            print("  Updating bonds...")
            for bond in btn.mu_network.bond_labels:
                if bond not in ['batch', 'batch_out', 'batch_out2', 'x']:  # Skip non-learnable bonds
                    btn.update_bond_variational(bond)
            
            # Update tau
            print("  Updating tau...")
            btn.update_tau_variational(X, y, inputs_dict)
            
            # Evaluate
            y_pred = []
            for i in range(num_samples):
                sample_input = {'input': X[i:i+1]}
                y_i = btn.forward_mu(sample_input)
                y_pred.append(y_i.item() if y_i.numel() == 1 else y_i.squeeze().item())
            
            y_pred = torch.tensor(y_pred, dtype=torch.float64)
            mse = ((y_pred - y)**2).mean().item()
            E_tau = btn.get_tau_mean().item()
            
            print(f"  MSE: {mse:.6f}, E[τ]: {E_tau:.4f}")
        
        print("\n" + "="*70)
        print("Training completed successfully!")
        print("="*70)
        
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_simple_regression()
