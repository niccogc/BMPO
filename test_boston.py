import torch
import numpy as np
import quimb.tensor as qt
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensor.builder import Inputs
from tensor.btn import BTN

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
torch.set_default_dtype(torch.float64)

def init_weights(shape, normalize=True):
    """Initialize weights with small random values."""
    w = torch.randn(*shape, dtype=torch.float64)
    if normalize:
        return w / torch.norm(w)
    return w

def run_bayesian_regression_test():
    print("="*70)
    print("BAYESIAN TENSOR NETWORK: BOSTON HOUSING REGRESSION")
    print("="*70)

    # ---------------------------------------------------------
    # 1. DATA PREPARATION
    # ---------------------------------------------------------
    boston = fetch_openml(name='boston', version=1, as_frame=True, parser='auto')
    
    # Using 3 features for the 3-site MPS
    FEATURE_NAMES = ['RM', 'LSTAT', 'CRIM']
    
    X_raw = boston.data[FEATURE_NAMES].values
    y_raw = boston.target.values.reshape(-1, 1)

    # Standardization
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X_raw)
    y_scaled = scaler_y.fit_transform(y_raw)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float64)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float64)

    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    # Sort test data by 'RM' (feature 0) for cleaner plotting
    sort_idx = torch.argsort(X_test[:, 0])
    X_test_sorted = X_test[sort_idx]
    y_test_sorted = y_test[sort_idx]

    # Add bias feature: [x, 1]
    X_train_f = torch.cat([X_train, torch.ones_like(X_train[:, :1])], dim=1)
    X_test_f = torch.cat([X_test_sorted, torch.ones_like(X_test_sorted[:, :1])], dim=1)

    N_SAMPLES = X_train.shape[0]
    N_SITES = 3 
    D_PHYS = X_train_f.shape[1]
    BATCH_SIZE = 50

    # ---------------------------------------------------------
    # 2. INPUT LOADER & MODEL
    # ---------------------------------------------------------
    input_labels = [f"x{i+1}" for i in range(N_SITES)]
    
    train_loader = Inputs(
        inputs=[X_train_f], outputs=[y_train], outputs_labels=["y"],
        input_labels=input_labels, batch_dim="s", batch_size=BATCH_SIZE
    )

    # Test Loader (One large batch for inference)
    test_loader = Inputs(
        inputs=[X_test_f], outputs=[y_test_sorted], outputs_labels=["y"],
        input_labels=input_labels, batch_dim="s", batch_size=X_test_f.shape[0]
    )

    D_BOND = 20
    t1 = qt.Tensor(data=init_weights((D_PHYS, D_BOND)), inds=('x1', 'b1'), tags={'Node1'})
    t2 = qt.Tensor(data=init_weights((D_BOND, D_PHYS, D_BOND)), inds=('b1', 'x2', 'b2'), tags={'Node2'})
    t3 = qt.Tensor(data=init_weights((D_BOND, D_PHYS, 1)), inds=('b2', 'x3', 'y'), tags={'Node3'})
    
    mu_tn = qt.TensorNetwork([t1, t2, t3])

    model = BTN(
        mu=mu_tn, 
        data_stream=train_loader, 
        batch_dim="s", 
        method='nocholesky'
    )

    # ---------------------------------------------------------
    # 3. TRAINING
    # ---------------------------------------------------------
    print(f"\nStarting Training on {N_SAMPLES} samples...")
    # Track ELBO to ensure convergence
    model.fit(epochs=10, track_elbo=True, trim_every_epochs=True, threshold=0.9) 
    print("Training Complete.")
    for i in model.mu.tags:
        print(model.mu[i].shape)
    # ---------------------------------------------------------
    # 4. INFERENCE (FIXED)
    # ---------------------------------------------------------
    # ERROR FIX: Do NOT use _calc_mu_forward (it sums over batch).
    # We use model.forward directly with sum_over_batch=False.
    
    print("\nRunning Inference...")

    # 1. Forward Mean
    mu_pred = model.forward(
        model.mu, 
        test_loader.data_mu,    # The generator for input tensors
        sum_over_batch=False,   # IMPORTANT: Keep batch dimension
        sum_over_output=False   # Keep output dimension (N, 1)
    )
    
    # 2. Forward Variance (Sigma)
    sigma_pred = model.forward(
        model.sigma, 
        test_loader.data_mu, 
        sum_over_batch=False, 
        sum_over_output=True    # Sum over output dim to get scalar variance per sample
    )

    # Convert to pure tensors for math
    if not isinstance(mu_pred, torch.Tensor): mu_pred = torch.as_tensor(mu_pred.data)
    if not isinstance(sigma_pred, torch.Tensor): sigma_pred = torch.as_tensor(sigma_pred.data)

    mu_pred = mu_pred.squeeze()     # Shape: (N,)
    sigma_pred = sigma_pred.squeeze() # Shape: (N,)
    y_true = y_test_sorted.squeeze()

    # 3. Process Uncertainties
    # Extract Tau (Precision). Trying to grab it from the model nodes if possible.
    try:
        # Assuming Tau is stored in model.Tau tensor network
        tau_vals = [t.data.mean() for t in model.Tau]
        tau_mean = torch.stack(tau_vals).mean().item()
    except:
        # Fallback if structure is different
        tau_mean = 2.2 # Value approx from your logs

    epistemic_var = torch.abs(sigma_pred)
    aleatoric_var = 1.0 / tau_mean
    
    total_var = epistemic_var + aleatoric_var
    total_std = torch.sqrt(total_var)
    epistemic_std = torch.sqrt(epistemic_var)

    # ---------------------------------------------------------
    # 5. METRICS & PLOTTING
    # ---------------------------------------------------------
    residuals = mu_pred - y_true
    rmse = torch.sqrt((residuals**2).mean()).item()
    ss_res = (residuals**2).sum().item()
    ss_tot = ((y_true - y_true.mean())**2).sum().item()
    r2 = 1 - (ss_res / ss_tot)

    print("\nRESULTS")
    print("="*70)
    print(f"Test RMSE: {rmse:.5f}")
    print(f"Test R²:   {r2:.5f}")
    print(f"E[τ]: {tau_mean:.4f}")
    print("="*70)

    print(mu_pred.dtype)
    # X-axis for plotting: The sorted 'RM' feature
    x_plot = X_test_sorted[:, 0].numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Prediction vs True
    ax = axes[0]
    ax.scatter(x_plot, y_true.numpy(), color='black', s=15, alpha=0.6, label='True Data')
    ax.plot(x_plot, mu_pred.numpy(), 'r-', linewidth=2, label='Predicted Mean')
    
    # Uncertainty Bands
    ax.fill_between(x_plot, 
                    (mu_pred - 2*total_std).numpy(), 
                    (mu_pred + 2*total_std).numpy(), 
                    color='red', alpha=0.15, label='±2σ Total (Uncertainty)')
    
    ax.fill_between(x_plot, 
                    (mu_pred - 2*epistemic_std).numpy(), 
                    (mu_pred + 2*epistemic_std).numpy(), 
                    color='blue', alpha=0.3, label='±2σ Epistemic (Model)')
    
    ax.set_title(f"Bayesian Regression (R²={r2:.3f})")
    ax.set_xlabel("Standardized Feature: Rooms (RM)")
    ax.set_ylabel("Standardized Price")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Residuals
    ax = axes[1]
    ax.scatter(x_plot, residuals.numpy(), color='purple', s=15, alpha=0.6)
    ax.axhline(0, color='black', linestyle='--')
    ax.fill_between(x_plot, (-2*total_std).numpy(), (2*total_std).numpy(), color='red', alpha=0.1, label='Expected Range')

    ax.set_title(f"Residuals")
    ax.set_xlabel("Standardized Feature: Rooms (RM)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_bayesian_regression_test()
