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
    # 1. DATA PREPARATION (Real Dataset)
    # ---------------------------------------------------------
    # Fetch Boston Housing
    boston = fetch_openml(name='boston', version=1, as_frame=True, parser='auto')
    
    # We use 3 features to fit the 3-site MPS structure
    # 'RM' (Rooms) is the dominant feature, useful for sorting plots
    FEATURE_NAMES = ['RM', 'LSTAT', 'CRIM']
    
    X_raw = boston.data[FEATURE_NAMES].values
    y_raw = boston.target.values.reshape(-1, 1)

    # Standardization (Critical for Tensor Networks)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X_raw)
    y_scaled = scaler_y.fit_transform(y_raw)

    # Convert to Torch
    X_tensor = torch.tensor(X_scaled, dtype=torch.float64)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float64)

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    # Sort test data by the first feature (RM) to make the line plots readable
    sort_idx = torch.argsort(X_test[:, 0])
    X_test_sorted = X_test[sort_idx]
    y_test_sorted = y_test[sort_idx]

    # Feature Engineering: Append '1' for bias -> [x, 1]
    # Physical dim becomes N_FEATURES + 1
    X_train_f = torch.cat([X_train, torch.ones_like(X_train[:, :1])], dim=1)
    X_test_f = torch.cat([X_test_sorted, torch.ones_like(X_test_sorted[:, :1])], dim=1)

    N_SAMPLES = X_train.shape[0]
    N_SITES = 3 # 3 Features
    D_PHYS = X_train_f.shape[1]
    BATCH_SIZE = 50

    print(f"Features: {FEATURE_NAMES}")
    print(f"Train Size: {N_SAMPLES} | Test Size: {X_test.shape[0]}")

    # ---------------------------------------------------------
    # 2. INPUT LOADER & MODEL SETUP
    # ---------------------------------------------------------
    input_labels = [f"x{i+1}" for i in range(N_SITES)]
    
    # Training Loader
    train_loader = Inputs(
        inputs=[X_train_f],
        outputs=[y_train],
        outputs_labels=["y"],
        input_labels=input_labels,
        batch_dim="s",
        batch_size=BATCH_SIZE
    )

    # Test Loader (treated as one big batch for evaluation)
    test_loader = Inputs(
        inputs=[X_test_f],
        outputs=[y_test_sorted], # Dummy outputs just to match structure
        outputs_labels=["y"],
        input_labels=input_labels,
        batch_dim="s",
        batch_size=X_test_f.shape[0]
    )

    # MPS Construction (3 Sites)
    D_BOND = 5
    
    # Simple MPS structure: Node1 - Node2 - Node3
    t1 = qt.Tensor(data=init_weights((D_PHYS, D_BOND)), inds=('x1', 'b1'), tags={'Node1'})
    t2 = qt.Tensor(data=init_weights((D_BOND, D_PHYS, D_BOND)), inds=('b1', 'x2', 'b2'), tags={'Node2'})
    t3 = qt.Tensor(data=init_weights((D_BOND, D_PHYS, 1)), inds=('b2', 'x3', 'y'), tags={'Node3'})
    
    mu_tn = qt.TensorNetwork([t1, t2, t3])

    # Initialize BTN
    model = BTN(
        mu=mu_tn, 
        data_stream=train_loader, 
        batch_dim="s", 
        method='cholesky'
    )

    # ---------------------------------------------------------
    # 3. TRAINING
    # ---------------------------------------------------------
    print("\nStarting Training...")
    EPOCHS = 20
    model.fit(epochs=EPOCHS, track_elbo=False)
    print("Training Complete.")

    # ---------------------------------------------------------
    # 4. INFERENCE (Using _calc_mu_forward / _calc_sigma_forward)
    # ---------------------------------------------------------
    # Note: We pass the test_loader as the 'inputs' argument to your methods
    
    # 1. Forward Mean
    mu_pred = model._calc_mu_forward(inputs=test_loader)
    
    # 2. Forward Variance (Sigma)
    sigma_pred = model._calc_sigma_forward(inputs=test_loader)

    # Ensure tensors
    if not isinstance(mu_pred, torch.Tensor): mu_pred = torch.as_tensor(mu_pred.data)
    if not isinstance(sigma_pred, torch.Tensor): sigma_pred = torch.as_tensor(sigma_pred.data)

    mu_pred = mu_pred.squeeze()
    sigma_pred = sigma_pred.squeeze()
    y_true = y_test_sorted.squeeze()

    # 3. Process Uncertainties
    # Epistemic Variance: Comes from model.sigma
    epistemic_var = torch.abs(sigma_pred) # Ensure positive
    epistemic_std = torch.sqrt(epistemic_var)

    # Aleatoric Variance: Comes from 1 / E[tau]
    # We assume 'model.Tau' exists after training. 
    # If Tau is a TN, we take the mean of its data.
    try:
        # Try getting mean from TN nodes if Tau is a TN
        tau_vals = [t.data.mean() for t in model.Tau]
        tau_mean = torch.stack(tau_vals).mean().item()
    except:
        # Fallback/Placeholder if Tau structure differs
        tau_mean = 10.0 

    aleatoric_var = 1.0 / tau_mean
    
    # Total Variance
    total_var = epistemic_var + aleatoric_var
    total_std = torch.sqrt(total_var)

    # ---------------------------------------------------------
    # 5. METRICS
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
    print(f"E[τ]: {tau_mean:.4f} (Implied Noise Std: {1/np.sqrt(tau_mean):.4f})")
    print(f"Mean Epistemic Std: {epistemic_std.mean():.5f}")
    print(f"Mean Total Std:     {total_std.mean():.5f}")
    print("="*70)

    # ---------------------------------------------------------
    # 6. PLOTTING
    # ---------------------------------------------------------
    # X-axis for plotting: The sorted 'RM' feature (index 0 of our test set)
    x_plot = X_test_sorted[:, 0].numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot 1: Prediction vs True with Uncertainty ---
    ax = axes[0]
    
    # Plot True Data
    ax.scatter(x_plot, y_true.numpy(), color='black', s=10, alpha=0.5, label='True Data')
    
    # Plot Predicted Mean
    ax.plot(x_plot, mu_pred.numpy(), 'r-', linewidth=1.5, label='Predicted Mean')
    
    # Plot Total Uncertainty (Aleatoric + Epistemic)
    ax.fill_between(x_plot, 
                    (mu_pred - 2*total_std).numpy(), 
                    (mu_pred + 2*total_std).numpy(), 
                    color='red', alpha=0.2, label='±2σ Total')
    
    # Plot Epistemic Uncertainty (Model Confidence)
    ax.fill_between(x_plot, 
                    (mu_pred - 2*epistemic_std).numpy(), 
                    (mu_pred + 2*epistemic_std).numpy(), 
                    color='blue', alpha=0.4, label='±2σ Epistemic')
    
    ax.set_title(f"Bayesian Regression (R²={r2:.3f})")
    ax.set_xlabel("Feature: Rooms (Standardized)")
    ax.set_ylabel("Target: Price (Standardized)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Residuals ---
    ax = axes[1]
    ax.scatter(x_plot, residuals.numpy(), color='purple', s=10, alpha=0.6)
    ax.axhline(0, color='black', linestyle='--')
    
    # Show the "expected" envelope of residuals based on total uncertainty
    ax.fill_between(x_plot, 
                    (-2*total_std).numpy(), 
                    (2*total_std).numpy(), 
                    color='red', alpha=0.1, label='Expected ±2σ Range')

    ax.set_title(f"Residuals (RMSE={rmse:.4f})")
    ax.set_xlabel("Feature: Rooms (Standardized)")
    ax.set_ylabel("Residual (Pred - True)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_bayesian_regression_test()
