import torch
import numpy as np
import quimb.tensor as qt
from tensor.builder import Inputs
from tensor.btn import BTN

# Set double precision for stability in polynomial fitting
torch.set_default_dtype(torch.float64)

def test_polynomial_learning():
    print("\n" + "="*60)
    print("TESTING 3rd DEGREE POLYNOMIAL REGRESSION WITH MPS")
    print("="*60)

    # ---------------------------------------------------------
    # 1. DATA GENERATION
    # ---------------------------------------------------------
    N_SAMPLES = 500
    BATCH_SIZE = 50
    
    # Generate X in range [-1, 1]
    # We use a slightly wider range to test generalization within the interval
    x_raw = 2 * torch.rand(N_SAMPLES, 1) - 1
    
    # Define Target Function: y = 2x^3 - x^2 + 0.5x + 0.2
    # This requires a polynomial degree of 3.
    # An MPS with 3 sites (each receiving inputs linear in x) can represent x^3.
    y_raw = 2 * (x_raw**3) - (x_raw**2) + 0.5 * x_raw + 0.2
    
    # Add small noise
    y_raw += 0.01 * torch.randn_like(y_raw)

    # ---------------------------------------------------------
    # 2. FEATURE ENGINEERING (Add constant)
    # ---------------------------------------------------------
    # We need to map scalar x -> feature vector [x, 1].
    # This allows the MPS to select 'x' or '1' at each site, forming terms like x*x*x, x*1*x, 1*1*1, etc.
    x_features = torch.cat([x_raw, torch.ones_like(x_raw)], dim=1)
    
    print(f"Input shape (with bias): {x_features.shape}") # (N, 2)
    print(f"Target shape: {y_raw.shape}")

    # ---------------------------------------------------------
    # 3. INPUT LOADER SETUP
    # ---------------------------------------------------------
    # We pass ONE input tensor but THREE input labels.
    # The Inputs class handles repetition automatically.
    # This simulates feeding the same x vector into 3 different MPS sites.
    input_labels = ["x1", "x2", "x3"]
    
    loader = Inputs(
        inputs=[x_features], 
        outputs=[y_raw],
        outputs_labels=["y"],
        input_labels=input_labels,
        batch_dim="s",
        batch_size=BATCH_SIZE
    )

    # ---------------------------------------------------------
    # 4. MPS CONSTRUCTION
    # ---------------------------------------------------------
    # 3 Nodes. Physical dimension = 2 (feature dim).
    # Bond dimension = 3 (sufficient for cubic poly).
    D_bond = 4 
    D_phys = 2
    
    # Initialize with small random weights normalized to prevent explosion
    def init_weights(shape):
        w = torch.randn(*shape)
        return w/torch.norm(w)

    # Node 1: Input 'x1', Bond 'b1'
    t1 = qt.Tensor(
        data=init_weights((D_phys, D_bond)), 
        inds=('x1', 'b1'), 
        tags={'Node1'}
    )
    
    # Node 2: Bond 'b1', Input 'x2', Bond 'b2'
    t2 = qt.Tensor(
        data=init_weights((D_bond, D_phys, D_bond)), 
        inds=('b1', 'x2', 'b2'), 
        tags={'Node2'}
    )
    
    # Node 3: Bond 'b2', Input 'x3', Output 'y'
    # Output dimension is 1 (scalar regression)
    t3 = qt.Tensor(
        data=init_weights((D_bond, D_phys, 1)), 
        inds=('b2', 'x3', 'y'), 
        tags={'Node3'}
    )
    
    mu_tn = qt.TensorNetwork([t1, t2, t3])
    
    # Initialize BTN
    # Mark inputs as not trainable (only weights are trained)
    # Actually, BTN treats nodes in 'mu' as variational parameters (Mean) by default.
    # The input tensors come from 'loader' and are contracted with these nodes.
    model = BTN(
        mu=mu_tn, 
        data_stream=loader, 
        batch_dim="s",
        method = 'cholesky',
    )

    print(f"Model Initialized. Training on {N_SAMPLES} samples...")
    print(f"Initial MSE: {model._calc_mu_mse().item():.6f}")

    # ---------------------------------------------------------
    # 5. TRAINING
    # ---------------------------------------------------------
    EPOCHS = 10
    
    # The fit function handles the Variational Inference updates
    # (Update Tau -> Update Sigma -> Update Mu -> Update Bonds)
    model.fit(epochs=EPOCHS)
    
    # ---------------------------------------------------------
    # 6. EVALUATION
    # ---------------------------------------------------------
    final_mse = model._calc_mu_mse().item()/model.data.samples
    print("\n" + "-"*30)
    print(f"Final MSE: {final_mse:.6f}")
    
    # Compare predictions on a small batch
    print("-"*30)
    print("Sample Predictions vs Truth:")
    
    # Get predictions
    # forward returns shape (N, 1) usually
    preds = model.forward(model.mu, model.data.data_mu, sum_over_batch=False, sum_over_output=False)
    
    # Convert to torch for slicing
    preds_data = torch.as_tensor(preds.data)
    
    for i in range(5):
        x_val = x_raw[i, 0].item()
        y_true = y_raw[i, 0].item()
        y_pred = preds_data[i, 0].item()
        print(f"x: {x_val:+.3f} | True: {y_true:+.3f} | Pred: {y_pred:+.3f}")

    # Simple assertion for success (MSE should drop significantly)
    if final_mse < 0.1:
        print("\n✅ SUCCESS: Model learned the polynomial structure.")
    else:
        print("\n⚠️ WARNING: Model might need more epochs or tuning.")

if __name__ == "__main__":
    test_polynomial_learning()
