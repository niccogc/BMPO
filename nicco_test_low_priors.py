import torch
import numpy as np
import quimb.tensor as qt
from tensor.builder import Inputs
from tensor.btn import BTN

# Set precision
torch.set_default_dtype(torch.float64)

def test_weak_priors_learning():
    print("\n" + "="*60)
    print("TESTING POLYNOMIAL REGRESSION WITH WEAK PRIORS")
    print("="*60)

    # 1. Data Generation
    N_SAMPLES = 500
    BATCH_SIZE = 50
    
    x_raw = 2 * torch.rand(N_SAMPLES, 1) - 1
    # y = 2x^3 - x^2 + 0.5x + 0.2
    y_raw = 2 * (x_raw**3) - (x_raw**2) + 0.5 * x_raw + 0.2
    y_raw += 0.01 * torch.randn_like(y_raw)

    x_features = torch.cat([x_raw, torch.ones_like(x_raw)], dim=1)
    
    loader = Inputs(
        inputs=[x_features], 
        outputs=[y_raw],
        outputs_labels=["y"],
        input_labels=["x1", "x2", "x3"],
        batch_dim="s",
        batch_size=BATCH_SIZE
    )

    # 2. Model Init
    D_bond = 4 
    D_phys = 2
    
    # Init weights (small random)
    def init_weights(shape):
        return torch.randn(*shape) * 0.1

    t1 = qt.Tensor(init_weights((D_phys, D_bond)), inds=('x1', 'b1'), tags={'Node1'})
    t2 = qt.Tensor(init_weights((D_bond, D_phys, D_bond)), inds=('b1', 'x2', 'b2'), tags={'Node2'})
    t3 = qt.Tensor(init_weights((D_bond, D_phys, 1)), inds=('b2', 'x3', 'y'), tags={'Node3'})
    
    mu_tn = qt.TensorNetwork([t1, t2, t3])
    model = BTN(mu=mu_tn, data_stream=loader, batch_dim="s")

    print(f"Initial MSE: {model._calc_mu_mse().item():.6f}")

    # ---------------------------------------------------------
    # 3. WEAKEN PRIORS (Crucial Step)
    # ---------------------------------------------------------
    print("\n>>> Weakening Priors (q_bonds)...")
    
    # We want Theta (Precision) to be small. Theta ~ E[lambda] = alpha / beta.
    # Let's set alpha=1e-4, beta=1.0 -> E[lambda] = 1e-4.
    # This effectively removes the L2 regularization.
    
    WEAK_ALPHA = 1e-4
    WEAK_BETA = 1.0
    
    for bond_name, dist in model.q_bonds.items():
        # dist is a GammaDistribution
        # Modify its parameters directly
        new_alpha = torch.full_like(dist.concentration.data, WEAK_ALPHA)
        new_beta = torch.full_like(dist.rate.data, WEAK_BETA)
        
        dist.concentration.modify(data=new_alpha)
        dist.rate.modify(data=new_beta)
        
    # Also reset p_bonds (Priors) to match, so KL doesn't blow up (optional but good practice)
    for bond_name, dist in model.p_bonds.items():
        new_alpha = torch.full_like(dist.concentration.data, WEAK_ALPHA)
        new_beta = torch.full_like(dist.rate.data, WEAK_BETA)
        dist.concentration.modify(data=new_alpha)
        dist.rate.modify(data=new_beta)

    print(f"Priors set to Alpha={WEAK_ALPHA}, Beta={WEAK_BETA} (Mean={WEAK_ALPHA/WEAK_BETA})")

    # ---------------------------------------------------------
    # 4. TRAINING
    # ---------------------------------------------------------
    EPOCHS = 15
    print(f"\nTraining for {EPOCHS} epochs...")
    
    # Track MSE
    for i in range(EPOCHS):
        model.fit(epochs=1)
        mse = model.mse.item() # mse is updated inside fit
        print(f"Epoch {i+1}: MSE = {mse:.6f}")
        
    # ---------------------------------------------------------
    # 5. EVALUATION
    # ---------------------------------------------------------
    final_mse = model._calc_mu_mse().item()
    print("\n" + "-"*30)
    print(f"Final MSE: {final_mse:.6f}")
    
    preds = model.forward(model.mu, model.data.data_mu, sum_over_batch=False, sum_over_output=False)
    preds_data = torch.as_tensor(preds.data)
    
    print("Sample Predictions:")
    for i in range(5):
        x_val = x_raw[i, 0].item()
        y_true = y_raw[i, 0].item()
        y_pred = preds_data[i, 0].item()
        print(f"x: {x_val:+.3f} | True: {y_true:+.3f} | Pred: {y_pred:+.3f}")

    if final_mse < 0.5:
        print("\n✅ SUCCESS: Model learned with weak priors.")
    else:
        print("\n❌ FAILURE: Model still not learning effectively.")

if __name__ == "__main__":
    test_weak_priors_learning()
