import torch
import numpy as np
import quimb.tensor as qt
from tensor.builder import Inputs
from tensor.btn import BTN
from tensor_old.bayesian_mpo_builder import create_bayesian_tensor_train

# Set precision
torch.set_default_dtype(torch.float64)

def compare_implementations():
    print("\n" + "="*80)
    print(" ⚖️  STRICT EQUIVALENCE TEST: OLD vs NEW IMPLEMENTATION")
    print("="*80)

    # ---------------------------------------------------------
    # 1. SHARED DATA GENERATION
    # ---------------------------------------------------------
    N_SAMPLES = 10
    
    # Raw X in range [-1, 1]
    x_raw = 2 * torch.rand(N_SAMPLES, 1) - 1
    
    # Target (dummy for initialization check)
    y_raw = x_raw ** 2 
    
    # Feature Engineering: [1, x]
    # Note: We align both models to use [1, x] order for consistency
    x_features = torch.cat([torch.ones_like(x_raw), x_raw], dim=1)
    
    print(f"Data Shape: {x_features.shape}")

    # ---------------------------------------------------------
    # 2. INITIALIZE OLD MODEL (Factory Method)
    # ---------------------------------------------------------
    print("\n[1] Initializing OLD Model (Reference)...")
    
    NUM_BLOCKS = 3
    BOND_DIM = 4
    INPUT_DIM = 2 # Feature dim
    
    old_model = create_bayesian_tensor_train(
        num_blocks=NUM_BLOCKS,
        bond_dim=BOND_DIM,
        input_features=INPUT_DIM,
        output_shape=1,
        constrict_bond=False,
        tau_alpha=torch.tensor(2.0),
        tau_beta=torch.tensor(1.0),
        dtype=torch.float64,
        seed=42, 
        random_priors=True
    )

    # ---------------------------------------------------------
    # 3. BRIDGE: EXTRACT WEIGHTS
    # ---------------------------------------------------------
    # We extract the 'mu' (mean) tensors from the old model 
    # to seed the new model exactly the same way.
    
    # mu_nodes is expected to be a list of Tensors/Parameters
    # [print(node.tensor) for node in old_model.mu_nodes]
    old_weights = [node.tensor.detach() for node in old_model.mu_nodes]
    
    print("    Extracted weights from Old Model:")
    for i, w in enumerate(old_weights):
        print(f"    Node {i} shape: {tuple(w.shape)}")

    # ---------------------------------------------------------
    # 4. INITIALIZE NEW MODEL (Quimb/BTN)
    # ---------------------------------------------------------
    print("\n[2] Initializing NEW Model (Target)...")

    # A. Setup Inputs
    # We define input labels mapping to the 3 nodes
    input_labels = ["x0", "x1", "x2"]
    
    loader = Inputs(
        inputs=[x_features], 
        outputs=[y_raw],
        outputs_labels=["y"],
        input_labels=input_labels,
        batch_dim="s",
        batch_size=N_SAMPLES
    )

    # B. Construct Graph using Extracted Weights
    # We must match indices: (LeftBond, Feature, RightBond)
    # Old model usually structures:
    # Node 0: (Feature, Bond_R) or (1, Feature, Bond_R) -> adjust if needed
    # Middle: (Bond_L, Feature, Bond_R)
    # Last:   (Bond_L, Feature, Output)
    
    tensors = []
    
    # Node 0
    # Assuming Old Node 0 is (Phys, Bond). 
    # We map Phys->'x0', Bond->'b0'
    t0 = qt.Tensor(
        data=old_weights[0].squeeze().clone(), 
        inds=('x0', 'b0'), 
        tags={'Node0'}
    )
    tensors.append(t0)
    
    # Node 1 (Middle)
    # Assuming Old Node 1 is (Bond_L, Phys, Bond_R)
    t1 = qt.Tensor(
        data=old_weights[1].squeeze().clone(), 
        inds=('b0', 'x1', 'b1'), 
        tags={'Node1'}
    )
    tensors.append(t1)
    
    # Node 2 (Last)
    # Assuming Old Node 2 is (Bond_L, Phys, Output)
    t2 = qt.Tensor(
        data=old_weights[2].squeeze().unsqueeze(-1).clone(), 
        inds=('b1', 'x2', 'y'), 
        tags={'Node2'}
    )
    tensors.append(t2)

    mu_tn = qt.TensorNetwork(tensors)

    # C. Create BTN
    new_model = BTN(
        mu=mu_tn, 
        data_stream=loader, 
        batch_dim="s",
        method='cholesky'
    )

    # ---------------------------------------------------------
    # 5. EXECUTION & COMPARISON
    # ---------------------------------------------------------
    print("\n[3] Running Forward Passes...")

    # --- Run Old ---
    # forward_mu usually expects tensor input
    pred_old = old_model.forward_mu(x_features, to_tensor=True)
    if isinstance(pred_old, torch.Tensor):
        pred_old = pred_old.detach()
    
    # --- Run New ---
    # forward returns the contraction result
    pred_new_tn = new_model.forward(
        new_model.mu, 
        new_model.data.data_mu, 
        sum_over_batch=False, 
        sum_over_output=False
    )

    pred_new = pred_new_tn.data # Extract torch tensor from QT tensor

    # --- Compare Values ---
    print("\n" + "-"*30)
    print("RESULTS")
    print("-"*30)

    # 1. Output Tensor Equality
    print(f"Old Output Shape: {pred_old.shape}")
    print(f"New Output Shape: {pred_new.shape}")
    
    # Squeeze to ensure shapes match (N,1) vs (N,)
    diff = torch.abs(pred_old.reshape(-1) - pred_new.reshape(-1))
    max_diff = diff.max().item()
    
    print(f"Max Difference in Predictions: {max_diff:.10e}")
    
    is_output_match = max_diff < 1e-14
    print(f"✅ Outputs Match: {is_output_match}")

    # 2. MSE Calculation Check
    # Old MSE calculation logic: ((pred - y)**2).mean()
    mse_old = ((pred_old.reshape(-1) - y_raw.reshape(-1))**2).mean().item()
    # New MSE calculation logic: Internal method
    mse_new = new_model._calc_mu_mse()/new_model.data.samples
    
    print(f"\nOld MSE: {mse_old:.10f}")
    print(f"New MSE: {mse_new.item():.10f}")
    
    is_mse_match = abs(mse_old - mse_new) < 1e-9
    print(f"✅ MSE Match:     {is_mse_match}")

    if is_output_match and is_mse_match:
        print("\n🎉 SUCCESS: The implementations are mathematically identical.")
    else:
        print("\n⚠️ FAILURE: Discrepancies detected. Check tensor index ordering.")

if __name__ == "__main__":
    compare_implementations()
