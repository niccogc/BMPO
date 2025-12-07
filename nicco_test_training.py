import torch
import numpy as np
import quimb.tensor as qt
from tensor.builder import Inputs
from tensor.btn import BTN

# Set precision
torch.set_default_dtype(torch.float64)

def debug_training_dynamics():
    print("\n" + "="*60)
    print("DEBUGGING TRAINING DYNAMICS (PROFOUND)")
    print("="*60)

    # 1. Setup minimal problem (Polynomial Regression)
    B = 50 
    x_raw = 2 * torch.rand(B, 1) - 1
    y_raw = 2 * (x_raw**3) - (x_raw**2) + 0.5 * x_raw + 0.2
    x_features = torch.cat([x_raw, torch.ones_like(x_raw)], dim=1) 
    
    loader = Inputs(
        inputs=[x_features], 
        outputs=[y_raw],
        outputs_labels=["y"],
        input_labels=["x1", "x2", "x3"], 
        batch_dim="s",
        batch_size=B
    )

    # 2. Init Model
    D_bond = 2
    D_phys = 2
    
    # Init weights slightly larger to see if they shrink
    t1 = qt.Tensor(torch.randn(D_phys, D_bond)*0.5, inds=('x1', 'b1'), tags={'Node1'})
    t2 = qt.Tensor(torch.randn(D_bond, D_phys, D_bond)*0.5, inds=('b1', 'x2', 'b2'), tags={'Node2'})
    t3 = qt.Tensor(torch.randn(D_bond, D_phys, 1)*0.5, inds=('b2', 'x3', 'y'), tags={'Node3'})
    
    mu_tn = qt.TensorNetwork([t1, t2, t3])
    model = BTN(mu=mu_tn, data_stream=loader, batch_dim="s")
    
    initial_mse = model._calc_mu_mse().item()
    print(f"Initial MSE: {initial_mse:.6f}")

    # 3. Deep Dive into Node2 Update
    node_tag = 'Node2'
    print(f"\n--- Debugging Update for {node_tag} ---")
    
    # Calculate components
    tau_mean = model.get_tau_mean().item()
    print(f"E[tau]: {tau_mean:.4f}")
    
    # A. PRIOR TERM (Theta)
    theta = model.theta_block_computation(node_tag)
    # Theta represents the diagonal precision from the bond priors.
    # Summing it gives us the total "force" pulling weights to 0 from the prior.
    tr_prior = theta.data.abs().sum().item()
    
    # B. DATA TERM (Mu Outer + Sigma Env)
    # This represents the curvature/precision coming from the data.
    # Precision_Data = E[tau] * (Sigma_Env + Mu_Outer)
    sigma_env = model.get_environment(
        tn=model.sigma, target_tag=node_tag, 
        input_generator=model.data.data_sigma, 
        sum_over_batch=True, sum_over_output=True
    )
    mu_outer = model.outer_operation(
        tn=model.mu, node_tag=node_tag,
        input_generator=model.data.data_mu,
        sum_over_batches=True
    )
    
    # We need the trace of the Data Precision term.
    # Since these tensors are "diagonal-ish" in the outer-product space (b, b'),
    # we can estimate magnitude by looking at their norms or traces.
    
    # Flatten to inspect properly
    precision_tensor = model.compute_precision_node(node_tag)
    inds = precision_tensor.inds
    unprimed = sorted([i for i in inds if not i.endswith('_prime')])
    primed = [f"{i}_prime" for i in unprimed]
    d = int(np.prod([precision_tensor.ind_size(i) for i in unprimed]))
    
    # Helper to get trace of a tensor (b, b')
    def get_trace_from_tensor(t):
        mat = t.transpose(*unprimed, *primed).data.reshape(d, d)
        return torch.trace(mat).item()

    tr_mu_outer = get_trace_from_tensor(mu_outer)
    tr_sigma_env = get_trace_from_tensor(sigma_env)
    
    # Total Data Trace
    tr_data = tau_mean * (tr_mu_outer + tr_sigma_env)
    
    print("\n--- Strength Analysis ---")
    print(f"Prior Strength (Trace Theta): {tr_prior:.4e}")
    print(f"Data Strength (Trace Data):   {tr_data:.4e}")
    print(f"  > From Mu Outer:            {tau_mean * tr_mu_outer:.4e}")
    print(f"  > From Sigma Env:           {tau_mean * tr_sigma_env:.4e}")
    
    if tr_prior > 0:
        ratio = tr_data / tr_prior
        print(f"\nRatio (Data / Prior): {ratio:.4f}")
        if ratio < 1.0:
            print("❌ CONCLUSION: Prior is dominating! The model ignores data and shrinks to 0.")
            print("   Fixes:")
            print("   1. Reduce prior strength (smaller alpha/beta in q_bonds init).")
            print("   2. Increase E[tau] (initial noise precision).")
            print("   3. Increase batch size / number of samples (Data term scales with N).")
        else:
            print("✅ Data signal is strong enough.")
            
    # C. Update Simulation
    old_mu_vec = model.mu[node_tag].data.reshape(-1)
    model.update_sigma_node(node_tag)
    model.update_mu_node(node_tag)
    new_mu_vec = model.mu[node_tag].data.reshape(-1)
    
    norm_old = torch.norm(old_mu_vec).item()
    norm_new = torch.norm(new_mu_vec).item()
    
    print(f"\n--- Weights Movement ---")
    print(f"Old Norm: {norm_old:.4f}")
    print(f"New Norm: {norm_new:.4f}")
    print(f"Shrinkage: {norm_new / norm_old:.2%}")

if __name__ == "__main__":
    debug_training_dynamics()
