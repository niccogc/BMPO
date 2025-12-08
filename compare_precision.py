import torch
import numpy as np
import quimb.tensor as qt
from tensor.builder import Inputs
from tensor.btn import BTN
from tensor_old.bayesian_mpo_builder import create_bayesian_tensor_train

torch.set_default_dtype(torch.float64)

def compare_precision_components_fixed():
    print("\n" + "="*80)
    print(" 🔬 COMPONENT-WISE DEBUGGING: PRECISION & INITIALIZATION (FIXED)")
    print("="*80)

    # ---------------------------------------------------------
    # 1. SETUP DATA
    # ---------------------------------------------------------
    N_SAMPLES = 10
    x = torch.rand(N_SAMPLES, dtype=torch.float64) * 2 - 1
    y = 2.0 * x**3 - 1.0 * x**2 + 0.5 + 0.01 * torch.randn(N_SAMPLES)
    
    X_old = torch.stack([torch.ones_like(x), x], dim=1)
    X_new = X_old
    y_new = y.unsqueeze(1)
    
    # ---------------------------------------------------------
    # 2. INIT OLD MODEL
    # ---------------------------------------------------------
    print("[1] Initializing OLD Model...")
    old_model = create_bayesian_tensor_train(
        num_blocks=3, bond_dim=4, input_features=2, output_shape=1,
        dtype=torch.float64, seed=42, random_priors=True
    )

    old_mu_weights = [n.tensor.detach().clone() for n in old_model.mu_nodes]
    old_sigma_weights = [n.tensor.detach().clone() for n in old_model.sigma_nodes]

    # ---------------------------------------------------------
    # 3. INIT NEW MODEL
    # ---------------------------------------------------------
    print("[2] Initializing NEW Model...")
    input_labels = ["x0", "x1", "x2"]
    loader = Inputs(inputs=[X_new], outputs=[y_new], outputs_labels=["y"],
                    input_labels=input_labels, batch_dim="s", batch_size=N_SAMPLES)

    # --- Helper for Mu ---
    def format_weight(w, is_last=False):
        w = w.squeeze()
        if is_last and w.dim() == 2:
            w = w.unsqueeze(-1)
        return w

    # A. Construct Mu Network (Standard)
    t0 = qt.Tensor(data=format_weight(old_mu_weights[0]), inds=('x0', 'b0'), tags={'Node0'})
    t1 = qt.Tensor(data=format_weight(old_mu_weights[1]), inds=('b0', 'x1', 'b1'), tags={'Node1'})
    t2 = qt.Tensor(data=format_weight(old_mu_weights[2], True), inds=('b1', 'x2', 'y'), tags={'Node2'})
    mu_tn = qt.TensorNetwork([t0, t1, t2])

    new_model = BTN(mu=mu_tn, data_stream=loader, batch_dim="s", method='cholesky')

    # B. REASSIGN SIGMA TENSOR NETWORK (CORRECTED: ALTERNATING STRATEGY)
    print("    -> Re-building Sigma TensorNetwork from Old Weights (Alternating Indices)...")
    
    # Node 0
    # Old: (Phys, Phys', Bond_R, Bond_R')
    w0 = old_sigma_weights[0].squeeze() 
    sig_t0 = qt.Tensor(data=w0, inds=('x0', 'x0_prime', 'b0', 'b0_prime'), tags={'Node0'})
    
    # Node 1
    # Old: (Bond_L, Bond_L', Phys, Phys', Bond_R, Bond_R')
    w1 = old_sigma_weights[1]
    sig_t1 = qt.Tensor(data=w1, inds=('b0', 'b0_prime', 'x1', 'x1_prime', 'b1', 'b1_prime'), tags={'Node1'})
    
    # Node 2
    # Old: (Bond_L, Bond_L', Phys, Phys', Out, Out') -> Squeeze Out(1), Add y(1)
    w2 = old_sigma_weights[2].squeeze().unsqueeze(-1)
    sig_t2 = qt.Tensor(data=w2, inds=('b1', 'b1_prime', 'x2', 'x2_prime', 'y'), tags={'Node2'})
    
    # Assign new TN
    new_model.sigma = qt.TensorNetwork([sig_t0, sig_t1, sig_t2])

    # C. Sync Hyperparameters
    new_model.q_tau.concentration = old_model.tau_alpha
    new_model.q_tau.rate = old_model.tau_beta

    # ---------------------------------------------------------
    # 4. RUN COMPARISON
    # ---------------------------------------------------------
    IDX = 1
    TAG = "Node1"
    print(f"\n[3] Comparing Components for {TAG}...")

    # --- OLD MODEL EXECUTION ---
    res_old = old_model.update_block_variational(IDX, X_old, y)
    
    old_prec_matrix = res_old["precision"]
    old_sigma_env = res_old["J_sigma"]    
    old_mu_env = res_old["J_mu_outer"]    
    old_theta = old_model.compute_theta_tensor(IDX).flatten()
    old_theta_matrix = torch.diag(old_theta)

    # --- NEW MODEL EXECUTION ---
    
    # 1. TAU
    tau_old = old_model.get_tau_mean().item()
    tau_new = new_model.get_tau_mean().item()
    print(f"  A. Tau Mean Match: {abs(tau_old - tau_new) < 1e-12}")

    # 2. THETA (Prior)
    theta_tn = new_model.theta_block_computation(TAG)
    theta_new_vec = theta_tn.data.flatten() 
    theta_new_matrix = torch.diag(theta_new_vec)
    
    diff_theta = torch.abs(old_theta_matrix - theta_new_matrix).max()
    print(f"  B. Theta (Prior) Max Diff: {diff_theta:.10e}")

    # 3. SIGMA ENVIRONMENT
    sigma_env_tn = new_model.get_environment(
         tn=new_model.sigma,
         target_tag=TAG,
         input_generator=new_model.data.data_sigma,
         copy=False,
         sum_over_batch=True,
         sum_over_output=True
    )
    # Transpose to match old layout
    # Current indices: b0, b0', x1, x1', b1, b1'
    # Target indices:  b0, x1, b1, b0', x1', b1'  (Group Unprimes | Group Primes)
    sigma_env_tn = sigma_env_tn.transpose('b0', 'x1', 'b1', 'b0_prime', 'x1_prime', 'b1_prime')
    sigma_env_new = sigma_env_tn.data.reshape(old_sigma_env.shape)
    
    diff_sigma_env = torch.abs(old_sigma_env - sigma_env_new).max()
    print(f"  C. Sigma Env Max Diff:     {diff_sigma_env:.10e}")

    # 4. MU OUTER ENVIRONMENT
    mu_env_tn = new_model.outer_operation(
        tn=new_model.mu,
        node_tag=TAG,
        input_generator=new_model.data.data_mu,
        sum_over_batches=True
    )
    # Transpose same way as Sigma
    mu_env_tn = mu_env_tn.transpose('b0', 'x1', 'b1', 'b0_prime', 'x1_prime', 'b1_prime')
    mu_env_new = mu_env_tn.data.reshape(old_mu_env.shape)
    
    diff_mu_env = torch.abs(old_mu_env - mu_env_new).max()
    print(f"  D. Mu Env Max Diff:        {diff_mu_env:.10e}")

    # 5. FINAL PRECISION RECONSTRUCTION
    prec_manual_new = tau_new * (sigma_env_new + mu_env_new) + theta_new_matrix
    diff_total = torch.abs(old_prec_matrix - prec_manual_new).max()
    print(f"\n  E. Total Precision Diff:   {diff_total:.10e}")

    if diff_total < 1e-9:
        print("\n✅ SUCCESS: Components align perfectly.")
    else:
        print("\n❌ FAILURE: Check the component with the high error above.")

if __name__ == "__main__":
    compare_precision_components_fixed()
