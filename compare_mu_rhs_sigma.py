import torch
import numpy as np
import quimb.tensor as qt
from tensor.builder import Inputs
from tensor.btn import BTN
from tensor_old.bayesian_mpo_builder import create_bayesian_tensor_train

torch.set_default_dtype(torch.float64)

def check_tensor_aligned(name, target_in, new_tn, expected_inds):
    """
    Contracts new_tn, aligns indices to 'expected_inds', flattens, and compares.
    """
    print(f"\n>>> CHECK: {name}")
    
    # 1. Contract/Extract Data
    if isinstance(new_tn, qt.TensorNetwork):
        t_new = new_tn.contract()
    elif isinstance(new_tn, qt.Tensor):
        t_new = new_tn
    else:
        t_new = qt.Tensor(new_tn, inds=expected_inds) 

    print(f"    Raw New Inds: {t_new.inds}")
    
    # 2. Transpose to Target Order & Flatten
    try:
        t_aligned = t_new.transpose(*expected_inds).squeeze()
        t_flat = t_aligned.data.reshape(-1)
        
        # FIX: Ensure target is also flattened for comparison
        target_flat = target_in.reshape(-1)
        
        diff = torch.abs(target_flat - t_flat).max().item()
        print(f"    Max Diff: {diff:.6e}")
        
        if diff < 1e-9:
            print("    ✅ MATCH")
            return True
        else:
            print("    ❌ FAIL")
            return False
    except Exception as e:
        print(f"    ❌ Alignment Failed: {e}")
        return False

def compare_modified_returns():
    print("="*80)
    print(" 🏆 FINAL VERIFICATION: MODIFIED _get_mu_update")
    print("="*80)

    # 1. SETUP
    N = 10
    x = torch.rand(N, dtype=torch.float64) * 2 - 1
    y = 2.0 * x**3 - 1.0 * x**2 + 0.5
    X_old = torch.stack([torch.ones_like(x), x], dim=1)
    
    print("[1] Init Models...")
    old = create_bayesian_tensor_train(3, 4, 2, 1, dtype=torch.float64, seed=42, random_priors=False)
    
    # Copy Weights
    old_mu = [n.tensor.detach().clone() for n in old.mu_nodes]
    old_sig = [n.tensor.detach().clone() for n in old.sigma_nodes]

    loader = Inputs([X_old], [y.unsqueeze(1)], ["y"], ["x0", "x1", "x2"], "s", N)
    
    def fmt(w, last=False):
        w = w.squeeze()
        if last and w.dim()==2: w = w.unsqueeze(-1)
        return w

    t0 = qt.Tensor(fmt(old_mu[0]), inds=('x0', 'b0'), tags={'Node0'})
    t1 = qt.Tensor(fmt(old_mu[1]), inds=('b0', 'x1', 'b1'), tags={'Node1'})
    t2 = qt.Tensor(fmt(old_mu[2], True), inds=('b1', 'x2', 'y'), tags={'Node2'})
    new = BTN(qt.TensorNetwork([t0, t1, t2]), loader, "s", method='cholesky')

    # Assign Sigma (Alternating Strategy)
    w0 = old_sig[0].squeeze()
    s0 = qt.Tensor(w0, inds=('x0', 'x0_prime', 'b0', 'b0_prime'), tags={'Node0'})
    w1 = old_sig[1]
    s1 = qt.Tensor(w1, inds=('b0', 'b0_prime', 'x1', 'x1_prime', 'b1', 'b1_prime'), tags={'Node1'})
    w2 = old_sig[2].squeeze().unsqueeze(-1)
    s2 = qt.Tensor(w2, inds=('b1', 'b1_prime', 'x2', 'x2_prime', 'y'), tags={'Node2'})
    new.sigma = qt.TensorNetwork([s0, s1, s2])
    
    new.q_tau.concentration = old.tau_alpha
    new.q_tau.rate = old.tau_beta

    # 3. RUN UPDATES
    IDX = 2
    TAG = "Node2"
    print(f"\n[2] Running Update for {TAG}...")
    
    # --- OLD MODEL ---
    res_old = old.update_block_variational(IDX, X_old, y)
    
    target_mu = res_old['mu_new']
    target_rhs = res_old['rhs']
    target_sigma = old.sigma_nodes[IDX].tensor 

    # --- NEW MODEL ---
    new.update_sigma_node(TAG)
    mu_new_val, rhs_tn, sigma_tn = new._get_mu_update(TAG, True)
    
    # ----------------------------------------------------------------
    # 4. COMPARISONS
    # ----------------------------------------------------------------
    
    # A. SIGMA
    print("\n>>> CHECK: Sigma (Covariance)")
    sig_old_sq = target_sigma.squeeze()
    sig_new_sq = sigma_tn.data.squeeze()
    sig_new_aligned = sig_new_sq.permute(0, 2, 1, 3)
    
    diff_sig = torch.abs(sig_old_sq - sig_new_aligned).max().item()
    print(f"    Match Diff: {diff_sig:.6e}")
    if diff_sig < 1e-9: print("    ✅ MATCH")
    else:               print("    ❌ FAIL")

    # B. RHS (GRADIENT)
    rhs_scaled = rhs_tn 
    check_tensor_aligned("RHS (Gradient, Scaled)", target_rhs, rhs_scaled, ['b1_prime', 'x2_prime', 'y'])

    # C. MU (WEIGHTS)
    target_mu = target_mu.squeeze()
    mu_new_val = mu_new_val.squeeze()
    
    check_tensor_aligned("Updated Mu", target_mu, mu_new_val, ['b1', 'x2'])

if __name__ == "__main__":
    compare_modified_returns()
