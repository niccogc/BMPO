import torch
import numpy as np
import quimb.tensor as qt
from tensor.builder import Inputs
from tensor.btn import BTN
from tensor_old.bayesian_mpo_builder import create_bayesian_tensor_train

torch.set_default_dtype(torch.float64)

def debug_contraction_logic():
    print("="*80)
    print(" 🔧 DEBUGGING SIGMA @ RHS CONTRACTION")
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

    # Assign Sigma
    w0 = old_sig[0].squeeze()
    s0 = qt.Tensor(w0, inds=('x0', 'x0_prime', 'b0', 'b0_prime'), tags={'Node0'})
    w1 = old_sig[1]
    s1 = qt.Tensor(w1, inds=('b0', 'b0_prime', 'x1', 'x1_prime', 'b1', 'b1_prime'), tags={'Node1'})
    w2 = old_sig[2].squeeze().unsqueeze(-1)
    s2 = qt.Tensor(w2, inds=('b1', 'b1_prime', 'x2', 'x2_prime', 'y'), tags={'Node2'})
    new.sigma = qt.TensorNetwork([s0, s1, s2])
    
    new.q_tau.concentration = old.tau_alpha
    new.q_tau.rate = old.tau_beta

    # 2. GET DATA FROM UPDATE
    TAG = "Node2"
    IDX = 2
    print(f"\n[2] Extracting Tensors for {TAG}...")
    
    # Old Target
    res_old = old.update_block_variational(IDX, X_old, y)
    target_mu = res_old['mu_new'] # (2, 2)

    # New Components
    new.update_sigma_node(TAG)
    _, rhs_tn, sigma_tn = new._get_mu_update(TAG)
    
    # Get Raw Data
    rhs_data = rhs_tn.contract().data # Shape likely (2, 2, 1) or (2, 2)
    sig_data = sigma_tn.data          # Shape likely (1, 2, 2, 2, 2)
    tau = new.get_tau_mean().item()
    
    print(f"    RHS Shape:   {tuple(rhs_data.shape)}")
    print(f"    Sigma Shape: {tuple(sig_data.shape)}")
    print(f"    Tau:         {tau}")
    print(f"    Target Mu:   {tuple(target_mu.shape)}")

    # 3. MANUAL CONTRACTION TESTS
    print("\n[3] Testing Contractions...")
    
    # Prepare Tensors Squeezed for Matrix Math
    # RHS: (2, 2, 1) -> (2, 2) -> Flatten to (4)
    rhs_vec = rhs_data.squeeze().reshape(-1) * tau
    
    # Sigma: (1, 2, 2, 2, 2) -> (2, 2, 2, 2) -> (4, 4)
    # But WHICH order? 
    # Indices: ('y', 'b1', 'x2', 'b1_prime', 'x2_prime')
    # We want to multiply Sigma @ RHS.
    # A. Assume Sigma is (Rows=Unprimes, Cols=Primes) -> (b1, x2) x (b1', x2')
    #    This matches the invert_ordered_tensor logic we fixed.
    #    So we reshape to (4_unprime, 4_prime)
    sig_sq = sig_data.squeeze() # (b1, x2, b1', x2') based on previous logs?
    # Wait, check indices from TN
    print(f"    Sigma Indices: {sigma_tn.inds}") 
    # If inds are ('y', 'b1', 'x2', 'b1_prime', 'x2_prime')
    # And we assume 'y' is dim 0.
    
    # --- TEST A: Standard (Sigma @ RHS) ---
    # We want: mu_{b1, x2} = sum_{b1', x2'} Sigma_{b1, x2, b1', x2'} * RHS_{b1', x2'}
    # This assumes Sigma connects Unprimes (output) to Primes (input/RHS)
    # Permute Sigma to (b1, x2, b1', x2') -> (4, 4)
    # Note: earlier we saw Sigma inds were ('y', 'b1', 'x2', 'b1_prime', 'x2_prime')
    # Let's ensure we map: Output(0,1) | Input(2,3)
    
    # Permute to: b1, x2, b1', x2'
    # indices mapping: b1=1, x2=2, b1'=3, x2'=4 (if y=0)
    # If y is squeezed, shift by -1.
    
    # Let's try contracting properly using einsum based on shape
    # RHS is (2, 2) -> (b1, x2)
    # Sigma is (2, 2, 2, 2) -> (b1, x2, b1', x2')
    
    # Case 1: Sigma * RHS (Standard Matrix Vector)
    # Flatten Sigma to (4, 4) where rows=(b1, x2)
    sig_mat_1 = sig_sq.reshape(4, 4) 
    mu_1 = sig_mat_1 @ rhs_vec
    diff_1 = torch.abs(target_mu.reshape(-1) - mu_1).max().item()
    
    print(f"    [Test 1] Sigma(flat) @ RHS(flat): Diff = {diff_1:.6e} {'✅' if diff_1 < 1e-9 else '❌'}")

    # Case 2: Sigma.T * RHS (Swap input/output indices of Sigma)
    # Maybe Sigma rows are primes and cols are unprimes?
    # Or RHS corresponds to unprimes and we want primes out?
    mu_2 = sig_mat_1.T @ rhs_vec
    diff_2 = torch.abs(target_mu.reshape(-1) - mu_2).max().item()
    
    print(f"    [Test 2] Sigma.T @ RHS:           Diff = {diff_2:.6e} {'✅' if diff_2 < 1e-9 else '❌'}")

    # Case 3: Permuted Sigma
    # Maybe Sigma is stored as (b1, b1', x2, x2')? (Alternating)
    # Let's try permuting Sigma from Alternating -> Block
    # Current (supposed): (b1, x2, b1', x2')
    # Try interpreting raw data as (b1, b1', x2, x2') -> Permute -> (b1, x2, b1', x2')
    # 0, 2, 1, 3
    sig_alt = sig_sq.permute(0, 2, 1, 3).reshape(4, 4)
    mu_3 = sig_alt @ rhs_vec
    diff_3 = torch.abs(target_mu.reshape(-1) - mu_3).max().item()
    print(f"    [Test 3] Sigma(Alt->Block) @ RHS: Diff = {diff_3:.6e} {'✅' if diff_3 < 1e-9 else '❌'}")

    # 4. FIX SUGGESTION
    if diff_1 < 1e-9:
        print("\n✅ CONCLUSION: Contraction indices are correct, but maybe Quimb output labels were swapped?")
    elif diff_2 < 1e-9:
        print("\n✅ CONCLUSION: Sigma is Transposed relative to RHS. Swap output indices in contraction.")
    elif diff_3 < 1e-9:
        print("\n✅ CONCLUSION: Sigma is Alternating. Permute Sigma before contracting.")

if __name__ == "__main__":
    debug_contraction_logic()
