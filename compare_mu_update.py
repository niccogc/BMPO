import torch
import numpy as np
import quimb.tensor as qt
from tensor.builder import Inputs
from tensor.btn import BTN
from tensor_old.bayesian_mpo_builder import create_bayesian_tensor_train

torch.set_default_dtype(torch.float64)

def check_permutations(name, target_flat, new_tn, inds_map):
    """
    Contracts new_tn and checks which index permutation matches the target_flat vector.
    inds_map: list of indices e.g. ['b1', 'x2', 'y']
    """
    print(f"\n>>> DIAGNOSTIC: {name}")
    
    # 1. Get New Tensor Data
    if isinstance(new_tn, qt.TensorNetwork):
        t_new = new_tn.contract()
    else:
        t_new = new_tn

    print(f"    Raw New Inds:  {t_new.inds}")
    print(f"    Raw New Shape: {t_new.shape}")
    print(f"    Target Shape:  {target_flat.shape}")

    # 2. Check Permutation 1: (Bond, Phys, Out) -> ('b1', 'x2', 'y')
    # This matches the standard MPS storage (Left, Input, Right/Out)
    perm1_inds = tuple(inds_map)
    t1 = t_new.transpose(*perm1_inds)
    diff1 = torch.abs(target_flat - t1.data.reshape(-1)).max().item()
    
    # 3. Check Permutation 2: (Phys, Bond, Out) -> ('x2', 'b1', 'y')
    # This swaps the bond and physical dimension
    perm2_inds = (inds_map[1], inds_map[0], inds_map[2])
    t2 = t_new.transpose(*perm2_inds)
    diff2 = torch.abs(target_flat - t2.data.reshape(-1)).max().item()

    print(f"\n    [Permutation 1] {perm1_inds}:")
    print(f"       Diff: {diff1:.6e} {'✅ MATCH' if diff1 < 1e-9 else '❌'}")
    
    print(f"    [Permutation 2] {perm2_inds}:")
    print(f"       Diff: {diff2:.6e} {'✅ MATCH' if diff2 < 1e-9 else '❌'}")

    return diff1 < 1e-9 or diff2 < 1e-9

def run_rhs_debug():
    print("="*80)
    print(" 🏹 NODE 2: RHS PERMUTATION CHECK")
    print("="*80)

    # 1. SETUP
    N = 10
    x = torch.rand(N, dtype=torch.float64) * 2 - 1
    y = 2.0 * x**3 - 1.0 * x**2 + 0.5 + 0.01 * torch.randn(N)
    X_old = torch.stack([torch.ones_like(x), x], dim=1)
    
    # 2. INIT MODELS
    print("[1] Init Models...")
    old = create_bayesian_tensor_train(3, 4, 2, 1, dtype=torch.float64, seed=42, random_priors=False)
    
    # Copy Weights
    def fmt(w, last=False):
        w = w.squeeze()
        if last and w.dim()==2: w = w.unsqueeze(-1)
        return w
    
    old_mu = [n.tensor.detach().clone() for n in old.mu_nodes]
    old_sig = [n.tensor.detach().clone() for n in old.sigma_nodes]

    # New Model Init
    loader = Inputs([X_old], [y.unsqueeze(1)], ["y"], ["x0", "x1", "x2"], "s", N)
    t0 = qt.Tensor(fmt(old_mu[0]), inds=('x0', 'b0'), tags={'Node0'})
    t1 = qt.Tensor(fmt(old_mu[1]), inds=('b0', 'x1', 'b1'), tags={'Node1'})
    t2 = qt.Tensor(fmt(old_mu[2], True), inds=('b1', 'x2', 'y'), tags={'Node2'})
    new = BTN(qt.TensorNetwork([t0, t1, t2]), loader, "s", method='cholesky')

    # Copy Sigma (Alternating)
    w0 = old_sig[0].squeeze()
    s0 = qt.Tensor(w0, inds=('x0', 'x0_prime', 'b0', 'b0_prime'), tags={'Node0'})
    w1 = old_sig[1]
    s1 = qt.Tensor(w1, inds=('b0', 'b0_prime', 'x1', 'x1_prime', 'b1', 'b1_prime'), tags={'Node1'})
    w2 = old_sig[2].squeeze().unsqueeze(-1)
    s2 = qt.Tensor(w2, inds=('b1', 'b1_prime', 'x2', 'x2_prime', 'y'), tags={'Node2'})
    new.sigma = qt.TensorNetwork([s0, s1, s2])
    
    new.q_tau.concentration = old.tau_alpha
    new.q_tau.rate = old.tau_beta

    # 3. RUN UPDATE
    TAG = "Node2"
    IDX = 2
    print(f"\n[2] Running Update for {TAG}...")
    
    # Old
    res_old = old.update_block_variational(IDX, X_old, y)
    target_rhs = res_old['rhs']
    target_mu = res_old['mu_new']

    # New
    # Note: We must update Sigma first because Mu update depends on it? 
    # Actually RHS calculation usually only depends on Data + Other Mu nodes.
    # But let's run Sigma update to be safe and consistent with previous steps.
    new.update_sigma_node(TAG)
    
    # --- CHECK RHS ---
    mu_idx = new.mu[TAG].inds # ('b1', 'x2', 'y')
    rhs_new = new.forward_with_target(
        new.data.data_mu_y, new.mu, 'dot', 
        sum_over_batch=True, output_inds=mu_idx
    )
    # Scale by Tau
   
    # We pass the expected indices for Node 2: Bond, Phys, Out
    check_permutations("RHS (Gradient)", target_rhs, rhs_new, ['b1', 'x2', 'y'])

    # --- CHECK MU ---
    new.update_mu_node(TAG)
    mu_new_tn = new.mu[TAG]
    
    check_permutations("Updated Mu", target_mu, mu_new_tn, ['b1', 'x2', 'y'])

if __name__ == "__main__":
    run_rhs_debug()
