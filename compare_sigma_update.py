import torch
import numpy as np
import quimb.tensor as qt
from tensor.builder import Inputs
from tensor.btn import BTN
from tensor_old.bayesian_mpo_builder import create_bayesian_tensor_train

torch.set_default_dtype(torch.float64)

def check_sigma_node2(old_sigma, new_sigma_tn):
    print("\n" + "="*60)
    print(" 🕵️  NODE 2 SIGMA VERIFICATION (Order-Aware)")
    print("="*60)
    
    # 1. Prepare OLD Sigma
    # Old model stores: (Bond, Bond', Phys, Phys', Out, Out')
    # We squeeze the Out dims -> (Bond, Bond', Phys, Phys')
    t_old = old_sigma.squeeze()
    print(f"Old (Squeezed) Shape: {tuple(t_old.shape)}")

    # 2. Prepare NEW Sigma
    # Indices seen in log: ('y', 'b1', 'x2', 'b1_prime', 'x2_prime')
    # Shape seen in log: (1, 2, 2, 2, 2)
    print(f"New Indices: {new_sigma_tn.inds}")
    
    # A. Squeeze the 'y' dimension (Size 1)
    # We use .squeeze() without arguments to remove ALL dimensions of size 1 
    # (safer than guessing if it's 0 or -1)
    t_new = new_sigma_tn.data.squeeze()
    
    print(f"New (Squeezed) Shape: {tuple(t_new.shape)}")

    # 3. Align Layouts
    # Old Layout: (Bond, Bond', Phys, Phys')  -> Indices: (b1, b1', x2, x2')
    # New Layout: (Bond, Phys, Bond', Phys')  -> Indices: (b1, x2, b1', x2')
    
    # We must permute New to match Old:
    # Current: 0=Bond, 1=Phys, 2=Bond', 3=Phys'
    # Target:  Bond(0), Bond'(2), Phys(1), Phys'(3)
    t_new_aligned = t_new.permute(0, 2, 1, 3)

    # 4. Compare
    diff = torch.abs(t_old - t_new_aligned).max().item()
    print(f"Match Diff: {diff:.6e}")
    
    if diff < 1e-9:
        print("✅ MATCH")
    else:
        print("❌ FAIL")
        print("   (If this fails, ensure you updated 'invert_ordered_tensor' with the fix provided earlier)")

def run_debug():
    # ... (Standard Init Code) ...
    N = 10
    x = torch.rand(N, dtype=torch.float64) * 2 - 1
    y = 2.0 * x**3 - 1.0 * x**2 + 0.5 + 0.01 * torch.randn(N)
    X_old = torch.stack([torch.ones_like(x), x], dim=1)
    
    print("[1] Init Models...")
    old = create_bayesian_tensor_train(3, 4, 2, 1, dtype=torch.float64, seed=42, random_priors=False)
    # ... (Copy weights exactly as before) ...
    # Copy Mu
    def fmt(w, last=False):
        w = w.squeeze()
        if last and w.dim()==2: w = w.unsqueeze(-1)
        return w
    
    old_mu = [n.tensor.detach().clone() for n in old.mu_nodes]
    old_sig = [n.tensor.detach().clone() for n in old.sigma_nodes]

    loader = Inputs([X_old], [y.unsqueeze(1)], ["y"], ["x0", "x1", "x2"], "s", N)
    t0 = qt.Tensor(fmt(old_mu[0]), inds=('x0', 'b0'), tags={'Node0'})
    t1 = qt.Tensor(fmt(old_mu[1]), inds=('b0', 'x1', 'b1'), tags={'Node1'})
    t2 = qt.Tensor(fmt(old_mu[2], True), inds=('b1', 'x2', 'y'), tags={'Node2'})
    new = BTN(qt.TensorNetwork([t0, t1, t2]), loader, "s", method='cholesky')

    # Copy Sigma
    w0 = old_sig[0].squeeze()
    s0 = qt.Tensor(w0, inds=('x0', 'x0_prime', 'b0', 'b0_prime'), tags={'Node0'})
    w1 = old_sig[1]
    s1 = qt.Tensor(w1, inds=('b0', 'b0_prime', 'x1', 'x1_prime', 'b1', 'b1_prime'), tags={'Node1'})
    w2 = old_sig[2].squeeze().unsqueeze(-1)
    s2 = qt.Tensor(w2, inds=('b1', 'b1_prime', 'x2', 'x2_prime', 'y'), tags={'Node2'})
    new.sigma = qt.TensorNetwork([s0, s1, s2])
    
    new.q_tau.concentration = old.tau_alpha
    new.q_tau.rate = old.tau_beta

    # Run Update
    TAG = "Node2"
    IDX = 2
    print(f"\n[2] Running Update {TAG}...")
    
    # Old
    old.update_block_variational(IDX, X_old, y)
    target_sigma = old.sigma_nodes[IDX].tensor

    # New
    new.update_sigma_node(TAG)
    new_sigma_tn = new.sigma[TAG]

    # Compare
    check_sigma_node2(target_sigma, new_sigma_tn)

if __name__ == "__main__":
    run_debug()
