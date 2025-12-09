import torch
import numpy as np
import quimb.tensor as qt
from tensor.builder import Inputs
from tensor.btn import BTN
from tensor_old.bayesian_mpo_builder import create_bayesian_tensor_train

torch.set_default_dtype(torch.float64)
# Threshold=inf ensures the full matrix is printed, no truncation
torch.set_printoptions(precision=4, linewidth=200, sci_mode=False, threshold=float('inf'))

def compare_rank2_matrices(old_tensor, new_tensor, node_name):
    """
    Takes BOTH tensors.
    Permutes BOTH from Alternating (A, A', B, B') to Block (A, B, ..., A', B').
    Prints BOTH matrices to prove they are identical and diagonal.
    """
    print(f"\n" + "="*60)
    print(f" >>> INSPECTING {node_name}")
    print("="*60)
    
    # --- 1. PREPARE OLD MATRIX ---
    ndim = old_tensor.ndim
    if ndim % 2 != 0: ndim -= 1 # Ignore 'y'
    perm = list(range(0, ndim, 2)) + list(range(1, ndim, 2))
    
    t_old_view = old_tensor[..., 0] if old_tensor.ndim % 2 != 0 else old_tensor
    mat_old = t_old_view.permute(*perm)
    rows = int(np.prod(mat_old.shape[:len(perm)//2]))
    mat_old = mat_old.reshape(rows, rows)

    # --- 2. PREPARE NEW MATRIX ---
    t_new_view = new_tensor[..., 0] if new_tensor.ndim % 2 != 0 else new_tensor
    mat_new = t_new_view.permute(*perm)
    mat_new = mat_new.reshape(rows, rows)

    # --- 3. PRINT OLD ---
    print(f"\n[OLD SIGMA] (Should be Diagonal):")
    print(mat_old)
    is_old_diag = torch.allclose(mat_old, torch.diag(torch.diagonal(mat_old)), atol=1e-10)
    print(f"> Is Diagonal: {is_old_diag}")

    # --- 4. PRINT NEW ---
    print(f"\n[NEW SIGMA] (Should match Old):")
    print(mat_new)
    is_new_diag = torch.allclose(mat_new, torch.diag(torch.diagonal(mat_new)), atol=1e-10)
    print(f"> Is Diagonal: {is_new_diag}")

    # --- 5. CHECK EQUALITY ---
    are_equal = torch.allclose(old_tensor, new_tensor, atol=1e-12)
    print(f"\n[RESULT] Identical: {are_equal}")
    if not are_equal:
        print("❌ MISMATCH DETECTED")

def run_both_check():
    # 1. Init Old (random_priors=False -> Diagonal)
    print("Initializing Old Model...")
    old = create_bayesian_tensor_train(3, 4, 2, 1, dtype=torch.float64, seed=42, random_priors=False)
    old_sig = [n.tensor.detach().clone() for n in old.sigma_nodes]

    # 2. Setup New Tensors (Copy + Label)
    print("Mapping to New Model...")
    
    # Node 0
    w0 = old_sig[0].squeeze()
    s0 = qt.Tensor(w0, inds=('x0', 'x0_prime', 'b0', 'b0_prime'), tags={'Node0'})
    
    # Node 1
    w1 = old_sig[1]
    s1 = qt.Tensor(w1, inds=('b0', 'b0_prime', 'x1', 'x1_prime', 'b1', 'b1_prime'), tags={'Node1'})
    
    # Node 2
    w2 = old_sig[2].squeeze().unsqueeze(-1)
    s2 = qt.Tensor(w2, inds=('b1', 'b1_prime', 'x2', 'x2_prime', 'y'), tags={'Node2'})

    # 3. COMPARE BOTH
    compare_rank2_matrices(w0, s0.data, "Node0")
    compare_rank2_matrices(w1, s1.data, "Node1")
    compare_rank2_matrices(w2, s2.data, "Node2")

if __name__ == "__main__":
    run_both_check()
