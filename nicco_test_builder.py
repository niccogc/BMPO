import torch
import numpy as np
import quimb.tensor as qt
from tensor.builder import Inputs, BTNBuilder

# Set precision
torch.set_default_dtype(torch.float64)

def check_sigma_initialization():
    print("\n" + "="*60)
    print("DEBUGGING SIGMA INITIALIZATION")
    print("="*60)

    # ---------------------------------------------------------
    # 1. SETUP: 3-Node Network (requires multiple bonds per node)
    # ---------------------------------------------------------
    # Node2 will have connections to Node1(b1) and Node3(b2).
    # This creates a tensor with multiple bond indices, which is 
    # where diagonal initialization logic is most likely to fail.
    
    D_b1 = 2
    D_b2 = 3
    
    # Dummy inputs just to satisfy builder
    x = torch.randn(2, 2)
    y = torch.randn(2, 1)
    
    # We define structure explicitly
    t1 = qt.Tensor(torch.randn(2, D_b1), inds=('x', 'b1'), tags={'Node1'})
    # Node2 has 2 bonds: b1, b2
    t2 = qt.Tensor(torch.randn(D_b1, D_b2), inds=('b1', 'b2'), tags={'Node2'})
    t3 = qt.Tensor(torch.randn(D_b2, 1), inds=('b2', 'y'), tags={'Node3'})
    
    mu_tn = qt.TensorNetwork([t1, t2, t3])
    
    # ---------------------------------------------------------
    # 2. BUILD MODEL
    # ---------------------------------------------------------
    print("Building model...")
    # We only need the builder to construct Sigma
    builder = BTNBuilder(mu=mu_tn, output_dimensions=['y'], batch_dim='s')
    _, _, _, _, sigma_tn = builder.build_model()
    
    # ---------------------------------------------------------
    # 3. INSPECT NODE2 SIGMA
    # ---------------------------------------------------------
    # Node2 Sigma should have indices: b1, b2, b1_prime, b2_prime
    # Shape: (D_b1, D_b2, D_b1, D_b2) -> (2, 3, 2, 3)
    # Total dim = 6. Flattened matrix (6, 6).
    
    sigma_node = sigma_tn['Node2']
    print(f"\nNode2 Sigma Indices: {sigma_node.inds}")
    print(f"Node2 Sigma Shape:   {sigma_node.shape}")
    
    # 1. Flatten to Matrix (using proper ordering)
    # We want rows=(b1, b2), cols=(b1_prime, b2_prime)
    # Or consistent sorting.
    
    # Identify prime/unprime pairs
    inds = sigma_node.inds
    unprimed = sorted([i for i in inds if not i.endswith('_prime')])
    primed = [f"{i}_prime" for i in unprimed]
    
    print(f"Flattening with Rows: {unprimed}, Cols: {primed}")
    
    # Permute to (b1, b2, b1_prime, b2_prime)
    sigma_ordered = sigma_node.transpose(*unprimed, *primed)
    
    # Reshape to Matrix
    row_dim = int(np.prod([sigma_node.ind_size(i) for i in unprimed]))
    col_dim = int(np.prod([sigma_node.ind_size(i) for i in primed]))
    
    sigma_mat = sigma_ordered.data.reshape(row_dim, col_dim)
    
    print(f"Matrix Shape: {sigma_mat.shape}")
    
    # ---------------------------------------------------------
    # 4. CHECKS
    # ---------------------------------------------------------
    
    # A. Symmetry
    is_sym = torch.allclose(sigma_mat, sigma_mat.T, atol=1e-8)
    print(f"Is Symmetric: {is_sym}")
    if not is_sym:
        print("Diff from symmetric:", (sigma_mat - sigma_mat.T).abs().max().item())

    # B. Eigenvalues (Positive Definite?)
    try:
        eigs = torch.linalg.eigvalsh(sigma_mat)
        min_eig = eigs.min().item()
        max_eig = eigs.max().item()
        
        print(f"Min Eigenvalue: {min_eig:.6e}")
        print(f"Max Eigenvalue: {max_eig:.6e}")
        
        if min_eig > 0:
            print("✅ Matrix is Positive Definite.")
        else:
            print("❌ Matrix is NOT Positive Definite (has non-positive eigenvalues).")
            
    except Exception as e:
        print(f"Eigenvalue computation failed: {e}")

    # C. Visualize Structure (Small matrix)
    print("\nMatrix Values (subset):")
    print(sigma_mat)
    
    # Check diagonal
    # For Identity initialization, we expect diagonal elements to be non-zero
    # and off-diagonal elements to be zero.
    
    is_diagonal = True
    for i in range(row_dim):
        for j in range(col_dim):
            if i != j and abs(sigma_mat[i, j]) > 1e-9:
                is_diagonal = False
                print(f"❌ Found off-diagonal non-zero at ({i}, {j}): {sigma_mat[i,j]}")
                break
        if not is_diagonal: break
    
    if is_diagonal:
        print("✅ Matrix structure is Diagonal (Correct for isotropic prior).")
    else:
        print("❌ Matrix structure is NOT Diagonal (Likely improper initialization logic).")
        print("   This confirms the hypothesis that _construct_sigma_topology logic is flawed.")

if __name__ == "__main__":
    check_sigma_initialization()
