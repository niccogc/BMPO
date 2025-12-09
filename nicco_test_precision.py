import torch
import numpy as np
import quimb.tensor as qt
from tensor.builder import Inputs
from tensor.btn import BTN

# Set precision
torch.set_default_dtype(torch.float64)

def debug_precision_computation():
    print("\n" + "="*60)
    print("DEBUGGING PRECISION MATRIX COMPUTATION")
    print("="*60)

    # ---------------------------------------------------------
    # 1. SETUP: Small 2-Node Network
    # ---------------------------------------------------------
    # Structure: Node1(x, b1) -- Node2(b1, y)
    # We focus on Node2 which has bond 'b1' and output 'y'.
    
    B_sz = 2
    D_b1 = 2
    D_out = 1 # Scalar output for simplicity
    
    # Random Inputs and Targets
    x = torch.randn(B_sz, 2)
    y_target = torch.randn(B_sz, D_out)
    
    loader = Inputs(
        inputs=[x], outputs=[y_target],
        outputs_labels=["y"], input_labels=["x"],
        batch_dim="s", batch_size=B_sz
    )
    
    # Create Tensors
    t1 = qt.Tensor(torch.randn(2, D_b1), inds=('x', 'b1'), tags={'Node1'})
    t2 = qt.Tensor(torch.randn(D_b1, D_out), inds=('b1', 'y'), tags={'Node2'})
    
    mu_tn = qt.TensorNetwork([t1, t2])
    btn = BTN(mu=mu_tn, data_stream=loader, batch_dim="s")
    
    print("Network: Node1(x,b1) -- Node2(b1,y)")
    
    # Force some values for distributions to be deterministic
    # q_tau: E[tau] = 1.0
    btn.q_tau.concentration=torch.tensor(2.0)
    btn.q_tau.rate=torch.tensor(2.0)
    tau_mean = btn.get_tau_mean().item()
    print(f"E[tau] = {tau_mean}")

    # q_bonds: b1 E[lambda] = 1.0
    btn.q_bonds['b1'].concentration.modify(data=torch.ones(D_b1))
    btn.q_bonds['b1'].rate.modify(data=torch.ones(D_b1))
    
    # ---------------------------------------------------------
    # 2. COMPONENT 1: THETA (Prior Precision)
    # ---------------------------------------------------------
    print("\n>>> Checking THETA (Prior)")
    
    # Node2 has indices ('b1', 'y'). 'y' is output, so excluded from Theta.
    # Theta should be diagonal matrix of E[lambda_b1].
    # In tensor form: vector (b1).
    # But precision matrix calculation expands it to diagonal (b1, b1_prime).
    
    # BTN computation
    theta_btn = btn.theta_block_computation('Node2') # Shape (b1,)
    print(f"Theta raw shape: {theta_btn.shape}")
    
    # Manual Theta
    # It is just E[lambda_b1]
    theta_manual = btn.q_bonds['b1'].mean().data
    
    if torch.allclose(theta_btn.data, theta_manual):
        print("✅ Theta raw matches.")
    else:
        print("❌ Theta raw mismatch.")

    # ---------------------------------------------------------
    # 3. COMPONENT 2: MU OUTER ENV (Data Term 1)
    # ---------------------------------------------------------
    print("\n>>> Checking MU OUTER ENV (Sum_n (T_i mu x_n) (x) (T_i mu x_n))")
    
    # This term corresponds to: sum_n Env_mu(n) * Env_mu(n)^T
    # Env for Node2 is contraction of Input(x) and Node1(x, b1).
    # Resulting Env indices: (s, b1).
    
    # A. BTN Calculation
    mu_env_outer_btn = btn.outer_operation(
        input_generator=btn.data.data_mu,
        tn=btn.mu,
        node_tag='Node2',
        sum_over_batches=True
    )
    # This returns shape (b1, b1_prime) ? Or (b1, b1)? 
    # Let's check indices.
    print(f"BTN Mu Outer Inds: {mu_env_outer_btn.inds}")
    mu_outer_data = torch.as_tensor(mu_env_outer_btn.data)
    
    # B. Manual Calculation
    # 1. Get raw input x
    x_data = x # (B, 2)
    # 2. Get Node1
    w1 = t1.data # (2, D_b1)
    # 3. Contract to get Env(Node2): x @ w1 -> (B, D_b1)
    env_mu_manual = torch.einsum('si, ib -> sb', x_data, w1)
    
    # 4. Outer product summed over batch
    # sum_s ( env_s^T @ env_s ) -> (b, b)
    mu_outer_manual = torch.einsum('sb, sc -> bc', env_mu_manual, env_mu_manual)
    
    # Compare
    # Note: BTN might return (b1, b1_prime) or (b1, b1). 
    # Since w1/x are random, this matrix should be symmetric positive semi-definite.
    
    print(f"Manual Shape: {tuple(mu_outer_manual.shape)}")
    print(f"BTN Shape:    {tuple(mu_outer_data.shape)}")
    
    if torch.allclose(mu_outer_data, mu_outer_manual, atol=1e-12):
        print("✅ Mu Outer Env matches.")
    else:
        print("❌ Mu Outer Env mismatch.")
        print(f"Diff: {(mu_outer_data - mu_outer_manual).abs().max()}")

    # ---------------------------------------------------------
    # 4. COMPONENT 3: SIGMA ENV (Data Term 2)
    # ---------------------------------------------------------
    print("\n>>> Checking SIGMA ENV (Sum_n (T_i Sigma x_n))")
    
    # This is the tricky part.
    # It corresponds to: sum_n T_i(Sigma) * (x_n (x) x_n)
    # Basically, propagating the input noise/correlation through the rest of the network.
    
    # The Sigma network has doubled indices.
    # Node1_Sigma shape: (x, x_prime, b1, b1_prime).
    # Input to Sigma network is x (x) x.
    
    # A. BTN Calculation
    # get_environment for Sigma network, targeting Node2.
    # sum_over_batch=True, sum_over_output=True.
    sigma_env_btn = btn.get_environment(
        tn=btn.sigma,
        target_tag='Node2',
        input_generator=btn.data.data_sigma,
        copy=False,
        sum_over_batch=True,
        sum_over_output=True # Output 'y' is removed
    )
    sigma_env_data = torch.as_tensor(sigma_env_btn.data)
    print(f"BTN Sigma Env Inds: {sigma_env_btn.inds}")
    
    # B. Manual Calculation
    # 1. Construct Node1 Sigma
    # We didn't explicitly set Sigma values in BTN init (it uses defaults or builder).
    # Let's inspect what's in btn.sigma['Node1'].
    sigma_node1 = btn.sigma['Node1']
    # Indices likely: (x, b1, x_prime, b1_prime) or similar.
    # We need to be sure about the order.
    # Let's just grab the tensor data and reshape carefully if needed.
    
    # Standard builder initialization for Sigma might be diagonal or random.
    # But here we need to know exactly what it is to verify contraction.
    # Let's assume standard contraction logic holds:
    # Input_Sigma = x_s (outer) x_s (inner) -> shape (B, x, x) effectively.
    # Contract with Node1_Sigma.
    
    # Input Sigma for batch s:
    # In BTN.inputs, data_sigma returns mu + prime tensors.
    # They are contracted with the network.
    # Effectively it computes:
    # sum_s  Input(s, x) * Input(s, x_prime) * Node1_Sigma(x, x_prime, b1, b1_prime)
    # -> Result(b1, b1_prime)
    
    # Let's replicate this.
    # 1. Input outer prod for each sample: x_s outer x_s
    input_outer = torch.einsum('si, sj -> sij', x_data, x_data) # (s, x, x)
    
    # 2. Get Node1 Sigma Data
    # Indices are strings. We need to map them.
    # Expected: x, b1, x_prime, b1_prime
    # Let's check:
    s1_inds = sigma_node1.inds
    s1_data = sigma_node1.data
    
    # Construct manual contraction based on index names
    # Map: x->i, x_prime->j, b1->a, b1_prime->b
    # Input: sij
    # Node1: inds map to letters
    # Output: ab (sum over s, i, j)
    
    # Create mapping
    ind_map = {'x': 'i', 'x_prime': 'j', 'b1': 'a', 'b1_prime': 'b'}
    ein_str_node = "".join([ind_map.get(k, '?') for k in s1_inds])
    
    full_ein = f"sij,{ein_str_node}->ab"
    
    sigma_env_manual = torch.einsum(full_ein, input_outer, s1_data)
    
    print(f"Manual Shape: {tuple(sigma_env_manual.shape)}")
    print(f"BTN Shape:    {tuple(sigma_env_data.shape)}")
    
    # Align if necessary
    # BTN output inds: likely (b1, b1_prime) or (b1_prime, b1)
    btn_inds = sigma_env_btn.inds
    perm = []
    # Map BTN inds to our manual dims (0->a/b1, 1->b/b1_prime)
    target_map = {'b1': 0, 'b1_prime': 1}
    try:
        perm = [target_map[k] for k in btn_inds]
        # BUT: our manual result is (a, b) which corresponds to (b1, b1_prime).
        # We need to permute MANUAL result to match BTN order if needed?
        # Actually we permute manual to match BTN is easier conceptual check.
        # Wait, if BTN is (b1_prime, b1), we want to compare with manual(b, a).
        
        # Let's map output indices to 0, 1
        # manual indices are ordered: 0->b1, 1->b1_prime
        
        # If BTN has (b1, b1_prime), perm is [0, 1].
        # If BTN has (b1_prime, b1), perm is [1, 0].
        
        # We want to transform manual (0, 1) -> BTN order.
        # No, we want to transform BTN data to match Manual (0, 1).
        # If btn is [1, 0], we permute it by [1, 0] to get back to [0, 1].
        pass
    except:
        pass

    # Simple check: try direct and transpose
    match = False
    if torch.allclose(sigma_env_data, sigma_env_manual, atol=1e-12):
        print("✅ Sigma Env matches (direct).")
        match = True
    elif torch.allclose(sigma_env_data, sigma_env_manual.T, atol=1e-12):
        print("✅ Sigma Env matches (transposed).")
        match = True
    else:
        print("❌ Sigma Env mismatch.")
        print(f"Diff: {(sigma_env_data - sigma_env_manual).abs().max()}")

    # ---------------------------------------------------------
    # 5. TOTAL PRECISION & POSITIVITY
    # ---------------------------------------------------------
    print("\n>>> Checking TOTAL PRECISION")
    
    # Precision = E[tau] * (Mu_Outer + Sigma_Env) + Theta_Diagonal
    
    # Reconstruct from BTN components (assuming alignment is handled by BTN logic)
    # We rely on the fact that BTN sums them, so they must be aligned internally.
    
    # We compute it manually using the verified manual components
    precision_manual = tau_mean * (mu_outer_manual + sigma_env_manual)
    
    # Add Theta to diagonal
    # theta_manual is vector (b1). precision is (b1, b1_prime).
    # We add to diagonal where b1 == b1_prime.
    # (Assuming indices 0 and 1 correspond to b1 and b1_prime)
    precision_manual.diagonal().add_(theta_manual)
    
    # Check Positive Definiteness
    try:
        L = torch.linalg.cholesky(precision_manual)
        print("✅ Precision Matrix is Positive Definite.")
    except Exception as e:
        print("❌ Precision Matrix is NOT Positive Definite!")
        # Debug Eigenvalues
        eigs = torch.linalg.eigvalsh(precision_manual)
        print(f"   Min Eig: {eigs.min().item()}")
        print(f"   Max Eig: {eigs.max().item()}")
        
        print("\n   Contributing Terms Eigenvalues:")
        print(f"   Mu Outer Min Eig: {torch.linalg.eigvalsh(mu_outer_manual).min().item()}")
        print(f"   Sigma Env Min Eig: {torch.linalg.eigvalsh(sigma_env_manual).min().item()}")
        
        # If Sigma Env is negative, that's the culprit.
        # Sigma network represents Covariance, should be PSD.
        # If initialized randomly without care, it might not be PSD.

if __name__ == "__main__":
    debug_precision_computation()
