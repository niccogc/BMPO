# pyright: ignore
import quimb.tensor as qt
import numpy as np

# --- 1. Setup Data ---
S_dim = 10  # Sample dimension (s)
Y_dim = 3   # Output dimension (y1)

# Create two random tensors with the same dimensions and indices
data_A = np.random.rand(S_dim, Y_dim)
data_B = np.random.rand(S_dim, Y_dim)
data_C = np.random.rand(S_dim, Y_dim)

T_A = qt.Tensor(data_A, inds=('s', 'y1'), tags={'A'})
T_B = qt.Tensor(data_B, inds=('s', 'y1'), tags={'B'})
T_C = qt.Tensor(data_C, inds=('s', 'y1'), tags={'C'})

# --- 2. Operation A: Tensor Network Contraction ---
# Contract the network, keeping only 'y1' (sums over 's')
tn = T_A & T_B & T_C
result_A = tn.contract(output_inds=['y1'])

# --- 3. Operation B: Manual Element-wise Product and Sum ---
# Step 1: Element-wise product (Hadamard product)
intermediate_C = data_A * data_B * data_C

# Step 2: Sum over the 's' axis (axis=0)
result_B = intermediate_C.sum(axis=0)

# --- 4. Verification ---
# Check if the results are numerically close
match = np.allclose(result_A.data, result_B)

print(f"Result A (TN Contraction): {result_A.data}")
print(f"Result B (Manual Array Op): {result_B}")
print(f"Do the results match? {match}")
