import numpy as np
from tensor.builder import Inputs 

# ==========================================
# SETUP: Generate Fake Data (100 samples)
# ==========================================
N = 100
x1 = np.random.randn(N, 4)
x2 = np.random.randn(N, 4)
y_simple = np.random.randn(N, 1) # Standard vector target

print(">>> RUNNING INPUT TESTS\n")

# ==========================================
# TEST 1: Standard (2 Inputs -> 2 Labels)
# ==========================================
print("--- Test 1: Standard Inputs ---")
loader_std = Inputs(
    inputs=[x1, x2], 
    outputs=[y_simple], 
    outputs_labels=["ly"], 
    input_labels=["x1", "x2"], 
    batch_size=32
)
print(loader_std)

# ==========================================
# TEST 2: Repeated (1 Input -> 3 Labels)
# ==========================================
print("\n--- Test 2: Repeated Inputs (1 Data -> 3 Nodes) ---")
x_all = np.random.randn(N, 4) # Single input source

loader_rep = Inputs(
    inputs=[x_all],   # List len 1 -> Repeated Mode
    outputs=[y_simple], 
    outputs_labels=["ly"], 
    input_labels=["a", "b", "c"], # Distribute to these 3 nodes
    batch_size=32
)
print(loader_rep)

# ==========================================
# TEST 3: Multidimensional Y + Multi-Labels
# ==========================================
print("\n--- Test 3: Complex Y (Matrix Output + 2 Labels) ---")

# Target is now a 3x5 matrix for every sample (Shape: 100, 3, 5)
y_complex = np.random.randn(N, 3, 5)

loader_complex = Inputs(
    inputs=[x1, x2], 
    outputs=[y_complex], 
    # Two labels for the two extra dimensions of Y
    outputs_labels=["row_idx", "col_idx"], 
    input_labels=["x1", "x2"], 
    batch_size=32
)
print(loader_complex)
