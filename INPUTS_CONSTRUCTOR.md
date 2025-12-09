## `Inputs` Class Reference

Handles batching, labeling, and tensor construction for Mu/Sigma networks.

### Constructor Arguments
* **`inputs`**: `List[Array]`. Source data for input nodes. Shape: `(N_samples, Features)`.
* **`outputs`**: `List[Array]`. Target data. Shape: `(N_samples, ...)`.
* **`input_labels`**: `List[str]`. Unique identifiers for input nodes (e.g., `['x1', 'x2']`).
* **`outputs_labels`**: `List[str]`. Identifiers for target dimensions excluding batch (e.g., `['label']` or `['row', 'col']`).
* **`batch_size`**: `int`. Number of samples per iteration.
* **`batch_dim`**: `str` (Default: `"s"`). Label for the batch index.

### Operating Modes
1.  **Standard Mapping:** `len(inputs) == len(input_labels)`.
    * Each input array is mapped 1:1 to its corresponding label.
2.  **Broadcast/Repeated:** `len(inputs) == 1` AND `len(input_labels) > 1`.
    * The single input array is duplicated for every label provided in `input_labels`.

### Generator Output
Iterating over the instance yields a tuple `(mu, sigma, y)` per batch:

1.  **`mu`**: `List[qt.Tensor]`
    * Contains tensors for the Mean network.
    * Indices: `(batch_dim, label)`.
2.  **`sigma`**: `List[qt.Tensor]`
    * Contains tensors for the Variance network (Inputs are doubled).
    * Indices: `(batch_dim, label)` AND `(batch_dim, label_prime)`.
3.  **`y`**: `qt.Tensor`
    * Target tensor.
    * Indices: `(batch_dim, *outputs_labels)`.

### Example

```py
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

```

```bash
python test_inputs_constructor.py
```
>>> RUNNING INPUT TESTS

--- Test 1: Standard Inputs ---
It is assumed that the Inputs and outputs in list are ordered as labels

>>> InputLoader Summary (Batch Size: 32, Samples: 100, Batches Number: 4)
TYPE     | SHAPE           | INDICES
------------------------------------------------------------
Target   | (32, 1)         | ('s', 'ly')
Mu       | (32, 4)         | [['s', 'x1'], ['s', 'x2']] ... (2 tensors)
Sigma    | (32, 4)         | [['s', 'x1'], ['s', 'x2'], ['s', 'x1_prime'], ['s', 'x2_prime']] ... (4 tensors)

--- Test 2: Repeated Inputs (1 Data -> 3 Nodes) ---
It is assumed that the Inputs and outputs in list are ordered as labels

>>> InputLoader Summary (Batch Size: 32, Samples: 100, Batches Number: 4)
TYPE     | SHAPE           | INDICES
------------------------------------------------------------
Target   | (32, 1)         | ('s', 'ly')
Mu       | (32, 4)         | [['s', 'a'], ['s', 'b'], ['s', 'c']] ... (3 tensors)
Sigma    | (32, 4)         | [['s', 'a'], ['s', 'b'], ['s', 'c'], ['s', 'a_prime'], ['s', 'b_prime'], ['s', 'c_prime']] ... (6 tensors)

--- Test 3: Complex Y (Matrix Output + 2 Labels) ---
It is assumed that the Inputs and outputs in list are ordered as labels

>>> InputLoader Summary (Batch Size: 32, Samples: 100, Batches Number: 4)
TYPE     | SHAPE           | INDICES
------------------------------------------------------------
Target   | (32, 3, 5)      | ('s', 'row_idx', 'col_idx')
Mu       | (32, 4)         | [['s', 'x1'], ['s', 'x2']] ... (2 tensors)
Sigma    | (32, 4)         | [['s', 'x1'], ['s', 'x2'], ['s', 'x1_prime'], ['s', 'x2_prime']] ... (4 tensors)
