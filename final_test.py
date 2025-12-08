import torch
import numpy as np
import matplotlib.pyplot as plt
import quimb.tensor as qt
from tensor.builder import Inputs
from tensor.btn import BTN

# Stability settings
torch.set_default_dtype(torch.float64)
np.random.seed(42)
torch.manual_seed(42)

# --- 1. Helper: Weight Initialization ---
def init_weights(shape):
    # Initialize small random weights
    w = torch.randn(*shape)
    return w / torch.norm(w)

# --- 2. Model Builders (Quimb Definitions) ---

def build_ring_mps(L, D_phys, D_bond):
    """
    Periodic MPS (Ring). 
    The last tensor connects back to the first via bond 'b_{L-1}'.
    Output 'y' is attached to the last site.
    """
    tensors = []
    for i in range(L):
        # Indices: Left Bond, Input, Right Bond
        # Ring closure: Bond -1 is Bond L-1
        b_left = f'b{i-1}' if i > 0 else f'b{L-1}'
        b_right = f'b{i}'
        inds = [b_left, f'x{i}', b_right]
        
        # Add output 'y' to the last tensor
        shape = [D_bond, D_phys, D_bond]
        if i == L - 1:
            inds.append('y')
            shape.append(1) # Scalar output
            
        tensors.append(qt.Tensor(
            data=init_weights(tuple(shape)),
            inds=tuple(inds),
            tags={f'Node{i}', 'Ring'}
        ))
    return qt.TensorNetwork(tensors)

def build_binary_tree(L, D_phys, D_bond):
    """
    Binary Tree Network. (Requires L to be power of 2).
    Leaves take inputs, merge upwards to a Root which outputs 'y'.
    """
    assert (L & (L - 1) == 0), "L must be power of 2 for Binary Tree"
    tensors = []
    
    # -- Layer 0: Leaves (Input -> Bond) --
    # Shape: (Input, Bond_Up)
    for i in range(L):
        tensors.append(qt.Tensor(
            data=init_weights((D_phys, D_bond)),
            inds=(f'x{i}', f'bond_0_{i}'),
            tags={f'Leaf{i}', 'Layer0'}
        ))

    # -- Intermediate Layers --
    # Merges 2 lower bonds -> 1 upper bond
    n_nodes = L
    layer = 0
    while n_nodes > 1:
        next_nodes = n_nodes // 2
        for i in range(next_nodes):
            b_left = f'bond_{layer}_{2*i}'
            b_right = f'bond_{layer}_{2*i+1}'
            
            # If this is the absolute top node, add 'y'
            if next_nodes == 1:
                b_out = 'y' # Output index
                # Shape: (Left, Right, Output) - No 'Up' bond, just Y
                shape = (D_bond, D_bond, 1)
                inds = (b_left, b_right, b_out)
            else:
                b_out = f'bond_{layer+1}_{i}'
                shape = (D_bond, D_bond, D_bond)
                inds = (b_left, b_right, b_out)
            
            tensors.append(qt.Tensor(
                data=init_weights(shape),
                inds=inds,
                tags={f'Node_{layer+1}_{i}', f'Layer{layer+1}'}
            ))
        n_nodes = next_nodes
        layer += 1
        
    return qt.TensorNetwork(tensors)

def build_peps_grid(Lx, Ly, D_phys, D_bond):
    """
    2D PEPS Grid.
    Sites connected N, S, E, W. 
    Output 'y' attached to bottom-right corner (Lx-1, Ly-1).
    """
    tensors = []
    for y in range(Ly):
        for x in range(Lx):
            # Standard Indices
            inds = [f'x{y*Lx + x}'] # Input
            shape = [D_phys]
            
            # Connectivity (Grid)
            # Right Bond (h_{x},{y})
            if x < Lx - 1:
                inds.append(f'h_{x}_{y}')
                shape.append(D_bond)
            # Left Bond (h_{x-1},{y})
            if x > 0:
                inds.append(f'h_{x-1}_{y}')
                shape.append(D_bond)
            # Down Bond (v_{x},{y})
            if y < Ly - 1:
                inds.append(f'v_{x}_{y}')
                shape.append(D_bond)
            # Up Bond (v_{x},{y-1})
            if y > 0:
                inds.append(f'v_{x}_{y-1}')
                shape.append(D_bond)
            
            # Output on last site
            if x == Lx - 1 and y == Ly - 1:
                inds.append('y')
                shape.append(1)
            
            tensors.append(qt.Tensor(
                data=init_weights(tuple(shape)),
                inds=tuple(inds),
                tags={f'Site_{x}_{y}', 'PEPS'}
            ))
            
    return qt.TensorNetwork(tensors)

# --- 3. Main Experiment ---

def run_comparison():
    print("\n" + "="*60)
    print("COMPARISON: RING vs TREE vs PEPS")
    print("="*60)

    # A. Data Generation (L=4 for square PEPS and Tree)
    L = 4 
    N_SAMPLES = 2000
    BATCH_SIZE = 200
    EPOCHS = 15

    # Simple target: Sum of cubes
    x_raw = 2 * torch.rand(N_SAMPLES, 1) - 1
    y_raw = 2 * (x_raw**3) - (x_raw**2) + 0.5 * x_raw
    y_raw += 0.05 * torch.randn_like(y_raw) # Noise

    # Features [x, 1]
    x_features = torch.cat([x_raw, torch.ones_like(x_raw)], dim=1)
    
    # Loader setup (L inputs)
    input_labels = [f"x1"]
    loader = Inputs(
        inputs=[x_features], 
        outputs=[y_raw],
        outputs_labels=["y"],
        input_labels=input_labels, # The loader repeats x_features for x0..x3
        batch_dim="s",
        batch_size=BATCH_SIZE
    )

    # B. Model Definitions
    D_bond = 4
    D_phys = 2
    
    models = {
        "Ring MPS": build_ring_mps(L, D_phys, D_bond),
        
        "Binary Tree": build_binary_tree(L, D_phys, D_bond),
        
        "PEPS (2x2)": build_peps_grid(2, 2, D_phys, D_bond)
    }

    history = {}

    # C. Training Loop
    for name, tn in models.items():
        print(f"\nTraining {name}...")
        
        # Initialize your BTN
        model = BTN(
            mu=tn, 
            data_stream=loader, 
            batch_dim="s",
            method='cholesky', # or 'qr' / 'svd' depending on your implementation
        )

        loss_curve = []
        
        # Train epoch by epoch to capture loss history
        for ep in range(EPOCHS):
            model.fit(epochs=1)
            # Calculate MSE on full dataset (or batch)
            mse = model._calc_mu_mse().item() / N_SAMPLES
            loss_curve.append(mse)
            print(f"  > Epoch {ep+1}: MSE = {mse:.5f}")
        
        history[name] = loss_curve

    # D. Visualization
    print("\nPlotting results...")
    plt.figure(figsize=(10, 6))
    for name, losses in history.items():
        plt.plot(losses, label=name, linewidth=2, marker='o', markersize=4)
    
    plt.title(f"Model Convergence (L={L}, Bond={D_bond})")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_comparison()
