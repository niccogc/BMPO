"""
Test Bayesian Tensor Network on different topologies: MPS, Ring, Binary Tree.

All three topologies train on the same 3rd degree polynomial: y = 2x³ - x² + 0.5
This demonstrates the generality of the BayesianTensorNetwork implementation.
"""

import torch
import quimb.tensor as qtn
import numpy as np
import matplotlib.pyplot as plt

from tensor.bayesian_tn import BayesianTensorNetwork
from tensor.bayesian_tn_builder import create_sigma_network


def create_mps_network(bond_dim, seed=42):
    """
    Create MPS: Block0(p0, r1) - Block1(r1, p1, r2) - Block2(r2, p2, y)
    """
    torch.manual_seed(seed)

    # Why dont you make of general data type?
    block0_data = torch.randn(2, bond_dim, dtype=torch.float64) * 0.1
    block1_data = torch.randn(bond_dim, 2, bond_dim, dtype=torch.float64) * 0.1
    block2_data = torch.randn(bond_dim, 2, 1, dtype=torch.float64) * 0.1
    
    Block0 = qtn.Tensor(data=block0_data, inds=('p0', 'r1'), tags='Block0')
    Block1 = qtn.Tensor(data=block1_data, inds=('r1', 'p1', 'r2'), tags='Block1')
    Block2 = qtn.Tensor(data=block2_data, inds=('r2', 'p2', 'y'), tags='Block2')
    
    mu_tn = qtn.TensorNetwork([Block0, Block1, Block2])
    learnable_tags = ['Block0', 'Block1', 'Block2']
    
    print("MPS Structure:")
    print("  Block0(p0, r1) - Block1(r1, p1, r2) - Block2(r2, p2, y)")
    for i, tag in enumerate(learnable_tags):
        node = mu_tn[tag]
        print(f"  {tag}: {node.shape} -> {node.inds}")
    
    return mu_tn, learnable_tags


def create_ring_network(bond_dim, seed=42):
    """
    Create Ring: Block0(p0, r1, r3) - Block1(r1, p1, r2) - Block2(r2, p2, r3, y)
    Forms a cycle: Block0 -- Block1 -- Block2 -- Block0
    """
    torch.manual_seed(seed)
    
    # Ring structure with 3 nodes
    block0_data = torch.randn(2, bond_dim, bond_dim, dtype=torch.float64) * 0.1  # (p0, r1, r3)
    block1_data = torch.randn(bond_dim, 2, bond_dim, dtype=torch.float64) * 0.1  # (r1, p1, r2)
    block2_data = torch.randn(bond_dim, 2, bond_dim, 1, dtype=torch.float64) * 0.1  # (r2, p2, r3, y)
    
    Block0 = qtn.Tensor(data=block0_data, inds=('p0', 'r1', 'r3'), tags='Block0')
    Block1 = qtn.Tensor(data=block1_data, inds=('r1', 'p1', 'r2'), tags='Block1')
    Block2 = qtn.Tensor(data=block2_data, inds=('r2', 'p2', 'r3', 'y'), tags='Block2')
    
    mu_tn = qtn.TensorNetwork([Block0, Block1, Block2])
    learnable_tags = ['Block0', 'Block1', 'Block2']
    
    print("Ring Structure:")
    print("  Block0(p0, r1, r3) - Block1(r1, p1, r2) - Block2(r2, p2, r3, y)")
    print("  Forms cycle: 0 -> 1 -> 2 -> 0")
    for i, tag in enumerate(learnable_tags):
        node = mu_tn[tag]
        print(f"  {tag}: {node.shape} -> {node.inds}")
    
    return mu_tn, learnable_tags


def create_tree_network(bond_dim, seed=42):
    """
    Create Binary Tree with 5 nodes:
    
               Node4(y, r4, r3)
                  /         \\
            [r4]             [r3]
              /                 \\
       Node3(r4, r1, r2)    Node2(r3, p3)
           /        \\
        [r1]        [r2]
         /            \\
    Node0(r1,p1)  Node1(r2,p2)
    
    Proper binary tree structure with 5 nodes, 3 inputs.
    """
    torch.manual_seed(seed)
    
    # Leaf nodes (3 leaves, each with input)
    node0_data = torch.randn(bond_dim, 2, dtype=torch.float64) * 0.1  # (r1, p1)
    node1_data = torch.randn(bond_dim, 2, dtype=torch.float64) * 0.1  # (r2, p2)
    node2_data = torch.randn(bond_dim, 2, dtype=torch.float64) * 0.1  # (r3, p3)
    
    # Internal nodes
    node3_data = torch.randn(bond_dim, bond_dim, bond_dim, dtype=torch.float64) * 0.1  # (r4, r1, r2)
    node4_data = torch.randn(1, bond_dim, bond_dim, dtype=torch.float64) * 0.1  # (y, r4, r3)
    
    Node0 = qtn.Tensor(data=node0_data, inds=('r1', 'p1'), tags='Node0')
    Node1 = qtn.Tensor(data=node1_data, inds=('r2', 'p2'), tags='Node1')
    Node2 = qtn.Tensor(data=node2_data, inds=('r3', 'p3'), tags='Node2')
    Node3 = qtn.Tensor(data=node3_data, inds=('r4', 'r1', 'r2'), tags='Node3')
    Node4 = qtn.Tensor(data=node4_data, inds=('y', 'r4', 'r3'), tags='Node4')
    
    mu_tn = qtn.TensorNetwork([Node0, Node1, Node2, Node3, Node4])
    learnable_tags = ['Node0', 'Node1', 'Node2', 'Node3', 'Node4']
    
    print("Binary Tree Structure (5 nodes):")
    print("           Node4(y, r4, r3)")
    print("              /         \\")
    print("           [r4]         [r3]")
    print("            /               \\")
    print("   Node3(r4,r1,r2)      Node2(r3,p3)")
    print("       /      \\")
    print("     [r1]    [r2]")
    print("      /        \\")
    print("Node0(r1,p1) Node1(r2,p2)")
    for i, tag in enumerate(learnable_tags):
        node = mu_tn[tag]
        print(f"  {tag}: {node.shape} -> {node.inds}")
    
    return mu_tn, learnable_tags


def train_and_evaluate(topology_name, mu_tn, learnable_tags, X, y, X_test, y_test_true, 
                       max_iter=40, verbose=False):
    """Train model on given topology and evaluate."""
    print(f"\n{'='*70}")
    print(f"TRAINING: {topology_name}")
    print(f"{'='*70}")
    
    # Create sigma network
    sigma_tn = create_sigma_network(mu_tn, learnable_tags, output_indices=['y'])
    
    # Determine which input indices exist in the network
    all_indices = set()
    for tensor in mu_tn.tensor_map.values():
        all_indices.update(tensor.inds)  # type: ignore
    
    # Map features to physical indices (p0, p1, p2) that exist
    input_indices = {}
    for idx in ['p0', 'p1', 'p2']:
        if idx in all_indices:
            input_indices['features'] = input_indices.get('features', []) + [idx]
    
    print(f"Input mapping: {input_indices}")
    
    # Determine which input indices exist in the network
    all_indices = set()
    for tensor in mu_tn.tensor_map.values():
        all_indices.update(tensor.inds)  # type: ignore
    
    # Map features to physical indices that exist (p1, p2, p3, etc.)
    input_indices = {}
    for idx in ['p0', 'p1', 'p2', 'p3']:
        if idx in all_indices:
            input_indices['features'] = input_indices.get('features', []) + [idx]
    
    print(f"Input mapping: {input_indices}")
    
    # Create model
    model = BayesianTensorNetwork(
        mu_tn=mu_tn,
        sigma_tn=sigma_tn,
        input_indices=input_indices,
        output_indices=['y'],
        learnable_tags=learnable_tags,
        tau_alpha=torch.tensor(2.0, dtype=torch.float64),
        tau_beta=torch.tensor(1.0, dtype=torch.float64),
        dtype=torch.float64
    )
    
    print(f"Model: {len(learnable_tags)} learnable nodes")
    print(f"Bonds: {model.mu_network.bond_labels}")
    print()
    
    # Train
    inputs_dict = {'features': X}
    history = model.fit(
        X=None,
        y=y,
        inputs_dict=inputs_dict,
        max_iter=max_iter,
        verbose=verbose
    )
    
    # Evaluate
    inputs_test = {'features': X_test}
    mu_pred = model.forward_mu(inputs_test)
    sigma_pred = model.forward_sigma(inputs_test)
    
    tau_mean = model.get_tau_mean().item()
    aleatoric_var = 1.0 / tau_mean
    total_var = sigma_pred + aleatoric_var
    total_std = torch.sqrt(total_var)
    epistemic_std = torch.sqrt(sigma_pred)
    
    # Metrics
    residuals = mu_pred - y_test_true
    rmse = torch.sqrt((residuals**2).mean()).item()
    ss_res = (residuals**2).sum().item()
    ss_tot = ((y_test_true - y_test_true.mean())**2).sum().item()
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"\nResults:")
    print(f"  RMSE: {rmse:.5f}")
    print(f"  R²: {r2:.5f}")
    print(f"  E[τ]: {tau_mean:.4f} (noise std: {1/np.sqrt(tau_mean):.4f})")
    print(f"  Epistemic std: {epistemic_std.mean():.5f}")
    print(f"  Total std: {total_std.mean():.5f}")
    
    return {
        'model': model,
        'history': history,
        'mu_pred': mu_pred,
        'total_std': total_std,
        'epistemic_std': epistemic_std,
        'rmse': rmse,
        'r2': r2,
        'tau_mean': tau_mean
    }


def test_all_topologies():
    """Test MPS, Ring, and Binary Tree on the same polynomial."""
    print("\n" + "="*70)
    print("BAYESIAN TENSOR NETWORK: TOPOLOGY COMPARISON")
    print("="*70)
    print("Polynomial: y = 2x³ - x² + 0.5")
    print("="*70)
    
    # Hyperparameters
    num_samples = 100
    noise_std = 0.1
    bond_dim = 6
    max_iter = 40
    seed = 42
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"\nHyperparameters:")
    print(f"  Samples: {num_samples}, Noise: {noise_std}")
    print(f"  Bond dim: {bond_dim}, Iterations: {max_iter}")
    print()
    
    # Generate data
    x = torch.rand(num_samples, dtype=torch.float64) * 2 - 1
    y_true = 2.0 * x**3 - 1.0 * x**2 + 0.5
    y = y_true + noise_std * torch.randn(num_samples, dtype=torch.float64)
    X = torch.stack([torch.ones_like(x), x], dim=1)
    
    x_test = torch.linspace(-1.2, 1.2, 200, dtype=torch.float64)
    y_test_true = 2.0 * x_test**3 - 1.0 * x_test**2 + 0.5
    X_test = torch.stack([torch.ones_like(x_test), x_test], dim=1)
    
    # Test each topology
    results = {}
    
    # 1. MPS
    print("\n" + "="*70)
    print("TOPOLOGY 1: Matrix Product State (MPS)")
    print("="*70)
    mu_tn, learnable_tags = create_mps_network(bond_dim, seed)
    results['MPS'] = train_and_evaluate(
        'MPS', mu_tn, learnable_tags, X, y, X_test, y_test_true, 
        max_iter=max_iter, verbose=True
    )
    
    # 2. Ring
    print("\n" + "="*70)
    print("TOPOLOGY 2: Ring (Cyclic)")
    print("="*70)
    mu_tn, learnable_tags = create_ring_network(bond_dim, seed)
    results['Ring'] = train_and_evaluate(
        'Ring', mu_tn, learnable_tags, X, y, X_test, y_test_true, 
        max_iter=max_iter, verbose=True
    )
    
    # 3. Binary Tree
    print("\n" + "="*70)
    print("TOPOLOGY 3: Binary Tree")
    print("="*70)
    mu_tn, learnable_tags = create_tree_network(bond_dim, seed)
    results['Tree'] = train_and_evaluate(
        'Tree', mu_tn, learnable_tags, X, y, X_test, y_test_true, 
        max_iter=max_iter, verbose=True
    )
    
    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    print(f"{'Topology':<15} {'R²':<10} {'RMSE':<10} {'E[τ]':<10}")
    print("-"*70)
    for name, res in results.items():
        print(f"{name:<15} {res['r2']:<10.5f} {res['rmse']:<10.5f} {res['tau_mean']:<10.4f}")
    print("="*70)
    
    # Plot comparison
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, (name, res) in enumerate(results.items()):
            ax = axes[idx]
            
            # Scatter training data
            ax.scatter(x.numpy(), y.numpy(), alpha=0.5, s=20, c='gray', label='Training data')
            
            # True function
            ax.plot(x_test.numpy(), y_test_true.numpy(), 'k-', linewidth=2, label='True')
            
            # Predictions
            mu_pred = res['mu_pred']
            total_std = res['total_std']
            epistemic_std = res['epistemic_std']
            
            ax.plot(x_test.numpy(), mu_pred.numpy(), 'r-', linewidth=2, label='Predicted')
            ax.fill_between(x_test.numpy(),
                            (mu_pred - 2*total_std).numpy(),
                            (mu_pred + 2*total_std).numpy(),
                            alpha=0.3, color='red', label='±2σ total')
            ax.fill_between(x_test.numpy(),
                            (mu_pred - 2*epistemic_std).numpy(),
                            (mu_pred + 2*epistemic_std).numpy(),
                            alpha=0.5, color='blue', label='±2σ epistemic')
            
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'{name} (R²={res["r2"]:.3f})', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = 'test_polynomial_topologies_comparison.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved: {filename}")
    except Exception as e:
        print(f"Plotting failed: {e}")
    
    # Verify all work well
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    for name, res in results.items():
        r2_ok = res['r2'] > 0.85
        rmse_ok = res['rmse'] < 0.5
        status = "✓ PASS" if (r2_ok and rmse_ok) else "✗ FAIL"
        print(f"{name:<15} {status}")
        if not (r2_ok and rmse_ok):
            print(f"  R²={res['r2']:.4f} (need > 0.85), RMSE={res['rmse']:.4f} (need < 0.5)")
    
    all_pass = all(res['r2'] > 0.85 and res['rmse'] < 0.5 for res in results.values())
    
    if all_pass:
        print("\n" + "="*70)
        print("✓ SUCCESS: All topologies learned the polynomial successfully!")
        print("="*70)
    else:
        print("\n✗ Some topologies failed verification")
    
    return results


if __name__ == '__main__':
    results = test_all_topologies()
    
    print("\nFinal Summary:")
    for name, res in results.items():
        print(f"  {name}: R²={res['r2']:.4f}, RMSE={res['rmse']:.4f}")
