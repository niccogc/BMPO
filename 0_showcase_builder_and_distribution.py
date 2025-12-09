# type: ignore
import numpy as np
import quimb.tensor as qt
from tensor.btn import BTN

def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def inspect_quimb_tensor(qt_tensor, label="Tensor"):
    if not isinstance(qt_tensor, qt.Tensor):
        print(f"  [!] {label} is not a quimb.Tensor (Type: {type(qt_tensor)})")
        return

    data_summary = ""
    if hasattr(qt_tensor.data, 'shape'):
        data_summary = f"Shape: {qt_tensor.data.shape}"
    
    tags_str = ", ".join(sorted(list(qt_tensor.tags)))
    print(f"  > {label}:")
    print(f"    - Tags:    [{tags_str}]")
    print(f"    - Inds:    {qt_tensor.inds}")
    print(f"    - Data:    {type(qt_tensor.data)} | {data_summary}")

def main():
    print_header("1. SETTING UP MOCK TENSOR NETWORK (MU)")
    
    # Configuration
    x_dim = 4      # Input feature dimension
    bond_dim = 5   # Bond dimension between nodes
    y_dim = 2      # Output dimension
    batch_dim = 's'
    
    # Define Tensors: T1 -[k1]- T2 -[k2]- T3
    # Inputs will be attached later, this is the core weight network
    data_t1 = np.random.rand(x_dim, bond_dim)
    data_t2 = np.random.rand(bond_dim, x_dim, y_dim, bond_dim)
    data_t3 = np.random.rand(bond_dim, y_dim)

    # Note: T1 and T2 have 'x' indices which are input dimensions (treated as leaves/bonds for distribution purposes if not fixed)
    t1 = qt.Tensor(data_t1, inds=('x1', 'k1'), tags={'T1', 'LAYER_1'})
    t2 = qt.Tensor(data_t2, inds=('k1', 'x2', 'y1', 'k2'), tags={'T2', 'LAYER_2'})
    t3 = qt.Tensor(data_t3, inds=('k2', 'y2'), tags={'T3', 'LAYER_3'})
    
    mu_tn = t1 & t2 & t3
    print("Network Topology Created: T1(x1,k1) <-> T2(k1,x2,y1,k2) <-> T3(k2,y2)")

    print_header("2. INITIALIZING BTN (BUILDER EXECUTION)")
    output_dims = ['y1', 'y2']
    
    # This triggers the Builder to:
    # 1. Scan bonds
    # 2. Create Gamma Priors/Posteriors for bonds
    # 3. Create Sigma Topology (Covariance TN)
    # 4. Create Gaussian Priors/Posteriors for Nodes
    model = BTN(mu_tn, output_dimensions=output_dims, batch_dim=batch_dim)
    print("BTN Model Initialized successfully.")

    print_header("3. INSPECTING EDGE DISTRIBUTIONS (GAMMA)")
    # Inspecting P (Prior) and Q (Posterior) for bonds
    # Expectation: distributions on 'k1', 'k2' (internal) and 'x1', 'x2' (leaves/inputs)
    
    bonds_to_inspect = ['k1', 'k2', 'x1']
    
    for bond in bonds_to_inspect:
        print(f"\n--- Bond: {bond} ---")
        
        # PRIOR
        if bond in model.p_bonds:
            p_dist = model.p_bonds[bond]
            print(f"  [Prior P({bond})]")
            inspect_quimb_tensor(p_dist.concentration, "Concentration (Alpha)")
            inspect_quimb_tensor(p_dist.rate, "Rate (Beta)")
            
            # Check Mean Calculation
            mean_val = p_dist.mean()
            if isinstance(mean_val, qt.Tensor):
                print(f"    -> Mean E[x]: quimb.Tensor with inds {mean_val.inds}")
        else:
            print(f"  [!] No Prior found for {bond}")

        # POSTERIOR
        if bond in model.q_bonds:
            q_dist = model.q_bonds[bond]
            print(f"  [Posterior Q({bond})]")
            inspect_quimb_tensor(q_dist.concentration, "Concentration (Alpha)")
            inspect_quimb_tensor(q_dist.rate, "Rate (Beta)")
            
            # KL Divergence Check
            kl = q_dist.expected_log_prob(concentration_p=p_dist.concentration, rate_p=p_dist.rate)
            if isinstance(kl, qt.Tensor):
                print(f"    -> KL Term (E_q[log p]): quimb.Tensor with inds {kl.inds}")

    print_header("4. INSPECTING SIGMA TOPOLOGY (COVARIANCE)")
    # Sigma network mirrors Mu but with prime indices for non-output dimensions
    
    # Let's look at T2_sigma. 
    # Original T2 inds: k1, x2, y1, k2
    # Output: y1
    # Expected Sigma inds: k1, x2, k2, k1_prime, x2_prime, k2_prime, y1
    
    target_sigma_tag = "T2_sigma"
    if target_sigma_tag in model.sigma.tag_map:
        sigma_t2 = model.sigma[target_sigma_tag]
        print(f"Found Tensor for tag '{target_sigma_tag}':")
        inspect_quimb_tensor(sigma_t2, "Sigma Tensor")
        
        # Verify Prime Indices
        prime_indices = [ix for ix in sigma_t2.inds if "_prime" in ix]
        print(f"  > Detected Prime Indices: {prime_indices}")
    else:
        print(f"Tag {target_sigma_tag} not found in Sigma Network.")

    print_header("5. INSPECTING NODE DISTRIBUTIONS (GAUSSIAN)")
    # Nodes are stored by a tuple of their sorted tags.
    
    # Key for T2
    node_key = tuple(sorted(['T2', 'LAYER_2'])) 
    
    if node_key in model.q_nodes:
        print(f"Found Node Distribution for key: {node_key}")
        q_node = model.q_nodes[node_key]
        p_node = model.p_nodes[node_key]
        
        print("\n  [Posterior Q(Node)]")
        # Loc should be the original tensor data wrapped in qt.Tensor
        inspect_quimb_tensor(q_node.loc, "Mean (Loc)")
        # Covariance should be the Sigma Tensor from the Sigma TN
        inspect_quimb_tensor(q_node.covariance_matrix, "Covariance (Sigma)")
        
        print("\n  [Prior P(Node)]")
        inspect_quimb_tensor(p_node.loc, "Mean (Loc - Zero initialized)")
        # Prior covariance is None (implicit precision)
        print(f"  > Covariance: {p_node.covariance_matrix}")
        
    else:
        print(f"Node key {node_key} not found. Available keys:")
        for k in model.q_nodes.keys():
            print(f"  - {k}")

    print_header("6. VERIFICATION COMPLETE")

if __name__ == "__main__":
    main()
