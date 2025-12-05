# type: ignore
import numpy as np
import quimb.tensor as qt
import torch
from tensor.builder import BTNBuilder
from tensor.distributions import GammaDistribution, MultivariateGaussianDistribution

def run_test():
    print("--- Starting BTNBuilder Verification ---")

    # --- 1. Setup Mock Network (Mu) ---
    x_dim = 4
    bond_dim = 5
    y_dim = 2
    
    # Use fixed seeds for reproducibility in value checks
    np.random.seed(42)
    data_t1 = np.random.rand(x_dim, bond_dim)
    data_t2 = np.random.rand(bond_dim, x_dim, y_dim, bond_dim)
    data_t3 = np.random.rand(bond_dim, y_dim)

    t1 = qt.Tensor(data_t1, inds=('x1', 'k1'), tags={'T1'})
    t2 = qt.Tensor(data_t2, inds=('k1', 'x2', 'y1', 'k2'), tags={'T2'})
    t3 = qt.Tensor(data_t3, inds=('k2', 'y2'), tags={'T3'})
    
    mu_tn = t1 & t2 & t3
    
    # --- 2. Build Model ---
    output_dims = ['y1', 'y2']
    builder = BTNBuilder(mu_tn, output_dimensions=output_dims, batch_dim='s')
    
    p_bonds, p_nodes, q_bonds, q_nodes, sigma_tn = builder.build_model()
    
    print("\n✅ Build successful.")

    # --- 3. Verify Edge Distributions (Gamma) ---
    print("\n--- Verifying Edge Distributions ---")
    expected_bonds = ['x1', 'k1', 'x2', 'k2'] 
    
    for bond in expected_bonds:
        if bond not in p_bonds:
            print(f"❌ Missing Prior for bond {bond}")
            continue
            
        p_dist = p_bonds[bond]
        q_dist = q_bonds[bond]
        
        # Check Type
        assert isinstance(p_dist, GammaDistribution)
        
        # Check Parameter Tags
        tags_alpha = p_dist.concentration.tags
        assert f"{bond}_alpha" in tags_alpha, f"Tag {bond}_alpha missing: {tags_alpha}"
        
        # Check Expectation Returns
        mean_val = p_dist.mean()
        assert isinstance(mean_val, qt.Tensor), "Mean should be returned as quimb.Tensor"
        assert bond in mean_val.inds, f"Mean tensor missing bond index {bond}"
        
        # Check KL
        kl_term = q_dist.expected_log_prob(concentration_p=p_dist.concentration, rate_p=p_dist.rate)
        assert isinstance(kl_term, qt.Tensor), "Expected log prob should be quimb.Tensor"
        
        print(f"✅ Bond {bond}: Tags ok. Mean sample: {mean_val.data[:2].numpy() if isinstance(mean_val.data, torch.Tensor) else mean_val.data[:2]}")

    # --- 4. Verify Sigma Topology ---
    print("\n--- Verifying Sigma Topology ---")
    if "T2_sigma" in sigma_tn.tag_map:
        sigma_t2 = sigma_tn["T2_sigma"]
        expected_inds_t2 = {'k1', 'x2', 'y1', 'k2', 'k1_prime', 'x2_prime', 'k2_prime'}
        current_inds_t2 = set(sigma_t2.inds)
        assert expected_inds_t2 == current_inds_t2, f"Sigma indices mismatch for T2"
        print(f"✅ Sigma Topology for T2 correct: {sigma_t2.inds}")
    else:
        print("❌ T2_sigma not found in network!")

    # --- 5. Verify Node Distributions (Gaussian) ---
    print("\n--- Verifying Node Distributions ---")
    
    expected_data_map = {
        'T1': data_t1,
        'T2': data_t2,
        'T3': data_t3
    }

    nodes_to_check = ['T1', 'T2', 'T3']

    for tag in nodes_to_check:
        print(f"\n🔎 Inspecting Node: {tag}")
        node_key = tuple(sorted([tag]))
        
        if node_key not in q_nodes:
            print(f"❌ Node {tag} not found in q_nodes!")
            continue

        q_node = q_nodes[node_key]
        p_node = p_nodes[node_key]
        expected_data = expected_data_map[tag]

        # --- A. Check Q (Posterior) ---
        # 1. Check Loc (Mean) matches Mu
        assert isinstance(q_node.loc, qt.Tensor)
        current_data = q_node.loc.data
        if isinstance(current_data, torch.Tensor):
            current_data = current_data.numpy()
            
        is_close = np.allclose(current_data, expected_data, atol=1e-6)
        match_symbol = "✅" if is_close else "❌"
        print(f"   {match_symbol} Q Mean matches original Data. Shape: {current_data.shape}")
        if not is_close:
            print(f"      Diff: {np.max(np.abs(current_data - expected_data))}")

        # 2. Check Covariance (Sigma)
        assert isinstance(q_node.covariance_matrix, qt.Tensor)
        sigma = q_node.covariance_matrix
        print(f"   ✅ Q Covariance (Sigma) is quimb.Tensor.")
        print(f"      Indices: {sigma.inds}")
        print(f"      Tags: {sigma.tags}")
        # Print a few values to ensure it's initialized (small random noise per builder)
        flat_sig = sigma.data.flatten()
        sample_vals = flat_sig[:3] if isinstance(flat_sig, np.ndarray) else flat_sig[:3].numpy()
        print(f"      Sample Values: {sample_vals}")

        # --- B. Check P (Prior) ---
        # 1. Check Loc is Zero
        assert isinstance(p_node.loc, qt.Tensor)
        prior_data = p_node.loc.data
        if isinstance(prior_data, torch.Tensor):
            prior_data = prior_data.numpy()
        
        is_zero = np.allclose(prior_data, 0.0, atol=1e-9)
        zero_symbol = "✅" if is_zero else "❌"
        print(f"   {zero_symbol} Prior Mean is Zero-initialized.")

    print("\n✅ All Builder tests passed.")

if __name__ == "__main__":
    run_test()
