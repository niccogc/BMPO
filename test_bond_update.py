"""
Test the update_bond_variational method.
"""

import torch
from tensor.bayesian_mpo_builder import create_bayesian_tensor_train


def test_bond_update():
    print("=" * 70)
    print("Testing Bond Variational Update")
    print("=" * 70)
    
    bmpo = create_bayesian_tensor_train(
        num_blocks=3,
        bond_dim=4,
        input_features=5,
        output_shape=1,
        seed=42
    )
    
    # Test 1: Check update formulas for each bond
    print("\nTest 1: Update Formulas")
    print("-" * 70)
    
    for label in sorted(bmpo.mu_mpo.distributions.keys()):
        print(f"\nBond '{label}':")
        
        # Get info
        bond_dist = bmpo.mu_mpo.distributions[label]
        bond_size = len(bond_dist['concentration'])
        nodes_info = bmpo.get_nodes_for_bond(label)
        block_indices = nodes_info['mu_nodes']
        N_b = len(block_indices)
        
        print(f"  Size: {bond_size}")
        print(f"  Blocks: {block_indices}, N_b = {N_b}")
        
        # Store old values
        old_conc = bond_dist['concentration'].clone()
        old_rate = bond_dist['rate'].clone()
        
        # Get prior
        prior = bmpo.prior_bond_params[label]
        conc_p = prior['concentration0']
        rate_p = prior['rate0']
        
        print(f"  Prior: concentration0 = {conc_p[0]:.2f}, rate0 = {rate_p[0]:.2f}")
        print(f"  Before: concentration = {old_conc[0]:.2f}, rate = {old_rate[0]:.2f}")
        
        # Update
        bmpo.update_bond_variational(label)
        
        # Check new values
        new_conc = bmpo.mu_mpo.distributions[label]['concentration']
        new_rate = bmpo.mu_mpo.distributions[label]['rate']
        
        print(f"  After: concentration = {new_conc[0]:.2f}, rate = {new_rate[0]:.2f}")
        
        # Verify concentration formula
        expected_conc = conc_p + N_b * bond_size
        print(f"\n  Concentration check:")
        print(f"    Expected: {conc_p[0]:.2f} + {N_b} × {bond_size} = {expected_conc[0]:.2f}")
        print(f"    Actual: {new_conc[0]:.2f}")
        print(f"    Match: {torch.allclose(new_conc, expected_conc)}")
        
        # Verify rate formula (can't easily check exact value without computing partial traces)
        print(f"\n  Rate check:")
        print(f"    Started from rate_p = {rate_p[0]:.2f}")
        print(f"    Updated to: {new_rate[0]:.2f}")
        print(f"    Changed by: {(new_rate[0] - rate_p[0]).item():.4f}")
    
    # Test 2: Verify dimensions
    print("\n" + "=" * 70)
    print("Test 2: Verify Output Dimensions")
    print("=" * 70)
    
    for label in sorted(bmpo.mu_mpo.distributions.keys()):
        bond_dist = bmpo.mu_mpo.distributions[label]
        bond_size = len(bond_dist['concentration'])
        
        conc = bond_dist['concentration']
        rate = bond_dist['rate']
        
        print(f"\nBond '{label}' (size {bond_size}):")
        print(f"  Concentration shape: {conc.shape}")
        print(f"  Rate shape: {rate.shape}")
        assert conc.shape == torch.Size([bond_size]), "Concentration shape wrong!"
        assert rate.shape == torch.Size([bond_size]), "Rate shape wrong!"
    
    # Test 3: Verify rate is positive (necessary for valid Gamma)
    print("\n" + "=" * 70)
    print("Test 3: Check Valid Gamma Parameters")
    print("=" * 70)
    
    all_valid = True
    for label in sorted(bmpo.mu_mpo.distributions.keys()):
        bond_dist = bmpo.mu_mpo.distributions[label]
        conc = bond_dist['concentration']
        rate = bond_dist['rate']
        
        conc_positive = torch.all(conc > 0)
        rate_positive = torch.all(rate > 0)
        
        print(f"\nBond '{label}':")
        print(f"  All concentrations > 0: {conc_positive}")
        print(f"  All rates > 0: {rate_positive}")
        
        if not (conc_positive and rate_positive):
            print(f"  WARNING: Invalid Gamma parameters!")
            print(f"    Concentration range: [{conc.min().item():.4f}, {conc.max().item():.4f}]")
            print(f"    Rate range: [{rate.min().item():.4f}, {rate.max().item():.4f}]")
            all_valid = False
    
    if all_valid:
        print("\n✓ All parameters valid for Gamma distributions!")
    else:
        print("\n✗ Some parameters invalid - need to check update formula!")
    
    # Test 4: Multiple updates (check convergence behavior)
    print("\n" + "=" * 70)
    print("Test 4: Multiple Updates")
    print("=" * 70)
    
    label = 'r1'
    print(f"\nBond '{label}': Running 3 updates")
    
    for i in range(3):
        bond_dist = bmpo.mu_mpo.distributions[label]
        conc = bond_dist['concentration'][0].item()
        rate = bond_dist['rate'][0].item()
        
        print(f"  Iteration {i}: concentration = {conc:.4f}, rate = {rate:.4f}")
        
        if i < 2:  # Don't update after last print
            bmpo.update_bond_variational(label)
    
    print("\n" + "=" * 70)
    print("✓ Bond variational update tests completed!")
    print("=" * 70)


if __name__ == "__main__":
    test_bond_update()
