"""
Debug sigma network structure.

Check:
1. Are sigma tensors created correctly with doubled indices?
2. Do they have the right shape?
3. Are they initialized correctly?
4. What happens when we add inputs?
"""

import torch
import numpy as np
import quimb.tensor as qtn

from tensor.bayesian_tn_builder import create_sigma_network


def test_sigma_network_structure():
    """Test that sigma network is created correctly."""
    print("\n" + "="*70)
    print("DEBUG: Sigma Network Structure")
    print("="*70)
    
    # Create simple mu network: 2 blocks MPS
    # Block A: (p1, r1)
    # Block B: (r1, p2)
    
    A_data = np.random.randn(2, 3) * 0.1  # (p1=2, r1=3)
    B_data = np.random.randn(3, 2) * 0.1  # (r1=3, p2=2)
    
    A = qtn.Tensor(data=A_data, inds=('p1', 'r1'), tags='A')  # type: ignore
    B = qtn.Tensor(data=B_data, inds=('r1', 'p2'), tags='B')  # type: ignore
    
    mu_tn = qtn.TensorNetwork([A, B])
    
    print("\n1. Mu network structure:")
    print(f"   A: shape={A.shape}, inds={A.inds}")  # type: ignore
    print(f"   B: shape={B.shape}, inds={B.inds}")  # type: ignore
    
    # Create sigma network
    print("\n2. Creating sigma network...")
    sigma_tn = create_sigma_network(mu_tn, ['A', 'B'])
    
    print("\n3. Sigma network structure:")
    for tid, tensor in sigma_tn.tensor_map.items():
        tag = list(tensor.tags)[0] if tensor.tags else tid
        print(f"   {tag}: shape={tensor.shape}, inds={tensor.inds}")  # type: ignore
    
    # Check expected structure
    print("\n4. Verification:")
    
    A_sigma = sigma_tn['A_sigma']  # type: ignore
    expected_shape_A = (2, 2, 3, 3)  # (p1o, p1i, r1o, r1i)
    expected_inds_A = ('p1o', 'p1i', 'r1o', 'r1i')
    
    print(f"   A_sigma:")
    print(f"     Expected shape: {expected_shape_A}")
    print(f"     Actual shape:   {A_sigma.shape}")  # type: ignore
    print(f"     Expected inds:  {expected_inds_A}")
    print(f"     Actual inds:    {A_sigma.inds}")  # type: ignore
    
    if A_sigma.shape == expected_shape_A:  # type: ignore
        print("     ✓ Shape correct!")
    else:
        print("     ✗ Shape wrong!")
    
    if A_sigma.inds == expected_inds_A:  # type: ignore
        print("     ✓ Indices correct!")
    else:
        print("     ✗ Indices wrong!")
    
    # Check diagonal initialization
    print("\n5. Checking diagonal initialization:")
    A_sigma_data = A_sigma.data  # type: ignore
    
    # Check a diagonal element (0,0,0,0)
    diag_00 = A_sigma_data[0, 0, 0, 0]
    # Check an off-diagonal element (0,1,0,0)
    off_diag = A_sigma_data[0, 1, 0, 0]
    
    print(f"   Diagonal element [0,0,0,0]: {diag_00}")
    print(f"   Off-diagonal [0,1,0,0]: {off_diag}")
    
    if diag_00 > 0 and off_diag == 0:
        print("   ✓ Diagonal initialization correct!")
    else:
        print("   ✗ Diagonal initialization wrong!")
    
    # Check that r1o and r1i contract
    print("\n6. Checking index connectivity:")
    from collections import Counter
    all_inds = []
    for tensor in sigma_tn.tensor_map.values():
        all_inds.extend(tensor.inds)  # type: ignore
    
    ind_counts = Counter(all_inds)
    print(f"   Index counts: {dict(ind_counts)}")
    print(f"   r1o appears: {ind_counts['r1o']} times (should be 2)")
    print(f"   r1i appears: {ind_counts['r1i']} times (should be 2)")
    
    if ind_counts['r1o'] == 2 and ind_counts['r1i'] == 2:
        print("   ✓ Rank indices connect correctly!")
    else:
        print("   ✗ Rank indices wrong!")


def test_sigma_forward():
    """Test sigma forward pass with inputs."""
    print("\n" + "="*70)
    print("DEBUG: Sigma Forward Pass")
    print("="*70)
    
    # Same network as above
    A_data = np.random.randn(2, 3) * 0.1
    B_data = np.random.randn(3, 2) * 0.1
    
    A = qtn.Tensor(data=A_data, inds=('p1', 'r1'), tags='A')  # type: ignore
    B = qtn.Tensor(data=B_data, inds=('r1', 'p2'), tags='B')  # type: ignore
    
    mu_tn = qtn.TensorNetwork([A, B])
    sigma_tn = create_sigma_network(mu_tn, ['A', 'B'])
    
    # Create input data
    input_data = np.array([1.0, 2.0])  # 2 features
    
    print("\n1. Input data:")
    print(f"   Shape: {input_data.shape}")
    print(f"   Values: {input_data}")
    
    # Add inputs to sigma network
    print("\n2. Adding inputs to sigma network:")
    
    # For sigma, we need inputs for p1o, p1i, p2o, p2i
    sigma_with_inputs = sigma_tn.copy()
    
    # p1 outer
    input_p1o = qtn.Tensor(data=input_data, inds=('p1o',), tags='input_p1o')  # type: ignore
    sigma_with_inputs &= input_p1o
    print(f"   Added input_p1o: shape={input_p1o.shape}, inds={input_p1o.inds}")  # type: ignore
    
    # p1 inner
    input_p1i = qtn.Tensor(data=input_data, inds=('p1i',), tags='input_p1i')  # type: ignore
    sigma_with_inputs &= input_p1i
    print(f"   Added input_p1i: shape={input_p1i.shape}, inds={input_p1i.inds}")  # type: ignore
    
    # p2 outer
    input_p2o = qtn.Tensor(data=input_data, inds=('p2o',), tags='input_p2o')  # type: ignore
    sigma_with_inputs &= input_p2o
    print(f"   Added input_p2o: shape={input_p2o.shape}, inds={input_p2o.inds}")  # type: ignore
    
    # p2 inner
    input_p2i = qtn.Tensor(data=input_data, inds=('p2i',), tags='input_p2i')  # type: ignore
    sigma_with_inputs &= input_p2i
    print(f"   Added input_p2i: shape={input_p2i.shape}, inds={input_p2i.inds}")  # type: ignore
    
    # Check what indices remain
    print("\n3. Network state after adding inputs:")
    all_inds = []
    for tensor in sigma_with_inputs.tensor_map.values():
        all_inds.extend(tensor.inds)  # type: ignore
    
    from collections import Counter
    ind_counts = Counter(all_inds)
    print(f"   All indices: {dict(ind_counts)}")
    
    free_inds = [ind for ind, count in ind_counts.items() if count == 1]
    print(f"   Free indices (uncontracted): {free_inds}")
    print(f"   Expected: [] (all contracted)")
    
    # Contract
    print("\n4. Contracting sigma network:")
    try:
        result = sigma_with_inputs.contract(optimize='greedy')
        print(f"   Result type: {type(result)}")
        
        if hasattr(result, 'data'):
            result_val = result.data  # type: ignore
        else:
            result_val = result
            
        print(f"   Result value: {result_val}")
        print(f"   Result shape: {np.shape(result_val)}")
        print("   ✓ Contraction successful!")
    except Exception as e:
        print(f"   ✗ Contraction failed: {e}")
        import traceback
        traceback.print_exc()


def test_mu_vs_sigma_structure():
    """Compare mu and sigma structures side by side."""
    print("\n" + "="*70)
    print("DEBUG: Mu vs Sigma Comparison")
    print("="*70)
    
    # Create mu network
    A_data = np.random.randn(2, 3) * 0.1
    B_data = np.random.randn(3, 2) * 0.1
    
    A = qtn.Tensor(data=A_data, inds=('p1', 'r1'), tags='A')  # type: ignore
    B = qtn.Tensor(data=B_data, inds=('r1', 'p2'), tags='B')  # type: ignore
    
    mu_tn = qtn.TensorNetwork([A, B])
    sigma_tn = create_sigma_network(mu_tn, ['A', 'B'])
    
    print("\n1. Side-by-side comparison:")
    print("\n   MU NETWORK:")
    for tid, tensor in mu_tn.tensor_map.items():
        tag = list(tensor.tags)[0] if tensor.tags else tid
        print(f"     {tag:10s}: shape={str(tensor.shape):15s} inds={tensor.inds}")  # type: ignore
    
    print("\n   SIGMA NETWORK:")
    for tid, tensor in sigma_tn.tensor_map.items():
        tag = list(tensor.tags)[0] if tensor.tags else tid
        print(f"     {tag:10s}: shape={str(tensor.shape):15s} inds={tensor.inds}")  # type: ignore
    
    print("\n2. Topology comparison:")
    print("   Mu:    A(p1, r1) -- B(r1, p2)")
    print("   Sigma: A(p1o,p1i, r1o,r1i) -- B(r1o,r1i, p2o,p2i)")
    print("   ")
    print("   Both r1o and r1i should connect A_sigma to B_sigma!")


if __name__ == "__main__":
    test_sigma_network_structure()
    test_sigma_forward()
    test_mu_vs_sigma_structure()
    
    print("\n" + "="*70)
    print("DEBUG COMPLETE")
    print("="*70)
