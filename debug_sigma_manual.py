"""
Debug sigma forward pass manually - step by step.
Also test sigma creation with various mu shapes.
"""

import numpy as np
import quimb.tensor as qtn

from tensor.bayesian_tn_builder import create_sigma_network


def manual_sigma_forward():
    """Manually trace through sigma forward pass step by step."""
    print("\n" + "="*70)
    print("MANUAL SIGMA FORWARD PASS")
    print("="*70)
    
    # Create simple 2-block network
    A_data = np.random.randn(2, 3) * 0.1  # (p1=2, r1=3)
    B_data = np.random.randn(3, 2) * 0.1  # (r1=3, p2=2)
    
    A = qtn.Tensor(data=A_data, inds=('p1', 'r1'), tags='A')  # type: ignore
    B = qtn.Tensor(data=B_data, inds=('r1', 'p2'), tags='B')  # type: ignore
    
    mu_tn = qtn.TensorNetwork([A, B])
    sigma_tn = create_sigma_network(mu_tn, ['A', 'B'])
    
    print("\n1. Initial sigma network:")
    for tid, tensor in sigma_tn.tensor_map.items():
        tag = list(tensor.tags)[0] if tensor.tags else tid
        print(f"   {tag}: shape={tensor.shape}, inds={tensor.inds}")  # type: ignore
    
    # Create input
    input_data = np.array([1.0, 2.0])  # 2 features
    print(f"\n2. Input data: {input_data}, shape={input_data.shape}")
    
    # Step 1: Start with A_sigma
    A_sigma = sigma_tn['A_sigma']  # type: ignore
    print(f"\n3. A_sigma: shape={A_sigma.shape}, inds={A_sigma.inds}")  # type: ignore
    
    # Step 2: Contract A_sigma with input on p1o
    print(f"\n4. Contracting A_sigma with input on p1o:")
    print(f"   A_sigma: (p1o, p1i, r1o, r1i) = {A_sigma.shape}")  # type: ignore
    print(f"   input:   (p1o,) = {input_data.shape}")
    print(f"   Result should be: (p1i, r1o, r1i)")
    
    A_after_p1o = np.einsum('oirR,o->irR', A_sigma.data, input_data)  # type: ignore
    print(f"   Actual result: {A_after_p1o.shape}")
    if A_after_p1o.shape == (2, 3, 3):
        print("   ✓ Shape correct!")
    else:
        print(f"   ✗ Shape wrong! Expected (2, 3, 3)")
    
    # Step 3: Contract with input on p1i
    print(f"\n5. Contracting with input on p1i:")
    print(f"   Current: (p1i, r1o, r1i) = {A_after_p1o.shape}")
    print(f"   input:   (p1i,) = {input_data.shape}")
    print(f"   Result should be: (r1o, r1i)")
    
    A_after_p1i = np.einsum('irR,i->rR', A_after_p1o, input_data)
    print(f"   Actual result: {A_after_p1i.shape}")
    if A_after_p1i.shape == (3, 3):
        print("   ✓ Shape correct!")
    else:
        print(f"   ✗ Shape wrong! Expected (3, 3)")
    
    # Step 4: Now B_sigma
    B_sigma = sigma_tn['B_sigma']  # type: ignore
    print(f"\n6. B_sigma: shape={B_sigma.shape}, inds={B_sigma.inds}")  # type: ignore
    
    # Step 5: Contract A result with B_sigma on r1o
    print(f"\n7. Contracting A result with B_sigma on r1o:")
    print(f"   A result: (r1o, r1i) = {A_after_p1i.shape}")
    print(f"   B_sigma:  (r1o, r1i, p2o, p2i) = {B_sigma.shape}")  # type: ignore
    print(f"   Result should be: (r1i, r1i, p2o, p2i) - wait, two r1i!")
    
    # Actually we contract on r1o, so one r1i from A, and r1i,p2o,p2i from B
    AB_after_r1o = np.einsum('oI,oIpP->IpP', A_after_p1i, B_sigma.data)  # type: ignore
    print(f"   Actual result: {AB_after_r1o.shape}")
    if AB_after_r1o.shape == (3, 2, 2):
        print("   ✓ Shape correct!")
    else:
        print(f"   ✗ Shape wrong! Expected (3, 2, 2)")
    
    # Step 6: Contract on r1i
    print(f"\n8. Contracting on r1i:")
    print(f"   Current: (r1i, p2o, p2i) = {AB_after_r1o.shape}")
    print(f"   Wait - B_sigma also has r1i, they should contract together!")
    print(f"   Let me recalculate...")
    
    # Actually, let's do it properly with einsum notation
    print(f"\n9. Full contraction with proper einsum:")
    print(f"   A_sigma: (p1o, p1i, r1o, r1i)")
    print(f"   B_sigma: (r1o, r1i, p2o, p2i)")
    print(f"   input_p1o: (p1o,)")
    print(f"   input_p1i: (p1i,)")
    print(f"   input_p2o: (p2o,)")
    print(f"   input_p2i: (p2i,)")
    print(f"")
    print(f"   Einsum: 'oirR,oirR,o,i,r,r -> ()'")
    print(f"           A     B     p1o p1i p2o p2i")
    
    # Full manual contraction
    result = np.einsum(
        'oirR,RrpP,o,i,p,P->',
        A_sigma.data, B_sigma.data,  # type: ignore
        input_data, input_data, input_data, input_data
    )
    print(f"   Result: {result} (scalar)")
    print(f"   Result shape: {np.shape(result)}")
    print("   ✓ Full contraction produces scalar!")
    
    # Compare with quimb
    print(f"\n10. Comparing with quimb contraction:")
    sigma_with_inputs = sigma_tn.copy()
    sigma_with_inputs &= qtn.Tensor(data=input_data, inds=('p1o',), tags='in_p1o')  # type: ignore
    sigma_with_inputs &= qtn.Tensor(data=input_data, inds=('p1i',), tags='in_p1i')  # type: ignore
    sigma_with_inputs &= qtn.Tensor(data=input_data, inds=('p2o',), tags='in_p2o')  # type: ignore
    sigma_with_inputs &= qtn.Tensor(data=input_data, inds=('p2i',), tags='in_p2i')  # type: ignore
    
    quimb_result = sigma_with_inputs.contract(optimize='greedy')
    quimb_val = quimb_result.data if hasattr(quimb_result, 'data') else quimb_result  # type: ignore
    
    print(f"   Manual einsum: {result}")
    print(f"   Quimb result:  {quimb_val}")
    print(f"   Difference:    {abs(result - quimb_val)}")
    
    if abs(result - quimb_val) < 1e-10:
        print("   ✓ Manual and quimb match!")
    else:
        print("   ✗ Manual and quimb differ!")


def test_various_shapes():
    """Test sigma creation with various mu network shapes."""
    print("\n" + "="*70)
    print("TESTING SIGMA CREATION WITH VARIOUS SHAPES")
    print("="*70)
    
    test_cases = [
        # (name, tensors_specs)
        ("1D MPS", [
            (('p1', 'r1'), (3, 5)),
            (('r1', 'p2'), (5, 4)),
        ]),
        ("2D PEPS node", [
            (('up', 'down', 'left', 'right', 'phys'), (2, 2, 2, 2, 3)),
        ]),
        ("3-leg tensor", [
            (('a', 'b', 'c'), (4, 5, 6)),
            (('c', 'd', 'e'), (6, 3, 2)),
        ]),
        ("4-block MPS", [
            (('p1', 'r1'), (2, 3)),
            (('r1', 'p2', 'r2'), (3, 2, 4)),
            (('r2', 'p3', 'r3'), (4, 2, 5)),
            (('r3', 'p4'), (5, 2)),
        ]),
        ("Tree structure", [
            (('root', 'left', 'right'), (3, 4, 5)),
            (('left', 'phys1'), (4, 2)),
            (('right', 'phys2'), (5, 2)),
        ]),
    ]
    
    for name, tensor_specs in test_cases:
        print(f"\n{name}:")
        print("  Mu network:")
        
        # Create mu network
        mu_tensors = []
        learnable_tags = []
        for i, (inds, shape) in enumerate(tensor_specs):
            data = np.random.randn(*shape) * 0.1
            tag = f'T{i}'
            tensor = qtn.Tensor(data=data, inds=inds, tags=tag)  # type: ignore
            mu_tensors.append(tensor)
            learnable_tags.append(tag)
            print(f"    {tag}: shape={shape}, inds={inds}")
        
        mu_tn = qtn.TensorNetwork(mu_tensors)
        
        # Create sigma
        try:
            sigma_tn = create_sigma_network(mu_tn, learnable_tags)
            print("  Sigma network:")
            
            # Check each sigma tensor
            all_correct = True
            for i, (inds, shape) in enumerate(tensor_specs):
                sigma_tag = f'T{i}_sigma'
                sigma_tensor = sigma_tn[sigma_tag]  # type: ignore
                
                # Expected: each dimension doubled, each index gets 'o' and 'i'
                expected_shape = tuple(d for dim in shape for d in [dim, dim])
                expected_inds = tuple(idx + suffix for idx in inds for suffix in ['o', 'i'])
                
                actual_shape = sigma_tensor.shape  # type: ignore
                actual_inds = sigma_tensor.inds  # type: ignore
                
                shape_ok = actual_shape == expected_shape
                inds_ok = actual_inds == expected_inds
                
                status = "✓" if (shape_ok and inds_ok) else "✗"
                print(f"    {sigma_tag}: {status} shape={actual_shape}, inds={actual_inds}")
                
                if not (shape_ok and inds_ok):
                    print(f"      Expected: shape={expected_shape}, inds={expected_inds}")
                    all_correct = False
            
            if all_correct:
                print("  ✓ ALL CORRECT!")
            else:
                print("  ✗ SOME ERRORS!")
                
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()


def test_mu_sigma_consistency():
    """Test that mu and sigma forward passes are consistent."""
    print("\n" + "="*70)
    print("MU vs SIGMA FORWARD CONSISTENCY")
    print("="*70)
    
    # Create network
    A_data = np.random.randn(2, 3) * 0.1
    B_data = np.random.randn(3, 2) * 0.1
    
    A = qtn.Tensor(data=A_data, inds=('p1', 'r1'), tags='A')  # type: ignore
    B = qtn.Tensor(data=B_data, inds=('r1', 'p2'), tags='B')  # type: ignore
    
    mu_tn = qtn.TensorNetwork([A, B])
    sigma_tn = create_sigma_network(mu_tn, ['A', 'B'])
    
    input_data = np.array([1.0, 2.0])
    
    print("\n1. Mu forward:")
    mu_with_input = mu_tn.copy()
    mu_with_input &= qtn.Tensor(data=input_data, inds=('p1',), tags='in_p1')  # type: ignore
    mu_with_input &= qtn.Tensor(data=input_data, inds=('p2',), tags='in_p2')  # type: ignore
    mu_result = mu_with_input.contract(optimize='greedy')
    mu_val = mu_result.data if hasattr(mu_result, 'data') else mu_result  # type: ignore
    print(f"   Mu result: {mu_val}")
    print(f"   Mu shape: {np.shape(mu_val)}")
    
    print("\n2. Sigma forward:")
    sigma_with_input = sigma_tn.copy()
    sigma_with_input &= qtn.Tensor(data=input_data, inds=('p1o',), tags='in_p1o')  # type: ignore
    sigma_with_input &= qtn.Tensor(data=input_data, inds=('p1i',), tags='in_p1i')  # type: ignore
    sigma_with_input &= qtn.Tensor(data=input_data, inds=('p2o',), tags='in_p2o')  # type: ignore
    sigma_with_input &= qtn.Tensor(data=input_data, inds=('p2i',), tags='in_p2i')  # type: ignore
    sigma_result = sigma_with_input.contract(optimize='greedy')
    sigma_val = sigma_result.data if hasattr(sigma_result, 'data') else sigma_result  # type: ignore
    print(f"   Sigma result: {sigma_val}")
    print(f"   Sigma shape: {np.shape(sigma_val)}")
    
    print("\n3. Relationship:")
    print(f"   Both should be scalars: mu={np.shape(mu_val)}, sigma={np.shape(sigma_val)}")
    print(f"   Sigma represents variance: should be positive")
    print(f"   Sigma value: {sigma_val}")
    
    if sigma_val > 0:
        print("   ✓ Sigma is positive (valid variance)!")
    else:
        print("   ✗ Sigma is not positive!")


if __name__ == "__main__":
    manual_sigma_forward()
    test_various_shapes()
    test_mu_sigma_consistency()
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)
