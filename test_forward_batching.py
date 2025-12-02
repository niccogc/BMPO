"""
Test forward pass with inputs and batching.

Tests:
1. Single sample forward (mu)
2. Batched forward (mu)
3. Single sample forward (sigma)
4. Batched forward (sigma)
5. Check shapes at each step
"""

import torch
import numpy as np
import quimb.tensor as qtn

from tensor.bayesian_tn import BayesianTensorNetwork
from tensor.bayesian_tn_builder import create_sigma_network


def test_single_sample_mu():
    """Test mu forward with single sample."""
    print("\n" + "="*70)
    print("TEST 1: Single Sample - Mu Forward")
    print("="*70)
    
    # Create simple network: A(p1, r1) -- B(r1, p2)
    A_data = np.random.randn(2, 3) * 0.1  # (p1=2, r1=3)
    B_data = np.random.randn(3, 2) * 0.1  # (r1=3, p2=2)
    
    A = qtn.Tensor(data=A_data, inds=('p1', 'r1'), tags='A')  # type: ignore
    B = qtn.Tensor(data=B_data, inds=('r1', 'p2'), tags='B')  # type: ignore
    
    mu_tn = qtn.TensorNetwork([A, B])
    sigma_tn = create_sigma_network(mu_tn, ['A', 'B'])
    
    # Create BayesianTensorNetwork
    btn = BayesianTensorNetwork(
        mu_tn=mu_tn,
        sigma_tn=sigma_tn,
        input_indices={'features': ['p1', 'p2']},
        learnable_tags=['A', 'B'],
        device=torch.device('cpu'),
        dtype=torch.float64
    )
    
    # Single input sample (no batch dimension)
    input_single = torch.tensor([1.0, 2.0], dtype=torch.float64)
    
    print(f"\n1. Input shape: {input_single.shape}")
    print(f"   Input values: {input_single}")
    
    try:
        result = btn.forward_mu({'features': input_single})
        print(f"\n2. Output shape: {result.shape}")
        print(f"   Output value: {result.item()}")
        print(f"   Expected: scalar (no dimensions)")
        
        if result.shape == torch.Size([]):
            print("   ✓ Single sample produces scalar!")
            return True
        else:
            print(f"   ✗ Expected scalar, got shape {result.shape}")
            return False
            
    except Exception as e:
        print(f"\n   ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batched_mu():
    """Test mu forward with batched samples."""
    print("\n" + "="*70)
    print("TEST 2: Batched - Mu Forward")
    print("="*70)
    
    # Same network as above
    A_data = np.random.randn(2, 3) * 0.1
    B_data = np.random.randn(3, 2) * 0.1
    
    A = qtn.Tensor(data=A_data, inds=('p1', 'r1'), tags='A')  # type: ignore
    B = qtn.Tensor(data=B_data, inds=('r1', 'p2'), tags='B')  # type: ignore
    
    mu_tn = qtn.TensorNetwork([A, B])
    sigma_tn = create_sigma_network(mu_tn, ['A', 'B'])
    
    btn = BayesianTensorNetwork(
        mu_tn=mu_tn,
        sigma_tn=sigma_tn,
        input_indices={'features': ['p1', 'p2']},
        learnable_tags=['A', 'B'],
        device=torch.device('cpu'),
        dtype=torch.float64
    )
    
    # Batched input
    batch_size = 5
    input_batch = torch.randn(batch_size, 2, dtype=torch.float64)
    
    print(f"\n1. Input shape: {input_batch.shape}")
    print(f"   Batch size: {batch_size}")
    
    try:
        result = btn.forward_mu({'features': input_batch})
        print(f"\n2. Output shape: {result.shape}")
        print(f"   Output values (first 3): {result[:3]}")
        print(f"   Expected shape: ({batch_size},)")
        
        if result.shape == torch.Size([batch_size]):
            print("   ✓ Batched forward produces correct shape!")
            
            # Check that each sample is processed independently
            # Process samples one by one
            individual_results = []
            for i in range(batch_size):
                single_result = btn.forward_mu({'features': input_batch[i]})
                individual_results.append(single_result.item())
            
            individual_tensor = torch.tensor(individual_results, dtype=torch.float64)
            diff = (result - individual_tensor).abs().max().item()
            
            print(f"\n3. Comparing batch vs individual processing:")
            print(f"   Max difference: {diff:.2e}")
            
            if diff < 1e-10:
                print("   ✓ Batch and individual give same results!")
                return True
            else:
                print("   ✗ Batch and individual differ!")
                print(f"   Batch: {result[:3]}")
                print(f"   Individual: {individual_tensor[:3]}")
                return False
        else:
            print(f"   ✗ Wrong shape!")
            return False
            
    except Exception as e:
        print(f"\n   ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_sample_sigma():
    """Test sigma forward with single sample."""
    print("\n" + "="*70)
    print("TEST 3: Single Sample - Sigma Forward")
    print("="*70)
    
    # Same network
    A_data = np.random.randn(2, 3) * 0.1
    B_data = np.random.randn(3, 2) * 0.1
    
    A = qtn.Tensor(data=A_data, inds=('p1', 'r1'), tags='A')  # type: ignore
    B = qtn.Tensor(data=B_data, inds=('r1', 'p2'), tags='B')  # type: ignore
    
    mu_tn = qtn.TensorNetwork([A, B])
    sigma_tn = create_sigma_network(mu_tn, ['A', 'B'])
    
    btn = BayesianTensorNetwork(
        mu_tn=mu_tn,
        sigma_tn=sigma_tn,
        input_indices={'features': ['p1', 'p2']},
        learnable_tags=['A', 'B'],
        device=torch.device('cpu'),
        dtype=torch.float64
    )
    
    # Single input
    input_single = torch.tensor([1.0, 2.0], dtype=torch.float64)
    
    print(f"\n1. Input shape: {input_single.shape}")
    
    try:
        result = btn.forward_sigma({'features': input_single})
        print(f"\n2. Output shape: {result.shape}")
        print(f"   Output value: {result.item()}")
        print(f"   Expected: scalar, positive (variance)")
        
        if result.shape == torch.Size([]):
            if result.item() > 0:
                print("   ✓ Single sample produces positive scalar!")
                return True
            else:
                print(f"   ✗ Sigma not positive: {result.item()}")
                return False
        else:
            print(f"   ✗ Expected scalar, got shape {result.shape}")
            return False
            
    except Exception as e:
        print(f"\n   ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batched_sigma():
    """Test sigma forward with batched samples."""
    print("\n" + "="*70)
    print("TEST 4: Batched - Sigma Forward")
    print("="*70)
    
    # Same network
    A_data = np.random.randn(2, 3) * 0.1
    B_data = np.random.randn(3, 2) * 0.1
    
    A = qtn.Tensor(data=A_data, inds=('p1', 'r1'), tags='A')  # type: ignore
    B = qtn.Tensor(data=B_data, inds=('r1', 'p2'), tags='B')  # type: ignore
    
    mu_tn = qtn.TensorNetwork([A, B])
    sigma_tn = create_sigma_network(mu_tn, ['A', 'B'])
    
    btn = BayesianTensorNetwork(
        mu_tn=mu_tn,
        sigma_tn=sigma_tn,
        input_indices={'features': ['p1', 'p2']},
        learnable_tags=['A', 'B'],
        device=torch.device('cpu'),
        dtype=torch.float64
    )
    
    # Batched input
    batch_size = 5
    input_batch = torch.randn(batch_size, 2, dtype=torch.float64)
    
    print(f"\n1. Input shape: {input_batch.shape}")
    
    try:
        result = btn.forward_sigma({'features': input_batch})
        print(f"\n2. Output shape: {result.shape}")
        print(f"   Output values (first 3): {result[:3]}")
        print(f"   Expected shape: ({batch_size},)")
        print(f"   All positive? {(result > 0).all().item()}")
        
        if result.shape == torch.Size([batch_size]):
            if (result > 0).all():
                print("   ✓ Batched sigma produces correct shape and positive values!")
                return True
            else:
                print("   ✗ Some sigma values are not positive!")
                return False
        else:
            print(f"   ✗ Wrong shape!")
            return False
            
    except Exception as e:
        print(f"\n   ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_polynomial_features():
    """Test with polynomial features (same input, different indices)."""
    print("\n" + "="*70)
    print("TEST 5: Polynomial Features (Same Input, Multiple Indices)")
    print("="*70)
    
    # Create 3-block MPS like in BayesianMPO
    # Each block contracts with a different p index
    A_data = np.random.randn(2, 3) * 0.1  # (p1, r1)
    B_data = np.random.randn(3, 2, 4) * 0.1  # (r1, p2, r2)
    C_data = np.random.randn(4, 2) * 0.1  # (r2, p3)
    
    A = qtn.Tensor(data=A_data, inds=('p1', 'r1'), tags='A')  # type: ignore
    B = qtn.Tensor(data=B_data, inds=('r1', 'p2', 'r2'), tags='B')  # type: ignore
    C = qtn.Tensor(data=C_data, inds=('r2', 'p3'), tags='C')  # type: ignore
    
    mu_tn = qtn.TensorNetwork([A, B, C])
    sigma_tn = create_sigma_network(mu_tn, ['A', 'B', 'C'])
    
    # Same input contracts with p1, p2, p3 (polynomial expansion)
    btn = BayesianTensorNetwork(
        mu_tn=mu_tn,
        sigma_tn=sigma_tn,
        input_indices={'features': ['p1', 'p2', 'p3']},  # Same input, 3 indices!
        learnable_tags=['A', 'B', 'C'],
        device=torch.device('cpu'),
        dtype=torch.float64
    )
    
    print(f"\n1. Network: 3 blocks, each with different p index")
    print(f"   input_indices: {btn.input_indices}")
    print(f"   Same input data contracts with p1, p2, AND p3")
    
    # Test single sample
    input_single = torch.tensor([1.0, 2.0], dtype=torch.float64)
    
    try:
        mu_result = btn.forward_mu({'features': input_single})
        print(f"\n2. Single sample - Mu forward:")
        print(f"   Output shape: {mu_result.shape}")
        print(f"   Output value: {mu_result.item()}")
        
        # Test batch
        batch_size = 4
        input_batch = torch.randn(batch_size, 2, dtype=torch.float64)
        
        mu_batch = btn.forward_mu({'features': input_batch})
        print(f"\n3. Batched - Mu forward:")
        print(f"   Input shape: {input_batch.shape}")
        print(f"   Output shape: {mu_batch.shape}")
        print(f"   Expected: ({batch_size},)")
        
        if mu_batch.shape == torch.Size([batch_size]):
            print("   ✓ Polynomial features work correctly!")
            return True
        else:
            print("   ✗ Wrong shape!")
            return False
            
    except Exception as e:
        print(f"\n   ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING FORWARD PASS WITH INPUTS AND BATCHING")
    print("="*70)
    
    results = []
    results.append(("Single sample mu", test_single_sample_mu()))
    results.append(("Batched mu", test_batched_mu()))
    results.append(("Single sample sigma", test_single_sample_sigma()))
    results.append(("Batched sigma", test_batched_sigma()))
    results.append(("Polynomial features", test_polynomial_features()))
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:25s}: {status}")
    
    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED")
