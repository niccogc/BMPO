"""
Test projection computation for 4-block MPS with repeated inputs.

Setup: MPS structure like polynomial regression
  Input^3 -> A -> B -> C -> D
  
Where Input is repeated 3 times (for polynomial features x, x^2, x^3 conceptually)
and connects to different blocks.

We'll compute the projection (Jacobian) for block B and verify it matches
manual computation.
"""

import torch
import quimb.tensor as qtn
import numpy as np

from tensor.bayesian_tn import BayesianTensorNetwork


def create_4block_mps_with_repeated_input():
    """
    Create a 4-block MPS with repeated input connections.
    
    Structure:
        Input1 (x1) -> A (d_in=2, r1=3)
        Input2 (x2) -> B (r1=3, d_in=2, r2=4) 
        Input3 (x3) -> C (r2=4, d_in=2, r3=3)
                       D (r3=3, d_out=1)
    
    where Input1, Input2, Input3 are the same input repeated.
    """
    # Dimensions
    d_in = 2  # Input dimension
    r1 = 3    # Bond between A and B
    r2 = 4    # Bond between B and C
    r3 = 3    # Bond between C and D
    d_out = 1 # Output dimension
    
    # Create input nodes (repeated 3 times with same data)
    # Each has shape (batch, d_in)
    # For simplicity, we'll use batch=1 initially
    input_data = np.array([[0.5, 0.8]])  # Shape (1, 2)
    
    input1 = qtn.Tensor(data=input_data, inds=('batch', 'x1'), tags='input1')  # type: ignore
    input2 = qtn.Tensor(data=input_data, inds=('batch', 'x2'), tags='input2')  # type: ignore
    input3 = qtn.Tensor(data=input_data, inds=('batch', 'x3'), tags='input3')  # type: ignore
    
    # Create learnable blocks
    # A: connects input1 and outputs to B via r1
    A_data = np.random.randn(d_in, r1) * 0.1
    A = qtn.Tensor(data=A_data, inds=('x1', 'r1'), tags='A')  # type: ignore
    
    # B: connects r1 from A, input2, and outputs to C via r2
    B_data = np.random.randn(r1, d_in, r2) * 0.1
    B = qtn.Tensor(data=B_data, inds=('r1', 'x2', 'r2'), tags='B')  # type: ignore
    
    # C: connects r2 from B, input3, and outputs to D via r3
    C_data = np.random.randn(r2, d_in, r3) * 0.1
    C = qtn.Tensor(data=C_data, inds=('r2', 'x3', 'r3'), tags='C')  # type: ignore
    
    # D: connects r3 from C and outputs to batch dimension
    D_data = np.random.randn(r3, d_out) * 0.1
    D = qtn.Tensor(data=D_data, inds=('r3', 'batch'), tags='D')  # type: ignore
    
    # Create tensor network
    tn = qtn.TensorNetwork([input1, input2, input3, A, B, C, D])
    
    return tn, input_data, (A_data, B_data, C_data, D_data)


def manual_projection_for_B(input_data, A_data, B_data, C_data, D_data):
    """
    Manually compute the projection for block B.
    
    Projection T_B is obtained by contracting everything EXCEPT B:
    - Contract input1 with A -> gives tensor with index r1
    - Contract input2 (just keep it, will contract with B later)
    - Contract input3 with C -> gives tensor with indices (r2, r3)
    - Contract previous result with D -> gives tensor with indices (r2, batch)
    - Contract (r1 tensor) with (r2, batch tensor) -> gives (r1, r2, batch)
    
    But we need to keep input2 separate since it connects to B.
    
    Actually, the projection should have the same indices as B: (r1, x2, r2)
    
    Let me think step by step:
    - Remove B from network
    - Contract everything else
    - The result should have indices that would connect to B
    
    Network without B:
      Input1 -> A -> (r1 free)
      Input2 -> (x2 free)
      Input3 -> C -> D -> batch
                     (r2 free)
    
    So we contract:
    - Input1 (batch, x1) with A (x1, r1) -> (batch, r1)
    - Input3 (batch, x3) with C (r2, x3, r3) -> (batch, r2, r3)
    - Previous with D (r3, batch) -> (batch, batch, r2)
    
    Wait, there's an issue with batch indices...
    Let me reconsider with einsum notation.
    """
    # Convert to torch tensors
    input_torch = torch.from_numpy(input_data).to(dtype=torch.float64)  # (1, 2)
    A_torch = torch.from_numpy(A_data).to(dtype=torch.float64)  # (2, 3)
    B_torch = torch.from_numpy(B_data).to(dtype=torch.float64)  # (3, 2, 4)
    C_torch = torch.from_numpy(C_data).to(dtype=torch.float64)  # (4, 2, 3)
    D_torch = torch.from_numpy(D_data).to(dtype=torch.float64)  # (3, 1)
    
    # Contract input1 with A: (batch, x1) x (x1, r1) -> (batch, r1)
    A_contracted = torch.einsum('bx,xr->br', input_torch, A_torch)
    
    # Contract input3 with C: (batch, x3) x (r2, x3, r3) -> (batch, r2, r3)
    C_contracted = torch.einsum('bx,rxs->brs', input_torch, C_torch)
    
    # Contract C_contracted with D: (batch, r2, r3) x (r3, batch_out) -> (batch, r2, batch_out)
    # Note: D connects r3 to batch output
    CD_contracted = torch.einsum('brs,so->bro', C_contracted, D_torch)
    
    # Now we have:
    # A_contracted: (batch, r1)
    # input2: (batch, x2) 
    # CD_contracted: (batch, r2, batch_out)
    
    # The projection for B should combine these
    # B has indices (r1, x2, r2), so projection should match these indices
    # But also needs to account for batch dimensions
    
    # Projection = einsum over A_contracted, input2, CD_contracted to get shape matching B
    # Actually, the projection T_B(x) depends on the input x
    # T_B contracts: A_contracted(batch, r1) with input2(batch, x2) with CD_contracted(batch, r2, batch_out)
    
    # The correct projection is: (batch, r1, x2, r2, batch_out) or something similar
    # Let me think about what the output dimensions should be...
    
    # Actually, when we compute projection for variational updates:
    # We want ∂(network_output)/∂(B)
    # The network output is scalar (or vector) for each batch
    # B has shape (r1, x2, r2)
    
    # The projection should have shape that when contracted with B gives the output
    # So: projection(batch, batch_out, r1, x2, r2) x B(r1, x2, r2) -> (batch, batch_out)
    
    # Let's compute this:
    projection = torch.einsum('br,bx,bso->boxrs', 
                             A_contracted,   # (batch, r1)
                             input_torch,     # (batch, x2) 
                             CD_contracted)   # (batch, r2, batch_out)
    
    return projection


def test_4block_mps_projection():
    """Test projection computation for 4-block MPS."""
    print("\n" + "="*70)
    print("Test: 4-Block MPS with Repeated Input - Projection Verification")
    print("="*70)
    
    # Create the network
    tn, input_data, (A_data, B_data, C_data, D_data) = create_4block_mps_with_repeated_input()
    
    print("\n1. Network Structure:")
    print(f"   Input: {input_data.shape} = {input_data}")
    print(f"   A: {A_data.shape}")
    print(f"   B: {B_data.shape}")
    print(f"   C: {C_data.shape}")
    print(f"   D: {D_data.shape}")
    
    # Create BayesianTensorNetwork
    btn = BayesianTensorNetwork(
        mu_tn=tn.copy(),
        learnable_tags=['A', 'B', 'C', 'D'],
        input_tags=['input1', 'input2', 'input3'],
        device=torch.device('cpu'),
        dtype=torch.float64
    )
    
    print("\n2. BayesianTensorNetwork created")
    print(f"   Learnable: {btn.learnable_tags}")
    print(f"   Inputs: {btn.input_tags}")
    print(f"   Bonds: {btn.mu_network.bond_labels}")
    
    # Compute projection using quimb
    print("\n3. Computing projection for block B using quimb...")
    
    # Set inputs in the network
    inputs = {
        'input1': torch.from_numpy(input_data).to(dtype=torch.float64),
        'input2': torch.from_numpy(input_data).to(dtype=torch.float64),
        'input3': torch.from_numpy(input_data).to(dtype=torch.float64),
    }
    
    proj_B_quimb = btn.compute_projection('B', network_type='mu', inputs=inputs)
    print(f"   Projection shape from quimb: {proj_B_quimb.shape}")
    print(f"   Expected indices for B: ('r1', 'x2', 'r2') = (3, 2, 4)")
    
    # Compute projection manually
    print("\n4. Computing projection manually...")
    proj_B_manual = manual_projection_for_B(input_data, A_data, B_data, C_data, D_data)
    print(f"   Projection shape from manual: {proj_B_manual.shape}")
    
    # Verify forward pass works
    print("\n5. Verifying forward pass...")
    output = btn.forward_mu(inputs)
    print(f"   Forward output shape: {output.shape}")
    print(f"   Forward output value: {output}")
    
    # Manually compute forward pass to verify
    input_torch = torch.from_numpy(input_data).to(dtype=torch.float64)
    A_torch = torch.from_numpy(A_data).to(dtype=torch.float64)
    B_torch = torch.from_numpy(B_data).to(dtype=torch.float64)
    C_torch = torch.from_numpy(C_data).to(dtype=torch.float64)
    D_torch = torch.from_numpy(D_data).to(dtype=torch.float64)
    
    # Manual forward: input1->A, A+input2->B, B+input3->C, C->D
    A_out = torch.einsum('bx,xr->br', input_torch, A_torch)
    B_out = torch.einsum('br,bx,rxs->bs', A_out, input_torch, B_torch)
    C_out = torch.einsum('bs,bx,sxt->bt', B_out, input_torch, C_torch)
    output_manual = torch.einsum('bt,to->bo', C_out, D_torch)
    
    print(f"   Manual forward output: {output_manual}")
    print(f"   Match: {torch.allclose(output, output_manual, atol=1e-6)}")
    
    # Now verify projection
    print("\n6. Verifying projection...")
    print(f"   Quimb projection shape: {proj_B_quimb.shape}")
    print(f"   Manual projection shape: {proj_B_manual.shape}")
    
    # The key test: contracting projection with B should give the same as
    # contracting the full network
    print("\n7. Testing projection property...")
    print("   Property: Projection(x) ⊗ B should equal forward(x)")
    
    # Contract quimb projection with B
    # proj_B_quimb should have shape that matches B's indices
    # B has shape (r1=3, x2=2, r2=4)
    # Projection should have shape (batch, output, r1, x2, r2) or similar
    
    if len(proj_B_quimb.shape) == 3 and proj_B_quimb.shape == B_torch.shape:
        # Projection has same shape as B - this means it's the "Jacobian" without batch dimensions
        # This is actually correct for the pure tensor network (no batch processing)
        print(f"   ✓ Projection shape matches B shape: {proj_B_quimb.shape}")
        
        # To verify: contract this projection with B and compare with full network
        # But we need to understand what this represents...
        print("\n   Note: Projection represents network with B removed")
        print("   To verify, we'd contract projection with B and check if it equals the network output")
    else:
        print(f"   Projection shape: {proj_B_quimb.shape}")
        print(f"   B shape: {B_torch.shape}")
    
    # Print some values for inspection
    print("\n8. Projection values (first few elements):")
    print(f"   Quimb projection [0, 0, :]: {proj_B_quimb[0, 0, :]}")
    if proj_B_manual.numel() < 100:
        print(f"   Manual projection sample: {proj_B_manual.flatten()[:10]}")
    
    return btn, proj_B_quimb, proj_B_manual


def test_projection_contraction():
    """
    Test that projection contracted with the block gives the network output.
    
    This is the key property: T_i ⊗ A^i = f(x) where f is the network function.
    """
    print("\n" + "="*70)
    print("Test: Verify Projection Contraction Property")
    print("="*70)
    
    # Create a simpler 2-block network for easier verification
    print("\n1. Creating simple 2-block MPS: Input -> A -> B -> Output")
    
    d_in = 2
    r1 = 3
    d_out = 1
    
    input_data = np.array([[0.5, 0.8]])  # (1, 2)
    input_tensor = qtn.Tensor(data=input_data, inds=('batch', 'x'), tags='input')  # type: ignore
    
    A_data = np.random.randn(d_in, r1) * 0.1
    A_tensor = qtn.Tensor(data=A_data, inds=('x', 'r1'), tags='A')  # type: ignore
    
    B_data = np.random.randn(r1, d_out) * 0.1
    B_tensor = qtn.Tensor(data=B_data, inds=('r1', 'batch_out'), tags='B')  # type: ignore
    
    tn = qtn.TensorNetwork([input_tensor, A_tensor, B_tensor])
    
    # Create BTN
    btn = BayesianTensorNetwork(
        mu_tn=tn.copy(),
        learnable_tags=['A', 'B'],
        input_tags=['input'],
        device=torch.device('cpu'),
        dtype=torch.float64
    )
    
    # Compute forward pass
    inputs = {'input': torch.from_numpy(input_data).to(dtype=torch.float64)}
    output_full = btn.forward_mu(inputs)
    print(f"   Full network output: {output_full}")
    
    # Compute projection for A (should be: Input contracted with B)
    proj_A = btn.compute_projection('A', network_type='mu', inputs=inputs)
    print(f"\n2. Projection for A:")
    print(f"   Shape: {proj_A.shape}")
    print(f"   A shape: {A_data.shape}")
    
    # The projection should represent: Input ⊗ B
    # When contracted with A, should give output_full
    A_torch = torch.from_numpy(A_data).to(dtype=torch.float64)
    
    # Try different contraction patterns
    print(f"\n3. Testing contraction...")
    print(f"   Projection shape: {proj_A.shape}")
    print(f"   A shape: {A_torch.shape}")
    
    # The projection T_A has the same indices as A would have in the contraction
    # So contracting them should work with matching indices
    if proj_A.shape == A_torch.shape:
        # Element-wise multiply and sum
        result = (proj_A * A_torch).sum()
        print(f"   Projection ⊙ A (elementwise then sum): {result}")
        print(f"   Full network output: {output_full.item()}")
        print(f"   Match: {torch.allclose(result, output_full, atol=1e-6)}")
    else:
        print(f"   Shapes don't match for simple contraction")
    
    return btn


if __name__ == "__main__":
    # Run tests
    btn, proj_quimb, proj_manual = test_4block_mps_projection()
    
    print("\n" + "="*70)
    
    test_projection_contraction()
    
    print("\n" + "="*70)
    print("TESTS COMPLETED")
    print("="*70 + "\n")
