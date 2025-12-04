import quimb.tensor as qtn
import numpy as np

dim = 5
# Tensors are 1D vectors with the same index 'y1'
T_A = qtn.Tensor(np.arange(dim), inds=('y1',), tags={'A'})
T_B = qtn.Tensor(np.ones(dim) * 2, inds=('y1',), tags={'B'})

tn = T_A & T_B

# Contraction: Explicitly keep 'y1' in the output
result_tensor = tn.contract(output_inds=['y1'])

# Manual Element-wise Check (Hadamard Product)
expected_output = T_A.data * T_B.data 

print(f"Result Tensor Data: {result_tensor.data}")
print(f"Manual Check: {expected_output}")

# The output indices remain 'y1' and the shape is (5,)
print(f"Indices: {result_tensor.inds}")
