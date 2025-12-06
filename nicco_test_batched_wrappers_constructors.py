import numpy as np
from tensor.builder import Inputs 
from tensor.btn import BTN
import quimb.tensor as qt
# ==========================================
# SETUP: Generate Fake Data (100 samples)
# ==========================================
N = 1000
x1 = np.random.randn(N, 4)
x2 = np.random.randn(N, 4)
x3 = np.random.randn(N, 4)
y = np.random.randn(N, 2,3) # Standard vector target

print(">>> RUNNING INPUT TESTS\n")

# ==========================================
# TEST 1: Standard (2 Inputs -> 2 Labels)
# ==========================================
print("--- Test 1: Standard Inputs ---")
loader = Inputs(
    inputs=[x1, x2, x3], 
    outputs=[y], 
    outputs_labels=["y1", "y2"], 
    input_labels=["x1", "x2", "x3"], 
    batch_size=32
)


node1 = qt.Tensor(np.random.randn(4, 3, 5), inds=('x1', 'r1', 'r3'), tags={'node1'})
node2 = qt.Tensor(np.random.randn(4, 2, 3, 3), inds=('x2', 'r2', 'r1', 'y2'), tags={'node2'})
node3 = qt.Tensor(np.random.randn(4, 2, 5, 2), inds=('x3', 'r2', 'r3','y1'), tags={'node3'})

mu = qt.TensorNetwork([node1, node2, node3])

model = BTN(mu, loader)
print("="*50)
print('\n Loader')
print(loader)
print("="*50)
generator = model.data
forward_pass_mu = model.forward(model.mu, model.data.data_mu, True, True)
print(forward_pass_mu)
print(" WOrkzz forward mu")
print("="*50)
forward_pass_sigma= model.forward(model.sigma, model.data.data_sigma, True, True)
print(forward_pass_sigma)
print(" WOrkzz forward sigma")
tag = 'node2'
tn = model.mu
env_mu = model.get_environment(tn, tag, generator.data_mu, True, True, True)
print(tn[tag].inds)
print(env_mu.inds)

print("="*10)
outer_op = model.outer_operation(tn=tn, input_generator= generator.data_mu, node_tag=tag, sum_over_batches=True)
print(outer_op)
print("="*10)
print(model.get_tau_mean())
print("=*="*20)
print("la precision")
precision = model.compute_precision_node(tag)
print(precision)
