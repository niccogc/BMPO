import numpy as np
from tensor.builder import Inputs 
from tensor.btn import BTN
import quimb.tensor as qt
import torch
test = qt.Tensor(, inds = ["s, x1"])
torch.set_default_dtype(torch.float64)   # or torch.float64
# ==========================================
# SETUP: Generate Fake Data (100 samples)
# ==========================================
<<<<<<< HEAD
N = 1000
x1 = np.random.randn(N, 3)
x2 = np.random.randn(N, 3)
x3 = np.random.randn(N, 3)
=======
N = 500
x1 = torch.rand(N, 1)
x2 = torch.rand(N, 1)
# x3 = torch.randn(N, 3)
>>>>>>> quimb

y = (3*x1).sum(axis=1, keepdims=True)
# Add a column of ones
x1 = torch.cat([x1, torch.ones((N, 1))], dim = -1)
x2 = torch.cat([x2, torch.ones((N, 1))], dim = -1)
# y keeps shape (N, 1) as product over the original + ones
print(">>> RUNNING INPUT TESTS\n")

# ==========================================
# TEST 1: Standard (2 Inputs -> 2 Labels)
# ==========================================
print("--- Test 1: Standard Inputs ---")
loader = Inputs(
    inputs=[x1, x2], 
    outputs=[y], 
    outputs_labels=["y1"], 
    input_labels=["x2", "x3"], 
    batch_size=32
)

def norm(shape):
    rand = torch.randn(*shape)
    return rand/torch.norm(rand)

# node1 = qt.Tensor(norm((4, 3)), inds=('x1', 'r2'), tags={'node1'})
# node2 = qt.Tensor(norm((4, 3, 5)), inds=('x2', 'r2', 'r3'), tags={'node2'})
node2 = qt.Tensor(norm((2, 3)), inds=('x2','r1'), tags={'node2'})
node3 = qt.Tensor(norm((2, 3, 1)), inds=('x3','r1','y1'), tags={'node3'})
mu = qt.TensorNetwork([node2, node3])

model = BTN(mu, loader)
print("="*50)
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
# tag = 'node1'
# tn = model.mu
# env_mu = model.get_environment(tn, tag, generator.data_mu, True, True, True)
# print(tn[tag].inds)
# print(env_mu.inds)
# print("="*10)
# outer_op = model.outer_operation(tn=tn, input_generator= generator.data_mu, node_tag=tag, sum_over_batches=True)
# print(outer_op)
# print("="*40)
# print(model.get_tau_mean())
# precision = model.compute_precision_node(tag)
# print("=*=*"*20)
# print("la precision")
# print(precision.inds)
# print("=*=*"*20)
# print(model.sigma[tag].inds)
# print("="*40)
# previous_sigma = model.sigma[tag].copy()
# model.update_sigma_node(tag)
# delta = model.sigma[tag] - previous_sigma
# print("piccino piccino updatino")
# print("Pre update")
# print(previous_sigma.inds)
# print("Post update")
# print(model.sigma[tag].inds)
# print("update_value")
# print(delta.data.sum())
# print("="*40)
# previous_mu = model.mu[tag].copy()
# model.update_mu_node(tag)
# delta = model.mu[tag] - previous_mu
# print("Pre update")
# print(previous_mu.inds)
# print("Post update")
# print(model.mu[tag].inds)
# print("update_value")
# print(delta.data.sum())
# print("="*10)
# bond_tag = 'r2'
# print("==="*10)
# print("="*20)
# concentration_pre = model.q_bonds[bond_tag].concentration.copy()
# rate_pre = model.q_bonds[bond_tag].rate.copy()
# model.update_bond(bond_tag)
# concentration_post = model.q_bonds[bond_tag].concentration.copy()
# delta = concentration_post - concentration_pre
# rate_post = model.q_bonds[bond_tag].rate.copy()
# delta_rate = rate_post - rate_pre
# print("DELTA CONCENTRATION")
# print(delta.data)
# print("DELTA RATE")
# print(delta_rate.data)
# print("==="*10)
# tau_c = model.q_tau.concentration
# tau_rate = model.q_tau.rate
# model.update_tau()
# up_tau_c = model.q_tau.concentration
# up_tau_rate = model.q_tau.rate
# delta_c, delta_r = tau_c - up_tau_c, tau_rate - up_tau_rate
# print("DELTA TAU CONCENTRATION")
# print(-delta_c)
# print("DELTA TAU RATE")
# print(-delta_r)
# print(model.mu.num_tensors)

model.fit(15)

forward_pass_mu = model.forward(model.mu, model.data.data_mu, False, True)
# print(forward_pass_mu)
