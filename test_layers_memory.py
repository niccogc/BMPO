"""
Test memory sharing in standard TensorTrainLayer from layers.py
"""
import torch
from tensor.layers import TensorTrainLayer

# Create a standard TensorTrainLayer
tt = TensorTrainLayer(num_carriages=3, bond_dim=4, input_features=5, output_shape=1, seed=42)

print(f'Number of input nodes: {len(tt.input_node_layer.nodes)}')
print(f'Input node shapes:')
for i, node in enumerate(tt.input_node_layer.nodes):
    print(f'  Node {i}: shape {node.tensor.shape}, labels {node.dim_labels}')

print(f'\nChecking memory sharing BEFORE forward:')
first_ptr = tt.input_node_layer.nodes[0].tensor.data_ptr()
for i, node in enumerate(tt.input_node_layer.nodes):
    shares = node.tensor.data_ptr() == first_ptr
    status = "✓ SHARES" if shares else "✗ DIFFERENT"
    print(f'  Node {i} data_ptr: {node.tensor.data_ptr()} - {status}')

# Test forward pass with shared input
print(f'\nTesting forward pass:')
x = torch.randn(10, 5)
output = tt.forward(x)
print(f'Output shape: {output.shape}')

print(f'\nAfter forward, checking memory sharing:')
first_ptr = tt.input_node_layer.nodes[0].tensor.data_ptr()
x_ptr = x.data_ptr()
print(f'Input x data_ptr: {x_ptr}')
for i, node in enumerate(tt.input_node_layer.nodes):
    shares_first = node.tensor.data_ptr() == first_ptr
    shares_x = node.tensor.data_ptr() == x_ptr
    print(f'  Node {i} data_ptr: {node.tensor.data_ptr()} - shares with node[0]: {shares_first}, shares with x: {shares_x}')
