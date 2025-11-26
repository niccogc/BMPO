# Input Node Memory Sharing in BMPO

## Summary

Input nodes in tensor networks (both standard TensorTrain and Bayesian MPO) **do NOT share memory at construction time**. Instead, memory sharing happens **dynamically during forward pass** via the `set_input()` method.

## How It Works

### 1. Construction Time
When nodes are created (e.g., in `bayesian_mpo_builder.py` or `layers.py`):
```python
# Each input node gets its own tensor
input_node_1 = TensorNode((1, features), ['s', 'p1'])  # Own memory
input_node_2 = TensorNode((1, features), ['s', 'p2'])  # Different memory  
input_node_3 = TensorNode((1, features), ['s', 'p3'])  # Different memory
```

At this point, all nodes have **different data pointers**.

### 2. Forward Pass
When `forward(x)` is called, it internally calls `set_input(x)`:

```python
def set_input(self, x):
    """Sets the input tensor for the network."""
    if isinstance(x, tuple) or isinstance(x, list):
        # Different tensor for each node
        for node, tensor in zip(self.input_nodes, x):
            node.set_tensor(tensor)
    else:
        # SAME tensor for ALL nodes (memory sharing!)
        for node in self.input_nodes:
            node.set_tensor(x)
```

After `set_input(x)` is called with a single tensor `x`, **all input nodes point to the same tensor**, sharing memory.

## Test Results

### Before Forward Pass
```
Node 0 data_ptr: 93825093215936 - ✓ SHARES (with itself)
Node 1 data_ptr: 93825094823744 - ✗ DIFFERENT  
Node 2 data_ptr: 93825094000704 - ✗ DIFFERENT
```

### After Forward Pass  
```
Input x data_ptr: 93825085797568
Node 0 data_ptr: 93825085797568 - shares with x: True
Node 1 data_ptr: 93825085797568 - shares with x: True
Node 2 data_ptr: 93825085797568 - shares with x: True
✓ All input nodes share memory with input x!
```

## Implications

1. **Efficiency**: Only one copy of input data exists in memory during forward pass
2. **Correctness**: All input nodes see the same data, as required for tensor contraction
3. **Σ-MPO**: The same mechanism works for Σ-MPO input nodes (both 'o' and 'i' variants)

## Conclusion

The current implementation is **CORRECT**. Input nodes are created independently but share memory dynamically during forward pass. This is the intended behavior and matches how `layers.py` works.

The previous concern about "input nodes NOT sharing memory" was based on checking node memory **before** the forward pass, which is expected behavior.
