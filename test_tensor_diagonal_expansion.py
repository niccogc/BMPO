
import quimb as qu
import quimb.tensor as qtn
import numpy as np

def expand_all_indices_to_diagonal(tensor: qtn.Tensor) -> qtn.Tensor:
    """
    Expands all indices of a tensor into diagonal pairs.
    For each original index 'x', it creates a pair ('x', 'x_prime')
    such that the new tensor is non-zero only when x == x_prime for all pairs.
    """
    new_tensor = tensor.copy()
    original_inds = list(new_tensor.inds) # Get a copy of indices to iterate over

    for old_ind in original_inds:
        primed_ind = old_ind + '_prime'

        # Expand the original index into the desired diagonal pair
        # The original values are placed along the diagonal of (old_ind, primed_ind)
        new_tensor = new_tensor.new_ind_pair_diag(old_ind, old_ind, primed_ind)

    return new_tensor

def test_expand_all_indices_to_diagonal():
    # Test case 1: A simple 2D tensor
    data_2d = np.array([[1, 2], [3, 4]])
    tensor_2d = qtn.Tensor(data_2d, inds=('a', 'b'))

    expanded_tensor_2d = expand_all_indices_to_diagonal(tensor_2d)
    print(expanded_tensor_2d)
    expected_inds_2d = ('a', 'a_prime', 'b', 'b_prime')
    assert expanded_tensor_2d.inds == expected_inds_2d

    expected_shape_2d = (2, 2, 2, 2)
    assert expanded_tensor_2d.shape == expected_shape_2d

    # Convert to dense array for easier verification
    dense_expanded_2d = expanded_tensor_2d.to_dense(('a', 'b'),('a_prime', 'b_prime'))

    # Verify some values
    # Original: tensor_2d['a'=0, 'b'=0] = 1
    # Expected: dense_expanded_2d[0, 0, 0, 0] = 1
    print(dense_expanded_2d)

    # Test case 2: A 3D tensor
    data_3d = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    tensor_3d = qtn.Tensor(data_3d, inds=('x', 'y', 'z'))

    expanded_tensor_3d = expand_all_indices_to_diagonal(tensor_3d)

    expected_inds_3d = ('x', 'x_prime', 'y', 'y_prime', 'z', 'z_prime')
    assert expanded_tensor_3d.inds == expected_inds_3d

    expected_shape_3d = (2, 2, 3, 3, 4, 4)
    assert expanded_tensor_3d.shape == expected_shape_3d
    idx = ['x', 'y', 'z']
    primeidx = [i + '_prime' for i in idx]
    dense_expanded_3d = expanded_tensor_3d.to_dense(idx,primeidx)
    print(dense_expanded_3d)

    print("All tests passed!")

if __name__ == '__main__':
    test_expand_all_indices_to_diagonal()
