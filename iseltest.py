import quimb.tensor as qtn
import torch

def test_isel_behavior():
    print("--- 1. Setup ---")
    # Create a Tensor with torch backend
    # Shape (2, 6, 6), indices ('x', 'bond', 'y')
    data = torch.randn(2, 6, 6)
    T = qtn.Tensor(data=data, inds=('x', 'bond', 'y'), tags={'TEST'})
    
    print(f"Original: Shape={T.data.shape}, Inds={T.inds}")
    
    # INDICES WE WANT TO KEEP
    # We want to keep indices 0, 2, 4 from the 'bond' dimension (axis 1)
    indices_list = [0, 2, 4]

    print("\n--- 2. Testing isel with LIST (Fancy Indexing) ---")
    try:
        # Clone to avoid messing up original
        T_list = T.copy()
        # This is the line failing in your code
        T_list.isel({'bond': indices_list}, inplace=True)
        
        print(f"SUCCESS (List): Shape={T_list.data.shape}, Inds={T_list.inds}")
        if 'bond' not in T_list.inds:
             print(">> WARNING: 'bond' label was REMOVED (Collapsed)!")
        else:
             print(">> OK: 'bond' label preserved.")
             
    except Exception as e:
        print(f"FAIL (List): {e}")

    print("\n--- 3. Testing isel with SLICE ---")
    try:
        T_slice = T.copy()
        # Slicing keeps the range 0-2
        T_slice.isel({'bond': slice(0, 3)}, inplace=True)
        
        print(f"SUCCESS (Slice): Shape={T_slice.data.shape}, Inds={T_slice.inds}")
    except Exception as e:
        print(f"FAIL (Slice): {e}")

    print("\n--- 4. Testing Manual Workaround (The 'Bypass') ---")
    try:
        T_manual = T.copy()
        axis = T_manual.inds.index('bond')
        
        # Manual Fancy Indexing on Data
        # [:, indices_list, :]
        slices = [slice(None)] * T_manual.data.ndim
        slices[axis] = indices_list
        new_data = T_manual.data[tuple(slices)]
        
        # Modify data directly
        T_manual.modify(data=new_data)
        
        print(f"SUCCESS (Manual): Shape={T_manual.data.shape}, Inds={T_manual.inds}")
    except Exception as e:
        print(f"FAIL (Manual): {e}")

if __name__ == "__main__":
    test_isel_behavior()
