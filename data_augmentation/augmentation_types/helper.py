import torch
import numpy as np
# from more_itertools import grouper
# from torchvision import tv_tensors
# from torchvision.transforms import v2
# from torchvision.transforms.v2 import functional as F

def to_tensor(poly):
    """Convert polygon (list/ndarray/tensor) to float32 torch tensor shape (N,2)."""
    if isinstance(poly, torch.Tensor):
        return poly.to(torch.float32)
    padded_poly = padding(poly)
    arr = np.asarray(padded_poly, dtype=np.float32)
    return torch.from_numpy(arr)

# Temporary function
def padding(poly):
    # Step A: Find the maximum number of items (e.g., max bounding boxes)
    max_items = max(len(row) for row in poly) # max_items is 3
    fixed_dims = 2 # The dimension of each item (e.g., [xn, yn])

    # Step B: Pad the lists to the maximum length
    padded_list = []
    for row in poly:
        # Calculate how many padding items are needed
        padding_needed = max_items - len(row)
        # Create the padding items (a list of [0, 0])
        padding_item = [0] * fixed_dims
        padding = [padding_item] * padding_needed

        # Append the padding to the original items
        padded_list.append(row + padding)
    return padded_list