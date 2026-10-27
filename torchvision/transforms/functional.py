"""
Minimal torchvision.transforms.functional stub compatible with transformers.
"""


def pil_to_tensor(pic):
    import torch
    import numpy as np

    if hasattr(pic, "__array__"):
        arr = np.array(pic)
        return torch.from_numpy(arr)
    raise NotImplementedError("pil_to_tensor only supports array-like objects in this stub")
