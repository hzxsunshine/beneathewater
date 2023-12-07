import torch
import numpy as np

from waternet.data import transform as preprocess_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def arr2ten_noeinops(arr):
    """Converts (N)HWC numpy array into torch Tensor:
    1. Divide by 255
    2. Rearrange dims: HWC -> 1CHW or NHWC -> NCHW
    """
    ten = torch.from_numpy(arr) / 255
    if len(ten.shape) == 3:
        # ten = rearrange(ten, "h w c -> 1 c h w")
        ten = torch.permute(ten, (2, 0, 1))
        ten = torch.unsqueeze(ten, dim=0)
    elif len(ten.shape) == 4:
        # ten = rearrange(ten, "n h w c -> n c h w")
        ten = torch.permute(ten, (0, 3, 1, 2))
    return ten

def pre_process(rgb_arr, ref):
    wb, gc, he = preprocess_transform(rgb_arr)
    rgb_ten = arr2ten_noeinops(rgb_arr)
    wb_ten = arr2ten_noeinops(wb)
    gc_ten = arr2ten_noeinops(gc)
    he_ten = arr2ten_noeinops(he)
    ref_ten = arr2ten_noeinops(ref)
    return rgb_ten, wb_ten, he_ten, gc_ten, ref_ten
    
def post_process(ten):
    arr = ten.cpu().detach().numpy()
    arr = np.clip(arr, 0, 1)
    # arr = arr - np.min(arr)
    # arr = arr / np.max(arr)
    arr = (arr * 255).astype(np.uint8)
    # arr = rearrange(arr, "n c h w -> n h w c")
    arr = np.transpose(arr, (0, 2, 3, 1))
    return arr

def process(img, waternet):
    rgb_ten, wb_ten, he_ten, gc_ten, _ = pre_process(img, img)
    rgb_ten, wb_ten, he_ten, gc_ten = rgb_ten.to(device), wb_ten.to(device), he_ten.to(device), gc_ten.to(device)
    out_ten = waternet(rgb_ten, wb_ten, he_ten, gc_ten)
    return post_process(out_ten)[0]