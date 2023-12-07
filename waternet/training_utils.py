import torch
import albumentations as A
import cv2
from pathlib import Path
import os

from waternet.data import transform as preprocess_transform
from typing import Optional
import numpy as np
from configs.constants import contrastive_pairs

def ten2arr_noeinops(ten):
    """Convert NCHW torch Tensor into NHWC numpy array:
    1. Multiply by 255, clip and change dtype to unsigned int
    2. Rearrange dims: NCHW -> NHWC
    """
    arr = ten.cpu().detach().numpy()
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    # arr = rearrange(arr, "n c h w -> n h w c")
    arr = np.transpose(arr, (0, 2, 3, 1))
    return arr

def arr2ten(arr):
    """Converts (N)HWC numpy array into torch Tensor:
    1. Divide by 255
    2. Rearrange dims: HWC -> CHW or NHWC -> NCHW
    """
    ten = torch.from_numpy(arr) / 255
    if len(ten.shape) == 3:
        # ten = rearrange(ten, "h w c -> 1 c h w")
        ten = torch.permute(ten, (2, 0, 1))

    elif len(ten.shape) == 4:
        # ten = rearrange(ten, "n h w c -> n c h w")
        ten = torch.permute(ten, (0, 3, 1, 2))
    return ten


def ten2arr(ten):
    """Convert NCHW torch Tensor into NHWC numpy array:
    1. Multiply by 255, clip and change dtype to unsigned int
    2. Rearrange dims: CHW -> HWC or NCHW -> NHWC
    """
    arr = ten.cpu().detach().numpy()
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)

    if len(arr.shape) == 3:
        # arr = rearrange(arr, "c h w -> h w c")
        arr = np.transpose(arr, (1, 2, 0))
    elif len(arr.shape) == 4:
        # arr = rearrange(arr, "n c h w -> n h w c")
        arr = np.transpose(arr, (0, 2, 3, 1))

    return arr


def get_pairs(raw_row, ref_row):
    # selected_pairs = []
    input_text = ""
    ref_scores = []
    for pair in contrastive_pairs:
        if pair[0] not in raw_row.index:
            # print(f"Pair {pair[0]} not found in raw row. Skipping...")
            continue
        if pair[0] not in ref_row.index:
            # print(f"Pair {pair[0]} not found in ref row. Skipping...")
            continue
        raw_positive_score = raw_row[pair[0]] # good score
        ref_positive_score = ref_row[pair[0]] # good score

        ref_scores.append(ref_positive_score)

        if raw_positive_score < ref_positive_score:
            # selected_pairs.append([pair[0], ref_positive_score - raw_positive_score])
            # limit to 2 decimal places
            input_text += f"{pair[0]} {ref_positive_score:.1f}, "

    return input_text, ref_scores



class UIEBDataset(torch.utils.data.Dataset):
    """UIEBDataset."""

    def __init__(
        self,
        raw_df,
        ref_df,
        im_height: Optional[int] = None,
        im_width: Optional[int] = None,
        transform=None,
    ):
        """
        legacy=True to replicate the paper's parameters
        """
        assert raw_df.shape[0] == ref_df.shape[0], "Raw and reference image counts do not match!"

        if transform is not None:
            self.transform = transform
        else:
            # No legacy augmentations
            # Paper uses flipping and rotation transforms to obtain 7 augmented versions of data
            # Rotate by 90, 180, 270 degs, hflip, vflip? Not very clear
            # This is as close as it gets without having to go out of my way to reproduce exactly 7 augmented versions
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                ]
            )
        self.raw_df = raw_df
        self.ref_df = ref_df
        self.im_height = im_height
        self.im_width = im_width

    def __len__(self):
        return self.raw_df.shape[0]

    def __getitem__(self, idx):
        # Load image
        raw_row = self.raw_df.iloc[idx]
        ref_row = self.ref_df.iloc[idx]
        raw_im = cv2.imread(raw_row["raw_path"])
        ref_im = cv2.imread(ref_row["reference_path"])
        

        if (self.im_width is not None) and (self.im_height is not None):
            # Resize accordingly
            raw_im = cv2.resize(raw_im, (self.im_width, self.im_height))
            ref_im = cv2.resize(ref_im, (self.im_width, self.im_height))
        else:
            # Else resize image to be mult of VGG, required by VGG
            im_w, im_h = raw_im.shape[0], raw_im.shape[1]
            vgg_im_w, vgg_im_h = int(im_w / 32) * 32, int(im_h / 32) * 32
            raw_im = cv2.resize(raw_im, (vgg_im_w, vgg_im_h))
            ref_im = cv2.resize(ref_im, (vgg_im_w, vgg_im_h))

        # Convert BGR to RGB for OpenCV
        raw_im = cv2.cvtColor(raw_im, cv2.COLOR_BGR2RGB)
        ref_im = cv2.cvtColor(ref_im, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            transformed = self.transform(image=raw_im, mask=ref_im)
            raw_im, ref_im = transformed["image"], transformed["mask"]
        else:
            pass

        # Preprocessing transforms
        wb, gc, he = preprocess_transform(raw_im)

        # Scale to 0 - 1 float, convert to torch Tensor
        raw_ten = arr2ten(raw_im)
        wb_ten = arr2ten(wb)
        gc_ten = arr2ten(gc)
        he_ten = arr2ten(he)
        ref_ten = arr2ten(ref_im)

        # Was gonna make this a tuple until I realized how confused future me would be
        return {
            "raw": raw_ten,
            "wb": wb_ten,
            "gc": gc_ten,
            "he": he_ten,
            "ref": ref_ten,
        }
