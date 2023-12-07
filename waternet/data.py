import cv2
import numpy as np
from typing import Tuple


def white_balance_transform(im_rgb):
    """
    Requires HWC uint8 input
    Originally in SimplestColorBalance.m
    """
    # This section basically reshapes into vectors per channel I think?

    # if RGB
    if len(im_rgb.shape) == 3:
        R = np.sum(im_rgb[:, :, 0], axis=None)
        G = np.sum(im_rgb[:, :, 1], axis=None)
        B = np.sum(im_rgb[:, :, 2], axis=None)

        maxpix = max(R, G, B)
        ratio = np.array([maxpix / R, maxpix / G, maxpix / B])

        satLevel1 = 0.005 * ratio
        satLevel2 = 0.005 * ratio

        m, n, p = im_rgb.shape
        im_rgb_flat = np.zeros(shape=(p, m * n))
        for i in range(0, p):
            im_rgb_flat[i, :] = np.reshape(im_rgb[:, :, i], (1, m * n))

    # if grayscale
    else:
        satLevel1 = np.array([0.001])
        satLevel2 = np.array([0.005])
        m, n = im_rgb.shape
        p = 1
        im_rgb_flat = np.reshape(im_rgb, (1, m * n))

    # Initialize an array for the white balanced image
    wb = np.zeros(shape=im_rgb_flat.shape)

    # Apply white balance to each color channel
    for ch in range(p):
        # Calculate the quantiles for the current color channel
        q = [satLevel1[ch], 1 - satLevel2[ch]]
        tiles = np.quantile(im_rgb_flat[ch, :], q)

        # Apply white balance transformation
        temp = im_rgb_flat[ch, :]
        temp[temp < tiles[0]] = tiles[0]
        temp[temp > tiles[1]] = tiles[1]
        wb[ch, :] = temp

        # Normalize the color channel to the range [0, 255]
        bottom = min(wb[ch, :])
        top = max(wb[ch, :])
        if top - bottom > 0:
            wb[ch, :] = (wb[ch, :] - bottom) * 255 / (top - bottom)
        else:
            wb[ch, :] = 0

    # Reshape the white balanced image back to the original shape
    if len(im_rgb.shape) == 3:
        outval = np.zeros(shape=im_rgb.shape)
        for i in range(p):
            outval[:, :, i] = np.reshape(wb[i, :], (m, n))

    else:
        outval = np.reshape(wb, (m, n))

    return outval.astype(np.uint8)


def gamma_correction(im):
    gc = np.power(im / 255, 0.7)
    gc = np.clip(255 * gc, 0, 255)
    gc = gc.astype(np.uint8)
    return gc


def histeq(im_rgb):
    # Convert the image from RGB to LAB color space
    im_lab = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2LAB)

    # Create a CLAHE (Contrast Limited Adaptive Histogram Equalization) object
    clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(8, 8))

    # Apply the CLAHE to the L channel of the LAB image
    el = clahe.apply(im_lab[:, :, 0])

    # Copy the original LAB image
    im_he = im_lab.copy()

    # Replace the L channel with the histogram equalized version
    im_he[:, :, 0] = el

    # Convert the image back to RGB color space
    im_he_rgb = cv2.cvtColor(im_he, cv2.COLOR_LAB2RGB)

    # Return the histogram equalized image
    return im_he_rgb


def transform(rgb) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    transform(rgb) -> wb, gc, he
    """
    # Convenience wrapper
    wb = white_balance_transform(rgb)
    gc = gamma_correction(rgb)
    he = histeq(rgb)

    return wb, gc, he
