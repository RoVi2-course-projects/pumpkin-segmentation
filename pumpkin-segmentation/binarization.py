#!/usr/bin/python3
import cv2
import numpy as np

def hsv_binarization(image, h_max=20, s_min=120, v_min=110):
    """Convert image to HSV and binarize it."""
    # Convert from RGB to HSV format.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Filter the pixels according to the HSV values.
    h_bin = (hsv[:, :, 0] < h_max).astype(np.uint8)
    s_bin = (hsv[:, :, 1] > s_min).astype(np.uint8)
    v_bin = (hsv[:, :, 2] > v_min).astype(np.uint8)
    binarized = h_bin * s_bin * v_bin
    return binarized
