#!/usr/bin/python3
import cv2
import numpy as np

def bgr_binarization(image, b_min=30, g_min=50, g_max=120, r_min=120):
    """Binarize the input image filtering the BGR values."""
    # Filter the pixels according to the BGR values.
    b_bin = (image[:, :, 0] < b_min).astype(np.uint8)
    g_bin = ((image[:, :, 1]>g_min) * (image[:, :, 1]<g_max)).astype(np.uint8)
    r_bin = (image[:, :, 2] > r_min).astype(np.uint8)
    binarized = b_bin * g_bin * r_bin
    return binarized

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
