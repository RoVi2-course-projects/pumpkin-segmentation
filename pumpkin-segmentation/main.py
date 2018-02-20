#!/usr/bin/python3
# Standard libraries
import numpy as np
import cv2
# Local libraries
import binarization

def main():
    """Main routine."""
    # Read photo
    image = cv2.imread("./photos/DJI_0237.JPG")
    # Binarize the image.
    binarized = binarization.hsv_binarization(image)
    cv2.imwrite("./photos/result.png", 255*binarized)
    return


if __name__ == "__main__":
    main()
