#!/usr/bin/python3
# Standard libraries
import numpy as np
import cv2
# Local libraries
import binarization
import get_mean_std_dev


def main():
    """Main routine."""
    # Read photo
    training_image_path = "./photos/DJI_0237_copy_better.png"
    image_path = "./photos/DJI_0237.JPG"
    image = cv2.imread(image_path)
    # Get the statistics of the pumpkins.
    stats = get_mean_std_dev.get_mean_st_dev(training_image_path, image_path)
    h_thresholds = get_mean_std_dev.get_thresholds(stats[1][0], stats[3][0])
    s_thresholds = get_mean_std_dev.get_thresholds(stats[1][1], stats[3][1])
    v_thresholds = get_mean_std_dev.get_thresholds(stats[1][2], stats[3][2])
    # Binarize the image.
    binarized = binarization.hsv_binarization(image, h_thresholds, s_thresholds,
                                              v_thresholds)
    cv2.imwrite("./photos/result.png", 255*binarized)
    return


if __name__ == "__main__":
    main()
